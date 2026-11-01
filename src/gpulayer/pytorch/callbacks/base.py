"""The ``Callback`` base class: hooks into the training loop, named as in PyTorch Lightning.

Hooks run on every rank. Epoch metrics are averaged across ranks before ``on_train_epoch_end``,
so callbacks that decide from them (like ``EarlyStopping``) agree; the Trainer also combines
``trainer.should_stop`` across ranks, so one rank asking to stop stops all of them.

Only hooks a callback overrides are called; the others cost nothing per step.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from gpulayer.pytorch.module import Module
    from gpulayer.pytorch.trainer import Trainer


class Callback:
    """Subclass and override the hooks you need. All hooks are optional."""

    @property
    def state_key(self) -> str:
        """Key of this callback's ``state_dict`` in checkpoints."""
        return type(self).__qualname__

    def state_dict(self) -> dict[str, Any]:
        """State saved in checkpoints and restored when resuming."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore what ``state_dict`` returned, when resuming from a checkpoint."""

    def setup(self, trainer: Trainer, pl_module: Module, stage: str) -> None: ...
    def teardown(self, trainer: Trainer, pl_module: Module, stage: str) -> None: ...
    def on_fit_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_fit_end(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_sanity_check_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_sanity_check_end(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_train_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_train_end(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None: ...

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        """After training and validation of an epoch; ``trainer.callback_metrics`` is set."""

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: Module, batch: Any, batch_idx: int
    ) -> None: ...

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """After each training batch. ``outputs`` is ``{"loss": loss}`` (or the dict that
        ``training_step`` returned). Callbacks written for GPUlayer 0.7 with the signature
        ``(trainer, model, loss, batch_idx)`` still work."""

    def on_validation_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_validation_end(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Module) -> None: ...

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: Module,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None: ...

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None: ...

    def on_test_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_test_end(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_test_epoch_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_test_epoch_end(self, trainer: Trainer, pl_module: Module) -> None: ...

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: Module,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None: ...

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None: ...

    def on_predict_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_predict_end(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_predict_epoch_start(self, trainer: Trainer, pl_module: Module) -> None: ...
    def on_predict_epoch_end(self, trainer: Trainer, pl_module: Module) -> None: ...

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: Module,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None: ...

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None: ...

    def on_before_backward(
        self, trainer: Trainer, pl_module: Module, loss: torch.Tensor
    ) -> None: ...

    def on_after_backward(self, trainer: Trainer, pl_module: Module) -> None: ...

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: Module, optimizer: torch.optim.Optimizer
    ) -> None: ...

    def on_before_zero_grad(
        self, trainer: Trainer, pl_module: Module, optimizer: torch.optim.Optimizer
    ) -> None: ...

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: Module, checkpoint: dict[str, Any]
    ) -> None: ...

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: Module, checkpoint: dict[str, Any]
    ) -> None: ...

    def on_exception(
        self, trainer: Trainer, pl_module: Module, exception: BaseException
    ) -> None: ...


def overrides(cb: Any, name: str) -> bool:
    if name in getattr(cb, "__dict__", {}):  # set on the instance (LambdaCallback)
        return True
    fn = getattr(type(cb), name, None)
    return fn is not None and fn is not getattr(Callback, name, None)


def legacy_batch_end(cb: Callback) -> bool:
    """True for ``on_train_batch_end(trainer, model, loss, batch_idx)`` (GPUlayer 0.7)."""
    try:
        params = inspect.signature(cb.on_train_batch_end).parameters
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) == 4 and not any(p.kind == p.VAR_POSITIONAL for p in params.values())


def _call(callbacks: list[Callback], hook: str, *args: Any) -> None:
    for cb in callbacks:
        if overrides(cb, hook):
            getattr(cb, hook)(*args)


def _document_hooks() -> None:
    from gpulayer.pytorch.hooks import apply_hook_docs

    apply_hook_docs(Callback, "Callback form, with ``trainer, pl_module`` first. ")


_document_hooks()
