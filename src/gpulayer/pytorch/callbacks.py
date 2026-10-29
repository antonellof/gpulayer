"""Trainer callbacks: hooks into the training loop, plus early stopping and best checkpoints.

    trainer = Trainer(max_epochs=50, callbacks=[
        EarlyStopping(monitor="val_loss", patience=3),
        ModelCheckpoint(monitor="val_loss"),
    ])

Hooks run on every rank. Epoch metrics are averaged across ranks before ``on_train_epoch_end``,
so callbacks that decide from them (like ``EarlyStopping``) agree; the Trainer also combines
``trainer.should_stop`` across ranks, so one rank asking to stop stops all of them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gpulayer.errors import ValidationFailed

if TYPE_CHECKING:
    import torch

    from gpulayer.pytorch.module import Module
    from gpulayer.pytorch.trainer import Trainer


class Callback:
    """Subclass and override the hooks you need. All hooks are optional."""

    def on_fit_start(self, trainer: Trainer, model: Module) -> None: ...

    def on_train_epoch_start(self, trainer: Trainer, model: Module) -> None: ...

    def on_train_batch_end(
        self, trainer: Trainer, model: Module, loss: torch.Tensor, batch_idx: int
    ) -> None: ...

    def on_train_epoch_end(self, trainer: Trainer, model: Module) -> None:
        """After training and validation of an epoch; ``trainer.callback_metrics`` is set."""

    def on_fit_end(self, trainer: Trainer, model: Module) -> None: ...


class _Monitor(Callback):
    def __init__(self, monitor: str, mode: Literal["min", "max"], min_delta: float) -> None:
        if mode not in ("min", "max"):
            raise ValidationFailed(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor, self.mode, self.min_delta = monitor, mode, abs(min_delta)
        self.best_score: float | None = None

    def _current(self, trainer: Trainer) -> float:
        if self.monitor not in trainer.callback_metrics:
            known = ", ".join(sorted(trainer.callback_metrics)) or "none"
            raise ValidationFailed(
                f"{type(self).__name__} monitors {self.monitor!r}, which was not logged "
                f"(logged: {known}); call self.log({self.monitor!r}, ...) in your Module"
            )
        return float(trainer.callback_metrics[self.monitor])

    def _improved(self, value: float) -> bool:
        if math.isnan(value):
            return False
        if self.best_score is None:
            return True
        if self.mode == "min":
            return value < self.best_score - self.min_delta
        return value > self.best_score + self.min_delta


class EarlyStopping(_Monitor):
    """Stop when ``monitor`` has not improved by more than ``min_delta`` for ``patience``
    epochs in a row."""

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        mode: Literal["min", "max"] = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        super().__init__(monitor, mode, min_delta)
        if patience < 1:
            raise ValidationFailed("patience must be >= 1")
        self.patience = patience
        self.wait = 0
        self.stopped_epoch: int | None = None

    def on_train_epoch_end(self, trainer: Trainer, model: Module) -> None:
        value = self._current(trainer)
        if self._improved(value):
            self.best_score, self.wait = value, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True
            trainer._say(
                f"[gpulayer] early stopping at epoch {trainer.current_epoch}: {self.monitor} "
                f"did not improve for {self.patience} epoch(s) (best {self.best_score:.4f})"
            )


class ModelCheckpoint(_Monitor):
    """Save ``checkpoints/<filename>`` whenever ``monitor`` improves (rank 0 only).

    The file has the same format as ``last.ckpt``, so ``Module.load_from_checkpoint`` and
    ``Trainer.fit(ckpt_path=...)`` accept it. ``best_model_path`` points at it afterwards,
    also when training ran on GPUlayer workers.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        mode: Literal["min", "max"] = "min",
        filename: str = "best.ckpt",
        min_delta: float = 0.0,
    ) -> None:
        super().__init__(monitor, mode, min_delta)
        if "/" in filename or "\\" in filename or not filename.endswith(".ckpt"):
            raise ValidationFailed("filename must be a plain name ending in .ckpt")
        self.filename = filename
        self.best_model_path: Path | None = None

    def on_train_epoch_end(self, trainer: Trainer, model: Module) -> None:
        value = self._current(trainer)
        if not self._improved(value):
            return
        self.best_score = value
        path = trainer.save_checkpoint(self.filename)
        if path is not None:
            self.best_model_path = path

    def on_fit_end(self, trainer: Trainer, model: Module) -> None:
        if self.best_model_path is None and trainer.launch_result is not None:
            # Trained on workers: the checkpoint came back with the job's outputs.
            found = sorted(trainer.launch_result.output_dir.rglob(self.filename))
            self.best_model_path = found[0] if found else None


def _call(callbacks: list[Callback], hook: str, *args: Any) -> None:
    for cb in callbacks:
        getattr(cb, hook)(*args)
