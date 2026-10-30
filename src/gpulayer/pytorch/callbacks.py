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
    """Save checkpoints of the best epochs by ``monitor`` (rank 0 only).

    With ``save_top_k=1`` (the default) the best epoch is kept as ``checkpoints/<filename>``.
    With ``save_top_k=N`` the N best are kept as ``checkpoints/<stem>-epoch=E.ckpt`` and worse
    ones are deleted. The files have the same format as ``last.ckpt``, so
    ``Module.load_from_checkpoint`` and ``Trainer.fit(ckpt_path=...)`` accept them.
    ``best_model_path`` points at the best one afterwards, also when training ran on GPUlayer
    workers; ``best_k_models`` maps each kept file to its score.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        mode: Literal["min", "max"] = "min",
        filename: str = "best.ckpt",
        min_delta: float = 0.0,
        save_top_k: int = 1,
    ) -> None:
        super().__init__(monitor, mode, min_delta)
        if "/" in filename or "\\" in filename or not filename.endswith(".ckpt"):
            raise ValidationFailed("filename must be a plain name ending in .ckpt")
        if save_top_k < 1:
            raise ValidationFailed("save_top_k must be >= 1")
        self.filename = filename
        self.save_top_k = save_top_k
        self.best_model_path: Path | None = None
        self.best_k_models: dict[Path, float] = {}

    def _kept_name(self, epoch: int) -> str:
        if self.save_top_k == 1:
            return self.filename
        return f"{self.filename[: -len('.ckpt')]}-epoch={epoch}.ckpt"

    def _worst(self) -> tuple[Path, float]:
        pick = max if self.mode == "min" else min
        return pick(self.best_k_models.items(), key=lambda kv: kv[1])

    def on_train_epoch_end(self, trainer: Trainer, model: Module) -> None:
        value = self._current(trainer)
        if math.isnan(value):
            return
        if self.save_top_k == 1:
            if not self._improved(value):
                return
        elif len(self.best_k_models) >= self.save_top_k:
            worst_path, worst = self._worst()
            better = (
                value < worst - self.min_delta
                if self.mode == "min"
                else (value > worst + self.min_delta)
            )
            if not better:
                return
        if self._improved(value):
            self.best_score = value
        path = trainer.save_checkpoint(self._kept_name(trainer.current_epoch))
        if path is None:  # not rank 0
            return
        if self.save_top_k > 1:
            self.best_k_models[path] = value
            if len(self.best_k_models) > self.save_top_k:
                worst_path, _ = self._worst()
                del self.best_k_models[worst_path]
                worst_path.unlink(missing_ok=True)
            pick = min if self.mode == "min" else max
            self.best_model_path = pick(self.best_k_models.items(), key=lambda kv: kv[1])[0]
        else:
            self.best_k_models = {path: value}
            self.best_model_path = path

    def on_fit_end(self, trainer: Trainer, model: Module) -> None:
        if self.best_model_path is None and trainer.launch_result is not None:
            # Trained on workers: the checkpoints came back with the job's outputs.
            out = trainer.launch_result.output_dir
            if self.save_top_k == 1:
                found = sorted(out.rglob(self.filename))
                self.best_model_path = found[0] if found else None
            else:
                import torch

                stem = self.filename[: -len(".ckpt")]
                for path in sorted(out.rglob(f"{stem}-epoch=*.ckpt")):
                    # Each checkpoint carries the metric history up to its epoch.
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    epoch_metrics = state.get("history", [{}])[-1]
                    self.best_k_models[path] = float(epoch_metrics.get(self.monitor, math.nan))
                if self.best_k_models:
                    pick = min if self.mode == "min" else max
                    self.best_model_path = pick(
                        self.best_k_models.items(),
                        key=lambda kv: kv[1] if not math.isnan(kv[1]) else math.inf,
                    )[0]


class LearningRateMonitor(Callback):
    """Record the learning rate each optimizer parameter group used in an epoch, as ``lr`` (one
    group) or ``lr-<i>``, in ``trainer.history`` and ``trainer.callback_metrics``. The value is
    read when the epoch starts, before an epoch-interval scheduler steps."""

    def __init__(self) -> None:
        self._lrs: dict[str, float] = {}

    def on_train_epoch_start(self, trainer: Trainer, model: Module) -> None:
        optimizer = trainer._optimizer
        groups = optimizer.param_groups if optimizer is not None else []
        self._lrs = {
            ("lr" if len(groups) == 1 else f"lr-{i}"): float(g["lr"]) for i, g in enumerate(groups)
        }

    def on_train_epoch_end(self, trainer: Trainer, model: Module) -> None:
        trainer.callback_metrics.update(self._lrs)
        if trainer.history:
            trainer.history[-1].update(self._lrs)


def _call(callbacks: list[Callback], hook: str, *args: Any) -> None:
    for cb in callbacks:
        getattr(cb, hook)(*args)
