"""Learning-rate and batch-size search, named as in PyTorch Lightning's ``Tuner``.

    tuner = Tuner(trainer)
    finder = tuner.lr_find(model, train_loader)     # sets model.lr to the suggestion
    size = tuner.scale_batch_size(model, datamodule=dm)

Both run short trainings in this process, then put the model's weights back. They read the
loss every step, which waits for the device; that is fine for a search, not for training.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import torch

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.callbacks.base import Callback

if TYPE_CHECKING:
    from gpulayer.pytorch.module import DataModule, Module
    from gpulayer.pytorch.trainer import Trainer


class _LRFinder:
    """Result of ``lr_find``: ``results`` (``lr`` and ``loss`` per step) and
    ``suggestion()``, the rate where the smoothed loss falls fastest."""

    def __init__(self, lrs: list[float], losses: list[float]) -> None:
        self.results = {"lr": lrs, "loss": losses}

    def suggestion(self, skip_begin: int = 10, skip_end: int = 1) -> float | None:
        lrs = self.results["lr"][skip_begin : -skip_end or None]
        losses = self.results["loss"][skip_begin : -skip_end or None]
        pairs = [(lr, lo) for lr, lo in zip(lrs, losses, strict=True) if math.isfinite(lo)]
        if len(pairs) < 2:
            return None
        grads = [
            (pairs[i + 1][1] - pairs[i][1])
            / (math.log10(pairs[i + 1][0]) - math.log10(pairs[i][0]))
            for i in range(len(pairs) - 1)
        ]
        return pairs[min(range(len(grads)), key=grads.__getitem__)][0]

    def plot(self, suggest: bool = False, show: bool = False) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.results["lr"], self.results["loss"])
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")
        if suggest and self.suggestion() is not None:
            ax.axvline(self.suggestion(), color="red")
        if show:
            plt.show()
        return fig


class _LRSweep(Callback):
    def __init__(
        self, min_lr: float, max_lr: float, n: int, mode: str, threshold: float | None
    ) -> None:
        self.min_lr, self.max_lr, self.n, self.mode = min_lr, max_lr, n, mode
        self.threshold = threshold
        self.lrs: list[float] = []
        self.losses: list[float] = []
        self._avg = 0.0
        self._best = math.inf

    def _lr(self, i: int) -> float:
        r = i / max(self.n - 1, 1)
        if self.mode == "exponential":
            return self.min_lr * (self.max_lr / self.min_lr) ** r
        return self.min_lr + (self.max_lr - self.min_lr) * r

    def on_train_start(self, trainer: Trainer, pl_module: Module) -> None:
        trainer.lr_scheduler_configs = []  # the sweep sets the rate itself

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: Module, batch: Any, batch_idx: int
    ) -> None:
        lr = self._lr(trainer.global_step)
        for opt in trainer.optimizers:
            for g in opt.param_groups:
                g["lr"] = lr

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        loss = float(outputs["loss"])
        i = len(self.losses)
        self._avg = 0.98 * self._avg + 0.02 * loss
        smooth = self._avg / (1 - 0.98 ** (i + 1))
        self.lrs.append(trainer.optimizers[0].param_groups[0]["lr"])
        self.losses.append(smooth)
        self._best = min(self._best, smooth)
        if self.threshold is not None and (
            not math.isfinite(smooth) or (i > 1 and smooth > self.threshold * self._best)
        ):
            trainer.should_stop = True


def _set_attr(holders: list[Any], name: str, value: Any) -> bool:
    for h in holders:
        if h is None:
            continue
        if hasattr(h, name) and not callable(getattr(h, name)):
            setattr(h, name, value)
            return True
        hp = getattr(h, "hparams", None)
        if hp is not None and name in hp:
            hp[name] = value
            return True
    return False


def _get_attr(holders: list[Any], name: str) -> Any:
    for h in holders:
        if h is None:
            continue
        if hasattr(h, name) and not callable(getattr(h, name)):
            return getattr(h, name)
        hp = getattr(h, "hparams", None)
        if hp is not None and name in hp:
            return hp[name]
    return None


def _is_oom(e: BaseException) -> bool:
    oom = getattr(torch, "OutOfMemoryError", ())
    return (oom and isinstance(e, oom)) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e)
    )


class Tuner:
    """Searches that run short trainings with a Trainer's settings."""

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def _scratch(self, **overrides: Any) -> Trainer:
        from gpulayer.pytorch.trainer import Trainer

        t = self.trainer
        if t.will_launch:
            raise ValidationFailed(
                "the Tuner runs in this process: use devices=1, num_nodes=1, no target"
            )
        args: dict[str, Any] = {
            "accelerator": t.accelerator,
            "precision": t.precision,
            "max_epochs": 10**6,
            "enable_checkpointing": False,
            "logger": False,
            "log_every_n_steps": 0,
            "num_sanity_val_steps": 0,
            "limit_val_batches": 0,
            "accumulate_grad_batches": t.accumulate_grad_batches,
            "gradient_clip_val": t.gradient_clip_val,
            "default_root_dir": t.default_root_dir / "tuner",
        }
        args.update(overrides)
        return Trainer(**args)

    def lr_find(
        self,
        model: Module,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        dataloaders: Any = None,
        datamodule: DataModule | None = None,
        method: str = "fit",
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_training: int = 100,
        mode: str = "exponential",
        early_stop_threshold: float | None = 4.0,
        update_attr: bool = True,
        attr_name: str = "",
    ) -> _LRFinder:
        """Train ``num_training`` steps with the rate rising from ``min_lr`` to ``max_lr``
        (``"exponential"`` or ``"linear"``), recording the smoothed loss; stop early when it
        exceeds ``early_stop_threshold`` times the best. The weights are restored afterwards.
        With ``update_attr`` the suggestion is written to ``model.lr`` or
        ``model.learning_rate`` (or ``attr_name``, or the same key in ``model.hparams``)."""
        if mode not in ("exponential", "linear"):
            raise ValidationFailed("mode must be 'exponential' or 'linear'")
        sweep = _LRSweep(min_lr, max_lr, num_training, mode, early_stop_threshold)
        saved = copy.deepcopy(model.state_dict())
        trainer = self._scratch(max_steps=num_training, callbacks=[sweep])
        try:
            trainer.fit(model, train_dataloaders or dataloaders, None, datamodule=datamodule)
        finally:
            model.load_state_dict(saved)
            model.trainer = self.trainer
        finder = _LRFinder(sweep.lrs, sweep.losses)
        lr = finder.suggestion()
        if update_attr and lr is not None:
            names = [attr_name] if attr_name else ["lr", "learning_rate"]
            if not any(_set_attr([model], n, lr) for n in names):
                raise ValidationFailed(
                    "lr_find(update_attr=True) needs model.lr or model.learning_rate (or "
                    "attr_name=...) to write the suggestion to"
                )
        return finder

    def scale_batch_size(
        self,
        model: Module,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        dataloaders: Any = None,
        datamodule: DataModule | None = None,
        method: str = "fit",
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
    ) -> int:
        """Find the largest batch size that fits in memory: double it (``"power"``) until a
        step runs out of memory, then, with ``"binsearch"``, search between the last two.
        The loader must be rebuilt from ``batch_arg_name`` on the model, its ``hparams`` or
        the DataModule (a ``train_dataloader`` method or a loader function). The size is left
        on that attribute."""
        if mode not in ("power", "binsearch"):
            raise ValidationFailed("mode must be 'power' or 'binsearch'")
        source = train_dataloaders or dataloaders
        holders = [model, datamodule]
        if _get_attr(holders, batch_arg_name) is None:
            raise ValidationFailed(
                f"scale_batch_size needs a `{batch_arg_name}` attribute on the model, its "
                "hparams or the DataModule, read when the loader is built"
            )
        if source is not None and not callable(source):
            raise ValidationFailed(
                "scale_batch_size rebuilds the loader for each size: pass a function that "
                "returns it, a DataModule, or define train_dataloader() on the model"
            )
        saved = copy.deepcopy(model.state_dict())

        def fits(size: int) -> tuple[bool, bool]:
            """(fits in memory, the loader could still grow)."""
            _set_attr(holders, batch_arg_name, size)
            trainer = self._scratch(max_steps=steps_per_trial)
            try:
                trainer.fit(model, source, None, datamodule=datamodule)
            except BaseException as e:
                if not _is_oom(e):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False, False
            loader = trainer.train_dataloader
            n = len(loader) if hasattr(loader, "__len__") else math.inf
            return True, n > 1

        try:
            good, size, bad = 0, init_val, None
            for _ in range(max_trials):
                ok, grow = fits(size)
                if not ok:
                    bad = size
                    break
                good = size
                if not grow:
                    break  # one batch already holds the whole dataset
                size *= 2
            if mode == "binsearch" and bad is not None and good:
                low, high = good, bad
                while high - low > 1:
                    mid = (low + high) // 2
                    ok, _ = fits(mid)
                    low, high = (mid, high) if ok else (low, mid)
                good = low
            if not good:
                raise ValidationFailed(f"batch size {init_val} does not fit in memory")
        finally:
            model.load_state_dict(saved)
            model.trainer = self.trainer
        _set_attr(holders, batch_arg_name, good)
        return good


class LearningRateFinder(Callback):
    """Run ``Tuner.lr_find`` when ``fit`` starts (one process only) and train with the
    suggested rate."""

    def __init__(
        self,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_training_steps: int = 100,
        mode: str = "exponential",
        early_stop_threshold: float | None = 4.0,
        update_attr: bool = True,
        attr_name: str = "",
    ) -> None:
        self.kwargs: dict[str, Any] = {
            "min_lr": min_lr,
            "max_lr": max_lr,
            "num_training": num_training_steps,
            "mode": mode,
            "early_stop_threshold": early_stop_threshold,
            "update_attr": update_attr,
            "attr_name": attr_name,
        }
        self.optimal_lr: _LRFinder | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: Module) -> None:
        if trainer.world_size > 1:
            raise ValidationFailed("LearningRateFinder runs in one process; use Tuner before fit")
        self.optimal_lr = Tuner(trainer).lr_find(
            pl_module, trainer._train_src, datamodule=trainer.datamodule, **self.kwargs
        )
        trainer._reconfigure_optimizers(pl_module)


class BatchSizeFinder(Callback):
    """Run ``Tuner.scale_batch_size`` when ``fit`` starts (one process only) and train with
    the size found."""

    def __init__(
        self,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
    ) -> None:
        self.kwargs: dict[str, Any] = {
            "mode": mode,
            "steps_per_trial": steps_per_trial,
            "init_val": init_val,
            "max_trials": max_trials,
            "batch_arg_name": batch_arg_name,
        }
        self.optimal_batch_size: int | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: Module) -> None:
        if trainer.world_size > 1:
            raise ValidationFailed("BatchSizeFinder runs in one process; use Tuner before fit")
        self.optimal_batch_size = Tuner(trainer).scale_batch_size(
            pl_module, trainer._train_src, datamodule=trainer.datamodule, **self.kwargs
        )
        trainer._reload_train_loader()
