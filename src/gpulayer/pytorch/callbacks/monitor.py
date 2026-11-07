"""Callbacks that decide from a monitored metric: ``EarlyStopping`` and ``ModelCheckpoint``."""

from __future__ import annotations

import math
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.callbacks.base import Callback, Checkpoint
from gpulayer.pytorch.storage import as_path, is_remote, load_file

if TYPE_CHECKING:
    from gpulayer.pytorch.module import Module
    from gpulayer.pytorch.trainer import Trainer


class _Monitor(Callback):
    def __init__(
        self,
        monitor: str | None,
        mode: Literal["min", "max"],
        min_delta: float,
        strict: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValidationFailed(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor, self.mode, self.min_delta = monitor, mode, abs(min_delta)
        self.strict = strict
        self.best_score: float | None = None

    def _current(self, trainer: Trainer) -> float | None:
        assert self.monitor is not None
        if self.monitor not in trainer.callback_metrics:
            if not self.strict:
                return None
            known = ", ".join(sorted(trainer.callback_metrics)) or "none"
            raise ValidationFailed(
                f"{type(self).__name__} monitors {self.monitor!r}, which was not logged "
                f"(logged: {known}); call self.log({self.monitor!r}, ...) in your Module"
            )
        return float(trainer.callback_metrics[self.monitor])

    def _better(self, value: float, than: float | None) -> bool:
        if math.isnan(value):
            return False
        if than is None:
            return True
        if self.mode == "min":
            return value < than - self.min_delta
        return value > than + self.min_delta

    def _improved(self, value: float) -> bool:
        return self._better(value, self.best_score)


class EarlyStopping(_Monitor):
    """Stop when ``monitor`` has not improved by more than ``min_delta`` for ``patience``
    checks in a row. As in Lightning, it checks at the end of each training epoch when
    validation runs once per epoch (``val_check_interval`` unset or 1.0 and
    ``check_val_every_n_epoch=1``), and otherwise after each validation, so epochs without
    validation do not use up patience; ``check_on_train_epoch_end`` overrides this. Also
    stops at once when the value reaches
    ``stopping_threshold``, crosses ``divergence_threshold`` the wrong way, or (with
    ``check_finite``) becomes NaN or infinite. With ``strict=False`` a missing metric is
    skipped instead of raising."""

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: Literal["min", "max"] = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
    ) -> None:
        super().__init__(monitor, mode, min_delta, strict)
        if patience < 1:
            raise ValidationFailed("patience must be >= 1")
        self.patience = patience
        self.verbose = verbose
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.wait_count = 0
        self.stopped_epoch: int | None = None
        self.check_on_train_epoch_end = check_on_train_epoch_end

    # GPUlayer 0.7 name
    @property
    def wait(self) -> int:
        """Checks without improvement so far (``wait_count``)."""
        return self.wait_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.wait_count = state_dict.get("wait_count", 0)
        self.stopped_epoch = state_dict.get("stopped_epoch")
        self.best_score = state_dict.get("best_score")

    def _stop(self, trainer: Trainer, reason: str) -> None:
        self.stopped_epoch = trainer.current_epoch
        trainer.should_stop = True
        trainer._say(f"[gpulayer] early stopping at epoch {trainer.current_epoch}: {reason}")

    def _on_train_epoch_end(self, trainer: Trainer) -> bool:
        """Check at the training epoch's end (True) or after each validation (False)."""
        if self.check_on_train_epoch_end is not None:
            return self.check_on_train_epoch_end
        v = trainer.val_check_interval
        return (v is None or v == 1.0) and trainer.check_val_every_n_epoch == 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self._on_train_epoch_end(trainer):
            self._run_check(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: Module) -> None:
        if trainer.state.fn != "fit" or trainer.sanity_checking:
            return
        if not self._on_train_epoch_end(trainer):
            self._run_check(trainer)

    def _run_check(self, trainer: Trainer) -> None:
        value = self._current(trainer)
        if value is None:
            return
        name = self.monitor
        sign = 1 if self.mode == "min" else -1
        if self.check_finite and not math.isfinite(value):
            self._stop(trainer, f"{name} = {value} is not finite")
            return
        if self.stopping_threshold is not None and sign * (value - self.stopping_threshold) <= 0:
            self._stop(trainer, f"{name} = {value:.4f} reached the stopping threshold")
            return
        if (
            self.divergence_threshold is not None
            and sign * (value - self.divergence_threshold) >= 0
        ):
            self._stop(trainer, f"{name} = {value:.4f} crossed the divergence threshold")
            return
        if self._improved(value):
            if self.verbose:
                trainer._say(f"[gpulayer] {name} improved to {value:.4f}")
            self.best_score, self.wait_count = value, 0
            return
        self.wait_count += 1
        if self.wait_count >= self.patience:
            best = f"{self.best_score:.4f}" if self.best_score is not None else "none"
            self._stop(
                trainer, f"{name} did not improve for {self.patience} check(s) (best {best})"
            )


_FORMAT = re.compile(r"\{([^{}:]+)(:[^{}]*)?\}")


class ModelCheckpoint(_Monitor, Checkpoint):
    """Save checkpoints of the best results by ``monitor`` (written by rank 0).

    - ``save_top_k=1`` (default) keeps the best as ``checkpoints/<filename>``;
      ``save_top_k=N`` keeps the N best; ``-1`` keeps every one; ``0`` keeps none (use with
      ``save_last``).
    - ``filename`` is a plain name ending in ``.ckpt`` (default ``best.ckpt``; with
      ``save_top_k > 1`` files become ``best-epoch=E.ckpt``), or a Lightning template such
      as ``"{epoch}-{val_loss:.2f}"``, which gives ``epoch=3-val_loss=0.12.ckpt``.
    - ``monitor=None`` saves the latest checkpoint instead of the best.
    - When to check: every ``every_n_epochs`` epochs (default 1), or every
      ``every_n_train_steps`` optimizer steps, or every ``train_time_interval``.
    - ``save_last=True`` also writes ``last.ckpt`` each time it checks.
    - ``save_weights_only=True`` leaves the optimizer state out.
    - ``save_on_exception=True`` also saves when training stops with an exception
      (including Ctrl+C), under the name ``filename`` gives at that point, or
      ``<stem>-on_exception.ckpt`` when that name is a plain file name or one of the kept
      best files (so the best checkpoint is never overwritten); with ``save_last`` it also
      writes ``last.ckpt``. Single-process runs only: with several processes the others
      may be inside a collective, so nothing is saved.

    ``best_model_path`` points at the best file afterwards, also when training ran on
    GPUlayer workers; ``best_k_models`` maps each kept file to its score.
    """

    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: bool | None = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: Literal["min", "max"] = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
        min_delta: float = 0.0,
        save_on_exception: bool = False,
    ) -> None:
        super().__init__(monitor, mode, min_delta)
        filename = filename or "best.ckpt"
        if "/" in filename or "\\" in filename:
            raise ValidationFailed("filename must be a plain name, not a path")
        if save_top_k < -1:
            raise ValidationFailed("save_top_k must be >= -1")
        if (
            sum(x is not None for x in (every_n_train_steps, train_time_interval, every_n_epochs))
            > 1
        ):
            raise ValidationFailed(
                "set at most one of every_n_train_steps, train_time_interval and every_n_epochs"
            )
        self.dirpath = as_path(dirpath) if dirpath is not None else None
        self.filename = filename
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.save_on_exception = save_on_exception
        self._last_global_step_saved = 0
        self.auto_insert_metric_name = auto_insert_metric_name
        self.every_n_train_steps = every_n_train_steps
        self.train_time_interval = train_time_interval
        self.every_n_epochs = (
            1
            if every_n_epochs is None and not (every_n_train_steps or train_time_interval)
            else every_n_epochs
        )
        self.best_model_path: Path | None = None
        self.best_model_score: float | None = None
        self.last_model_path: Path | None = None
        self.best_k_models: dict[Path, float] = {}
        # Scores by file name, kept on every rank so all ranks take the same branch (with
        # FSDP, saving is a collective).
        self._scores: dict[str, float] = {}
        self._last_time = time.monotonic()

    @property
    def kth_best_model_path(self) -> Path | None:
        """The worst of the kept checkpoints (the next to be replaced)."""
        if not self.best_k_models:
            return None
        return self._worst(self.best_k_models)[0]

    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "scores": dict(self._scores),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("monitor") != self.monitor:
            return
        self.best_model_score = state_dict.get("best_model_score")
        self.best_score = self.best_model_score
        self._scores = dict(state_dict.get("scores") or {})

    # ---------------------------------------------------------------- naming

    def format_checkpoint_name(self, metrics: dict[str, Any], ver: int | None = None) -> str:
        """The file name for ``metrics`` (``epoch`` and ``step`` included)."""
        if self.filename.endswith(self.FILE_EXTENSION) and "{" not in self.filename:
            if self.save_top_k in (0, 1):
                return self.filename
            stem = self.filename[: -len(self.FILE_EXTENSION)]
            name = f"{stem}-epoch={metrics['epoch']}"
            if self.every_n_train_steps or self.train_time_interval:
                name += f"-step={metrics['step']}"
            return name + self.FILE_EXTENSION

        def fill(m: re.Match[str]) -> str:
            key, spec = m.group(1), m.group(2) or ""
            value = metrics.get(key, 0)
            text = format(value, spec[1:]) if spec else str(value)
            return f"{key}={text}" if self.auto_insert_metric_name else text

        stem = (
            self.filename[: -len(self.FILE_EXTENSION)]
            if self.filename.endswith(self.FILE_EXTENSION)
            else self.filename
        )
        return _FORMAT.sub(fill, stem).replace("/", "_") + self.FILE_EXTENSION

    def _templated(self) -> bool:
        return not (self.filename.endswith(self.FILE_EXTENSION) and "{" not in self.filename)

    # ---------------------------------------------------------------- saving

    def _worst(self, scores: dict[Any, float]) -> tuple[Any, float]:
        pick = max if self.mode == "min" else min
        return pick(scores.items(), key=lambda kv: kv[1] if not math.isnan(kv[1]) else math.inf)

    def _best(self, scores: dict[Any, float]) -> tuple[Any, float]:
        pick = min if self.mode == "min" else max
        return pick(scores.items(), key=lambda kv: kv[1])

    def _check(self, trainer: Trainer, pl_module: Module) -> None:
        if trainer.sanity_checking or trainer.fast_dev_run:
            return
        metrics = {
            **trainer.callback_metrics,
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        }
        if self.monitor is None:
            value: float | None = float(trainer.global_step)  # newest is best
            mode = "max"
        else:
            value = self._current(trainer)
            mode = self.mode
        if value is not None and self.save_top_k != 0 and not math.isnan(value):
            self._save_top_k(trainer, metrics, value, mode)
        if self.save_last:
            path = trainer.save_checkpoint(
                self.CHECKPOINT_NAME_LAST + self.FILE_EXTENSION,
                weights_only=self.save_weights_only,
                dirpath=self.dirpath,
            )
            self._last_global_step_saved = trainer.global_step
            if path is not None:
                self.last_model_path = path

    def _save_top_k(
        self, trainer: Trainer, metrics: dict[str, Any], value: float, mode: str
    ) -> None:
        name = self.format_checkpoint_name(metrics)
        if self.save_top_k != 1 and name in self._scores and self._templated():
            # The template gave the same name again: keep both, as name-v1.ckpt (Lightning).
            stem, v = name[: -len(self.FILE_EXTENSION)], 1
            while f"{stem}-v{v}{self.FILE_EXTENSION}" in self._scores:
                v += 1
            name = f"{stem}-v{v}{self.FILE_EXTENSION}"
        saved_mode, self.mode = self.mode, mode  # type: ignore[assignment]
        try:
            if self.save_top_k == -1 or len(self._scores) < self.save_top_k or name in self._scores:
                keep = name not in self._scores or self._better(value, self._scores[name])
                if self.save_top_k == 1 and self._scores:
                    keep = self._better(value, self._best(self._scores)[1])
            else:
                keep = self._better(value, self._worst(self._scores)[1])
            if not keep:
                return
            removed: str | None = None
            if self.save_top_k == 1:
                removed = next((n for n in self._scores if n != name), None)
                self._scores = {name: value}
            else:
                self._scores[name] = value
                if self.save_top_k > 0 and len(self._scores) > self.save_top_k:
                    removed = self._worst(self._scores)[0]
                    del self._scores[removed]
            best_name, best_value = self._best(self._scores)
            self.best_score = self.best_model_score = best_value
        finally:
            self.mode = saved_mode
        path = trainer.save_checkpoint(
            name, weights_only=self.save_weights_only, dirpath=self.dirpath
        )
        self._last_global_step_saved = trainer.global_step
        if path is None:  # not rank 0
            return
        folder = path.parent
        if removed is not None and removed != name:
            (folder / removed).unlink(missing_ok=True)
        self.best_k_models = {folder / n: s for n, s in self._scores.items()}
        self.best_model_path = folder / best_name
        if self.verbose:
            trainer._say(f"[gpulayer] saved {path.name} ({self.monitor}={value:.4f})")

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not trainer._stepped:
            return
        if self.every_n_train_steps and trainer.global_step % self.every_n_train_steps == 0:
            trainer._read_step_metrics()
            self._check(trainer, pl_module)
        elif self.train_time_interval is not None:
            now = time.monotonic()
            # Same decision on every rank: compare the time on rank 0's clock.
            due = trainer._broadcast_flag(
                now - self._last_time >= self.train_time_interval.total_seconds()
            )
            if due:
                self._last_time = now
                trainer._read_step_metrics()
                self._check(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self.every_n_epochs and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._check(trainer, pl_module)

    def on_exception(self, trainer: Trainer, pl_module: Module, exception: BaseException) -> None:
        """With ``save_on_exception``, save a checkpoint when training stops with an
        exception (not during the sanity check or ``fast_dev_run``, and not again for a step
        already saved)."""
        if (
            not self.save_on_exception
            or trainer.fast_dev_run
            or trainer.sanity_checking
            or self._last_global_step_saved == trainer.global_step
            # Only safe in one process: other ranks may be inside a collective.
            or trainer.world_size > 1
        ):
            return
        metrics = {
            **trainer.callback_metrics,
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        }
        name = self.format_checkpoint_name(metrics)
        if not self._templated() or name in self._scores:
            stem = self.filename.split("{", 1)[0].removesuffix(self.FILE_EXTENSION).rstrip("-_")
            name = f"{stem or 'checkpoint'}-on_exception{self.FILE_EXTENSION}"
        path = trainer.save_checkpoint(
            name, weights_only=self.save_weights_only, dirpath=self.dirpath
        )
        if self.save_last:
            last = trainer.save_checkpoint(
                self.CHECKPOINT_NAME_LAST + self.FILE_EXTENSION,
                weights_only=self.save_weights_only,
                dirpath=self.dirpath,
            )
            if last is not None:
                self.last_model_path = last
        self._last_global_step_saved = trainer.global_step
        if path is not None:
            trainer._say(
                f"[gpulayer] {type(exception).__name__} raised ({exception}); "
                f"saved checkpoint to {path}"
            )

    def on_fit_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self.best_model_path is not None or trainer.launch_result is None:
            return
        # Trained on workers: the checkpoints came back with the job's outputs.
        out: Any = trainer.launch_result.output_dir
        root = getattr(trainer, "default_root_dir", None)
        if is_remote(self.dirpath):  # the workers wrote to the store, not to job outputs
            out = self.dirpath
        elif root is not None and is_remote(root):
            out = root / "checkpoints"
        found: dict[Any, float] = {}
        for path in sorted(out.rglob(f"*{self.FILE_EXTENSION}")):
            if path.name == "last.ckpt" or path.name == RESUME_NAME:
                continue
            if self.filename.endswith(self.FILE_EXTENSION) and "{" not in self.filename:
                stem = self.filename[: -len(self.FILE_EXTENSION)]
                if path.name != self.filename and not path.name.startswith(f"{stem}-epoch="):
                    continue
            state = load_file(path, map_location="cpu", weights_only=False)
            score = (state.get("callbacks") or {}).get(self.state_key, {}).get("scores", {})
            if path.name in score:
                found[path] = float(score[path.name])
            else:
                # Each checkpoint carries the metric history up to its epoch.
                last = (state.get("history") or [{}])[-1]
                found[path] = float(last.get(self.monitor or "", math.nan))
        if found:
            self.best_k_models = found
            self.best_model_path = min(
                found,
                key=lambda p: (
                    (found[p] if self.mode == "min" else -found[p])
                    if not math.isnan(found[p])
                    else math.inf
                ),
            )
            self.best_model_score = found[self.best_model_path]


RESUME_NAME = "gpulayer-resume.ckpt"


class OnExceptionCheckpoint(Checkpoint):
    """Save a checkpoint when training stops with an exception (including Ctrl+C), so it can
    be resumed with ``fit(ckpt_path=...)``."""

    def __init__(self, dirpath: str | Path | None = None, filename: str = "on_exception") -> None:
        self.dirpath = as_path(dirpath) if dirpath is not None else None
        self.filename = filename if filename.endswith(".ckpt") else filename + ".ckpt"
        self.ckpt_path: Path | None = None

    def on_exception(self, trainer: Trainer, pl_module: Module, exception: BaseException) -> None:
        # Only safe in one process: other ranks may be inside a collective.
        if trainer.world_size > 1:
            return
        self.ckpt_path = trainer.save_checkpoint(self.filename, dirpath=self.dirpath)
