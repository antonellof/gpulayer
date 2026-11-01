"""Monitoring and utility callbacks: learning rate, time, throughput, device stats, loss
spikes, model summary, progress bar, prediction writing, accumulation schedules and lambdas.

Each one is opt-in. The per-step ones read values only on the steps where they log (every
``log_every_n_steps``), except ``SpikeDetection``, which reads the loss every step by design.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.callbacks.base import Callback

if TYPE_CHECKING:
    from gpulayer.pytorch.module import Module
    from gpulayer.pytorch.trainer import Trainer


class LearningRateMonitor(Callback):
    """Record the learning rate of each optimizer parameter group.

    ``logging_interval="epoch"`` (or ``None``, the default) records the rate used in each
    epoch as ``lr`` (one group) or ``lr-<i>``, in ``trainer.history`` and
    ``trainer.callback_metrics``; it is read when the epoch starts, before an epoch-interval
    scheduler steps. ``"step"`` also records it on every logged step (``metrics.csv`` and
    loggers). ``log_momentum`` and ``log_weight_decay`` add those values.
    ``log_key_prefix`` is put before every name (``"optim/"`` gives ``optim/lr``)."""

    def __init__(
        self,
        logging_interval: Literal["step", "epoch"] | None = None,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
        log_key_prefix: str | None = None,
    ) -> None:
        if logging_interval not in (None, "step", "epoch"):
            raise ValidationFailed("logging_interval must be None, 'step' or 'epoch'")
        self.logging_interval = logging_interval
        self.log_momentum = log_momentum
        self.log_weight_decay = log_weight_decay
        self.log_key_prefix = log_key_prefix or ""
        self._lrs: dict[str, float] = {}

    def _read(self, trainer: Trainer) -> dict[str, float]:
        out: dict[str, float] = {}
        opts = trainer.optimizers
        for o, opt in enumerate(opts):
            groups = opt.param_groups
            for i, g in enumerate(groups):
                key = "lr"
                if len(opts) > 1:
                    key += f"-{type(opt).__name__}" + (f"-{o}" if len(opts) > 1 else "")
                if len(groups) > 1:
                    key += f"-{i}"
                out[key] = float(g["lr"])
                if self.log_momentum:
                    m = g.get("momentum", (g.get("betas") or (None,))[0])
                    if m is not None:
                        out[key.replace("lr", "momentum", 1)] = float(m)
                if self.log_weight_decay and "weight_decay" in g:
                    out[key.replace("lr", "weight_decay", 1)] = float(g["weight_decay"])
        if self.log_key_prefix:
            return {f"{self.log_key_prefix}{k}": v for k, v in out.items()}
        return out

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._lrs = self._read(trainer)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.logging_interval == "step" and trainer._is_log_step:
            trainer.log_metrics(self._read(trainer))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self.logging_interval == "step":
            return
        trainer.callback_metrics.update(self._lrs)
        if trainer.history:
            trainer.history[-1].update(self._lrs)


def _parse_duration(duration: Any) -> timedelta | None:
    if duration is None or isinstance(duration, timedelta):
        return duration
    if isinstance(duration, dict):
        return timedelta(**duration)
    if isinstance(duration, str):
        parts = [int(p) for p in duration.split(":")]
        if len(parts) != 4:
            raise ValidationFailed("duration must be 'DD:HH:MM:SS'")
        d, h, m, s = parts
        return timedelta(days=d, hours=h, minutes=m, seconds=s)
    raise ValidationFailed("duration must be a timedelta, a dict or 'DD:HH:MM:SS'")


class Timer(Callback):
    """Stop training after ``duration`` (a ``timedelta``, a dict of timedelta arguments or
    ``"DD:HH:MM:SS"``), checked every step (``interval="step"``) or every epoch. With several
    processes it is checked at the end of each epoch, so every process stops together.
    ``time_elapsed(stage)`` and ``time_remaining(stage)`` report the times in seconds."""

    def __init__(
        self,
        duration: Any = None,
        interval: Literal["step", "epoch"] = "step",
        verbose: bool = True,
    ) -> None:
        if interval not in ("step", "epoch"):
            raise ValidationFailed("interval must be 'step' or 'epoch'")
        self._duration = _parse_duration(duration)
        self.interval = interval
        self.verbose = verbose
        self._start: dict[str, float] = {}
        self._end: dict[str, float] = {}
        self._offset = 0.0

    def start_time(self, stage: str = "train") -> float | None:
        """Monotonic start time of ``stage`` ("train", "validate", "test"), or None."""
        return self._start.get(stage)

    def end_time(self, stage: str = "train") -> float | None:
        """Monotonic end time of ``stage``, or None while it runs."""
        return self._end.get(stage)

    def time_elapsed(self, stage: str = "train") -> float:
        """Seconds spent in ``stage`` (training time includes time before a resume)."""
        start = self._start.get(stage)
        if start is None:
            return 0.0
        end = self._end.get(stage, time.monotonic())
        return end - start + (self._offset if stage == "train" else 0.0)

    def time_remaining(self, stage: str = "train") -> float | None:
        """Seconds left of ``duration``, or None without one."""
        if self._duration is None:
            return None
        return self._duration.total_seconds() - self.time_elapsed(stage)

    def state_dict(self) -> dict[str, Any]:
        return {"time_elapsed": {"train": self.time_elapsed("train")}}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._offset = float(state_dict.get("time_elapsed", {}).get("train", 0.0))

    def on_train_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._start["train"] = time.monotonic()

    def on_train_end(self, trainer: Trainer, pl_module: Module) -> None:
        self._end["train"] = time.monotonic()

    def on_validation_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._start["validate"] = time.monotonic()

    def on_validation_end(self, trainer: Trainer, pl_module: Module) -> None:
        self._end["validate"] = time.monotonic()

    def on_test_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._start["test"] = time.monotonic()

    def on_test_end(self, trainer: Trainer, pl_module: Module) -> None:
        self._end["test"] = time.monotonic()

    def _check(self, trainer: Trainer) -> None:
        if self._duration is None or trainer.should_stop:
            return
        if self.time_elapsed("train") >= self._duration.total_seconds():
            trainer.should_stop = True
            if self.verbose:
                trainer._say(f"[gpulayer] Timer: time limit of {self._duration} reached")

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.interval == "step" and trainer.world_size == 1:
            self._check(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        self._check(trainer)


class LambdaCallback(Callback):
    """A callback from functions: ``LambdaCallback(on_train_epoch_end=lambda t, m: ...)``."""

    def __init__(self, **hooks: Callable[..., Any]) -> None:
        for name, fn in hooks.items():
            if not hasattr(Callback, name) or name.startswith("_"):
                raise ValidationFailed(f"LambdaCallback: unknown hook {name!r}")
            setattr(self, name, fn)
        self._hooks = set(hooks)


class GradientAccumulationScheduler(Callback):
    """Change ``accumulate_grad_batches`` by epoch: ``{0: 8, 4: 4, 8: 1}`` accumulates 8
    batches from epoch 0, 4 from epoch 4 and none from epoch 8."""

    def __init__(self, scheduling: dict[int, int]) -> None:
        if not scheduling or any(
            not isinstance(k, int) or k < 0 or not isinstance(v, int) or v < 1
            for k, v in scheduling.items()
        ):
            raise ValidationFailed("scheduling maps epochs (>= 0) to batch counts (>= 1)")
        self.scheduling = dict(sorted(scheduling.items()))
        if 0 not in self.scheduling:
            self.scheduling = {0: 1, **self.scheduling}

    def going_to_accumulate_grad_batches(self) -> bool:
        """True if any epoch accumulates more than one batch."""
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, epoch: int) -> int:
        """Batches accumulated per optimizer step in ``epoch``."""
        value = 1
        for start, n in self.scheduling.items():
            if epoch >= start:
                value = n
        return value

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(trainer.current_epoch)


class ThroughputMonitor(Callback):
    """Record training throughput on every logged step: ``train/batches_per_sec``,
    ``train/samples_per_sec`` (with ``batch_size_fn``) and ``train/items_per_sec`` (with
    ``length_fn``, e.g. tokens), per process. Time is measured on the host between logged
    steps, so no device synchronisation is added."""

    def __init__(
        self,
        batch_size_fn: Callable[[Any], int] | None = None,
        length_fn: Callable[[Any], int] | None = None,
        window_size: int = 100,
    ) -> None:
        self.batch_size_fn = batch_size_fn
        self.length_fn = length_fn
        self.window_size = window_size
        self._t: float | None = None
        self._batches = self._samples = self._items = 0

    def on_train_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._t = time.perf_counter()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self._batches += 1
        if self.batch_size_fn is not None:
            self._samples += int(self.batch_size_fn(batch))
        if self.length_fn is not None:
            self._items += int(self.length_fn(batch))
        if not trainer._is_log_step or self._t is None:
            return
        now = time.perf_counter()
        dt = max(now - self._t, 1e-9)
        m = {"train/batches_per_sec": self._batches / dt}
        if self.batch_size_fn is not None:
            m["train/samples_per_sec"] = self._samples / dt
        if self.length_fn is not None:
            m["train/items_per_sec"] = self._items / dt
        trainer.log_metrics(m)
        self._t, self._batches, self._samples, self._items = now, 0, 0, 0


class DeviceStatsMonitor(Callback):
    """Record device memory on every logged step: on CUDA the allocated, reserved and peak
    memory in MB; on MPS the allocated and driver memory. ``cpu_stats=True`` adds CPU
    utilisation and RAM (needs ``psutil``). ``filter_keys`` keeps only the stats whose
    names contain one of the given strings (``{"memory_allocated"}`` keeps
    ``device/memory_allocated_mb``); a key that matches no stat gives one warning."""

    def __init__(self, cpu_stats: bool | None = None, filter_keys: set[str] | None = None) -> None:
        self.cpu_stats = cpu_stats
        self._filter_keys = set(filter_keys) if filter_keys is not None else None
        self._warned = False

    def _filter(self, stats: dict[str, float]) -> dict[str, float]:
        keys = self._filter_keys
        if keys is None:
            return stats
        if not self._warned:
            self._warned = True
            unmatched = sorted(k for k in keys if not any(k in name for name in stats))
            if unmatched:
                import warnings

                warnings.warn(
                    f"DeviceStatsMonitor: filter_keys {unmatched} match no device stat "
                    f"(available: {sorted(stats)}) and are ignored",
                    stacklevel=2,
                )
        return {name: v for name, v in stats.items() if any(k in name for k in keys)}

    def _stats(self, trainer: Trainer) -> dict[str, float]:
        import torch

        out: dict[str, float] = {}
        rt = trainer.runtime
        mb = 1024**2
        if rt is not None and rt.device.type == "cuda":
            dev = rt.device
            out["device/memory_allocated_mb"] = torch.cuda.memory_allocated(dev) / mb
            out["device/memory_reserved_mb"] = torch.cuda.memory_reserved(dev) / mb
            out["device/max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(dev) / mb
        elif rt is not None and rt.device.type == "mps":
            out["device/memory_allocated_mb"] = torch.mps.current_allocated_memory() / mb
            out["device/driver_memory_mb"] = torch.mps.driver_allocated_memory() / mb
        if self.cpu_stats or (
            self.cpu_stats is None and rt is not None and rt.device.type == "cpu"
        ):
            try:
                import psutil
            except ImportError:
                if self.cpu_stats:
                    raise ValidationFailed(
                        "DeviceStatsMonitor(cpu_stats=True) needs psutil"
                    ) from None
                return out
            out["cpu/percent"] = float(psutil.cpu_percent())
            out["cpu/ram_used_mb"] = psutil.Process().memory_info().rss / mb
        return out

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if trainer._is_log_step:
            stats = self._filter(self._stats(trainer))
            if stats:
                trainer.log_metrics(stats)


class TrainingSpikeException(RuntimeError):
    def __init__(self, batch_idx: int, value: float) -> None:
        super().__init__(f"training loss spiked at batch {batch_idx}: {value}")
        self.batch_idx, self.value = batch_idx, value


class SpikeDetection(Callback):
    """Raise ``TrainingSpikeException`` when the training loss jumps: above
    ``mean * (1 + rtol)`` plus ``atol`` of the last ``window`` losses (``mode="relative"``),
    or non-finite (``finite_only``). Reads the loss every step, which waits for the GPU, so
    use it while debugging a run."""

    def __init__(
        self,
        mode: Literal["relative", "absolute"] = "relative",
        window: int = 10,
        warmup: int = 1,
        atol: float | None = None,
        rtol: float | None = 2.0,
        exclude_batches_path: str | None = None,
        finite_only: bool = True,
    ) -> None:
        if mode not in ("relative", "absolute"):
            raise ValidationFailed("mode must be 'relative' or 'absolute'")
        self.mode, self.window, self.warmup = mode, window, warmup
        self.atol, self.rtol = atol, rtol
        self.finite_only = finite_only
        self.bad_batches: list[int] = []
        self._losses: deque[float] = deque(maxlen=window)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        value = float(loss)
        if self.finite_only and not math.isfinite(value):
            self.bad_batches.append(batch_idx)
            raise TrainingSpikeException(batch_idx, value)
        if len(self._losses) >= max(self.warmup, 1):
            mean = sum(self._losses) / len(self._losses)
            limit = (self.atol or 0.0) + (
                abs(mean) * (1 + (self.rtol or 0.0)) if self.mode == "relative" else abs(mean)
            )
            if value > limit:
                self.bad_batches.append(batch_idx)
                raise TrainingSpikeException(batch_idx, value)
        self._losses.append(value)


def summarize(model: Any, max_depth: int = 1) -> list[tuple[str, str, int, str]]:
    """Rows ``(name, type, parameters, mode)`` for the submodules up to ``max_depth``."""
    rows = []
    for name, m in model.named_modules():
        if not name:
            continue
        depth = name.count(".") + 1
        if max_depth >= 0 and depth > max_depth:
            continue
        n = sum(p.numel() for p in m.parameters())
        rows.append((name, type(m).__name__, n, "train" if m.training else "eval"))
    return rows


def _human(n: float) -> str:
    for unit in ("", " K", " M", " B", " T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000
    return f"{n:.1f} P"


class ModelSummary(Callback):
    """Print the layers and parameter counts when training starts (rank 0). ``max_depth=-1``
    lists every submodule. Extra keyword arguments go to ``summarize`` (``RichModelSummary``
    takes ``header_style``)."""

    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        self.max_depth = max_depth
        self._summarize_kwargs = summarize_kwargs

    def on_fit_start(self, trainer: Trainer, pl_module: Module) -> None:
        if not trainer.is_global_zero:
            return
        rows = summarize(pl_module, self.max_depth)
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        size_mb = sum(p.numel() * p.element_size() for p in pl_module.parameters()) / 1024**2
        modes = {"train": 0, "eval": 0}
        for m in pl_module.modules():
            modes["train" if m.training else "eval"] += 1
        summary_data: list[tuple[str, list[str]]] = [
            (" ", [str(i) for i in range(len(rows))]),
            ("Name", [r[0] for r in rows]),
            ("Type", [r[1] for r in rows]),
            ("Params", [_human(r[2]) for r in rows]),
            ("Mode", [r[3] for r in rows]),
        ]
        self.summarize(summary_data, total, trainable, size_mb, modes, **self._summarize_kwargs)

    @staticmethod
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        """Print the summary. ``summary_data`` holds the table as ``(column name, values)``
        pairs; ``model_size`` is in MB; ``total_training_modes`` counts the modules in
        ``"train"`` and ``"eval"`` mode. Override it to change the output."""
        columns = dict(summary_data)
        names, kinds = columns["Name"], columns["Type"]
        if names:
            w = max(len(n) for n in names)
            t = max(len(k) for k in kinds)
            print(f"  | {'Name':<{w}} | {'Type':<{t}} | Params | Mode", flush=False)
            for i, name, kind, n, mode in zip(
                columns[" "], names, kinds, columns["Params"], columns["Mode"], strict=True
            ):
                print(f"{i:<2}| {name:<{w}} | {kind:<{t}} | {n:>6} | {mode}")
        print(
            f"{_human(trainable_parameters)} trainable params, "
            f"{_human(total_parameters - trainable_parameters)} non-trainable, "
            f"{model_size:.3f} MB",
            flush=True,
        )


def _require_rich(what: str) -> None:
    try:
        import rich  # noqa: F401
    except ImportError:
        raise ValidationFailed(
            f"{what} needs the rich package: pip install rich (or use the plain one)"
        ) from None


class RichModelSummary(ModelSummary):
    """``ModelSummary`` printed as a ``rich`` table (needs ``pip install rich``).
    ``header_style`` sets the style of the table header (default ``"bold magenta"``)."""

    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        _require_rich("RichModelSummary")
        super().__init__(max_depth, **summarize_kwargs)

    @staticmethod
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        """Print the layers as a ``rich`` table, then the parameter counts and model size."""
        from rich import get_console
        from rich.table import Table

        console = get_console()
        table = Table(header_style=summarize_kwargs.get("header_style", "bold magenta"))
        for name, _ in summary_data:
            if name == " ":
                table.add_column(" ", style="dim")
            elif name == "Name":
                table.add_column("Name", justify="left", no_wrap=True)
            elif name == "Params":
                table.add_column("Params", justify="right")
            else:
                table.add_column(name)
        for row in zip(*(values for _, values in summary_data), strict=True):
            table.add_row(*row)
        console.print(table)
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(f"[bold]Trainable params[/]: {_human(trainable_parameters)}")
        grid.add_row(
            f"[bold]Non-trainable params[/]: {_human(total_parameters - trainable_parameters)}"
        )
        grid.add_row(f"[bold]Total params[/]: {_human(total_parameters)}")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {model_size:.3f}")
        grid.add_row(f"[bold]Modules in train mode[/]: {total_training_modes.get('train', 0)}")
        grid.add_row(f"[bold]Modules in eval mode[/]: {total_training_modes.get('eval', 0)}")
        console.print(grid)


class ProgressBar(Callback):
    """One progress line per epoch with the step count, rate and the values of the last
    logged step (no extra device reads). Redrawn in place on a terminal at most
    ``refresh_rate`` times per second; elsewhere (job logs) it prints nothing per step.
    Uses ``tqdm`` when installed."""

    def __init__(self, refresh_rate: float = 4.0, leave: bool = True) -> None:
        self.refresh_rate = refresh_rate
        self.leave = leave
        self._enabled = True
        self._bar: Any = None
        self._last = 0.0
        self._t0 = 0.0

    def disable(self) -> None:
        """Stop drawing the bar."""
        self._enabled = False

    def enable(self) -> None:
        """Draw the bar again."""
        self._enabled = True

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._active = self._enabled and trainer.is_global_zero and sys.stdout.isatty()
        if not self._active:
            return
        total = trainer.num_training_batches if trainer.num_training_batches != math.inf else None
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=total, desc=f"Epoch {trainer.current_epoch}", leave=self.leave)
        except ImportError:
            self._bar = None
        self._t0 = self._last = time.monotonic()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not getattr(self, "_active", False):
            return
        now = time.monotonic()
        if self._bar is not None:
            self._bar.update(1)
        if now - self._last < 1.0 / self.refresh_rate:
            return
        self._last = now
        shown = " ".join(f"{k}={v:.4g}" for k, v in trainer.progress_bar_metrics.items())
        if self._bar is not None:
            self._bar.set_postfix_str(shown, refresh=False)
        else:
            rate = (batch_idx + 1) / max(now - self._t0, 1e-9)
            sys.stdout.write(
                f"\rEpoch {trainer.current_epoch}: {batch_idx + 1} batches, {rate:.1f} it/s {shown}"
            )
            sys.stdout.flush()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if not getattr(self, "_active", False):
            return
        if self._bar is not None:
            self._bar.close()
        else:
            sys.stdout.write("\n" if self.leave else "\r")
            sys.stdout.flush()


TQDMProgressBar = ProgressBar


@dataclass
class RichProgressBarTheme:
    """Styles of the parts of ``RichProgressBar`` (``rich`` style strings or ``Style``s), as
    in Lightning: ``description`` (``Epoch N``), ``progress_bar``, ``progress_bar_finished``,
    ``progress_bar_pulse`` (unknown length), ``batch_progress`` (``10/50``), ``time``,
    ``processing_speed`` and ``metrics``; ``metrics_text_delimiter`` goes between metrics
    and ``metrics_format`` formats their values."""

    description: Any = ""
    progress_bar: Any = "#6206E0"
    progress_bar_finished: Any = "#6206E0"
    progress_bar_pulse: Any = "#6206E0"
    batch_progress: Any = ""
    time: Any = "dim"
    processing_speed: Any = "dim underline"
    metrics: Any = "italic"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".3f"


class RichProgressBar(ProgressBar):
    """``ProgressBar`` drawn with ``rich`` (needs ``pip install rich``): a bar per epoch with
    the batch count, elapsed time, speed and the values of the last logged step. As
    ``ProgressBar``, it draws only on a terminal (rank 0) and at most ``refresh_rate`` times
    per second (``0`` turns it off), with no extra device reads. ``leave=True`` keeps each
    finished epoch's bar on screen; ``theme`` is a ``RichProgressBarTheme`` and
    ``console_kwargs`` go to ``rich.console.Console``."""

    def __init__(
        self,
        refresh_rate: float = 4.0,
        leave: bool = False,
        theme: RichProgressBarTheme | None = None,
        console_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _require_rich("RichProgressBar")
        super().__init__(refresh_rate=refresh_rate, leave=leave)
        self.theme = theme or RichProgressBarTheme()
        self._console_kwargs = console_kwargs or {}
        self.progress: Any = None
        self._task: Any = None
        self._total: int | None = None
        self._done = 0

    def _metrics_text(self, trainer: Trainer) -> str:
        fmt = self.theme.metrics_format
        parts = []
        for k, v in trainer.progress_bar_metrics.items():
            try:
                parts.append(f"{k}: {v:{fmt}}")
            except (TypeError, ValueError):
                parts.append(f"{k}: {v}")
        return self.theme.metrics_text_delimiter.join(parts)

    def _draw(self, trainer: Trainer, now: float) -> None:
        rate = self._done / max(now - self._t0, 1e-9)
        total = "?" if self._total is None else str(self._total)
        self.progress.update(
            self._task,
            completed=self._done,
            batches=f"{self._done}/{total}",
            speed=f"{rate:.2f}it/s",
            metrics=self._metrics_text(trainer),
        )
        self.progress.refresh()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Module) -> None:
        self._stop()
        self._active = (
            self._enabled
            and self.refresh_rate > 0
            and trainer.is_global_zero
            and sys.stdout.isatty()
        )
        if not self._active:
            return
        from rich.console import Console
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

        th = self.theme
        n = trainer.num_training_batches
        self._total = None if n == math.inf else int(n)
        self.progress = Progress(
            TextColumn("{task.description}", style=th.description),
            BarColumn(
                complete_style=th.progress_bar,
                finished_style=th.progress_bar_finished,
                pulse_style=th.progress_bar_pulse,
            ),
            TextColumn("{task.fields[batches]}", style=th.batch_progress),
            TimeElapsedColumn(),
            TextColumn("{task.fields[speed]}", style=th.processing_speed),
            TextColumn("{task.fields[metrics]}", style=th.metrics, markup=False),
            console=Console(**self._console_kwargs),
            auto_refresh=False,  # redrawn from on_train_batch_end, rate-limited
            transient=not self.leave,
        )
        self.progress.start()
        self._task = self.progress.add_task(
            f"Epoch {trainer.current_epoch}",
            total=self._total,
            batches=f"0/{'?' if self._total is None else self._total}",
            speed="",
            metrics="",
        )
        self._done = 0
        self._t0 = self._last = time.monotonic()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Module, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if not getattr(self, "_active", False) or self.progress is None:
            return
        self._done += 1
        if self._total is None and trainer.num_training_batches != math.inf:
            # Known only once the epoch's loop starts (the first epoch).
            self._total = int(trainer.num_training_batches)
            self.progress.update(self._task, total=self._total)
        now = time.monotonic()
        if now - self._last < 1.0 / self.refresh_rate:
            return
        self._last = now
        self._draw(trainer, now)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self.progress is not None:
            self._draw(trainer, time.monotonic())
        self._stop()

    def on_train_end(self, trainer: Trainer, pl_module: Module) -> None:
        self._stop()

    def on_exception(self, trainer: Trainer, pl_module: Module, exception: BaseException) -> None:
        self._stop()

    def _stop(self) -> None:
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self._task = None


class BasePredictionWriter(Callback):
    """Write predictions as they are made, instead of keeping them in memory. Subclass and
    implement ``write_on_batch_end`` and/or ``write_on_epoch_end``; ``write_interval`` is
    ``"batch"``, ``"epoch"`` or ``"batch_and_epoch"``."""

    def __init__(
        self, write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch"
    ) -> None:
        if write_interval not in ("batch", "epoch", "batch_and_epoch"):
            raise ValidationFailed("write_interval must be 'batch', 'epoch' or 'batch_and_epoch'")
        self.interval = write_interval

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        prediction: Any,
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write one batch's prediction; ``batch_indices`` are its samples' dataset indices."""
        raise NotImplementedError

    def write_on_epoch_end(
        self, trainer: Trainer, pl_module: Module, predictions: list[Any], batch_indices: Any
    ) -> None:
        """Write all predictions at the end, with each batch's dataset indices."""
        raise NotImplementedError

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: Module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.interval in ("batch", "batch_and_epoch"):
            indices = trainer._predict_indices[batch_idx] if trainer._predict_indices else None
            self.write_on_batch_end(
                trainer, pl_module, outputs, indices, batch, batch_idx, dataloader_idx
            )

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: Module) -> None:
        if self.interval in ("epoch", "batch_and_epoch"):
            self.write_on_epoch_end(
                trainer, pl_module, trainer._predictions, trainer._predict_indices
            )
