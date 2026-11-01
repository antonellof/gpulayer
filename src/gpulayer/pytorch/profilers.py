"""Profilers for the training loop, named as in PyTorch Lightning.

    Trainer(profiler="simple")      # time per loop action, printed at the end
    Trainer(profiler="advanced")    # cProfile per action
    Trainer(profiler="pytorch")     # torch.profiler: operators, devices, optional trace

Without a profiler the loop has no profiling code on its path. With ``dirpath`` the report is
also written to ``<dirpath>/<filename>.txt`` (rank 0).
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gpulayer.errors import ValidationFailed


class Profiler:
    """Base class: implement ``start(action)``, ``stop(action)`` and ``summary()``."""

    def __init__(self, dirpath: str | Path | None = None, filename: str | None = None) -> None:
        self.dirpath = Path(dirpath) if dirpath is not None else None
        self.filename = filename

    def start(self, action_name: str) -> None:
        """Start timing ``action_name``."""

    def stop(self, action_name: str) -> None:
        """Stop timing ``action_name``."""

    def summary(self) -> str:
        """The report as text."""
        return ""

    @contextmanager
    def profile(self, action_name: str) -> Iterator[None]:
        """Context manager: time the code inside as ``action_name``."""
        self.start(action_name)
        try:
            yield
        finally:
            self.stop(action_name)

    def setup(self, stage: str, local_rank: int | None = None, log_dir: str | None = None) -> None:
        """Called when a run starts; ``log_dir`` becomes ``dirpath`` when none was given."""
        if self.dirpath is None and log_dir is not None:
            self.dirpath = Path(log_dir)

    def describe(self) -> str:
        """The report, also written to ``dirpath/filename`` when both are set."""
        text = self.summary()
        if text and self.dirpath is not None and self.filename:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            (self.dirpath / f"{self.filename}.txt").write_text(text)
        return text

    def teardown(self, stage: str | None = None) -> None:
        """Called when the run ends, before the report is printed."""


class PassThroughProfiler(Profiler):
    """Does nothing."""


class SimpleProfiler(Profiler):
    """Wall time of each loop action: calls, mean and total seconds, share of the run."""

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        extended: bool = True,
    ) -> None:
        super().__init__(dirpath, filename)
        self.recorded_durations: dict[str, list[float]] = defaultdict(list)
        self._current: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self.extended = extended

    def start(self, action_name: str) -> None:
        if action_name in self._current:
            raise ValidationFailed(f"profiler action {action_name!r} is already running")
        self._current[action_name] = time.perf_counter()

    def stop(self, action_name: str) -> None:
        start = self._current.pop(action_name, None)
        if start is None:
            raise ValidationFailed(f"profiler action {action_name!r} was not started")
        self.recorded_durations[action_name].append(time.perf_counter() - start)

    def summary(self) -> str:
        total = time.perf_counter() - self._t0
        rows = sorted(
            ((k, len(v), sum(v)) for k, v in self.recorded_durations.items()),
            key=lambda r: -r[2],
        )
        if not rows:
            return ""
        w = max(len(r[0]) for r in rows + [("Action", 0, 0.0)])
        lines = [
            "Profiler report",
            f"{'Action':<{w}}  {'Calls':>7}  {'Mean (s)':>10}  {'Total (s)':>10}  {'%':>6}",
        ]
        for name, n, tot in rows:
            lines.append(
                f"{name:<{w}}  {n:>7}  {tot / n:>10.6f}  {tot:>10.4f}  {100 * tot / total:>5.1f}%"
            )
        lines.append(f"{'Total':<{w}}  {'':>7}  {'':>10}  {total:>10.4f}")
        return "\n".join(lines)


class AdvancedProfiler(Profiler):
    """``cProfile`` per action: the functions that took the most time in each."""

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        line_count_restriction: float = 1.0,
        dump_stats: bool = False,
    ) -> None:
        super().__init__(dirpath, filename)
        self.profiled_actions: dict[str, cProfile.Profile] = {}
        self.line_count_restriction = line_count_restriction

    def start(self, action_name: str) -> None:
        # One cProfile can run at a time (Python 3.12+): a nested action is already inside
        # the outer one's profile, so only the outermost action is profiled.
        self._stack = getattr(self, "_stack", [])
        self._stack.append(action_name)
        if len(self._stack) == 1:
            self.profiled_actions.setdefault(action_name, cProfile.Profile()).enable()

    def stop(self, action_name: str) -> None:
        stack = getattr(self, "_stack", [])
        if not stack or stack[-1] != action_name:
            raise ValidationFailed(f"profiler action {action_name!r} was not started")
        stack.pop()
        if not stack:
            self.profiled_actions[action_name].disable()

    def summary(self) -> str:
        parts = []
        for name, pr in self.profiled_actions.items():
            s = io.StringIO()
            pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(
                self.line_count_restriction
            )
            parts.append(f"Profile stats for: {name}\n{s.getvalue()}")
        return "\n".join(parts)


class PyTorchProfiler(Profiler):
    """``torch.profiler`` over the whole fit, with each loop action as a labelled range.
    ``export_to_chrome=True`` writes a trace for chrome://tracing or Perfetto (rank 0).
    ``sort_by_key`` picks the table's order, e.g. ``"cuda_time_total"``."""

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        export_to_chrome: bool = False,
        row_limit: int = 20,
        sort_by_key: str | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        **profiler_kwargs: Any,
    ) -> None:
        super().__init__(dirpath, filename)
        self.export_to_chrome = export_to_chrome
        self.row_limit = row_limit
        self.sort_by_key = sort_by_key
        self._kwargs = {
            "record_shapes": record_shapes,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            **profiler_kwargs,
        }
        self._prof: Any = None
        self._ranges: dict[str, Any] = {}
        self._cuda = False

    def setup(self, stage: str, local_rank: int | None = None, log_dir: str | None = None) -> None:
        super().setup(stage, local_rank, log_dir)
        import torch
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
            self._cuda = True
        self._prof = profile(activities=activities, **self._kwargs)
        self._prof.__enter__()

    def start(self, action_name: str) -> None:
        import torch

        rf = torch.autograd.profiler.record_function(action_name)
        rf.__enter__()
        self._ranges[action_name] = rf

    def stop(self, action_name: str) -> None:
        rf = self._ranges.pop(action_name, None)
        if rf is not None:
            rf.__exit__(None, None, None)

    def teardown(self, stage: str | None = None) -> None:
        if self._prof is not None and self._ranges == {}:
            self._prof.__exit__(None, None, None)
            if self.export_to_chrome and self.dirpath is not None:
                self.dirpath.mkdir(parents=True, exist_ok=True)
                self._prof.export_chrome_trace(
                    str(self.dirpath / f"{self.filename or 'trace'}.pt.trace.json")
                )

    def summary(self) -> str:
        if self._prof is None:
            return ""
        key = self.sort_by_key or ("cuda_time_total" if self._cuda else "cpu_time_total")
        try:
            return str(self._prof.key_averages().table(sort_by=key, row_limit=self.row_limit))
        except Exception:  # the profiler has not finished collecting
            return ""


def resolve_profiler(profiler: str | Profiler | None) -> Profiler | None:
    if profiler is None or isinstance(profiler, Profiler):
        return profiler
    kinds = {"simple": SimpleProfiler, "advanced": AdvancedProfiler, "pytorch": PyTorchProfiler}
    if profiler not in kinds:
        raise ValidationFailed(f"profiler must be one of {sorted(kinds)} or a Profiler")
    return kinds[profiler]()
