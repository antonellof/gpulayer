"""Profilers for the training loop, named as in PyTorch Lightning.

    Trainer(profiler="simple")      # time per loop action, printed at the end
    Trainer(profiler="advanced")    # cProfile per action
    Trainer(profiler="pytorch")     # torch.profiler: operators, devices, optional trace
    Trainer(profiler="xla")         # torch_xla profiler server and traces, for TPUs

Without a profiler the loop has no profiling code on its path. With ``dirpath`` the report is
also written to ``<dirpath>/<filename>.txt`` (rank 0).
"""

from __future__ import annotations

import cProfile
import importlib
import inspect
import io
import os
import pstats
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gpulayer.errors import ValidationFailed
from gpulayer.pytorch.storage import RemotePath, as_path


class Profiler:
    """Base class: implement ``start(action)``, ``stop(action)`` and ``summary()``."""

    def __init__(self, dirpath: str | Path | None = None, filename: str | None = None) -> None:
        self.dirpath = as_path(dirpath) if dirpath is not None else None
        self.filename = filename
        self._local_rank: int | None = None

    @property
    def local_rank(self) -> int:
        """This process's rank on its machine: the one given to ``setup``, else ``LOCAL_RANK``
        (0 when unset)."""
        if self._local_rank is not None:
            return self._local_rank
        return int(os.environ.get("LOCAL_RANK", "0"))

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
        self._local_rank = local_rank
        if self.dirpath is None and log_dir is not None:
            self.dirpath = as_path(log_dir)

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
    ``sort_by_key`` picks the table's order, e.g. ``"cuda_time_total"``.

    ``record_module_names`` labels each submodule's forward with its class (a range per call,
    added only while profiling). ``group_by_input_shapes`` (with ``record_shapes=True``) splits
    the table by input shape; ``table_kwargs`` go to ``EventList.table``. ``emit_nvtx=True``
    emits NVTX ranges for ``nsys``/``nvprof`` instead (CUDA only, no table)."""

    def __init__(
        self,
        dirpath: str | Path | None = None,
        filename: str | None = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = False,
        row_limit: int = 20,
        sort_by_key: str | None = None,
        record_module_names: bool = True,
        table_kwargs: dict[str, Any] | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        **profiler_kwargs: Any,
    ) -> None:
        super().__init__(dirpath, filename)
        self.export_to_chrome = export_to_chrome
        self.row_limit = row_limit
        self.sort_by_key = sort_by_key
        self._group_by_input_shapes = group_by_input_shapes and record_shapes
        self._emit_nvtx = emit_nvtx
        self._record_module_names = record_module_names
        self._table_kwargs = dict(table_kwargs or {})
        from torch.autograd.profiler_util import EventList

        valid = set(inspect.signature(EventList.table).parameters) - {
            "self",
            "sort_by",
            "row_limit",
        }
        for key in self._table_kwargs:
            if key in ("sort_by", "row_limit"):
                raise ValidationFailed(f"table_kwargs: {key!r} is already an argument")
            if key not in valid:
                raise ValidationFailed(f"table_kwargs: {key!r} is not one of {sorted(valid)}")
        self._kwargs = {
            "record_shapes": record_shapes,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            **profiler_kwargs,
        }
        self._prof: Any = None
        self._parent: Any = None
        self._ranges: dict[str, Any] = {}
        self._hooks: list[Any] = []
        self._module_ranges: dict[int, list[Any]] = {}
        self._cuda = False

    def setup(self, stage: str, local_rank: int | None = None, log_dir: str | None = None) -> None:
        super().setup(stage, local_rank, log_dir)
        import torch
        from torch.profiler import ProfilerActivity, profile

        if self._emit_nvtx:
            if not torch.cuda.is_available():
                raise ValidationFailed("PyTorchProfiler(emit_nvtx=True) needs CUDA")
            nvtx = torch.autograd.profiler.emit_nvtx
            params = inspect.signature(nvtx.__init__).parameters
            self._prof = nvtx(**{k: v for k, v in self._kwargs.items() if k in params})
            self._prof.__enter__()
            self._parent = torch.cuda.profiler.profile()
            self._parent.__enter__()
        else:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
                self._cuda = True
            self._prof = profile(activities=activities, **self._kwargs)
            self._prof.__enter__()
        if self._record_module_names:
            self._register_module_ranges()

    def _register_module_ranges(self) -> None:
        import torch
        from torch.nn.modules.module import (
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )

        ranges = self._module_ranges

        def pre(module: Any, args: Any) -> None:
            cls = type(module)
            rf = torch.autograd.profiler.record_function(f"[module]{cls.__module__}.{cls.__name__}")
            rf.__enter__()
            ranges.setdefault(id(module), []).append(rf)

        def post(module: Any, args: Any, output: Any) -> None:
            stack = ranges.get(id(module))
            if stack:
                stack.pop().__exit__(None, None, None)

        self._hooks = [register_module_forward_pre_hook(pre), register_module_forward_hook(post)]

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
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._module_ranges.clear()
        if self._prof is not None and self._ranges == {}:
            self._prof.__exit__(None, None, None)
            if self._parent is not None:
                self._parent.__exit__(None, None, None)
                self._parent = None
            if self.export_to_chrome and not self._emit_nvtx and self.dirpath is not None:
                self.dirpath.mkdir(parents=True, exist_ok=True)
                name = f"{self.filename or 'trace'}.pt.trace.json"
                if isinstance(self.dirpath, RemotePath):
                    with tempfile.TemporaryDirectory() as tmp:  # the profiler writes a local file
                        self._prof.export_chrome_trace(str(Path(tmp) / name))
                        (self.dirpath / name).put(Path(tmp) / name)
                else:
                    self._prof.export_chrome_trace(str(self.dirpath / name))

    def summary(self) -> str:
        if self._prof is None or self._emit_nvtx:
            return ""
        key = self.sort_by_key or ("cuda_time_total" if self._cuda else "cpu_time_total")
        try:
            data = self._prof.key_averages(group_by_input_shape=self._group_by_input_shapes)
            return str(data.table(sort_by=key, row_limit=self.row_limit, **self._table_kwargs))
        except Exception:  # the profiler has not finished collecting
            return ""


class XLAProfiler(Profiler):
    """torch_xla's profiler, for TPUs (needs ``torch_xla``). The first traced action starts the
    profiler server on ``port`` (local rank 0, one per machine), for TensorBoard's profile
    plugin or ``xp.trace`` to capture from. Training batches, backward, optimizer steps and
    evaluation loops are wrapped in ``xp.Trace``; ``validation_step``, ``test_step`` and
    ``predict_step`` actions in a numbered ``xp.StepTrace``."""

    STEP_FUNCTIONS = {"validation_step", "test_step", "predict_step"}
    RECORD_FUNCTIONS = {
        "training_step",
        "backward",
        "validation_step",
        "test_step",
        "predict_step",
        "run_training_batch",
        "optimizer_step",
        "run_validation",
        "run_test",
        "run_predict",
    }

    def __init__(self, port: int = 9012) -> None:
        try:
            self._xp: Any = importlib.import_module("torch_xla.debug.profiler")
        except ImportError as e:
            raise ValidationFailed("XLAProfiler needs torch_xla (a TPU VM)") from e
        super().__init__(dirpath=None, filename=None)
        self.port = port
        self.server: Any = None
        self._recording_map: dict[str, Any] = {}
        self._step_counts: dict[str, int] = {}
        self._start_trace = False

    def start(self, action_name: str) -> None:
        hook = action_name.split(".")[-1]
        if hook not in self.RECORD_FUNCTIONS:
            return
        if not self._start_trace:
            if self.local_rank == 0:
                self.server = self._xp.start_server(self.port)
            self._start_trace = True
        if hook in self.STEP_FUNCTIONS:
            self._step_counts[action_name] = self._step_counts.get(action_name, 0) + 1
            rec = self._xp.StepTrace(action_name, step_num=self._step_counts[action_name])
        else:
            rec = self._xp.Trace(action_name)
        rec.__enter__()
        self._recording_map[action_name] = rec

    def stop(self, action_name: str) -> None:
        rec = self._recording_map.pop(action_name, None)
        if rec is not None:
            rec.__exit__(None, None, None)


def resolve_profiler(profiler: str | Profiler | None) -> Profiler | None:
    if profiler is None or isinstance(profiler, Profiler):
        return profiler
    kinds: dict[str, type[Profiler]] = {
        "simple": SimpleProfiler,
        "advanced": AdvancedProfiler,
        "pytorch": PyTorchProfiler,
        "xla": XLAProfiler,
    }
    if profiler not in kinds:
        raise ValidationFailed(f"profiler must be one of {sorted(kinds)} or a Profiler")
    return kinds[profiler]()
