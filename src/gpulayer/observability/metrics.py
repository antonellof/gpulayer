"""Prometheus metrics for the control plane (text exposition format 0.0.4).

Gauges are computed from current state on every scrape. Counters and histograms are stored
in the metadata database, so they only ever increase, across restarts and garbage collection.
Grafana reads them through a Prometheus data source; see docs/guides/monitoring.md.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING

from gpulayer.contracts import (
    API_VERSION,
    JobState,
    TaskState,
    WorkerCapabilities,
    WorkerOrigin,
    WorkerState,
)
from gpulayer.storage.sqlite import parse_ts

if TYPE_CHECKING:
    from gpulayer.control_plane.service import ControlPlane
    from gpulayer.storage.sqlite import SqliteStore

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Seconds. Jobs range from sub-second functions to multi-hour training runs.
BUCKETS = (1, 5, 15, 60, 300, 900, 3600, 4 * 3600, 24 * 3600)

HELP = {
    "gpulayer_jobs_submitted_total": ("counter", "Jobs submitted."),
    "gpulayer_jobs_finished_total": ("counter", "Jobs that reached a terminal state."),
    "gpulayer_tasks_lost_total": ("counter", "Tasks whose worker stopped heartbeating."),
    "gpulayer_job_run_seconds": ("histogram", "Time from the first task starting to the end."),
    "gpulayer_job_queue_wait_seconds": ("histogram", "Time from submission to the first start."),
}


def observe(store: SqliteStore, name: str, seconds: float, labels: dict[str, str]) -> None:
    """Record one histogram observation as cumulative bucket counters."""
    for le in BUCKETS:
        if seconds <= le:
            store.inc(f"{name}_bucket", {**labels, "le": str(le)})
    store.inc(f"{name}_bucket", {**labels, "le": "+Inf"})
    store.inc(f"{name}_sum", labels, seconds)
    store.inc(f"{name}_count", labels)


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""

    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    return "{" + ",".join(f'{k}="{esc(str(v))}"' for k, v in sorted(labels.items())) + "}"


def _family(name: str) -> str:
    for suffix in ("_bucket", "_sum", "_count"):
        if name.endswith(suffix) and name[: -len(suffix)] in HELP:
            return name[: -len(suffix)]
    return name


def _render(
    samples: Iterable[tuple[str, dict[str, str], float]], meta: dict[str, tuple[str, str]]
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name, labels, value in samples:
        fam = _family(name)
        if fam not in seen:
            seen.add(fam)
            kind, help_text = meta.get(fam, ("gauge", ""))
            out.append(f"# HELP {fam} {help_text}")
            out.append(f"# TYPE {fam} {kind}")
        out.append(f"{name}{_labels(labels)} {value:g}")
    return out


def _sample_order(sample: tuple[str, dict[str, str], float]) -> tuple[object, ...]:
    """Group by family and label set; histogram buckets ascending with +Inf last."""
    name, labels, _ = sample
    rest = sorted((k, v) for k, v in labels.items() if k != "le")
    suffix_rank = {"_bucket": 0, "_sum": 1, "_count": 2}
    rank = next((r for sfx, r in suffix_rank.items() if name.endswith(sfx)), 0)
    le = labels.get("le")
    bound = float("inf") if le == "+Inf" else float(le) if le else 0.0
    return (_family(name), rest, rank, bound)


def _bytes(gb: float | None) -> float | None:
    return None if gb is None else round(gb * 1024**3)


def render(cp: ControlPlane, now: datetime | None = None) -> str:
    """The full /metrics payload."""
    now = now or cp.now()
    store = cp.store
    gauges: list[tuple[str, dict[str, str], float]] = []
    meta: dict[str, tuple[str, str]] = {
        "gpulayer_info": ("gauge", "Build information."),
        "gpulayer_jobs": ("gauge", "Jobs currently in each state."),
        "gpulayer_tasks": ("gauge", "Tasks currently assigned or running."),
        "gpulayer_workers": ("gauge", "Registered workers by state and origin."),
        "gpulayer_accelerators": ("gauge", "Usable accelerator devices on online workers."),
        "gpulayer_queue_oldest_seconds": ("gauge", "Age of the oldest queued job."),
        "gpulayer_scrape_duration_seconds": ("gauge", "Time spent producing this response."),
        "gpulayer_worker_load1": ("gauge", "1-minute load average reported by the worker."),
        "gpulayer_worker_cpu_count": ("gauge", "CPU cores on the worker."),
        "gpulayer_worker_memory_used_bytes": ("gauge", "Memory in use on the worker."),
        "gpulayer_worker_memory_total_bytes": ("gauge", "Total memory on the worker."),
        "gpulayer_worker_processes": ("gauge", "OS processes belonging to running tasks."),
        "gpulayer_worker_running_tasks": ("gauge", "Tasks the worker is running."),
        "gpulayer_gpu_utilization_ratio": ("gauge", "GPU utilisation (NVIDIA, 0 to 1)."),
        "gpulayer_gpu_memory_used_bytes": ("gauge", "GPU memory in use (NVIDIA)."),
        **HELP,
    }
    t0 = time.perf_counter()
    gauges.append(("gpulayer_info", {"api_version": API_VERSION}, 1))

    counts = {
        r["state"]: r["n"] for r in store.query("SELECT state, COUNT(*) n FROM jobs GROUP BY state")
    }
    for st in JobState:
        gauges.append(("gpulayer_jobs", {"state": st.value}, counts.get(st.value, 0)))

    tcounts = {
        r["state"]: r["n"]
        for r in store.query(
            "SELECT state, COUNT(*) n FROM tasks WHERE state IN (?,?) GROUP BY state",
            (TaskState.ASSIGNED.value, TaskState.RUNNING.value),
        )
    }
    for ts_ in (TaskState.ASSIGNED, TaskState.RUNNING):
        gauges.append(("gpulayer_tasks", {"state": ts_.value}, tcounts.get(ts_.value, 0)))

    workers: dict[tuple[str, str], int] = defaultdict(int)
    devices: dict[str, int] = defaultdict(int)
    for r in store.query("SELECT state, capabilities FROM workers WHERE revoked=0"):
        caps = WorkerCapabilities.model_validate_json(r["capabilities"])
        workers[(r["state"], caps.origin.value)] += 1
        if r["state"] != WorkerState.OFFLINE.value:
            for d in caps.accelerators:
                if d.execution_supported:
                    devices[d.accelerator.value] += 1
    # Every combination, so dashboards show 0 rather than "no data".
    for wstate in WorkerState:
        for origin in WorkerOrigin:
            n = workers.get((wstate.value, origin.value), 0)
            gauges.append(("gpulayer_workers", {"state": wstate.value, "origin": origin.value}, n))
    for accel in ("cuda", "mps", "rocm"):
        gauges.append(("gpulayer_accelerators", {"accelerator": accel}, devices.get(accel, 0)))

    # Per-worker resource gauges from each worker's latest sample (last minute only).
    names = {r["id"]: r["name"] for r in store.query("SELECT id, name FROM workers")}
    for wid, m in sorted(cp.latest_samples().items()):
        lbl = {"worker_id": wid, "worker": names.get(wid, wid)}
        for name, value in (
            ("gpulayer_worker_load1", m.load1),
            ("gpulayer_worker_cpu_count", m.cpu_count),
            ("gpulayer_worker_memory_used_bytes", _bytes(m.memory_used_gb)),
            ("gpulayer_worker_memory_total_bytes", _bytes(m.memory_total_gb)),
            ("gpulayer_worker_processes", m.processes),
            ("gpulayer_worker_running_tasks", m.running_tasks),
        ):
            if value is not None:
                gauges.append((name, lbl, float(value)))
        for g in m.gpus:
            glbl = {**lbl, "gpu": str(g.index)}
            if g.utilization_pct is not None:
                gauges.append(("gpulayer_gpu_utilization_ratio", glbl, g.utilization_pct / 100))
            if g.memory_used_gb is not None:
                gauges.append(
                    ("gpulayer_gpu_memory_used_bytes", glbl, _bytes(g.memory_used_gb) or 0)
                )

    oldest = store.one("SELECT MIN(created_at) t FROM jobs WHERE state=?", (JobState.QUEUED.value,))
    created = parse_ts(oldest["t"]) if oldest and oldest["t"] else None
    age = (now - created).total_seconds() if created else 0.0
    gauges.append(("gpulayer_queue_oldest_seconds", {}, max(age, 0.0)))

    counters = store.counters()
    lines = _render(gauges, meta)
    lines += _render(sorted(counters, key=_sample_order), meta)
    lines += _render(
        [("gpulayer_scrape_duration_seconds", {}, round(time.perf_counter() - t0, 6))], meta
    )
    return "\n".join(lines) + "\n"
