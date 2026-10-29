"""Cheap resource sampling for worker heartbeats, without extra dependencies or privileges.

- load average: ``os.getloadavg`` (Linux, macOS)
- memory: ``/proc/meminfo`` on Linux; ``sysctl hw.memsize`` + ``vm_stat`` on macOS
- processes: OS processes in the process groups of running tasks (``ps``)
- NVIDIA GPUs: ``nvidia-smi`` utilisation and memory. Apple GPU utilisation needs root
  (``powermetrics``), so it is not reported.

Every probe is best effort: a missing tool leaves that field empty.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterable

from gpulayer.contracts import Accelerator, GpuMetrics, WorkerCapabilities, WorkerMetrics

_GB = 1024**3


def _run(argv: list[str], timeout: float = 2.0) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def _memory_gb() -> tuple[float | None, float | None]:
    """(used, total) in GB."""
    if sys.platform.startswith("linux"):
        try:
            info = dict(
                (line.split(":")[0], int(line.split()[1]) * 1024)
                for line in open("/proc/meminfo")  # noqa: SIM115
                if line.split(":")[0] in ("MemTotal", "MemAvailable")
            )
        except (OSError, ValueError, IndexError):
            return None, None
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
        if total is None or avail is None:
            return None, None
        return round((total - avail) / _GB, 2), round(total / _GB, 2)
    if sys.platform == "darwin":
        total_s, vm = _run(["sysctl", "-n", "hw.memsize"]), _run(["vm_stat"])
        if not total_s or not vm:
            return None, None
        page = int(m.group(1)) if (m := re.search(r"page size of (\d+) bytes", vm)) else 4096
        pages = {
            k.strip(): int(v.strip().rstrip("."))
            for k, v in (line.split(":", 1) for line in vm.splitlines()[1:] if ":" in line)
            if v.strip().rstrip(".").isdigit()
        }
        # What Activity Monitor calls "memory used": app (active) + wired + compressed.
        used = sum(
            pages.get(k, 0)
            for k in ("Pages active", "Pages wired down", "Pages occupied by compressor")
        )
        return round(used * page / _GB, 2), round(int(total_s) / _GB, 2)
    return None, None


def _processes(pgids: Iterable[int]) -> int | None:
    groups = set(pgids)
    if not groups:
        return 0
    out = _run(["ps", "-A", "-o", "pgid="])
    if out is None:
        return None
    return sum(1 for line in out.split() if line.isdigit() and int(line) in groups)


def _nvidia() -> list[GpuMetrics]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                GpuMetrics(
                    index=int(parts[0]),
                    utilization_pct=float(parts[1]),
                    memory_used_gb=round(float(parts[2]) / 1024, 2),
                    memory_total_gb=round(float(parts[3]) / 1024, 2),
                )
            )
        except ValueError:
            continue
    return gpus


def sample(
    caps: WorkerCapabilities, task_pgids: Iterable[int], running_tasks: int
) -> WorkerMetrics:
    try:
        load1: float | None = round(os.getloadavg()[0], 2)
    except OSError:
        load1 = None
    used, total = _memory_gb()
    has_nvidia = any(d.accelerator == Accelerator.CUDA for d in caps.accelerators)
    return WorkerMetrics(
        load1=load1,
        cpu_count=os.cpu_count(),
        memory_used_gb=used,
        memory_total_gb=total,
        processes=_processes(task_pgids),
        running_tasks=running_tasks,
        gpus=_nvidia() if has_nvidia else [],
    )
