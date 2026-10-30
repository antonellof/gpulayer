"""Submit cost as the queue grows: one control plane, 30 four-slot workers, N submissions.

    python benchmarks/submit_queue.py [--jobs 2000 --workers 30 --slots 4]

Once every slot is busy, each submit should cost the same however many jobs are queued.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from gpulayer.artifacts.store import LocalArtifactStore
from gpulayer.contracts import (
    CommandJob,
    JobSpec,
    PlacementPolicy,
    WorkerCapabilities,
    WorkerOrigin,
)
from gpulayer.control_plane.service import ControlPlane
from gpulayer.storage.sqlite import SqliteStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--slots", type=int, default=4)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    cp = ControlPlane(SqliteStore(tmp / "db.sqlite"), LocalArtifactStore(tmp / "a"))
    for i in range(args.workers):
        caps = WorkerCapabilities(
            hostname=f"h{i}",
            os="linux",
            cpu_arch="x86_64",
            cpu_cores=4,
            python_version="3.12.0",
            origin=WorkerOrigin.EXISTING,
            max_concurrent_tasks=args.slots,
        )
        cp.register_worker(f"w{i}", caps)
    spec = JobSpec(
        workload=CommandJob(command=["true"]), placement=PlacementPolicy(target="remote")
    )
    step = max(1, args.jobs // 8)
    last = time.perf_counter()
    for n in range(1, args.jobs + 1):
        cp.submit(spec)
        if n % step == 0:
            now = time.perf_counter()
            print(f"queued {n:6d}: {(now - last) / step * 1000:6.2f} ms per submit")
            last = now


if __name__ == "__main__":
    main()
