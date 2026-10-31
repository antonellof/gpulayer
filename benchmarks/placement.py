"""Time the scheduler's placement of a distributed job (pure function, no processes).

    python benchmarks/placement.py [--workers 30 120 500] [--nodes 8] [--repeat 2000]

Workers are spread over zones in two regions, as on a cloud fleet; half of them are busy so
the group has to be found among the rest. Prints microseconds per `schedule()` call (median
of --repeat calls) and the locality of the chosen group. Run it on the same machine before
and after a scheduler change.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpulayer.contracts import (  # noqa: E402
    DistributedJob,
    JobSpec,
    PlacementPolicy,
    ResourceRequest,
    WorkerCapabilities,
    WorkerOrigin,
    WorkerState,
)
from gpulayer.control_plane.scheduler import WorkerView, schedule  # noqa: E402

ZONES = ["europe-west1-b", "europe-west1-c", "europe-west1-d", "europe-west4-a", "europe-west4-b"]


def fleet(n: int) -> list[WorkerView]:
    out = []
    for i in range(n):
        zone = ZONES[i % len(ZONES)]
        caps = dict(
            hostname=f"h{i}",
            os="linux",
            cpu_arch="x86_64",
            cpu_cores=4,
            memory_gb=15.0,
            python_version="3.12.3",
            frameworks={"torch": "2.15.0"},
            distributed_backends=["gloo"],
            launchers=["direct", "torchrun"],
            advertise_address=f"10.0.0.{i % 250}",
            origin=WorkerOrigin.EXISTING,
            max_concurrent_tasks=4,
        )
        # Only set location fields the installed contracts know, so this runs before and after.
        if "zone" in WorkerCapabilities.model_fields:
            caps.update(region=zone.rsplit("-", 1)[0], zone=zone)
        out.append(
            WorkerView(
                worker_id=f"wkr-{i:04d}",
                state=WorkerState.IDLE,
                capabilities=WorkerCapabilities(**caps),
                running_tasks=4 if i % 2 else 0,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[30, 120, 500])
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=2000)
    a = ap.parse_args()
    spec = JobSpec(
        workload=DistributedJob(entrypoint="t.py", num_workers=a.nodes, procs_per_worker=2),
        resources=ResourceRequest(),
        placement=PlacementPolicy(target="remote"),
    )
    print(f"# {platform.platform()}, Python {platform.python_version()}")
    print(f"# distributed job, {a.nodes} nodes; median of {a.repeat} schedule() calls")
    for n in a.workers:
        ws = fleet(n)
        d = schedule(spec, ws)
        times = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            schedule(spec, ws)
            times.append(time.perf_counter() - t)
        by_id = {w.worker_id: w for w in ws}
        zones = sorted({getattr(by_id[x].capabilities, "zone", None) or "?" for x in d.worker_ids})
        print(
            f"workers={n:4d}  {statistics.median(times) * 1e6:9.1f} us/call  "
            f"placed={d.placed}  zones={zones}"
        )


if __name__ == "__main__":
    main()
