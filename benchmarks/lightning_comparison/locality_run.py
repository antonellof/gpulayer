"""Does keeping a distributed group in one zone train faster? (ADR-0021)

Run on a fleet's GPUlayer server (benchmarks/cloud/gcp.sh), with ~/env.sh sourced, on a fleet
with idle workers in at least two zones of one region and one zone of another region:

    python locality_run.py [NODES]          # default 4; DEVICES=4, REPEAT=3, EPOCHS=2

The same training job as scaling.py (a CNN, gloo, DDP with two-level gradient sync), on NODES
machines picked three ways, run one at a time and interleaved so drift affects all equally:

  one zone       all machines in the zone with the most idle workers
  two zones      half in that zone, half in another zone of the same region
  two regions    half in that zone, half in another region

Then once more without pinning (group_locality="prefer"), to show which machines the
scheduler picks by itself. Prints the median wall time of each, from submission until the
job has finished, and the final training loss.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import gpulayer as gl

HERE = Path(__file__).resolve().parent
DEVICES = os.environ.get("DEVICES", "4")
REPEAT = int(os.environ.get("REPEAT", "3"))
EPOCHS = os.environ.get("EPOCHS", "2")


def client() -> gl.Client:
    return gl.Client(
        url=os.environ["GPULAYER_URL"],
        token=os.environ["GPULAYER_TOKEN"],
        fingerprint=os.environ.get("GPULAYER_FINGERPRINT"),
    )


def region(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def run(n: int, worker_ids: list[str] | None) -> tuple[float, float]:
    env = {**os.environ}
    if worker_ids:
        env["GL_WORKERS"] = ",".join(worker_ids)
    else:
        env.pop("GL_WORKERS", None)
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "scaling.py", "gl", str(n), DEVICES, EPOCHS],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
    )
    took = time.perf_counter() - t0
    if p.returncode:
        raise SystemExit(f"run failed: {p.stdout[-2000:]}{p.stderr[-2000:]}")
    m = re.findall(r"LOSS ([0-9.]+)", p.stdout)
    return took, float(m[-1]) if m else float("nan")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    c = client()
    idle = [w for w in c.workers() if w.state.value == "idle" and w.capabilities.zone]
    by_zone: dict[str, list] = {}
    for w in sorted(idle, key=lambda w: w.name):
        by_zone.setdefault(w.capabilities.zone, []).append(w)
    home = max(by_zone, key=lambda z: (len(by_zone[z]), z))
    same_region = [z for z in by_zone if z != home and region(z) == region(home)]
    other_region = [z for z in by_zone if region(z) != region(home)]
    half = n // 2
    if len(by_zone[home]) < n or not same_region or not other_region:
        raise SystemExit(f"need {n} idle workers in one zone plus two other zones: {by_zone}")
    zr, rr = same_region[0], other_region[0]
    if len(by_zone[zr]) < n - half or len(by_zone[rr]) < n - half:
        raise SystemExit(f"not enough idle workers in {zr} or {rr}")
    layouts = {
        "one zone": by_zone[home][:n],
        "two zones": by_zone[home][:half] + by_zone[zr][: n - half],
        "two regions": by_zone[home][:half] + by_zone[rr][: n - half],
    }
    print(f"# {n} nodes x {DEVICES} processes, {EPOCHS} epochs, median of {REPEAT}", flush=True)
    for name, ws in layouts.items():
        print(f"# {name}: {sorted({w.capabilities.zone for w in ws})}", flush=True)
    results: dict[str, list[tuple[float, float]]] = {k: [] for k in layouts}
    for _ in range(REPEAT):
        for name, ws in layouts.items():
            results[name].append(run(n, [w.worker_id for w in ws]))
    base = statistics.median(t for t, _ in results["one zone"])
    print(f"{'layout':12} {'seconds':>8} {'vs one zone':>12} {'loss':>8}", flush=True)
    for name, rs in results.items():
        t = statistics.median(x for x, _ in rs)
        print(f"{name:12} {t:8.1f} {t / base:11.2f}x {rs[-1][1]:8.4f}", flush=True)

    before = {j.job_id for j in c.jobs(limit=200)}
    took, loss = run(n, None)
    new = [j for j in c.jobs(limit=200) if j.job_id not in before]
    zones: list[str] = []
    if new:
        status = c.job(new[0].job_id).status()
        by_id = {w.worker_id: w for w in c.workers()}
        zones = sorted(
            {str(by_id[t.worker_id].capabilities.zone) for t in status.tasks if t.worker_id}
        )
        if status.scheduling:
            print(f"# unpinned: {status.scheduling.summary}", flush=True)
    print(f"unpinned     {took:8.1f} {took / base:11.2f}x {loss:8.4f}  zones={zones}", flush=True)


if __name__ == "__main__":
    main()
