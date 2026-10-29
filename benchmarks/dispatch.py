"""Dispatch overhead: how long GPUlayer adds around a job that does nothing.

    python benchmarks/dispatch.py --jobs 20 --workers 4

Measures, per job, submit -> process started (queue + dispatch), and submit -> the
client sees the final state (end to end). The job itself is `python -c pass`, so the
numbers are GPUlayer's overhead plus Python interpreter start-up (shown separately).
Runs in embedded mode in a throwaway GPULAYER_HOME.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import gpulayer as gl


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    t = time.perf_counter()
    for _ in range(5):
        subprocess.run([sys.executable, "-c", "pass"], check=True)
    python_start = (time.perf_counter() - t) / 5

    with tempfile.TemporaryDirectory() as home, gl.Client(home=home) as c:
        c.ensure_local_workers(args.workers)
        c.run(gl.command([sys.executable, "-c", "pass"]))  # warm-up
        # Idle latency: one job at a time on an idle worker (what a single job feels like).
        single = []
        for _ in range(10):
            t = time.perf_counter()
            c.submit(gl.command([sys.executable, "-c", "pass"])).wait()
            single.append(time.perf_counter() - t)
        submit_at: dict[str, float] = {}
        jobs = []
        t0 = time.perf_counter()
        for _ in range(args.jobs):
            submitted = time.time()
            job = c.submit(gl.command([sys.executable, "-c", "pass"]))
            submit_at[job.id] = submitted
            jobs.append(job)
        done_at = {}
        pending = {j.id: j for j in jobs}
        while pending:
            for jid, job in list(pending.items()):
                if job.status().state.terminal:
                    done_at[jid] = time.time()
                    del pending[jid]
            time.sleep(0.005)
        wall = time.perf_counter() - t0
        to_start, end_to_end = [], []
        for j in jobs:
            st = j.status()
            assert st.state.value == "succeeded", st
            started = (
                min(
                    datetime.fromisoformat(t.started_at).timestamp()
                    for t in st.tasks
                    if t.started_at
                )
                if isinstance(st.tasks[0].started_at, str)
                else min(t.started_at.timestamp() for t in st.tasks if t.started_at)
            )
            to_start.append(started - submit_at[j.id])
            end_to_end.append(done_at[j.id] - submit_at[j.id])

    print(f"jobs={args.jobs} workers={args.workers}")
    print(f"python start-up            {python_start * 1000:7.1f} ms")
    print(
        f"one job, idle worker       p50 {statistics.median(single) * 1000:7.1f} ms   "
        f"p95 {pct(single, 95) * 1000:7.1f} ms   (submit -> wait() returns)"
    )
    print("burst of jobs:")
    for name, xs in (("submit -> started", to_start), ("submit -> client sees done", end_to_end)):
        p50, p95 = statistics.median(xs) * 1000, pct(xs, 95) * 1000
        print(f"{name:26s} p50 {p50:7.1f} ms   p95 {p95:7.1f} ms")
    print(f"throughput                {args.jobs / wall:7.1f} jobs/s")


if __name__ == "__main__":
    main()
