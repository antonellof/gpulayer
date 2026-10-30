"""Strong scaling, GPUlayer vs PyTorch Lightning: the same training job on 1, 2, 4, 8 machines.

Run on a fleet's GPUlayer server (benchmarks/cloud/gcp.sh), with ~/env.sh sourced:

    python scaling_run.py [NODES ...]        # default: 1 2 4 8; DEVICES=4, REPEAT=2

Wall time is measured here, from start until the last process exits, one run at a time.
GPUlayer submits one job; Lightning is started with torchrun on each node over ssh
(key ~/.ssh/fleet, user gl), at the same moment.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from multinode import SSH, idle_workers

HERE = Path(__file__).resolve().parent
DEVICES = int(os.environ.get("DEVICES", "4"))
REPEAT = int(os.environ.get("REPEAT", "2"))
EPOCHS = os.environ.get("EPOCHS", "3")


def loss_of(text: str) -> float:
    m = re.findall(r"LOSS ([0-9.]+)", text)
    return float(m[-1]) if m else float("nan")


def run_gl(n: int, worker_ids: list[str]) -> tuple[float, float]:
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "scaling.py", "gl", str(n), str(DEVICES), EPOCHS],
        cwd=HERE,
        env={**os.environ, "GL_WORKERS": ",".join(worker_ids)},
        capture_output=True,
        text=True,
    )
    took = time.perf_counter() - t0
    if p.returncode:
        raise SystemExit(f"gl {n}: {p.stdout[-2000:]}{p.stderr[-2000:]}")
    return took, loss_of(p.stdout)


def run_pl(ips: list[str]) -> tuple[float, float]:
    n = len(ips)
    t0 = time.perf_counter()
    procs = [
        subprocess.Popen(
            [
                *SSH,
                f"gl@{ip}",
                "cd /opt/gpulayer/benchmarks/lightning_comparison && "
                f"/opt/gpulayer/.venv/bin/torchrun --nnodes {n} --nproc-per-node {DEVICES} "
                f"--node-rank {r} --master-addr {ips[0]} --master-port 29512 "
                f"scaling.py pl {n} {DEVICES} {EPOCHS}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for r, ip in enumerate(ips)
    ]
    outs = [p.communicate()[0] for p in procs]
    took = time.perf_counter() - t0
    if any(p.returncode for p in procs):
        raise SystemExit(f"pl {n}:\n" + "\n".join(o[-1500:] for o in outs))
    return took, loss_of(outs[0])


def main() -> None:
    nodes = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
    print(
        f"{'nodes':>5} {'procs':>5} {'GPUlayer s':>11} {'Lightning s':>12} "
        f"{'GL loss':>8} {'PL loss':>8}",
        flush=True,
    )
    base: dict[str, float] = {}
    for n in nodes:
        ws = idle_workers(n)  # the same machines for both trainers
        ips = [w.capabilities.advertise_address for w in ws]
        gl_runs, pl_runs = [], []
        for _ in range(REPEAT):  # interleaved, so drift affects both equally
            gl_runs.append(run_gl(n, [w.worker_id for w in ws]))
            pl_runs.append(run_pl(ips))
        g = statistics.median(t for t, _ in gl_runs)
        p = statistics.median(t for t, _ in pl_runs)
        base.setdefault("gl", g)
        base.setdefault("pl", p)
        print(
            f"{n:5} {n * DEVICES:5} {g:11.1f} {p:12.1f} {gl_runs[-1][1]:8.4f} "
            f"{pl_runs[-1][1]:8.4f}   speedup GL x{base['gl'] / g:.2f}, "
            f"PL x{base['pl'] / p:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
