"""Multi-node DDP: GPUlayer Trainer(num_nodes=N) vs PyTorch Lightning under torchrun.

Run on the GPUlayer server of a fleet (benchmarks/cloud/gcp.sh), with ~/env.sh sourced:

    python multinode.py [NODES ...]          # default: 2 4 8

GPUlayer: `python bench.py gl CASE STEPS N` here; the Trainer submits one job and the server
places it on N workers. Lightning: torchrun on N workers over ssh (key ~/.ssh/fleet, user gl),
started at the same moment, with the same model, data and steps. Runs are sequential, so
nothing else uses the workers. Wall time is measured here, from start to the last process
exiting; ms/step = (median wall(600) - median wall(100)) / 500, start-up = wall(100) - 100 x
step.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import gpulayer as gl

HERE = Path(__file__).resolve().parent
PY = sys.executable
SSH = [
    "ssh",
    "-i",
    os.path.expanduser("~/.ssh/fleet"),
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
]
SHORT, LONG, REPEAT = 100, 600, 3
CASES = os.environ.get("MN_CASES", "ddp-nodes ddp-nodes-big").split()


def workers(n: int) -> list[str]:
    c = gl.Client(
        url=os.environ["GPULAYER_URL"],
        token=os.environ["GPULAYER_TOKEN"],
        fingerprint=os.environ.get("GPULAYER_FINGERPRINT"),
    )
    ws = sorted((w for w in c.workers() if w.state.value == "idle"), key=lambda w: w.name)
    if len(ws) < n:
        raise SystemExit(f"need {n} idle workers, have {len(ws)}")
    return [w.capabilities.advertise_address for w in ws[:n]]


def run_gl(case: str, steps: int, n: int) -> float:
    t0 = time.perf_counter()
    p = subprocess.run(
        [PY, "bench.py", "gl", case, str(steps), str(n)], cwd=HERE, capture_output=True, text=True
    )
    if p.returncode:
        raise SystemExit(f"gl {case} {steps} {n}: {p.stdout[-2000:]}{p.stderr[-2000:]}")
    return time.perf_counter() - t0


def run_pl(case: str, steps: int, ips: list[str]) -> float:
    n, master = len(ips), ips[0]
    t0 = time.perf_counter()
    procs = [
        subprocess.Popen(
            [
                *SSH,
                f"gl@{ip}",
                f"cd /opt/gpulayer/benchmarks/lightning_comparison && "
                f"/opt/gpulayer/.venv/bin/torchrun --nnodes {n} --nproc-per-node 2 "
                f"--node-rank {r} --master-addr {master} --master-port 29511 "
                f"bench.py pl {case} {steps} {n}",
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
        raise SystemExit(f"pl {case} {steps} {n}:\n" + "\n".join(o[-1500:] for o in outs))
    return took


def main() -> None:
    nodes = [int(x) for x in sys.argv[1:]] or [2, 4, 8]
    print(f"{'case':14} {'nodes':>5} {'trainer':8} {'ms/step':>8} {'start-up s':>10}", flush=True)
    for case in CASES:
        for n in nodes:
            ips = workers(n)
            res = {}
            for fw in ("gl", "pl"):
                short, long_ = [], []
                for _ in range(REPEAT):
                    for steps, bucket in ((SHORT, short), (LONG, long_)):
                        took = run_gl(case, steps, n) if fw == "gl" else run_pl(case, steps, ips)
                        bucket.append(took)
                step = (statistics.median(long_) - statistics.median(short)) / (LONG - SHORT)
                startup = statistics.median(short) - SHORT * step
                res[fw] = step
                print(f"{case:14} {n:5} {fw:8} {step * 1000:8.2f} {startup:10.2f}", flush=True)
            g, p = res["gl"], res["pl"]
            faster = "gl faster" if g < p else "pl faster"
            print(f"{case:14} {n:5} ratio    {faster} x{max(g, p) / min(g, p):.2f}", flush=True)


if __name__ == "__main__":
    main()
