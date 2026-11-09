"""Run every case for both trainers: python run.py [cases...]. Fresh process per run.

TRAINERS (default "gl pl") picks the variants, e.g. TRAINERS="gl gl-base pl" (see bench.py).

ms/step = (median wall(LONG) - median wall(SHORT)) / (LONG - SHORT); LONG 600 and SHORT 100
by default (BENCH_LONG, BENCH_SHORT). Start-up = median wall(SHORT) - SHORT * step.
Wall time is measured outside the process, from start to exit.
"""

import os
import statistics
import subprocess
import sys
import time

CASES = sys.argv[1:] or ["mlp-fp32", "mlp-fp16", "cnn-fp16", "ddp-cpu-2"]
TRAINERS = os.environ.get("TRAINERS", "gl pl").split()
SHORT = int(os.environ.get("BENCH_SHORT", 100))  # fewer for large models: BENCH_SHORT=20
LONG = int(os.environ.get("BENCH_LONG", 600))
REPEAT = 3


def wall(fw, case, steps):
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "bench.py", fw, case, str(steps)], capture_output=True, text=True
    )
    took = time.perf_counter() - t0
    if p.returncode != 0:
        sys.exit(f"{fw} {case} {steps} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return took


print(f"{'case':10} {'trainer':8} {'ms/step':>8} {'start-up s':>10}", flush=True)
for case in CASES:
    res = {}
    for fw in TRAINERS:
        if case.endswith("-compile"):
            # One run thrown away first: it fills torch.compile's on-disk cache, which later
            # runs read, so that every measured run starts the same way.
            wall(fw, case, SHORT)
        # Interleave short and long runs so drift affects both equally.
        s, lg = [], []
        for _ in range(REPEAT):
            s.append(wall(fw, case, SHORT))
            lg.append(wall(fw, case, LONG))
        step = (statistics.median(lg) - statistics.median(s)) / (LONG - SHORT)
        startup = statistics.median(s) - SHORT * step
        res[fw] = (step, startup)
        print(f"{case:10} {fw:8} {step * 1000:8.2f} {startup:10.2f}", flush=True)
    if "gl" in res and "pl" in res:
        g, p = res["gl"][0], res["pl"][0]
        faster = "gl faster" if g < p else "pl faster"
        print(f"{case:10} ratio    {faster} x{max(g, p) / min(g, p):.2f}")
    if "gl" in res and "gl-base" in res:
        g, b = res["gl"][0], res["gl-base"][0]
        print(f"{case:10} gl vs gl-base  x{b / g:.2f}")
