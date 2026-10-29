"""Where the start-up time of a distributed job goes.

    python benchmarks/distributed_startup.py [--nodes 2 --procs 2 --repeat 3]

Runs a trivial torchrun job (import torch, init_process_group, barrier) on local workers and
splits its wall time into GPUlayer dispatch, torchrun start-up, torch import, process-group
init, torchrun shutdown and GPUlayer completion. Requires torch.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import gpulayer as gl

SCRIPT = """\
import sys, time
t0 = time.time()
import torch.distributed as dist
t1 = time.time()
dist.init_process_group("gloo")
t2 = time.time()
dist.barrier()
if dist.get_rank() == 0:
    sys.stdout.write(f"T {t0:.4f} {t1:.4f} {t2:.4f}\\n")
    sys.stdout.flush()
dist.destroy_process_group()
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp, "src")
        src.mkdir()
        (src / "t.py").write_text(SCRIPT)
        with gl.Client(home=Path(tmp, "home")) as c:
            c.ensure_local_workers(args.nodes)
            bundle = c.package(src)
            print(f"{args.nodes} node(s) x {args.procs} process(es), seconds")
            for _ in range(args.repeat):
                t_submit = time.time()
                job = c.submit(
                    gl.distributed(
                        "t.py", source=bundle, num_workers=args.nodes, procs_per_worker=args.procs
                    )
                )
                st = job.wait(300)
                seen = time.time()
                if st.state.value != "succeeded":
                    sys.exit(f"job {st.state.value}: {st.error}")
                line = next(x.line for x in job.logs() if x.line.startswith("T "))
                proc, imported, ready = (float(v) for v in line.split()[1:])
                started = min(t.started_at.timestamp() for t in st.tasks if t.started_at)
                finished = max(t.finished_at.timestamp() for t in st.tasks if t.finished_at)
                parts = {
                    "dispatch": started - t_submit,
                    "torchrun start": proc - started,
                    "import torch": imported - proc,
                    "init group": ready - imported,
                    "torchrun exit": finished - ready,
                    "completion": seen - finished,
                    "total": seen - t_submit,
                }
                print("  ".join(f"{k} {v:5.2f}" for k, v in parts.items()))


if __name__ == "__main__":
    main()
