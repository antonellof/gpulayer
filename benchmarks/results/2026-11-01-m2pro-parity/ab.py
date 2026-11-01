"""A/B step time of two GPUlayer checkouts: python ab.py OLD_TREE CASE.

OLD_TREE is a git worktree of the earlier version (its src/ goes first on PYTHONPATH); the other
side imports the current checkout. Runs alternate, fresh process each, 5 repeats.
"""

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

OLD_TREE, CASE = sys.argv[1], sys.argv[2]
BENCH = Path(__file__).resolve().parents[2] / "trainer_comparison" / "bench.py"


def wall(tag: str, steps: int) -> float:
    env = dict(os.environ)
    if tag == "old":
        env["PYTHONPATH"] = f"{OLD_TREE}/src"
    t = time.perf_counter()
    subprocess.run(
        [sys.executable, str(BENCH), "gl", CASE, str(steps)],
        env=env,
        check=True,
        capture_output=True,
        cwd=BENCH.parent,
    )
    return time.perf_counter() - t


res: dict[str, tuple[list[float], list[float]]] = {"old": ([], []), "new": ([], [])}
for _ in range(5):
    for tag in ("old", "new"):
        res[tag][0].append(wall(tag, 200))
        res[tag][1].append(wall(tag, 2200))
for tag, (short, long) in res.items():
    ms = (statistics.median(long) - statistics.median(short)) / 2000 * 1000
    print(CASE, tag, f"{ms:.3f} ms/step")
