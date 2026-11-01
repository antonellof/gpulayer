"""Step time of two GPUlayer trees, alternating runs: python hotpath.py DIR CASE REPS.

DIR holds the baseline tree in DIR/base (a git worktree of the earlier commit)."""

import os
import statistics
import subprocess
import sys

S, case, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
trees = {"base": f"{S}/base", "new": "/Users/d695663/Desktop/Dev/gpulayer"}
res: dict[str, dict[int, list[float]]] = {k: {100: [], 600: []} for k in trees}
for _ in range(reps):
    for steps in (100, 600):
        for name, tree in trees.items():
            env = {**os.environ, "PYTHONPATH": f"{tree}/src"}
            script = f"{tree}/benchmarks/trainer_comparison/bench.py"
            out = subprocess.run(
                [sys.executable, script, "gl", case, str(steps)],
                cwd=S,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            ).stdout
            fit = [float(line.split()[2]) for line in out.splitlines() if line.startswith("FIT")]
            res[name][steps].append(fit[0])
for name in trees:
    ms = (statistics.median(res[name][600]) - statistics.median(res[name][100])) / 500 * 1000
    print(f"{case} {name}: {ms:.3f} ms/step  (100: {res[name][100]}, 600: {res[name][600]})")
