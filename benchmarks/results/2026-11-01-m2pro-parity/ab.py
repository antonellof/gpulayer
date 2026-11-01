import os, statistics, subprocess, sys, time
S = sys.argv[1]; case = sys.argv[2]
bench = "/Users/d695663/Desktop/Dev/gpulayer/benchmarks/lightning_comparison/bench.py"
def wall(tag, steps):
    env = dict(os.environ)
    if tag == "old":
        env["PYTHONPATH"] = f"{S}/oldtree/src"
    t = time.perf_counter()
    subprocess.run([sys.executable, bench, "gl", case, str(steps)], env=env, check=True,
                   capture_output=True, cwd=os.path.dirname(bench))
    return time.perf_counter() - t
res = {"old": ([], []), "new": ([], [])}
for _ in range(5):
    for tag in ("old", "new"):
        res[tag][0].append(wall(tag, 200)); res[tag][1].append(wall(tag, 2200))
for tag, (s, l) in res.items():
    print(case, tag, f"{(statistics.median(l) - statistics.median(s)) / 2000 * 1000:.3f} ms/step")
