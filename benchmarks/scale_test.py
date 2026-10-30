"""Scale test against a GPUlayer server with many workers (see benchmarks/README.md).

    GPULAYER_URL=... GPULAYER_TOKEN=... GPULAYER_FINGERPRINT=... \\
        python benchmarks/scale_test.py [section...]

Sections: workers, burst, shards, sweep, distributed, failure, worker_loss. Prints WORKERS,
CHECK and RESULT lines and writes scale_results.json next to this file. worker_loss stops the
`gpulayer join` process on some workers over ssh (key ~/.ssh/fleet, user gl, as created by
benchmarks/cloud/gcp.sh); set SCALE_KILL=0 to skip it.
"""

from __future__ import annotations

import collections
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gpulayer as gl
from gpulayer.contracts import JobState, PlacementPolicy, ResourceRequest, RetryPolicy

HERE = Path(__file__).resolve().parent
c = gl.Client(
    url=os.environ["GPULAYER_URL"],
    token=os.environ["GPULAYER_TOKEN"],
    fingerprint=os.environ.get("GPULAYER_FINGERPRINT"),
)
OUT: dict = {}
REMOTE = PlacementPolicy(target="remote")

TASKS = """
import hashlib, socket, time, zlib

def tiny(i):
    return {"i": i, "host": socket.gethostname(), "secs": 0.0}

def shard(i, size_mb=96):
    t0 = time.time()
    data = hashlib.sha256(str(i).encode()).digest() * (size_mb * 32768)
    packed = zlib.compress(data, 6)
    return {"i": i, "host": socket.gethostname(), "crc": zlib.crc32(packed),
            "secs": time.time() - t0}

def train(lr, steps=3000, seed=0):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x = torch.randn(2048, 32)
    y = x @ torch.randn(32, 1) + 0.1 * torch.randn(2048, 1)
    net = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, 2048, (128,))
        loss = torch.nn.functional.mse_loss(net(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(net(x), y).item()
    return {"lr": lr, "loss": final if final == final else None,
            "host": socket.gethostname(), "secs": time.time() - t0}
"""

DIST = """
import os, socket, torch, torch.distributed as dist
dist.init_process_group("gloo")
r, w = dist.get_rank(), dist.get_world_size()
t = torch.tensor([float(r + 1)])
dist.all_reduce(t)
assert t.item() == w * (w + 1) / 2, t.item()
if r == 0:
    print(f"world={w} sum={t.item()} host={socket.gethostname()}", flush=True)
print(f"rank={r} host={socket.gethostname()}", flush=True)
if os.environ.get("FAIL_RANK") == str(r):
    raise SystemExit(3)
dist.barrier()
dist.destroy_process_group()
"""


def say(*a):
    print(*a, flush=True)


def online():
    return [w for w in c.workers() if w.state.value != "offline"]


def bundle(files):
    d = Path(tempfile.mkdtemp())
    for name, text in files.items():
        (d / name).write_text(text)
    return c.package(d)


def names():
    return {w.worker_id: w.name for w in c.workers()}


def run_batch(label, specs, *, threads=16):
    """Submit with a few threads (as a real client would), then wait for every job."""
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        jobs = list(ex.map(c.submit, specs))
    submitted = time.time() - t0
    states = list(ThreadPoolExecutor(threads).map(lambda j: j.wait(3600), jobs))
    wall = time.time() - t0
    who = names()
    per_worker = collections.Counter()
    results, failed, retried = [], 0, 0
    for j, st in zip(jobs, states, strict=True):
        if st.state != JobState.SUCCEEDED:
            failed += 1
            continue
        retried += st.attempts > 1
        per_worker[who.get(st.tasks[-1].worker_id, "?")] += 1
        results.append(j.result().return_value)
    busy = sum(r["secs"] for r in results)
    spread = sorted(per_worker.values())
    say(
        f"RESULT {label}: {len(jobs)} jobs, {failed} failed, {retried} retried, submit "
        f"{submitted:.1f}s, wall {wall:.1f}s, {len(jobs) / wall:.1f} jobs/s, task time "
        f"{busy:.1f}s, {len(per_worker)} workers used (min {spread[0] if spread else 0}, "
        f"max {spread[-1] if spread else 0} jobs each)"
    )
    OUT[label] = {
        "jobs": len(jobs),
        "failed": failed,
        "retried": retried,
        "submit_s": round(submitted, 2),
        "wall_s": round(wall, 2),
        "task_s": round(busy, 2),
        "workers_used": len(per_worker),
        "per_worker": dict(per_worker),
    }
    return results, failed


def section_workers():
    want = int(os.environ.get("SCALE_WORKERS", "1"))
    deadline = time.time() + 600
    while len(online()) < want and time.time() < deadline:
        time.sleep(5)
    ws = online()
    slots = sum(w.capabilities.max_concurrent_tasks for w in ws)
    hosts = {w.capabilities.hostname for w in ws}
    zones = collections.Counter(w.capabilities.labels.get("zone", "?") for w in ws)
    say(f"WORKERS {len(ws)} online on {len(hosts)} hosts, {slots} slots")
    OUT["workers"] = {"online": len(ws), "slots": slots}
    assert len(ws) >= want, f"only {len(ws)} of {want} workers online"
    say("CHECK workers PASS", len(ws), dict(zones))


def section_burst():
    src = bundle({"tasks.py": TASKS})
    n = int(os.environ.get("SCALE_BURST", "5000"))
    specs = [gl.function("tasks:tiny", i, source=src, placement=REMOTE) for i in range(n)]
    _, failed = run_batch(f"burst_{n}_tiny", specs)
    say("CHECK burst", "PASS" if not failed else "FAIL")


def section_shards():
    src = bundle({"tasks.py": TASKS})
    n = int(os.environ.get("SCALE_SHARDS", "2000"))
    specs = [gl.function("tasks:shard", i, source=src, placement=REMOTE) for i in range(n)]
    res, failed = run_batch(f"shards_{n}", specs)
    med = statistics.median(r["secs"] for r in res) if res else 0
    say("CHECK shards", "PASS" if not failed else "FAIL", f"median task {med:.2f}s")


def section_sweep():
    src = bundle({"tasks.py": TASKS})
    n = int(os.environ.get("SCALE_SWEEP", "480"))
    req = ResourceRequest(accelerator="cpu", frameworks={"torch": ">=2.1"})
    lrs = [round(10 ** (-3 + 2.0 * k / (n - 1)), 6) for k in range(n)]
    specs = [
        gl.function("tasks:train", lr, source=src, resources=req, placement=REMOTE) for lr in lrs
    ]
    res, failed = run_batch(f"sweep_{n}_trainings", specs)
    ok = [r for r in res if r["loss"] is not None]
    best = min(ok, key=lambda r: r["loss"])
    say(
        "CHECK sweep",
        "PASS" if not failed else "FAIL",
        f"best lr {best['lr']} loss {best['loss']:.4f}, diverged {len(res) - len(ok)}",
    )


def section_distributed():
    src = bundle({"d.py": DIST})
    for nodes in (8, 16):
        for launcher in ("direct", "torchrun"):
            t0 = time.time()
            job = c.submit(
                gl.distributed(
                    "d.py",
                    source=src,
                    num_workers=nodes,
                    procs_per_worker=2,
                    accelerator="cpu",
                    placement=REMOTE,
                    launcher=launcher,
                    timeout_seconds=600,
                )
            )
            st = job.wait(900)
            took = time.time() - t0
            hosts = {x.line.split("host=")[1] for x in job.logs() if x.line.startswith("rank=")}
            ok = st.state == JobState.SUCCEEDED and len(hosts) == nodes
            say(
                f"CHECK distributed-{nodes}x2-{launcher} {'PASS' if ok else 'FAIL'} "
                f"world={nodes * 2} hosts={len(hosts)} {took:.2f}s {st.error or ''}"
            )
            OUT[f"distributed_{nodes}x2_{launcher}_s"] = round(took, 2)


def section_failure():
    src = bundle({"d.py": DIST})
    t0 = time.time()
    job = c.submit(
        gl.distributed(
            "d.py",
            source=src,
            num_workers=8,
            procs_per_worker=2,
            accelerator="cpu",
            placement=REMOTE,
            env={"FAIL_RANK": "11"},
            timeout_seconds=600,
        )
    )
    job.wait(900)
    deadline = time.time() + 120
    while time.time() < deadline and not all(t.state.terminal for t in job.status().tasks):
        time.sleep(0.5)
    st = job.status()
    states = collections.Counter(t.state.value for t in st.tasks)
    ok = st.state == JobState.FAILED and set(states) <= {"failed", "cancelled"}
    took = time.time() - t0
    say(
        f"CHECK rank-failure-stops-8-machines {'PASS' if ok else 'FAIL'} {dict(states)} {took:.1f}s"
    )
    OUT["failure_8_nodes_s"] = round(took, 1)


def section_worker_loss():
    """Kill the worker process on a few machines while jobs run; retried jobs must finish."""
    if os.environ.get("SCALE_KILL", "1") == "0":
        return
    src = bundle({"tasks.py": TASKS})
    n = int(os.environ.get("SCALE_LOSS_JOBS", "600"))
    specs = [
        gl.function(
            "tasks:shard", i, 192, source=src, placement=REMOTE, retry=RetryPolicy(max_attempts=3)
        )
        for i in range(n)
    ]
    victims = online()[:3]

    def kill_soon():
        time.sleep(8)
        for w in victims:
            subprocess.run(
                [
                    "ssh",
                    "-i",
                    os.path.expanduser("~/.ssh/fleet"),
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "LogLevel=ERROR",
                    f"gl@{w.capabilities.advertise_address}",
                    "sudo pkill -9 -f '[g]pulayer join'",
                ],
                check=False,
                timeout=30,
            )
        say(f"KILLED worker processes on {[w.name for w in victims]}")

    import threading

    threading.Thread(target=kill_soon, daemon=True).start()
    res, failed = run_batch(f"worker_loss_{n}_shards_3_killed", specs)
    retried = OUT[f"worker_loss_{n}_shards_3_killed"]["retried"]
    say(
        "CHECK worker-loss",
        "PASS" if not failed and retried else "FAIL",
        f"{retried} jobs retried after their worker was lost, {failed} failed",
    )


SECTIONS = {
    "workers": section_workers,
    "burst": section_burst,
    "shards": section_shards,
    "sweep": section_sweep,
    "distributed": section_distributed,
    "failure": section_failure,
    "worker_loss": section_worker_loss,
}
failed = []
for name in sys.argv[1:] or list(SECTIONS):
    try:
        SECTIONS[name]()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        failed.append(name)
        say(f"CHECK {name} FAIL {type(e).__name__}: {str(e)[:300]}")
    (Path.home() / "scale_results.json").write_text(json.dumps(OUT, indent=2))
say("SUMMARY", "all passed" if not failed else f"failed: {failed}")
