"""Multi-machine and scale tests against a GPUlayer server with workers on 2+ machines.

Run on any machine that reaches the server (see benchmarks/README.md):

    GPULAYER_URL=... GPULAYER_TOKEN=... GPULAYER_FINGERPRINT=... python mm_test.py [section...]

Sections: workers, placement, distributed, failure, scale, sweep. Prints CHECK/RESULT lines.
"""

from __future__ import annotations

import collections
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import gpulayer as gl
from gpulayer.contracts import JobState, PlacementPolicy, ResourceRequest

HERE = Path(__file__).resolve().parent
c = gl.Client(
    url=os.environ["GPULAYER_URL"],
    token=os.environ["GPULAYER_TOKEN"],
    fingerprint=os.environ.get("GPULAYER_FINGERPRINT"),
)
WORKERS = {w.worker_id: w.name for w in c.workers()}
# One online worker on each of two different machines, for the cross-machine tests.
_per_host: dict[str, str] = {}
for _w in c.workers():
    if _w.state.value != "offline":
        _per_host.setdefault(_w.capabilities.hostname, _w.worker_id)
TWO_HOSTS = PlacementPolicy(target="remote", worker_ids=list(_per_host.values())[:2])
OUT: dict = {}


def say(*a):
    print(*a, flush=True)


def where(st) -> list[str]:
    tasks = sorted(st.tasks, key=lambda t: t.node_rank)
    return [WORKERS.get(t.worker_id, t.worker_id) for t in tasks]


def bundle(files: dict[str, str]):
    d = Path(tempfile.mkdtemp())
    for name, text in files.items():
        (d / name).write_text(text)
    return c.package(d)


def section_workers():
    ws = c.workers()
    for w in ws:
        caps = w.capabilities
        devs = ",".join(d.name for d in caps.accelerators) or "cpu"
        say(
            f"WORKER {w.name} {w.state.value} host={caps.hostname} "
            f"slots={caps.max_concurrent_tasks} devices={devs} addr={caps.advertise_address}"
        )
    assert len({w.capabilities.hostname for w in ws}) >= 2, "need workers on two machines"
    hosts = {w.capabilities.hostname for w in ws}
    say("CHECK workers PASS", len(ws), "workers on", len(hosts), "hosts")


def section_placement():
    code = "import socket; print('HOST', socket.gethostname())"
    gpu = c.submit(gl.command(["python", "-c", code], accelerator="cuda", gpus=1, target="remote"))
    st = gpu.wait(300)
    assert st.state == JobState.SUCCEEDED, (st.error, gpu.logs_text())
    host = [x for x in gpu.logs_text().splitlines() if x.startswith("HOST")]
    say("CHECK cuda-job-lands-on-gpu-worker PASS", where(st), host)
    # A request no worker satisfies is rejected with a reason per worker, not run on the wrong one.
    bad = c.submit(
        gl.command(
            ["true"],
            resources=ResourceRequest(accelerator="cuda", gpu_count=4),
            placement=PlacementPolicy(target="remote", on_unschedulable="fail"),
        )
    )
    st = bad.wait(120)
    assert st.state == JobState.FAILED, st.state
    say("CHECK impossible-request-rejected PASS", (st.error or "")[:160].replace("\n", " "))


DIST = """
import os, socket, torch, torch.distributed as dist
dist.init_process_group("gloo")
r, w = dist.get_rank(), dist.get_world_size()
t = torch.tensor([float(r + 1)])
dist.all_reduce(t)
print(f"rank={r} world={w} sum={t.item()} host={socket.gethostname()}", flush=True)
assert t.item() == w * (w + 1) / 2
if os.environ.get("FAIL_RANK") == str(r):
    raise SystemExit(3)
dist.barrier()
dist.destroy_process_group()
"""


def section_distributed():
    src = bundle({"d.py": DIST})
    for launcher in ("direct", "torchrun"):
        t0 = time.time()
        job = c.submit(
            gl.distributed(
                "d.py",
                source=src,
                num_workers=2,
                procs_per_worker=2,
                accelerator="cpu",
                placement=TWO_HOSTS,
                launcher=launcher,
                timeout_seconds=300,
            )
        )
        st = job.wait(400)
        took = time.time() - t0
        lines = [x.line for x in job.logs() if x.line.startswith("rank=")]
        hosts = sorted({x.split("host=")[1] for x in lines})
        assert st.state == JobState.SUCCEEDED, (st.error, job.logs_text()[-2000:])
        assert len(lines) == 4 and len(hosts) == 2, (lines, hosts)
        say(f"CHECK distributed-2-machines-{launcher} PASS world=4 hosts={hosts} {took:.2f}s")
        OUT[f"distributed_{launcher}_s"] = round(took, 2)


def section_failure():
    src = bundle({"d.py": DIST})
    t0 = time.time()
    job = c.submit(
        gl.distributed(
            "d.py",
            source=src,
            num_workers=2,
            procs_per_worker=2,
            accelerator="cpu",
            placement=TWO_HOSTS,
            env={"FAIL_RANK": "3"},
            timeout_seconds=300,
        )
    )
    st = job.wait(400)
    deadline = time.time() + 90
    while time.time() < deadline and not all(
        t.state.terminal for t in c.job(job.id).status().tasks
    ):
        time.sleep(0.5)
    st = c.job(job.id).status()
    states = {WORKERS.get(t.worker_id): t.state.value for t in st.tasks}
    assert st.state == JobState.FAILED and set(states.values()) <= {"failed", "cancelled"}, states
    say(f"CHECK rank-failure-stops-both-machines PASS {states} {time.time() - t0:.1f}s")


TASKS = """
import hashlib, os, socket, time, zlib

def shard(i, size_mb=4):
    # A data-processing shard: generate, compress, checksum.
    t0 = time.time()
    data = (hashlib.sha256(str(i).encode()).digest() * (size_mb * 32768))
    packed = zlib.compress(data, 6)
    return {"i": i, "host": socket.gethostname(), "ratio": len(data) / len(packed),
            "crc": zlib.crc32(packed), "secs": time.time() - t0}

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
    return {"lr": lr, "loss": final, "host": socket.gethostname(), "secs": time.time() - t0}
"""


def run_batch(label, specs):
    t0 = time.time()
    jobs = c.submit_batch(specs)
    results, per_worker, failed = [], collections.Counter(), 0
    for j in jobs:
        st = j.wait(1800)
        if st.state != JobState.SUCCEEDED:
            failed += 1
            continue
        per_worker[where(st)[0]] += 1
        results.append(j.result().return_value)
    wall = time.time() - t0
    busy = sum(r["secs"] for r in results)
    say(
        f"RESULT {label}: {len(jobs)} jobs, {failed} failed, wall {wall:.1f}s, "
        f"{len(jobs) / wall:.1f} jobs/s, summed task time {busy:.1f}s "
        f"(x{busy / wall:.1f} parallel), per worker {dict(per_worker)}"
    )
    OUT[label] = {
        "jobs": len(jobs),
        "failed": failed,
        "wall_s": round(wall, 2),
        "task_s": round(busy, 2),
        "per_worker": dict(per_worker),
    }
    assert failed == 0
    return results


def section_scale():
    src = bundle({"tasks.py": TASKS})
    n = int(os.environ.get("SCALE_N", "500"))
    specs = [
        gl.function(
            "tasks:shard", i, source=src, accelerator="cpu", target="remote", name=f"shard-{i}"
        )
        for i in range(n)
    ]
    res = run_batch(f"scale_{n}_shards", specs)
    say(
        "CHECK scale PASS",
        len(res),
        "results, median task",
        f"{statistics.median(r['secs'] for r in res):.2f}s",
    )


def section_heavy():
    src = bundle({"tasks.py": TASKS})
    specs = [
        gl.function(
            "tasks:shard", i, 96, source=src, accelerator="cpu", target="remote", name=f"heavy-{i}"
        )
        for i in range(200)
    ]
    res = run_batch("heavy_200_shards_96MB", specs)
    say(
        "CHECK heavy PASS",
        len(res),
        "results, median task",
        f"{statistics.median(r['secs'] for r in res):.2f}s",
    )


def section_sweep():
    src = bundle({"tasks.py": TASKS})
    lrs = [round(10 ** (-3 + 2.5 * k / 95), 6) for k in range(96)]
    req = ResourceRequest(accelerator="cpu", frameworks={"torch": ">=2.1"})
    specs = [
        gl.function("tasks:train", lr, source=src, resources=req, target="remote", name=f"lr-{lr}")
        for lr in lrs
    ]
    res = run_batch("sweep_96_trainings", specs)
    ok = [r for r in res if r["loss"] is not None and r["loss"] == r["loss"]]
    say(f"RESULT sweep diverged trials (NaN loss): {len(res) - len(ok)}")
    best = min(ok, key=lambda r: r["loss"])
    say(f"CHECK sweep PASS best lr {best['lr']} loss {best['loss']:.4f}")


SECTIONS = {
    "workers": section_workers,
    "placement": section_placement,
    "distributed": section_distributed,
    "failure": section_failure,
    "scale": section_scale,
    "heavy": section_heavy,
    "sweep": section_sweep,
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
(HERE / "multi_machine_results.json").write_text(json.dumps(OUT, indent=2))
say("SUMMARY", "all passed" if not failed else f"failed: {failed}")
