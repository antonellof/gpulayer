"""Validate several GPUs on one NVIDIA machine: interconnect detection, GPU choice and NCCL.

    python scripts/validate_multigpu.py            # on a machine with 2 or more NVIDIA GPUs

Prints CHECK lines (PASS/FAIL) and exits non-zero on a failure. Checks:

1. topology  `nvidia-smi topo -m` is parsed into links for every pair of GPUs, both ways.
2. choice    for each task size from 2 to the GPU count, which GPUs GPUlayer gives the task,
             their slowest link, and what the lowest indices would have given.
3. assigned  a 2-GPU job (or all GPUs) runs with CUDA_VISIBLE_DEVICES set to exactly the
             GPUs chosen in step 2.
4. nccl      a distributed job, 1 node x N processes, backend nccl: every rank all-reduces its
             rank number and checks the sum, so NCCL ran across all the GPUs.

Across machines, run the same NCCL job with `--nodes 2` on a server with two GPU workers
(benchmarks/cloud/gcp.sh cluster): `gpulayer submit --distributed --nodes 2
--procs-per-node 1 --backend nccl --accelerator cuda --gpus 1 ...`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gpulayer as gl
from gpulayer.contracts import Accelerator, JobState, ResourceRequest, WorkerCapabilities
from gpulayer.control_plane.scheduler import link_rank, pick_devices
from gpulayer.gpu.discovery import detect_capabilities

failures = 0


def check(name: str, ok: bool, detail: str) -> None:
    global failures
    failures += 0 if ok else 1
    print(f"CHECK {name} {'PASS' if ok else 'FAIL'} {detail}", flush=True)


NCCL_SCRIPT = """
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
t = torch.tensor([float(rank)], device="cuda")
dist.all_reduce(t)
want = world * (world - 1) / 2
print(f"RANK {rank} of {world} on cuda:{local} ({torch.cuda.get_device_name(local)}) "
      f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')} sum={t.item()} want={want}", flush=True)
assert t.item() == want
dist.destroy_process_group()
"""

VISIBLE_SCRIPT = "import os; print('VISIBLE', os.environ.get('CUDA_VISIBLE_DEVICES'))"


def check_topology(caps: WorkerCapabilities) -> dict[int, list[int]]:
    """Checks 1 and 2; returns the GPUs chosen for each task size."""
    gpus = caps.devices(Accelerator.CUDA)
    n = len(gpus)
    missing = [
        (a.index, b.index)
        for a in gpus
        for b in gpus
        if a.index != b.index
        and (b.index not in a.links or a.links[b.index] != b.links.get(a.index))
    ]
    links = {d.index: d.links for d in gpus}
    check("topology", not missing, f"{n} GPUs, links {links}, missing or asymmetric: {missing}")

    def weakest(ids: list[int]) -> str:
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
        worst = min(pairs, key=lambda p: link_rank(links[p[0]].get(p[1])))
        return links[worst[0]].get(worst[1], "?")

    chosen_for: dict[int, list[int]] = {}
    for k in range(2, n + 1):
        r = ResourceRequest(accelerator=Accelerator.CUDA, gpu_count=k)
        chosen = pick_devices(r, caps, set()) or []
        chosen_for[k] = chosen
        lowest = sorted(d.index for d in gpus)[:k]
        ok = len(chosen) == k and link_rank(weakest(chosen)) >= link_rank(weakest(lowest))
        check(
            f"choice-{k}",
            ok,
            f"chosen {chosen} (slowest link {weakest(chosen)}), lowest indices {lowest} "
            f"(slowest link {weakest(lowest)})",
        )
    return chosen_for


def main() -> int:
    caps = detect_capabilities()
    gpus = caps.devices(Accelerator.CUDA)
    n = len(gpus)
    try:
        print(subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        print("nvidia-smi not found")
    if n < 2:
        print(f"needs 2 or more NVIDIA GPUs, found {n}")
        return 2

    chosen_for = check_topology(caps)

    with tempfile.TemporaryDirectory() as tmp, gl.Client(home=Path(tmp) / "home") as c:
        src = Path(tmp) / "src"
        src.mkdir()
        (src / "nccl_check.py").write_text(NCCL_SCRIPT)
        (src / "visible.py").write_text(VISIBLE_SCRIPT)
        bundle = c.package(src)
        c.ensure_local_worker()

        # 3. assigned
        k = 2
        job = c.submit(
            gl.command(
                ["python", "visible.py"], source=bundle, accelerator="cuda", gpus=k, name="visible"
            )
        )
        st = job.wait(300)
        got = next(
            (ln.split()[1] for ln in job.logs_text().splitlines() if ln.startswith("VISIBLE")), ""
        )
        want = ",".join(str(i) for i in chosen_for[k])
        check(
            "assigned",
            st.state == JobState.SUCCEEDED and got == want,
            f"CUDA_VISIBLE_DEVICES={got} expected {want} ({st.state.value})",
        )

        # 4. nccl
        t0 = time.perf_counter()
        job = c.submit(
            gl.distributed(
                "nccl_check.py",
                source=bundle,
                procs_per_worker=n,
                comm_backend="nccl",
                name="nccl-check",
                resources=ResourceRequest(
                    accelerator=Accelerator.CUDA,
                    gpu_count=n,
                    frameworks={"torch": ">=2.1", "torch.cuda": ">=0"},
                ),
            )
        )
        st = job.wait(600)
        logs = job.logs_text()
        ranks = sum(1 for ln in logs.splitlines() if "RANK " in ln and "sum=" in ln)
        check(
            "nccl",
            st.state == JobState.SUCCEEDED and ranks == n,
            f"1 node x {n} processes, {ranks} ranks reported, {st.state.value}, "
            f"{time.perf_counter() - t0:.1f} s",
        )
        if st.state != JobState.SUCCEEDED:
            print(logs[-3000:])
    print(f"SUMMARY {'all passed' if not failures else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
