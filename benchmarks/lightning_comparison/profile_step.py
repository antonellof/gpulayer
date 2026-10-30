"""Where does a distributed training step spend its time? Run as a GPUlayer distributed job:

    gpulayer submit --distributed --source benchmarks/lightning_comparison \
        --entrypoint profile_step.py --nodes 8 --procs-per-node 4

Same model, data and per-process batch as scaling.py. Rank 0 prints the median time per step
(slowest rank) for:

- compute: forward, backward and SGD step with no gradient exchange;
- flat: one all-reduce of the flattened gradients over every process (what DDP does);
- two-level: sum inside each machine, all-reduce among one process per machine, then copy
  back inside each machine;
- ddp and ddp+two-level: a full DDP step, without and with the two-level communication hook.

It also checks that both all-reduces give the same result.
"""

import os
import statistics
import time

import torch
import torch.distributed as dist
from torch import nn

from gpulayer.pytorch.hierarchical import HierarchicalAllReduce

GLOBAL_BATCH = 256  # as in scaling.py
torch.set_num_threads(1)


def make_net() -> nn.Module:  # the model in scaling.py
    def block(i: int, o: int) -> list[nn.Module]:
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]

    return nn.Sequential(
        *block(3, 32),
        *block(32, 32),
        nn.MaxPool2d(2),
        *block(32, 64),
        *block(64, 64),
        nn.MaxPool2d(2),
        *block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10),
    )


dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_WORLD_SIZE"])
STEPS, WARMUP = int(os.environ.get("STEPS", "60")), 10

comm = HierarchicalAllReduce.create(local)
torch.manual_seed(rank)
batch = GLOBAL_BATCH // world
x, y = torch.randn(batch, 3, 32, 32), torch.randint(0, 10, (batch,))


def timed(fn) -> float:
    """Median seconds per call over STEPS calls, taken on the slowest process."""
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(STEPS):
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        t = torch.tensor([time.perf_counter() - t0])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times.append(t.item())
    return statistics.median(times)


net = make_net()
opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)


def compute() -> None:
    opt.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(net(x), y).backward()
    opt.step()


compute()
flat = torch.cat([p.grad.flatten() for p in net.parameters()])
a, b = flat.clone(), flat.clone()
dist.all_reduce(a)
comm.all_reduce(b)
same = torch.allclose(a, b, rtol=1e-5, atol=1e-6)

results = {
    "compute": timed(compute),
    "flat": timed(lambda: dist.all_reduce(flat.clone())),
    "two-level": timed(lambda: comm.all_reduce(flat.clone())),
}
for name, hook in (("ddp", False), ("ddp+two-level", True)):
    model = nn.parallel.DistributedDataParallel(make_net())
    if hook:
        model.register_comm_hook(comm, HierarchicalAllReduce.hook)
    o = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    def step(model=model, o=o) -> None:
        o.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), y).backward()
        o.step()

    results[name] = timed(step)

if rank == 0:
    n = sum(p.numel() for p in net.parameters())
    print(
        f"PROFILE world={world} local={local} params={n} batch_per_proc={batch} same_result={same}",
        flush=True,
    )
    for k, v in results.items():
        print(f"PROFILE {k:14s} {v * 1000:8.2f} ms", flush=True)
dist.destroy_process_group()
