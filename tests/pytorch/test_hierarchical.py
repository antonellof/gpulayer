"""Two-level all-reduce: 4 CPU processes on one host, treated as 2 machines x 2 processes."""

import socket
import sys

import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch import nn  # noqa: E402

from gpulayer.pytorch.hierarchical import HierarchicalAllReduce  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="gloo over TCP on localhost")


def _worker(rank: int, world: int, port: int, out) -> None:
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    comm = HierarchicalAllReduce.create(2)
    t = torch.arange(10, dtype=torch.float32) * (rank + 1)
    flat, two = t.clone(), t.clone()
    dist.all_reduce(flat)
    comm.all_reduce(two)

    def train(hook: bool) -> torch.Tensor:
        torch.manual_seed(0)
        model = nn.parallel.DistributedDataParallel(
            nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        )
        if hook:
            model.register_comm_hook(comm, HierarchicalAllReduce.hook)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        g = torch.Generator().manual_seed(rank)
        for _ in range(5):
            x = torch.randn(4, 8, generator=g)
            opt.zero_grad()
            nn.functional.mse_loss(model(x), x.sum(1, keepdim=True)).backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    import os

    from gpulayer.pytorch.runtime import Runtime

    os.environ.update(
        {
            "WORLD_SIZE": str(world),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank % 2),
            "LOCAL_WORLD_SIZE": "2",
        }
    )
    auto, flat_rt = Runtime("cpu"), Runtime("cpu", gradient_sync="flat")
    wrapped = auto.module(nn.Linear(4, 1))  # registers the hook, creating the groups once
    wrapped(torch.randn(2, 4)).sum().backward()
    picks = (auto.uses_two_level_sync(), flat_rt.uses_two_level_sync())
    out.put(
        (rank, torch.equal(flat, two), torch.allclose(train(False), train(True), atol=1e-6), picks)
    )
    dist.destroy_process_group()


def test_two_level_all_reduce_matches_flat_and_ddp():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 4, port, out)) for r in range(4)]
    for p in procs:
        p.start()
    results = sorted(out.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(30)
    assert results == [(r, True, True, (True, False)) for r in range(4)]


def test_rejects_uneven_machines(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 6)
    with pytest.raises(ValueError, match="multiple of 4"):
        HierarchicalAllReduce.create(4)


def test_gradient_sync_choice(monkeypatch):
    from gpulayer.errors import ValidationFailed
    from gpulayer.pytorch import Trainer
    from gpulayer.pytorch.runtime import Runtime

    with pytest.raises(ValidationFailed, match="gradient_sync"):
        Runtime("cpu", gradient_sync="ring")
    with pytest.raises(ValidationFailed, match="gradient_sync"):
        Trainer(accelerator="cpu", gradient_sync="ring")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")  # one machine: nothing to save
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    assert not Runtime("cpu").uses_two_level_sync()
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert Runtime("cpu").uses_two_level_sync()
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    assert not Runtime("cpu").uses_two_level_sync()
    assert Runtime("cpu", gradient_sync="two_level").uses_two_level_sync()
