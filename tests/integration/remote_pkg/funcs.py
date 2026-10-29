"""Functions used by test_remote.py. They live in a file because GpuLayer ships source."""

import os

import gpulayer as gl


@gl.remote
def add(a: int, b: int = 0) -> int:
    return a + b


@gl.remote
def boom() -> None:
    raise ValueError("nope")


@gl.remote(nodes=2, procs_per_node=1)
def world_sum() -> dict:
    import torch
    import torch.distributed as dist

    dist.init_process_group(os.environ["GPULAYER_COMM_BACKEND"])
    t = torch.tensor([float(dist.get_rank() + 1)])
    dist.all_reduce(t)
    out = {"world": dist.get_world_size(), "sum": t.item()}
    dist.destroy_process_group()
    return out
