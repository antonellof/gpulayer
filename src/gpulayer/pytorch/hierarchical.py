"""Two-level gradient all-reduce for several processes per machine.

A flat all-reduce over N machines x P processes sends every gradient bucket around a ring of
N*P processes, most of them across the network. Summing inside each machine first, then
all-reducing among one process per machine, then copying the result back inside each
machine, crosses the network between N processes instead. The sum is the same up to
floating-point rounding order.

Processes must be numbered machine by machine (rank = machine * P + local rank), which is how
GPUlayer and torchrun assign them.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class HierarchicalAllReduce:
    local_size: int
    local_group: Any  # this machine's processes
    leader_group: Any  # local rank 0 of every machine; None on other processes
    leader: int  # global rank of this machine's local rank 0

    @classmethod
    def create(cls, local_size: int) -> "HierarchicalAllReduce":
        """Every process must call this, in the same order: new_group is a collective."""
        rank, world = dist.get_rank(), dist.get_world_size()
        if local_size < 1 or world % local_size:
            raise ValueError(f"world size {world} is not a multiple of {local_size} per machine")
        machines = world // local_size
        local_group = None
        for m in range(machines):
            ranks = list(range(m * local_size, (m + 1) * local_size))
            g = dist.new_group(ranks)
            if rank in ranks:
                local_group = g
        leaders = [m * local_size for m in range(machines)]
        leader_group = dist.new_group(leaders)
        leader = (rank // local_size) * local_size
        return cls(local_size, local_group, leader_group if rank == leader else None, leader)

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """Sum ``t`` over every process, in place."""
        if self.local_size > 1:
            dist.reduce(t, dst=self.leader, group=self.local_group)
        if self.leader_group is not None:
            dist.all_reduce(t, group=self.leader_group)
        if self.local_size > 1:
            dist.broadcast(t, src=self.leader, group=self.local_group)
        return t

    @staticmethod
    def hook(
        state: "HierarchicalAllReduce", bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        """DDP communication hook: average the bucket over every process. DDP checks these
        annotations at runtime, so this module does not postpone them."""
        t = state.all_reduce(bucket.buffer())
        t.div_(dist.get_world_size())
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(t)
        return fut
