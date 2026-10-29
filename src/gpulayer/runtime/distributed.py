"""Worker-group planning and torchrun launch construction for DistributedJob.

GpuLayer allocates the group, assigns node ranks and publishes the rendezvous endpoint;
PyTorch (torchrun + the chosen process-group backend) does the collective communication.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from gpulayer.contracts import DistributedJob, RendezvousInfo


@dataclass(frozen=True)
class NodePlan:
    node_rank: int
    worker_id: str
    global_ranks: tuple[int, ...]  # torch RANK values hosted on this node
    local_world_size: int


@dataclass(frozen=True)
class GroupPlan:
    world_size: int
    nodes: tuple[NodePlan, ...]


def plan_group(job: DistributedJob, worker_ids: list[str]) -> GroupPlan:
    """Deterministic rank layout: node r hosts ranks [r*P, (r+1)*P)."""
    if len(worker_ids) != job.num_workers:
        raise ValueError(f"need {job.num_workers} workers, got {len(worker_ids)}")
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("a worker may appear only once in a group")
    p = job.procs_per_worker
    nodes = tuple(
        NodePlan(
            node_rank=r,
            worker_id=w,
            global_ranks=tuple(range(r * p, (r + 1) * p)),
            local_world_size=p,
        )
        for r, w in enumerate(worker_ids)
    )
    return GroupPlan(world_size=job.world_size, nodes=nodes)


def free_port(host: str = "") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def torchrun_argv(
    python: str, job: DistributedJob, rdzv: RendezvousInfo, entrypoint_path: str
) -> list[str]:
    """Static rendezvous: every node gets rank 0's MASTER_ADDR/PORT and its node rank."""
    return [
        python,
        "-m",
        "torch.distributed.run",
        f"--nnodes={rdzv.num_nodes}",
        f"--nproc-per-node={rdzv.procs_per_node}",
        f"--node-rank={rdzv.node_rank}",
        f"--master-addr={rdzv.master_addr}",
        f"--master-port={rdzv.master_port}",
        "--max-restarts=0",  # fail_fast: no elastic restarts inside the group
        # Each rank's output is captured separately and re-emitted line by line with a
        # "[default<local rank>]:" prefix, so ranks never interleave mid-line and the worker
        # can tag every line with its global rank.
        "--tee=3",
        entrypoint_path,
        *job.args,
    ]
