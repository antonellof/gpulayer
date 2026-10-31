# Distributed jobs

A distributed job is one workload spread over several workers and processes. GPUlayer places the
whole group at once, assigns ranks, publishes a rendezvous address, starts every node, collects
per-node logs and outputs, and stops the group if any node fails.

Every process gets the standard torchrun variables `RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
`MASTER_ADDR` and `MASTER_PORT`. GPUlayer starts the processes itself by default
(`launcher="direct"`, about 1 s faster per job); `launcher="torchrun"` (or `--launcher
torchrun` on the CLI) starts them through torchrun instead. Any code that can use them works,
including non-training workloads that only need a process group.

## Three ways to start one

```python
# a function, called on every process; rank 0's return value is the result
@gl.remote(nodes=2, procs_per_node=4, accelerator="cuda")
def shard_work() -> dict: ...
```

```bash
# a script from the CLI
gpulayer submit --distributed --source ./app --entrypoint run.py --nodes 2 --procs-per-node 4
```

```python
# an existing script, with two lines at the top
import gpulayer.pytorch as glt
glt.launch(nodes=2, devices=4, accelerator="cuda")
```

For training, the [PyTorch integration](../pytorch/distributed.md) also handles DDP wrapping,
data sharding and checkpoints.

## Semantics

- The group is placed all at once or not at all; a partly free cluster does not start a job.
- All nodes must share the same torch version and the requested backend (`gloo` or `nccl`).
- Across machines, workers need an address their peers can reach (`--advertise-address`) and
  open ports between them.
- The group is kept as close together as possible: one network group, else one zone, else
  one region. `--group-locality zone` (or `network_group`, `region`) requires it; see
  [Scheduling](../architecture/scheduling.md#distributed-groups-and-network-locality).
- One failed or lost node fails the job; the others are stopped. Distributed jobs are never
  retried automatically.

Internals: [Distributed execution](../architecture/distributed.md).
