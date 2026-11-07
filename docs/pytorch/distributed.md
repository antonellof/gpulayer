# Distributed training

GPUlayer runs DistributedDataParallel: it places a group of compatible workers, assigns ranks
and starts every process with torchrun's environment. PyTorch does the communication.

## With the Trainer

```python
trainer = glt.Trainer(max_epochs=10, accelerator="cpu", devices=2)                  # 2 processes, this machine
trainer = glt.Trainer(max_epochs=10, accelerator="cpu", devices=2, num_nodes=2)     # 2 workers x 2
trainer = glt.Trainer(max_epochs=10, accelerator="cuda", devices=8, num_nodes=4,
                      target="remote")                                              # 32 GPUs
trainer.fit(model, train_loader, val_loader)
```

Each process joins the process group (`gloo` for CPU, `nccl` for CUDA), wraps the model in
`DistributedDataParallel` and gets a `DistributedSampler` shard. Metrics are combined across
ranks by their `reduce_fx` (a mean over all samples, a sum, a max or a min) and rank 0 saves
checkpoints. Your `Module` does not change.

If any node fails, the job fails and the other nodes stop; a partly finished group is never
retried. Ranks, rendezvous and launch are described in
[Distributed execution](../architecture/distributed.md).

## On SLURM and in notebooks

Under `srun`, tasks train in place: the ranks come from SLURM, the first node is the rendezvous,
and nothing is submitted. `devices` and `num_nodes` must match the tasks. To survive a time
limit, let SLURM warn the job and requeue it, and have the script continue:

```bash
#SBATCH --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --requeue --signal=TERM@120
srun python train.py        # with trainer.fit(model, loader, ckpt_path="last")
```

Simulated with SLURM's variables on one machine; not run on a cluster.

In a notebook on Linux, `Trainer(devices=N)` forks one process per device and returns the
weights to the notebook, which needs checkpoints on (the default). The processes inherit your
cells, so nothing has to be pickled, but CUDA must not have been used before `fit`. Tested in a
real Jupyter kernel on CPU, not on a GPU. On macOS it stops with an explanation, because forks of
a kernel crash there. `launch()` has no script to send from a notebook: export it to a `.py` file.

## Requirements

- All nodes need the same torch version and the requested backend.
- Across machines, start each worker with a reachable address
  (`gpulayer worker start --advertise-address 10.0.0.21`) and open the rendezvous, gloo and
  NCCL ports between them. Set `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` through
  `launch_options={"env": {...}}`.
- MPS is single-process only.

## Your own DDP code

Run an existing DDP script unchanged with [`launch()`](launch.md), as in
[`examples/launch_ddp_script/train_ddp.py`](../../examples/launch_ddp_script/train_ddp.py):

```python
import gpulayer.pytorch as glt
glt.launch(nodes=2, devices=4, accelerator="cuda", target="remote")

# your existing script: dist.init_process_group(...), DDP(model), DistributedSampler, ...
```

## What has been tested

| Scenario | Status |
|---|---|
| 1 node x 2 processes, 2 nodes x 2 processes, CPU/gloo, one machine | tested end to end |
| A failing rank fails the job and stops the other node | tested |
| Nodes on different machines | validated with gloo on up to 16 GCP VMs |
| NCCL on several GPUs in one machine | validated: all-reduce across 4× RTX 3070 and 2× A100 SXM4 as a GPUlayer job |
| Trainer DDP on several GPUs | validated: 1, 2 and 4 RTX 3070s in one machine, and 2 to 8 T4 machines (benchmarked against Lightning) |
| NCCL across machines | validated on 2, 4 and 8 AWS machines with one Tesla T4 each (DDP, FSDP) |
