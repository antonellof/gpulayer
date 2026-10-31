# GPUlayer

**Train PyTorch on every GPU you can reach: one Trainer, from your laptop to many machines.**

The GPUlayer `Trainer` runs the same PyTorch script in your process, on every GPU of a machine,
or across many machines: laptops, workstations, servers and cloud VMs. It picks machines and
GPUs that can train the model, starts every process, and brings back logs, checkpoints and
metrics.

```python
import gpulayer.pytorch as glt

trainer = glt.Trainer(max_epochs=5, accelerator="cuda", devices=4, num_nodes=2)
trainer.fit(model, train_loader)        # 2 machines x 4 GPUs, DDP over NCCL
```

NVIDIA CUDA, Apple Metal (MPS) and CPU are validated on hardware. AMD ROCm is implemented and
not yet validated.

> **Status: v0.7.1.** Validated on a Tesla T4, 4× RTX 3070, 2× A100 (NVLink), an Apple M2 Pro,
> and distributed CPU training on up to 16 Google Cloud machines. NCCL across machines is
> implemented but not validated, and cloud VMs are created on Google Cloud only. See
> [Status and roadmap](project/status.md) and
> [Performance and comparison](get-started/performance-and-comparison.md).

## What it does for a training run

| You want to | Without GPUlayer | With GPUlayer |
|---|---|---|
| Train on a GPU box from your laptop | ssh, copy code, rebuild the environment, tmux, copy checkpoints back | `python train.py --target remote` |
| Use every GPU of a machine | torchrun with the right process count and visible devices | `Trainer(devices=4)`, on the best-connected GPUs |
| Train across machines | torchrun on every node with matching ranks, address and port | `Trainer(num_nodes=2, devices=4)`, kept in one zone when possible |
| Use NVIDIA servers and Macs | separate setup per machine | `accelerator="cuda"` or `"mps"`, same script |
| Sweep learning rates | loops and ad hoc scripts | `train.map([...])` across free GPUs |
| Add a GPU machine | set it up by hand for each project | `gpulayer join <url> --token ...` |

A run that no machine can train is rejected before it starts, with the reason for each
worker (for example `torch on this worker cannot use CUDA`).

## How it works

![Your training goes through GPUlayer to the machines you can reach](assets/architecture-light.svg)

- **Your process** runs the `Trainer`, [`launch()`](pytorch/launch.md) for an existing DDP
  script, or the `gpulayer` CLI. A run states what it needs (accelerator, GPUs, machines, GPU
  memory), not which machine to use.
- **The control plane** keeps the queue and picks machines ([Scheduling](architecture/scheduling.md)).
  It runs inside your process on one machine, or as a server (`gpulayer up`, Docker or the
  [Helm chart](guides/kubernetes.md)) for several.
- **A worker on each machine** reports its GPUs and PyTorch build, starts one process per rank
  ([Distributed training](architecture/distributed.md)), streams logs and uploads checkpoints.
  Workers open every connection themselves, so machines can sit behind NAT. A cloud VM is a
  worker that joined, or one GPUlayer [created for the run](guides/cloud-providers.md).

The same engine runs [commands](workloads/commands.md) and [Python functions](workloads/functions.md)
as jobs. The [Architecture overview](architecture/overview.md) has the details.

## When not to use it

- **One GPU on the machine you are using.** A plain training loop is simpler.
- **Untrusted users.** Jobs run as normal processes, not in a sandbox.
- **Beyond what is validated.** NCCL across machines and AMD GPUs have not been run yet.

Code ships as files (the directory of your script), and function arguments and results must
be JSON.

## Where to go next

- [Quickstart](get-started/quickstart.md): one training file in your process, with DDP, on two nodes and on a GPU
- [Trainer and Module](pytorch/trainer.md)
- [Performance and comparison](get-started/performance-and-comparison.md)
- [Run a server](guides/self-hosting.md) to add machines
