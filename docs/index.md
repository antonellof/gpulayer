# GPUlayer

GPUlayer runs your workloads on every GPU you can reach: the one in your laptop, remote
servers, clusters, data centers, or cloud VMs. A workload can be a command in any
language, a Python function using any library, or a distributed job across several machines.
GPUlayer finds a machine that can run it, starts it there, and brings back the logs, outputs and
results.

```bash
gpulayer submit --accelerator cuda --source ./app -- ./render.sh        # any command
```

```python
@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})             # any library
def benchmark(n: int) -> dict: ...

benchmark.map([1024, 2048, 4096])
```

```python
trainer = glt.Trainer(accelerator="cuda", devices=4, num_nodes=2, target="remote")
trainer.fit(model, train_loader)                                         # PyTorch integration
```

> **Status: v0.1.0, the first release.** Local execution, scheduling, the server and worker protocol, and
> distributed jobs on one host are implemented and tested, on CPU and Apple Silicon. NVIDIA GPUs
> and multi-machine runs are implemented but not validated yet, and automatic cloud
> provisioning is simulated. See [Status and roadmap](project/status.md).

## Architecture

![Your workloads go through GPUlayer to the machines you can reach](assets/architecture-light.svg)

GPUlayer has three parts:

- **Clients** submit work: the `gpulayer` CLI, the Python SDK (`@gl.remote`, `gl.Client`), the
  PyTorch `Trainer` and `launch()`, and the [MCP server](guides/ai-agents.md) for AI agents. A
  job states what it needs (accelerator, GPU count and memory, frameworks), not which machine
  to use.
- **The control plane** keeps the job queue, picks a worker for each job and records the
  reason when no worker fits ([Scheduling](architecture/scheduling.md)). It stores code
  bundles, inputs, outputs and logs, and checks tokens. For work on one machine it runs inside
  your process; for several machines it runs as a server (`gpulayer up`).
- **Workers**, one per machine, report their GPUs and installed frameworks, ask the control
  plane for tasks, run them as normal processes (through `torchrun` for
  [distributed jobs](architecture/distributed.md)), stream logs back and upload outputs.
  Workers open every connection themselves, so they can sit behind NAT. A cloud VM is a
  worker that joined with `gpulayer join`.

A job goes from `queued` to `scheduled` when a worker fits, then `running`, and ends as
`succeeded`, `failed`, `cancelled` or `timed_out`. Frameworks still decide how a computation
runs on the hardware they are given; GPUlayer decides which hardware runs it. It works under
any framework, and adds a deeper integration for PyTorch training. The
[Architecture overview](architecture/overview.md) covers the components, data model and request
flow in detail.

## What you can do with it

| You want to | With GPUlayer |
|---|---|
| Run a script on a GPU that is not in front of you | `gpulayer submit --target remote -- python infer.py` |
| Use one interface for NVIDIA boxes and Apple Silicon Macs | ask for `cuda` or `mps`; each job lands where that accelerator and its frameworks exist |
| Spread a batch of tasks over all free GPUs | `fn.map([...])` |
| Know why a job cannot run before it starts | a rejection reason per worker, e.g. `framework 'jax' not installed` |
| Train a model on 8 GPUs across two machines | `Trainer(devices=4, num_nodes=2)` or `launch(nodes=2, devices=4)` |
| Add a GPU machine to the pool | `gpulayer join <url> --token ...` |

## Where to go next

- [Quickstart](get-started/quickstart.md): commands, functions and training, from this machine to several
- [Why GPUlayer](get-started/why-gpulayer.md): what it adds, and when not to use it
- [Commands and scripts](workloads/commands.md), [Python functions](workloads/functions.md), [Any GPU framework](workloads/frameworks.md)
- [PyTorch integration](pytorch/trainer.md)
- [Run a server](guides/self-hosting.md): add more machines
