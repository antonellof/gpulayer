# Remote functions and sweeps

`@gl.remote` turns a top-level function into something you can run on any compatible worker,
or many times in parallel.

```python
import os
import gpulayer as gl


@gl.remote(accelerator="mps", frameworks={"torch": ">=2.0"})
def train(lr: float, epochs: int = 200) -> dict:
    import torch
    device = torch.device(os.environ["GPULAYER_ACCELERATOR"])
    ...
    return {"lr": lr, "loss": loss.item()}


if __name__ == "__main__":
    train(0.1)                                   # a plain local call
    train.run(0.1)                               # on a worker; logs stream to your terminal
    results = train.map([0.3, 0.1, 0.03, 0.01])  # four jobs in parallel
    train.options(accelerator="cuda", target="remote").run(0.1)
```

## Methods

| Call | What it does |
|---|---|
| `train(*args)` | calls the function in your process, no GpuLayer involved |
| `train.run(*args, stream_logs=True)` | submits a job, streams its logs, returns the return value, raises `gl.RemoteError` on failure |
| `train.submit(*args)` | submits and returns a `Job` handle without waiting |
| `train.map(iterable, parallelism=None)` | one job per argument (or argument tuple); results in input order |
| `train.options(**overrides)` | a copy with other options |
| `train.spec(*args)` | the `JobSpec` that would be submitted |

With the embedded control plane, `map` starts up to four local workers so the jobs run in
parallel. With a remote control plane, the jobs spread over all compatible workers.

## Options

`@gl.remote(...)` and `.options(...)` accept: `accelerator`, `gpus`, `gpu_memory_gb`,
`memory_gb`, `cpu_cores`, `frameworks`, `python`, `target`, `fallback_targets`,
`worker_labels`, `on_unschedulable`, `provisioning`, `env`, `secrets`, `timeout_seconds`,
`retry`, `name`, `client`, `source`, `nodes`, `procs_per_node` and `comm_backend`.

## Distributed functions

With `nodes` or `procs_per_node` above 1, the function becomes a distributed job. torchrun
calls it on every process, and global rank 0's return value is the result:

```python
@gl.remote(nodes=2, procs_per_node=2)
def all_reduce_check() -> dict:
    import os, torch, torch.distributed as dist
    dist.init_process_group(os.environ["GPULAYER_COMM_BACKEND"])
    t = torch.tensor([float(dist.get_rank() + 1)])
    dist.all_reduce(t)
    return {"world": dist.get_world_size(), "sum": t.item()}

all_reduce_check.run()   # {'world': 4, 'sum': 10.0}
```

## Rules

- The function must be defined at module top level, in a file. GpuLayer uploads the file's
  directory and calls the function by name on the worker; it does not pickle closures.
- Arguments and return values must be JSON: numbers, strings, booleans, lists, dicts, `None`.
  Save tensors and models to `$GPULAYER_OUTPUT_DIR` instead of returning them.
- Imports used only by the function can live inside it, which keeps the submitting process
  light.

Full example: [`examples/remote_sweep/sweep.py`](../../examples/remote_sweep/sweep.py).
