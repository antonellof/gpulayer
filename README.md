<p align="center">
  <img src="docs/assets/logo.svg" alt="GPUlayer" width="300">
</p>

# GPUlayer

Run any workload on every GPU you can reach: this machine, a server down the hall, a cloud
VM, or several of them at once. A workload can be a command in any language, a Python function
using any library, or a distributed job. GPUlayer finds a machine that can run it, starts it
there, and brings back logs, outputs and results. PyTorch gets a deeper integration on top.

```bash
gpulayer submit --accelerator cuda --source ./app -- ./render.sh          # any command
```

```python
import gpulayer as gl

@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})               # any library
def matmul_benchmark(n: int) -> dict:
    import mlx.core as mx
    ...

matmul_benchmark.map([1024, 2048, 4096])                                  # in parallel
```

```python
import gpulayer.pytorch as glt                                            # PyTorch integration

trainer = glt.Trainer(max_epochs=5, accelerator="cuda", devices=4, num_nodes=2, target="remote")
trainer.fit(model, train_loader, val_loader)                              # 8 GPUs, two machines
```

> **Status: v0.1.0, the first release.** Tested on CPU and Apple Silicon (Metal: PyTorch MPS and MLX), with
> multiple workers on one host. NVIDIA GPUs and multi-machine runs are implemented but not
> validated yet; automatic cloud provisioning is simulated. See
> [Status and roadmap](docs/project/status.md).

## Install

**One command** (Linux, macOS; no sudo; installs into its own virtualenv):

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/gpulayer/main/install.sh | sh
gpulayer resources detect             # what this machine can run
```

Add `GPULAYER_EXTRAS=examples` before `sh` to include PyTorch, NumPy and MLX.

**Docker** (a server and workers, see [deploy/docker](deploy/docker)):

```bash
docker build -t gpulayer .
docker run -d -p 8470:8470 -v gpulayer-data:/data gpulayer up --no-worker
```

**From source**, to run the examples or work on GPUlayer:

```bash
git clone https://github.com/antonellof/gpulayer.git && cd gpulayer
uv venv && source .venv/bin/activate
uv pip install -e '.[dev,examples]'   # GPUlayer plus what the examples use (torch, numpy, mlx on Apple Silicon)
```

More options in [Installation](docs/get-started/installation.md).

## Quickstart

1. **Run a command as a job.** Any executable; inputs staged, outputs returned.

   ```bash
   gpulayer submit --source examples/shell_job --input examples/shell_job/sales.csv:sales.csv -- bash report.sh
   ```

2. **Run Python functions in parallel**, with any library (NumPy here, MLX on the Apple GPU in
   `examples/mlx_metal`).

   ```bash
   python examples/numpy_batch/estimate.py
   ```

3. **Train a PyTorch model in this process.** Picks CUDA, MPS or CPU automatically.

   ```bash
   python examples/trainer_regression/train.py
   ```

4. **Same file, two processes with DDP.** GPUlayer starts a local worker, launches torchrun,
   shards the data, and loads the trained weights back into your `model`.

   ```bash
   python examples/trainer_regression/train.py --accelerator cpu --devices 2
   ```

5. **Two nodes.** Two workers, ranks and rendezvous handled for you.

   ```bash
   python examples/trainer_regression/train.py --accelerator cpu --nodes 2 --devices 2
   ```

6. **Ask for hardware you don't have.** The job is rejected before it starts, with a reason
   per worker (`needs 1 usable cuda device(s), worker has 0`).

   ```bash
   python examples/trainer_regression/train.py --accelerator cuda --target local
   ```

7. **A PyTorch sweep** with `@gl.remote`:

   ```bash
   python examples/remote_sweep/sweep.py
   ```

8. **Add more machines.** On a machine that will be the server (it also runs jobs):

   ```bash
   gpulayer up
   ```

   It prints a `gpulayer join ...` line to paste on other GPU machines or cloud VMs, and an
   `export GPULAYER_URL=... GPULAYER_TOKEN=... GPULAYER_FINGERPRINT=...` line for your laptop.
   After that, the same commands as above run on the server's workers.

9. **See what ran**, in the terminal or the web dashboard.

   ```bash
   gpulayer job list
   gpulayer job status <job-id>
   gpulayer dashboard          # web UI: jobs, live logs, placement reasons, workers, GPUs
   ```

   Servers also expose Prometheus metrics at `/metrics`, with a ready-made Grafana dashboard
   in [`deploy/monitoring`](deploy/monitoring) ([guide](docs/guides/monitoring.md)).

   ![GPUlayer dashboard: jobs, workers, CPU load, memory and task processes over time](docs/assets/dashboard/overview-dark.png)

The [full quickstart](docs/get-started/quickstart.md) shows the output of each step.

## Where GPUlayer sits

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/where-gpulayer-sits-dark.svg">
  <img alt="Your workloads go through GPUlayer to the machines you can reach" src="docs/assets/where-gpulayer-sits-light.svg">
</picture>

Frameworks (PyTorch, JAX, MLX, CuPy, ...) decide how a computation uses the hardware they are
given. GPUlayer decides which hardware runs it, gets code and data there, starts it, and returns
the results: for training, inference batches, rendering, simulation or data processing.

| Situation | Without | With GPUlayer |
|---|---|---|
| Code on a laptop, GPU somewhere else | ssh, copy code, rebuild env, tmux, copy results back | `gpulayer submit --target remote -- <cmd>` or `python train.py --target remote` |
| Several GPU machines | track free boxes by hand; jobs crash on the wrong GPU, driver or library | one queue; each job lands where its accelerator and frameworks exist, or is rejected with the reason |
| NVIDIA boxes and Apple Silicon Macs | separate scripts and setup per machine | ask for `cuda` or `mps`; one interface |
| A batch of independent tasks | loops and ad-hoc scripts | `fn.map([...])` in parallel |
| Multi-node training | `torchrun` on every node with matching ranks, address and port | `Trainer(num_nodes=2, devices=4)` |

On a single machine with a single GPU, running your code directly is simpler. GPUlayer runs jobs
that finish (long-running services are planned) and is not a sandbox for untrusted code.
Details: [Why GPUlayer](docs/get-started/why-gpulayer.md).

## Ways to use it

| API | For |
|---|---|
| [`gpulayer submit -- <cmd>`](docs/workloads/commands.md) | any command, script or binary |
| [`@gl.remote`](docs/workloads/functions.md) | Python functions with any library: `.run()`, `.map()`, distributed |
| [`Trainer` + `Module`](docs/pytorch/trainer.md) | PyTorch training: one file, any device, any target |
| [`launch()`](docs/pytorch/launch.md) | an existing PyTorch script: two lines at the top |

## Examples

| Example | Shows |
|---|---|
| [`trainer_regression`](examples/trainer_regression) | the smallest `Module` + `Trainer` |
| [`trainer_image_classifier`](examples/trainer_image_classifier) | a CNN on generated images, accuracy, DDP |
| [`trainer_char_lm`](examples/trainer_char_lm) | a small transformer: LR schedule, clipping, mixed precision, resume, sampling |
| [`runtime_custom_loop`](examples/runtime_custom_loop) | your own loop with `Runtime` handling devices and DDP |
| [`launch_ddp_script`](examples/launch_ddp_script) | a plain DDP script made portable with `launch()` |
| [`remote_sweep`](examples/remote_sweep) | a PyTorch sweep with `@gl.remote` and `.map()` |
| [`mlx_metal`](examples/mlx_metal) | MLX on the Apple GPU (no PyTorch) |
| [`numpy_batch`](examples/numpy_batch) | NumPy chunks in parallel with `.map()` |
| [`shell_job`](examples/shell_job) | a shell script with input and output files |

## Documentation

The docs live in [`docs/`](docs/) and are published by the site in [`site/`](site/)
(`cd site && npm install && npm run dev`).

- Get started: [Introduction](docs/index.md) · [Installation](docs/get-started/installation.md) · [Quickstart](docs/get-started/quickstart.md) · [Core concepts](docs/get-started/concepts.md)
- Workloads: [Commands](docs/workloads/commands.md) · [Python functions](docs/workloads/functions.md) · [Any GPU framework](docs/workloads/frameworks.md) · [Distributed jobs](docs/workloads/distributed.md)
- PyTorch integration: [Trainer](docs/pytorch/trainer.md) · [Distributed training](docs/pytorch/distributed.md) · [launch()](docs/pytorch/launch.md) · [Runtime](docs/pytorch/runtime.md)
- Guides: [Run a server](docs/guides/self-hosting.md) · [Apple Silicon](docs/guides/apple-silicon.md) · [NVIDIA CUDA](docs/guides/nvidia-cuda.md) · [Cloud](docs/guides/cloud-providers.md)
- Architecture: [Overview](docs/architecture/overview.md) · [Scheduling](docs/architecture/scheduling.md) · [Security](docs/architecture/security.md) · [Decisions (ADRs)](docs/decisions/README.md)
- Reference: [Python API](docs/reference/python-api.md) · [CLI](docs/reference/cli.md) · [JobSpec](docs/reference/job-spec.md)

## Development

```bash
.venv/bin/pytest                    # unit, integration, distributed; hardware tests self-skip
.venv/bin/ruff check src tests examples && .venv/bin/mypy src
```

See [Contributing](docs/project/contributing.md). No license has been chosen yet
([ADR-0012](docs/decisions/ADR-0012-licensing-and-project-governance.md)); until one is added,
no rights are granted beyond viewing.
