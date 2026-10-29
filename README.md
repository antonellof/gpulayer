# GpuLayer

Run PyTorch on any GPU you can reach: this machine, a GPU server down the hall, a cloud VM, or
several of them at once. Write one training file; GpuLayer decides where it can run, starts it
there, and brings back logs, metrics, checkpoints and the trained model.

```python
import gpulayer.pytorch as glt

class Regressor(glt.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

trainer = glt.Trainer(max_epochs=5)                                  # this process, best device
trainer = glt.Trainer(max_epochs=5, devices=2, accelerator="cpu")    # DDP, 2 processes
trainer = glt.Trainer(max_epochs=5, accelerator="cuda", devices=4,
                      num_nodes=2, target="remote")                  # 8 GPUs on two machines
trainer.fit(model, train_loader, val_loader)
```

> **Status: pre-alpha.** Tested on CPU and Apple Silicon (MPS), with multiple workers on one
> host. NVIDIA GPUs and multi-machine runs are implemented but not validated yet; automatic
> cloud provisioning is simulated. See [Status and roadmap](docs/project/status.md).

## Install

```bash
git clone https://github.com/antonellof/gpulayer.git && cd gpulayer
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]' torch
gpulayer resources detect          # what this machine can run
```

## Quickstart

1. **Train in this process.** Picks CUDA, MPS or CPU automatically.

   ```bash
   python examples/trainer_regression/train.py
   ```

2. **Same file, two processes with DDP.** GpuLayer starts a local worker, launches torchrun,
   shards the data, and loads the trained weights back into your `model`.

   ```bash
   python examples/trainer_regression/train.py --accelerator cpu --devices 2
   ```

3. **Two nodes.** Two workers, ranks and rendezvous handled for you.

   ```bash
   python examples/trainer_regression/train.py --accelerator cpu --nodes 2 --devices 2
   ```

4. **Ask for hardware you don't have.** The job is rejected before it starts, with a reason
   per worker (`needs 1 usable cuda device(s), worker has 0`).

   ```bash
   python examples/trainer_regression/train.py --accelerator cuda --target local
   ```

5. **A parallel sweep** with `@gl.remote`:

   ```bash
   python examples/remote_sweep/sweep.py
   ```

6. **Add more machines.** On a machine that will be the server (it also runs jobs):

   ```bash
   gpulayer up
   ```

   It prints a `gpulayer join ...` line to paste on other GPU machines or cloud VMs, and an
   `export GPULAYER_URL=... GPULAYER_TOKEN=... GPULAYER_FINGERPRINT=...` line for your laptop.
   After that, the same commands as above run on the server's workers.

7. **See what ran.**

   ```bash
   gpulayer job list
   gpulayer job status <job-id>
   gpulayer worker status
   ```

The [full quickstart](docs/get-started/quickstart.md) shows the output of each step.

## With and without GpuLayer

| Situation | Without | With GpuLayer |
|---|---|---|
| Code on a laptop, GPU elsewhere | ssh, copy code, rebuild env, tmux, copy checkpoints back | `python train.py --target remote` |
| Several GPU machines | track free boxes by hand; jobs crash on the wrong GPU or CUDA build | one queue; each job lands where it fits or is rejected with the reason |
| Multi-node training | torchrun on every node with matching ranks, address and port | `Trainer(num_nodes=2, devices=4)` |
| Hyperparameter sweep | sequential loop or custom scripts | `train.map([...])` in parallel |
| Mac, CPU and CUDA from one file | device, DDP and sampler boilerplate | `accelerator="auto"` |

On a single machine with a single GPU, plain PyTorch is simpler. GpuLayer is also not a
sandbox for untrusted code. Details: [Why GpuLayer](docs/get-started/why-gpulayer.md).

## Three ways to use it

| API | For |
|---|---|
| [`Trainer` + `Module`](docs/pytorch/trainer.md) | new training code: one file, any device, any target |
| [`launch()`](docs/pytorch/launch.md) | an existing PyTorch script: two lines at the top |
| [`@gl.remote`](docs/pytorch/remote-functions.md) | functions and sweeps: `.run()`, `.map()` |

## Examples

| Example | Shows |
|---|---|
| [`trainer_regression`](examples/trainer_regression) | the smallest `Module` + `Trainer` |
| [`trainer_image_classifier`](examples/trainer_image_classifier) | a CNN on generated images, accuracy, DDP |
| [`trainer_char_lm`](examples/trainer_char_lm) | a small transformer: LR schedule, clipping, mixed precision, resume, sampling |
| [`runtime_custom_loop`](examples/runtime_custom_loop) | your own loop with `Runtime` handling devices and DDP |
| [`launch_ddp_script`](examples/launch_ddp_script) | a plain DDP script made portable with `launch()` |
| [`remote_sweep`](examples/remote_sweep) | `@gl.remote` and `.map()` |

## Documentation

The docs live in [`docs/`](docs/) and are published by the site in [`site/`](site/)
(`cd site && npm install && npm run dev`).

- Get started: [Introduction](docs/index.md) · [Installation](docs/get-started/installation.md) · [Quickstart](docs/get-started/quickstart.md) · [Core concepts](docs/get-started/concepts.md)
- PyTorch: [Trainer](docs/pytorch/trainer.md) · [Distributed](docs/pytorch/distributed.md) · [launch()](docs/pytorch/launch.md) · [Runtime](docs/pytorch/runtime.md) · [Remote functions](docs/pytorch/remote-functions.md)
- Guides: [Run a server](docs/guides/self-hosting.md) · [Apple Silicon](docs/guides/apple-silicon.md) · [NVIDIA CUDA](docs/guides/nvidia-cuda.md) · [Cloud](docs/guides/cloud-providers.md) · [CLI jobs](docs/guides/jobs-and-cli.md)
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
