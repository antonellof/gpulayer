<p align="center">
  <img src="https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/logo.svg" alt="GPUlayer" width="300">
</p>

<p align="center">
  <b>Train PyTorch on every GPU you can reach.</b><br>
  One Trainer, from your laptop to many machines.<br>
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#performance-and-comparison">Performance</a> ·
  <a href="#for-ai-and-coding-agents">AI agents</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="https://www.gpulayer.dev/docs/">Docs</a> ·
  <a href="https://github.com/antonellof/gpulayer/tree/main/examples">Examples</a> ·
  <a href="https://github.com/antonellof/gpulayer/blob/main/CHANGELOG.md">Changelog</a>
</p>

The GPUlayer `Trainer` runs the same PyTorch script in your process, on every GPU of a
machine, or across many machines: laptops, workstations, servers and cloud VMs. It picks
machines and GPUs that can train the model, starts every process, and brings back logs,
checkpoints and metrics.

NVIDIA CUDA, Apple Metal (MPS) and CPU are validated on hardware. AMD ROCm is implemented and
not yet validated. Docs: [gpulayer.dev/docs](https://www.gpulayer.dev/docs/).

> **v0.7.1.** Validated on a Tesla T4, 4× RTX 3070, 2× A100 (NVLink), an Apple M2 Pro, and
> distributed CPU training on up to 16 Google Cloud machines. NCCL across machines is
> implemented but not validated. See
> [Status](https://github.com/antonellof/gpulayer/blob/main/docs/project/status.md).

## Install

```bash
pip install gpulayer
```

Check what this machine can train on:

```bash
gpulayer resources detect
```

To train with PyTorch, install it too: `pip install "gpulayer[torch]"`.

<details>
<summary>Other ways to install</summary>

| Method | Command |
|---|---|
| As a standalone CLI with [uv](https://docs.astral.sh/uv/) | `uv tool install gpulayer` |
| Install script (own virtual environment, no sudo) | `curl -fsSL https://raw.githubusercontent.com/antonellof/gpulayer/main/install.sh \| sh` |
| Docker | `docker pull ghcr.io/antonellof/gpulayer:0.7.1` (also `:0.7.1-torch`; see [deploy/docker](https://github.com/antonellof/gpulayer/tree/main/deploy/docker)) |
| Kubernetes (Helm) | `helm install gpulayer deploy/kubernetes/helm/gpulayer -n gpulayer --create-namespace` (see [Kubernetes](https://github.com/antonellof/gpulayer/blob/main/docs/guides/kubernetes.md)) |
| From source, to run the examples | `git clone https://github.com/antonellof/gpulayer && cd gpulayer && pip install -e ".[dev,examples]"` |

More in [Installation](https://github.com/antonellof/gpulayer/blob/main/docs/get-started/installation.md).
</details>

## Quickstart

```python
import gpulayer.pytorch as glt

trainer = glt.Trainer(max_epochs=5, accelerator="cuda", devices=4, num_nodes=2)
trainer.fit(model, train_loader)          # 2 machines x 4 GPUs, DDP over NCCL
```

```bash
# The same file in this process, as 2-process DDP, or on 2 nodes: only the flags change
python examples/trainer_regression/train.py
python examples/trainer_regression/train.py --accelerator cpu --devices 2
python examples/trainer_regression/train.py --accelerator cpu --nodes 2 --devices 2

# More machines: run this on one, then paste the printed `gpulayer join` line on the others
gpulayer up
```

The [full quickstart](https://github.com/antonellof/gpulayer/blob/main/docs/get-started/quickstart.md) shows the output of each step.

## Performance

Against PyTorch Lightning, with the same model, data and optimizer and each trainer's
defaults, the GPUlayer Trainer was faster in 10 of 11 GPU cases and in every CPU case.

![Training step time: GPUlayer and PyTorch Lightning](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/training-step-light.svg#gh-light-mode-only)
![Training step time: GPUlayer and PyTorch Lightning](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/training-step-dark.svg#gh-dark-mode-only)
![Several GPUs in one machine](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/multigpu-step-light.svg#gh-light-mode-only)
![Several GPUs in one machine](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/multigpu-step-dark.svg#gh-dark-mode-only)

| Hardware | GPUlayer Trainer against Lightning |
|---|---|
| Tesla T4 | 1.14 to 1.39x per step |
| 1, 2 and 4 RTX 3070s (DDP over NCCL) | 1.11 to 1.76x in 5 cases; 1.14x slower on one GPU with a CNN |
| Apple M2 Pro, GPU and CPU | 1.05 to 1.75x |
| 8 CPU machines on Google Cloud (gloo) | 11.9% faster (Lightning under torchrun) |

Keeping a 4-machine run in one zone instead of two regions made it 1.38x faster (42.5 s vs
58.8 s). Method and logs:
[Performance](https://github.com/antonellof/gpulayer/blob/main/docs/get-started/performance-and-comparison.md);
scripts in [`benchmarks/`](https://github.com/antonellof/gpulayer/blob/main/benchmarks/README.md).

## How it works

![Your training goes through GPUlayer to the machines you can reach](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/architecture-light.svg#gh-light-mode-only)
![Your training goes through GPUlayer to the machines you can reach](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/architecture-dark.svg#gh-dark-mode-only)

The Trainer submits a run that says what it needs (accelerator, number of GPUs and machines,
GPU memory). A control plane, inside your process or run as a server with `gpulayer up`,
picks machines that can train it: the best-connected GPUs of a machine, and machines in one
zone when it can. A worker on each machine starts the ranks, streams logs and uploads
checkpoints. Workers open every connection themselves, so machines can sit behind NAT. More
in the [Architecture overview](https://github.com/antonellof/gpulayer/blob/main/docs/architecture/overview.md).

The same engine runs plain commands and Python functions as jobs, and `gpulayer mcp` lets
coding agents start runs and read their results
([AI agents](https://github.com/antonellof/gpulayer/blob/main/docs/guides/ai-agents.md)).

With one GPU on the machine you are using, a plain training loop is simpler. Jobs are not
sandboxed, so use GPUlayer among people who trust each other.

## Development

```bash
uv pip install -e '.[dev,examples]'
.venv/bin/pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

See [Contributing](https://github.com/antonellof/gpulayer/blob/main/docs/project/contributing.md).

## License

Apache License 2.0, see [LICENSE](https://github.com/antonellof/gpulayer/blob/main/LICENSE).
