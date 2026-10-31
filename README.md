<p align="center">
  <img src="https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/logo.svg" alt="GPUlayer" width="300">
</p>

<p align="center">
  <b>Run any workload on every GPU (and CPU) you can reach.</b><br>
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#for-ai-and-coding-agents">AI agents</a> ·
  <a href="#why-gpulayer">Why GPUlayer</a> ·
  <a href="#performance-and-comparison">Performance</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="https://www.gpulayer.dev/docs/">Docs</a> ·
  <a href="https://github.com/antonellof/gpulayer/tree/main/examples">Examples</a> ·
  <a href="https://github.com/antonellof/gpulayer/blob/main/CHANGELOG.md">Changelog</a>
</p>

GPUlayer sends a command, a Python function or a distributed job to machines that can run
it (this one, remote servers, clusters, data centers, cloud VMs, or several at once), starts
it there and brings back logs, outputs and results. PyTorch gets a deeper integration on top.
Documentation: [gpulayer.dev/docs](https://www.gpulayer.dev/docs/).

> **v0.5.0.** Tested on CPU, Apple Silicon (PyTorch MPS and MLX) and an
> NVIDIA Tesla T4, on machines with several GPUs (4× RTX 3070, 2× A100 with NVLink: NCCL across all
> GPUs of a machine), and on up to 30 machines on Google Cloud. NCCL across machines is implemented
> but not validated yet; cloud provisioning covers Google Cloud. See
> [Status and roadmap](https://github.com/antonellof/gpulayer/blob/main/docs/project/status.md).

## Install

```bash
pip install gpulayer
```

That's it. Check what this machine can run:

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
| Docker | `docker pull ghcr.io/antonellof/gpulayer:0.5.0` (also `:0.5.0-torch`; see [deploy/docker](https://github.com/antonellof/gpulayer/tree/main/deploy/docker)) |
| Kubernetes (Helm) | `helm install gpulayer deploy/kubernetes/helm/gpulayer -n gpulayer --create-namespace` (see [Kubernetes](https://github.com/antonellof/gpulayer/blob/main/docs/guides/kubernetes.md)) |
| From source, to run the examples | `git clone https://github.com/antonellof/gpulayer && cd gpulayer && pip install -e ".[dev,examples]"` |

More in [Installation](https://github.com/antonellof/gpulayer/blob/main/docs/get-started/installation.md).
</details>

## Quickstart

```bash
# Any command as a job: code and inputs shipped, outputs returned
gpulayer submit --accelerator cuda --source ./app -- ./render.sh

# PyTorch: the same script in this process, as 2-process DDP, or on 2 nodes
python examples/trainer_regression/train.py
python examples/trainer_regression/train.py --accelerator cpu --devices 2
python examples/trainer_regression/train.py --accelerator cpu --nodes 2 --devices 2

# More machines: run this on the server, then paste the printed `gpulayer join` line elsewhere
gpulayer up

# See what ran
gpulayer dashboard
```

```python
import gpulayer as gl

@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})   # any library, any GPU
def benchmark(n: int) -> dict: ...

benchmark.map([1024, 2048, 4096])                           # in parallel
```

![GPUlayer dashboard](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/dashboard/overview-dark.png)

The [full quickstart](https://github.com/antonellof/gpulayer/blob/main/docs/get-started/quickstart.md) walks through each step with its output.

## For AI and coding agents

`gpulayer mcp` is an MCP server, so Claude Code, Claude Desktop, Cursor, Codex and VS Code
agents can list GPUs, submit jobs, wait, read logs and download outputs. An
[Agent Skill](https://github.com/antonellof/gpulayer/blob/main/integrations/skills/gpulayer/SKILL.md) covers agents that use the CLI.

```bash
claude mcp add gpulayer -- gpulayer mcp
```

See [AI and coding agents](https://github.com/antonellof/gpulayer/blob/main/docs/guides/ai-agents.md).

## Why GPUlayer

Most GPU code assumes the GPU is on the machine where you run it. With GPUlayer you describe
what a workload needs (an accelerator, GPU memory, frameworks) and it runs on a machine that
has it, wherever that machine is.

| Situation | Without GPUlayer | With GPUlayer |
|---|---|---|
| Code on a laptop, GPU elsewhere | ssh, copy code, rebuild the environment, tmux, copy results back | `gpulayer submit --target remote -- <cmd>` |
| Several GPU machines | track free machines by hand; jobs crash on the wrong GPU or library | one queue; each job lands where it can run, or is rejected with the reason |
| NVIDIA servers and Apple Silicon Macs | separate scripts and setup per machine | ask for `cuda` or `mps`; one interface |
| Many independent tasks | loops and ad hoc scripts | `fn.map([...])` across free GPUs |
| Multi-node training | `torchrun` on every node with matching ranks and addresses | `Trainer(num_nodes=2, devices=4)`; the machines are kept in one zone or rack when possible |

On one machine with one GPU, running your code directly is simpler. GPUlayer runs services
without a load balancer in front of them, and is not a sandbox for untrusted code. More in
[Why GPUlayer](https://www.gpulayer.dev/docs/#why-gpulayer).

## Performance and comparison

Measured on NVIDIA GPUs against PyTorch Lightning, with the same model, data and optimizer
and each trainer's defaults:

![Training step time: GPUlayer and PyTorch Lightning](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/training-step-light.svg#gh-light-mode-only)
![Training step time: GPUlayer and PyTorch Lightning](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/training-step-dark.svg#gh-dark-mode-only)
![Several GPUs in one machine](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/multigpu-step-light.svg#gh-light-mode-only)
![Several GPUs in one machine](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/multigpu-step-dark.svg#gh-dark-mode-only)

- One Tesla T4: GPUlayer's Trainer is 1.25 to 1.39x faster per step on small models and 1.14x
  on a compute-bound CNN, and starts 0.1 to 0.7 s sooner.
- 1, 2 and 4 RTX 3070s (DDP over NCCL): GPUlayer is faster in 5 of 6 cases, by 1.11 to 1.76x,
  with the lead growing with the number of GPUs; the 1-GPU CNN was 1.14x faster with Lightning
  on that host.
- NCCL across 4 GPUs and over NVLink (2x A100), and the best-connected GPU choice, validated
  on rented machines.
- One training job across machines on CPUs (gloo): GPUlayer trains 11.9% faster than
  Lightning under torchrun on 8 machines (50.5 s vs 57.3 s; 5.3% on 4, 1.6% on 2), and it
  starts every machine from one `Trainer(num_nodes=N)` call.
- A distributed job split across two GCP regions ran 16 to 24% slower than in one region;
  GPUlayer keeps a group in one zone or region when it can.
- Distributed jobs start 1.1 to 2.3 s faster than through torchrun; tested on up to 16
  machines (32 processes) and 30 workers across two GCP regions.
- Across two machines: a distributed job spanning both, fail-fast in 2.7 s, and 55.6 small
  jobs per second.

![One training job on more machines](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/strong-scaling-light.svg#gh-light-mode-only)
![One training job on more machines](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/perf/strong-scaling-dark.svg#gh-dark-mode-only)

Method, all charts and raw logs:
[Performance and comparison](https://github.com/antonellof/gpulayer/blob/main/docs/get-started/performance-and-comparison.md). Reproduce them
with [`benchmarks/`](https://github.com/antonellof/gpulayer/blob/main/benchmarks/README.md).

## Architecture

![Your workloads go through GPUlayer to the machines you can reach](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/architecture-light.svg#gh-light-mode-only)
![Your workloads go through GPUlayer to the machines you can reach](https://raw.githubusercontent.com/antonellof/gpulayer/main/docs/assets/architecture-dark.svg#gh-dark-mode-only)

GPUlayer has three parts:

- **Clients** submit work: the `gpulayer` CLI, the Python SDK (`@gl.remote`, `gl.Client`), the
  PyTorch `Trainer` and `launch()`, and the MCP server for AI agents. A job states what it
  needs (accelerator, GPU count and memory, frameworks), not which machine to use.
- **The control plane** keeps the job queue, picks a worker for each job and records the
  reason when no worker fits. Distributed groups stay in one network group, zone or region
  when possible, and multi-GPU jobs get the best-connected GPUs on a machine. It stores code bundles, inputs, outputs and logs, and checks
  tokens. For work on one machine it runs inside your process; for several machines it runs
  as a server (`gpulayer up`).
- **Workers**, one per machine, report their GPUs and installed frameworks, ask the control
  plane for tasks, run them as normal processes (one per rank for distributed jobs),
  stream logs back and upload outputs. Workers open every connection themselves, so they can
  sit behind NAT. A cloud VM is a worker that joined with `gpulayer join`.

A job goes from `queued` to `scheduled` when a worker fits, then `running`, and ends as
`succeeded`, `failed`, `cancelled` or `timed_out`. Frameworks such as PyTorch, JAX, MLX and
CuPy still decide how the computation uses the GPU; GPUlayer decides which GPU runs it. More in
[Architecture overview](https://github.com/antonellof/gpulayer/blob/main/docs/architecture/overview.md).

## Development

```bash
uv pip install -e '.[dev,examples]'
.venv/bin/pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

See [Contributing](https://github.com/antonellof/gpulayer/blob/main/docs/project/contributing.md).

## License

Apache License 2.0, see [LICENSE](https://github.com/antonellof/gpulayer/blob/main/LICENSE).
