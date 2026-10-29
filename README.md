<p align="center">
  <img src="docs/assets/logo.svg" alt="GPUlayer" width="300">
</p>

<p align="center">
  <b>Run any workload on every GPU you can reach.</b><br>
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#for-ai-and-coding-agents">AI agents</a> ·
  <a href="#where-gpulayer-sits">Where it sits</a> ·
  <a href="docs/index.md">Docs</a> ·
  <a href="examples">Examples</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

GPUlayer sends a command, a Python function or a distributed job to a machine that can run
it (this one, a server down the hall, a cloud VM, or several at once), starts it there and
brings back logs, outputs and results. PyTorch gets a deeper integration on top.

> **v0.1.0, the first release.** Tested on CPU and Apple Silicon (PyTorch MPS and MLX) and in
> Linux CI. NVIDIA GPUs and multi-machine runs are implemented but not validated yet; cloud
> provisioning is simulated. See [Status and roadmap](docs/project/status.md).

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/gpulayer/main/install.sh | sh
gpulayer resources detect      # what this machine can run
```

Or with Docker (`docker build -t gpulayer .`, see [deploy/docker](deploy/docker)), or from
source to run the examples (`uv pip install -e '.[dev,examples]'`).
More in [Installation](docs/get-started/installation.md).

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

![GPUlayer dashboard](docs/assets/dashboard/overview-dark.png)

The [full quickstart](docs/get-started/quickstart.md) walks through each step with its output.

## For AI and coding agents

`gpulayer mcp` is an MCP server, so Claude Code, Claude Desktop, Cursor, Codex and VS Code
agents can list GPUs, submit jobs, wait, read logs and download outputs. An
[Agent Skill](integrations/skills/gpulayer/SKILL.md) covers agents that use the CLI.

```bash
claude mcp add gpulayer -- gpulayer mcp
```

See [AI and coding agents](docs/guides/ai-agents.md).

## Where GPUlayer sits

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/where-gpulayer-sits-dark.svg">
  <img alt="Your workloads go through GPUlayer to the machines you can reach" src="docs/assets/where-gpulayer-sits-light.svg">
</picture>

Frameworks decide how a computation uses the hardware they are given; GPUlayer decides which
hardware runs it. On one machine with one GPU, running your code directly is simpler. See
[Why GPUlayer](docs/get-started/why-gpulayer.md).

## Development

```bash
uv pip install -e '.[dev,examples]'
.venv/bin/pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

See [Contributing](docs/project/contributing.md). No license has been chosen yet
([ADR-0012](docs/decisions/ADR-0012-licensing-and-project-governance.md)); until one is added,
no rights are granted beyond viewing.
