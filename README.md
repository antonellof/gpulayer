# GpuLayer

**One job model, one capability system, one scheduling layer — many execution environments.**

GpuLayer is an open-source GPU execution and orchestration platform. You describe a workload
and what it needs; GpuLayer finds a compatible worker (your laptop, a GPU box on your network,
a provider-provisioned cloud machine, or a group of them) and runs it, with logs, events,
results and artifacts flowing back through one SDK and CLI.

```python
import gpulayer as gl

with gl.Client() as client:                       # embedded control plane, no account needed
    src = client.package("examples/pytorch_train")
    job = client.submit(gl.command(["python", "train.py"], source=src,
                                   resources=gl.ResourceRequest(accelerator="mps")))
    for line in job.follow_logs():
        print(line.line)
    print(job.result().outputs)                   # model.pt, metrics.json as artifacts
```

> **Status: pre-alpha (milestone 1).** The core is implemented and tested; several paths are
> deliberately marked *simulated* or *planned*. See [What works today](#what-works-today) —
> nothing below is advertised beyond what the test suite demonstrates.

---

## Why

- **Unified interface.** The same `JobSpec`, SDK and CLI for local, remote and cloud targets.
- **Capability-aware placement.** Scheduling uses GPU vendor, device count and memory (and
  whether it is *unified* or *dedicated*), CUDA compute capability, OS/arch, Python and framework
  versions, launcher and communication backend — never "it has a GPU, ship it".
- **Explainable.** Every scheduling decision records why each worker was eligible or rejected.
- **Distributed is a workload type, not a batch.** A `DistributedJob` is gang-scheduled onto a
  compatible worker group, ranks are assigned, a rendezvous endpoint is published, and
  `torchrun` + PyTorch do the collectives. One failed node fails the whole job.
- **Honest about limits.** MPS is not CUDA. ROCm is detection-only. Leases are not
  exactly-once. Subprocesses are not a sandbox. The docs say so.

## Concepts

| Concept | Values | Meaning |
|---|---|---|
| **Execution target** | `local`, `remote`, `cloud` | *Where* a job may run. Fallback across targets only when explicitly listed. |
| **Workload type** | `independent`, `distributed` | *What* runs: one execution unit vs one coordinated multi-node workload. |
| **Workload kind** | `CommandJob`, `FunctionJob`, `DistributedJob` | A command, a `module:function` with JSON args, or a torchrun entrypoint. |
| **Worker origin** | `local`, `existing`, `provider` | How a worker joined; maps to the target it serves. |

Targets and workload types are orthogonal: a distributed job can run on local workers, and a
cloud job is not necessarily distributed.

## Install

Requires Python ≥ 3.10. CPU-only installs need no GPU libraries.

```bash
git clone <this repo> && cd gpulayer
uv venv && uv pip install -e '.[dev]'        # or: python -m venv .venv && pip install -e '.[dev]'
uv pip install torch                          # optional: MPS/CUDA probing, torchrun jobs, examples
```

## Quickstart

```bash
gpulayer resources detect                     # what this machine offers (and what it doesn't)

# Run a command locally; an in-process worker executes it and the CLI streams logs.
gpulayer submit -- python -c "import platform; print(platform.machine())"

# Inspect it afterwards
gpulayer job list
gpulayer job status <job-id>                  # state, placement explanation, tasks
gpulayer job logs <job-id>
gpulayer job events <job-id>
gpulayer job result <job-id> --download ./out

# Function job: module:function from a packaged directory; JSON in, JSON out
gpulayer submit --source examples/pytorch_train --function predict:predict --arg '[[1, 1]]'

# Queue now, execute later with a long-running worker
gpulayer submit --detach -- python long_task.py
gpulayer worker start
gpulayer job cancel <job-id>
```

Requirements that no worker can satisfy fail fast with a reason per worker, e.g.
`needs 1 usable cuda device(s), worker has 0`.

### Distributed PyTorch (DDP)

```bash
# One node, two processes:
gpulayer submit --distributed --source examples/pytorch_ddp --entrypoint train_ddp.py \
    --nodes 1 --procs-per-node 2 --framework 'torch>=2.0'

# Two nodes x two processes (world size 4): start two workers, then submit
gpulayer worker start --name node-a &
gpulayer worker start --name node-b &
gpulayer submit --detach --distributed --source examples/pytorch_ddp \
    --entrypoint train_ddp.py --nodes 2 --procs-per-node 2
```

### Self-hosted control plane with remote workers

```bash
# On the control-plane host
gpulayer server init                          # prints a client token and a worker join token ONCE
gpulayer server start --host 0.0.0.0 --tls-cert cert.pem --tls-key key.pem

# On each GPU machine
GPULAYER_JOIN_TOKEN=glj_... gpulayer --url https://cp.example:8470 worker start

# From your laptop
export GPULAYER_URL=https://cp.example:8470 GPULAYER_TOKEN=glc_...
gpulayer submit --target remote --accelerator cuda --gpus 1 -- python train.py
```

Plaintext HTTP on a non-loopback address is refused unless you pass
`--allow-insecure-remote`.

## Examples

| Example | Shows |
|---|---|
| [`examples/pytorch_train`](examples/pytorch_train) | Train an MLP on CPU / MPS / CUDA, download `model.pt`, then a JSON `FunctionJob` for inference — via the SDK |
| [`examples/pytorch_ddp`](examples/pytorch_ddp) | `DistributedDataParallel` with `DistributedSampler`, launched by GpuLayer through torchrun across 1 or more nodes |

## What works today

Status vocabulary from [CLAUDE.md](CLAUDE.md): *hardware-validated* · *tested* · *simulated* ·
*planned*.

| Capability | Status | Evidence |
|---|---|---|
| Local CPU execution (command / function), logs, results, artifacts, cancel, timeout | **tested** | `tests/integration/test_local_execution.py`, `test_cli.py` |
| Capability-aware deterministic scheduling, rejection reasons, gang scheduling | **tested** | `tests/unit/test_scheduler.py` |
| Leases, heartbeats, worker loss, opt-in retry, revocation, persistence | **tested** (simulated clock) | `tests/unit/test_control_plane.py` |
| Remote worker over authenticated HTTP (separate process, separate state dir) | **tested on one host** | `tests/integration/test_http_remote.py` |
| Remote worker on a *different machine* | **not yet validated** | protocol is network-ready; no cross-host test yet |
| Distributed torchrun/gloo, 2 nodes × 2 procs, fail-fast | **tested on one host** (real processes, loopback) | `tests/distributed/test_torchrun_multiprocess.py` |
| Multi-machine distributed | **planned** | needs a documented multi-host test environment |
| Apple MPS independent jobs | **hardware-validated** (Apple Silicon smoke test) | `tests/hardware/test_gpu_smoke.py` |
| Distributed MPS | **unsupported** by design (ADR-0006) | rejected at validation |
| NVIDIA CUDA detection & scheduling | **tested** (mocked `nvidia-smi` / torch probe) | `tests/unit/test_discovery.py` |
| NVIDIA CUDA execution (single GPU and NCCL) | **implemented, not hardware-validated** | smoke test exists, skipped without a GPU |
| AMD ROCm | **detection-only** | never scheduled |
| Cloud provisioning lifecycle (offer selection, single-use bootstrap tokens, reservation, release, cost estimates) | **simulated** with `FakeProvider` | `tests/unit/test_providers.py` |
| Real cloud provider adapter | **planned** — no vendor chosen (ADR-0010) | — |
| Hosted control plane | **planned** | — |

## Architecture at a glance

```
 SDK / CLI ──► Control plane (embedded in-process, or `gpulayer server` over HTTPS)
                 jobs · worker registry · leases · scheduler · coordination · provisioning
                 SQLite metadata  +  content-addressed artifact store
                        │  worker protocol (poll: heartbeat / acquire / start / logs / finish)
        ┌───────────────┼──────────────────────┐
   local worker    existing worker       provider worker ◄── ProviderAdapter (FakeProvider)
        └──► WorkerAgent ──► process group: command | function runner | torchrun
```

Read more:

- [Getting started](docs/getting-started.md)
- [Architecture overview](docs/architecture/overview.md) · [Execution modes](docs/architecture/execution-modes.md) ·
  [Remote workers](docs/architecture/remote-workers.md) · [Cloud & providers](docs/architecture/providers.md) ·
  [Distributed](docs/architecture/distributed.md) · [GPU compatibility & MPS](docs/architecture/gpu-compatibility.md) ·
  [Scheduling & cost](docs/architecture/scheduling-and-cost.md) · [Security & trust](docs/architecture/security.md)
- [Architecture decisions (ADRs)](docs/decisions/README.md)
- [Implementation plan & status](docs/development/implementation-plan.md) · [Contributing & testing](docs/development/contributing.md)
- Long-form design: [architecture_plan.md](architecture_plan.md)

## Development

```bash
.venv/bin/pytest                               # everything (hardware tests self-skip)
.venv/bin/pytest tests/unit                    # fast, no processes
.venv/bin/pytest -m distributed                # real torchrun runs (needs torch)
.venv/bin/pytest -m "hardware_mps or hardware_cuda"
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests && .venv/bin/mypy src
```

## License

Not yet chosen — see [ADR-0012](docs/decisions/ADR-0012-licensing-and-project-governance.md).
Until a license is added, no rights are granted beyond viewing.
