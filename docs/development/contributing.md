# Contributing, testing and releases

## Setup

```bash
uv venv && uv pip install -e '.[dev]' && uv pip install torch
```

## Quality gates (run before every commit)

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src
.venv/bin/pytest
```

## Test layout

| Directory | Kind | Needs |
|---|---|---|
| `tests/unit` | contracts, scheduler, discovery (mocked), control-plane lifecycle with **simulated** workers and a fake clock, providers via `FakeProvider` | nothing |
| `tests/integration` | **real** subprocess execution via SDK/CLI; HTTP server + worker in a separate process | nothing beyond the package |
| `tests/distributed` | **real** torchrun/gloo multi-process, 2-node topology on one host | `torch` (skips otherwise) |
| `tests/hardware` | MPS / CUDA smoke tests | Apple Silicon or NVIDIA GPU (skip otherwise) |

Markers: `integration`, `distributed`, `hardware_mps`, `hardware_cuda`. CPU-only CI runs
everything; hardware tests skip themselves. A passing unit test is not evidence of cloud
provisioning or multi-node GPU compatibility — say which kind of test backs a claim.

## Rules of the road

Read [CLAUDE.md](../../CLAUDE.md) and [architecture_plan.md](../../architecture_plan.md).
In short: typed contracts, provider logic out of the scheduler, no invented prices or
capabilities, documentation updated with behaviour, ADRs for architectural decisions
([docs/decisions](../decisions/README.md)), and implementation status labelled honestly.

## Releases

Not yet established. Versioning follows PEP 440 (`0.1.0.devN` during milestone 1). Contract
changes bump `API_VERSION` / `PROTOCOL_VERSION` and must document migration. A license and
release process will be set before any public release (ADR-0012).
