# Contributing, testing and releases

## Setup

```bash
uv venv && uv pip install -e '.[dev,examples]'
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
| `tests/unit` | contracts, scheduler, discovery (mocked), control-plane lifecycle with simulated workers and a fake clock, providers via `FakeProvider` | nothing |
| `tests/integration` | real subprocess execution via SDK/CLI; HTTP server + worker in a separate process | nothing beyond the package |
| `tests/distributed` | real torchrun/gloo multi-process run, 2-node topology on one host | `torch` (skips otherwise) |
| `tests/hardware` | MPS / CUDA smoke tests | Apple Silicon or NVIDIA GPU (skip otherwise) |

The markers are `integration`, `distributed`, `hardware_mps` and `hardware_cuda`. CPU-only CI
runs everything, and hardware tests skip themselves. A passing unit test says nothing about real
cloud provisioning or multi-node GPU compatibility, so name the kind of test behind any claim.

## Rules of the road

Read [CLAUDE.md](../../CLAUDE.md) and [architecture_plan.md](../../architecture_plan.md)
first. Keep contracts typed and provider logic out of the scheduler. Never invent prices or
capabilities. Update the documentation in the same change as the behaviour, record
architectural decisions as ADRs ([docs/decisions](../decisions/README.md)), and label the
implementation status of every feature accurately.

## Releases

GPUlayer is licensed under Apache-2.0 (ADR-0012). Versions follow PEP 440; before 1.0, minor
versions may change public APIs, with migration notes in `CHANGELOG.md`. Contract changes bump
`API_VERSION` / `PROTOCOL_VERSION` and must document migration.

To release: bump the version in `pyproject.toml`, `src/gpulayer/__init__.py` and
`site/src/lib/version.ts` (`tests/unit/test_version.py` checks they agree), move the
`Unreleased` CHANGELOG entries under the new version, wait for CI on `main`, then tag and
publish a GitHub release. `.github/workflows/publish.yml` builds the package and uploads it to
PyPI through trusted publishing.
