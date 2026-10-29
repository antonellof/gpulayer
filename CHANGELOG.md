# Changelog

All notable changes to GPUlayer. Versions follow [semantic versioning](https://semver.org);
before 1.0, minor versions may change public APIs, with migration notes here.

## Unreleased

- `gpulayer mcp`: a Model Context Protocol server on stdio, so AI and coding agents can list
  resources, submit commands (with a project directory and inputs), wait, read logs, cancel
  and download outputs; `--read-only` for monitoring. Tested with the official MCP Python SDK.
- Agent Skill for the CLI in `integrations/skills/gpulayer`.
- Guide: AI and coding agents.

## 0.1.0 (2026-10-29)

The first release. GPUlayer runs commands, Python functions and distributed jobs on the
machines you can reach, from a laptop to servers that joined a control plane.

### Run work

- Job types: commands in any language, Python functions (`@gl.remote`, `.map()`), and
  distributed jobs launched with torchrun (static rendezvous, gang scheduling, fail-fast).
- PyTorch integration: `Module`, `Trainer` (in-process, multi-process DDP, multi-node,
  remote), `Runtime` and `launch()`; gradient accumulation with DDP `no_sync`.
- Runtime profiles (`pytorch-cuda`, `pytorch-mps`, `jax-cuda`, `cupy-cuda`, `mlx`, ...).

### Place and run it well

- Capability-aware scheduler with per-worker reasons when a job cannot run.
- Several tasks per worker, each with its own reserved GPUs (default one slot per NVIDIA or
  AMD GPU); `spread` placement keeps machines evenly busy.
- Event-driven dispatch: long-poll worker acquire and job waits, background log shipping.
  One job on an idle worker: 626 ms to 31 ms; burst throughput 27 to 87 jobs/s (one Mac).
- Per-rank log streams for distributed jobs.

### Servers and workers

- `gpulayer up` (control plane with TLS and a pinned self-signed certificate) and
  `gpulayer join`; named, expiring, revocable tokens with roles (admin, submitter, join,
  metrics); rate-limited logins; worker artifact scoping; retention and `gpulayer gc`.

### Observe

- Web dashboard: jobs, live logs, placement reasons, workers, and activity charts (jobs,
  CPU load, memory, task processes, NVIDIA GPU utilisation).
- Prometheus `/metrics`, a Grafana dashboard and a Prometheus/Grafana Compose setup.

### Install

- `install.sh` (curl one-liner, private virtualenv, no sudo), a Docker image for server and
  workers (CPU or CUDA base) with Compose, and CI that tests all of it.

### Validation status

- Tested on CPU and Apple Silicon (PyTorch MPS and MLX), with several workers on one host,
  and in Linux CI including a Docker server/worker test.
- NVIDIA CUDA execution and multi-machine runs are implemented but not validated on hardware.
- Cloud provisioning is simulated (`FakeProvider`); cloud VMs can join with `gpulayer join`.
- Not a sandbox for untrusted code. See docs/project/status.md.
