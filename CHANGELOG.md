# Changelog

All notable changes to GPUlayer. Versions follow [semantic versioning](https://semver.org);
before 1.0, minor versions may change public APIs, with migration notes here.

## Unreleased

- `--gpus 0,1|none` on `gpulayer up`, `join` and `worker start`: a worker offers only those
  GPUs, so several workers on one machine no longer offer the same GPU (the known issue in
  0.2.0). `--gpus none` makes a CPU-only worker on a GPU machine.
- CUDA jobs get `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `CUDA_VISIBLE_DEVICES` selects the GPUs
  that were reserved (`nvidia-smi` numbering) on machines with mixed GPU models.
- Trainer: `max_steps`, `check_val_every_n_epoch`, `trainer.test()` with `Module.test_step`
  and `trainer.predict()` with `Module.predict_step` (ADR-0018).
- `ModelCheckpoint(save_top_k=N)` keeps the N best epochs (also after remote runs), and a
  `LearningRateMonitor` callback records each epoch's learning rate.
- `val_check_interval` (batches or a fraction of the epoch) and `metrics.csv` next to
  `metrics.json`, with step, mid-epoch validation and epoch rows.

## 0.2.0 (2026-10-30)

Validated on an NVIDIA GPU and across machines, faster distributed start-up and HTTP API, and
a measured comparison with PyTorch Lightning.

### Migration from 0.1.0

- Distributed jobs default to `launcher="direct"`. Upgrade the server and the workers
  together: 0.1.0 workers are not chosen for `direct` jobs, and a 0.1.0 server rejects them.
  Pass `launcher="torchrun"` (`--launcher torchrun`) to keep the old behaviour.

### Changes

- `gpulayer mcp`: a Model Context Protocol server on stdio, so AI and coding agents can list
  resources, submit commands (with a project directory and inputs), wait, read logs, cancel
  and download outputs; `--read-only` for monitoring. Tested with the official MCP Python SDK.
- Agent Skill for the CLI in `integrations/skills/gpulayer`.
- Guide: AI and coding agents.
- Trainer callbacks: `Callback` hooks, `EarlyStopping` and `ModelCheckpoint` (best checkpoint,
  also returned from remote runs); when one DDP rank asks to stop, every rank stops after the
  same epoch.
- Faster distributed jobs: `DistributedJob.launcher="direct"` is the new default. Workers
  start the ranks themselves, with the same environment torchrun sets, instead of through
  torchrun's elastic agent: a 2 x 2 job on one Mac takes 0.93 s instead of about 2.1 s.
  `launcher="torchrun"` (`--launcher torchrun`) keeps the old behaviour.
  Migration: workers and the server must both be upgraded to run `direct` jobs. v0.1.0
  workers report only `torchrun` and are not chosen for them; a v0.1.0 server rejects the
  new value, so submit with `launcher="torchrun"` there.
- Faster HTTP API: clients and workers keep their connection open, and the server sets
  `TCP_NODELAY` and runs TLS handshakes per connection. On Linux a request dropped from 43 ms
  to 0.6 ms, and 500 small jobs over two machines ran 7.1x faster (7.8 to 55.6 jobs/s).
- Validated on NVIDIA: a Tesla T4 on GCP (CUDA detection, CUDA jobs, Trainer in fp32, bf16,
  fp16 and with `torch.compile`, GPU reservation), and on two machines over a real network
  (distributed jobs across both, fail-fast, hundreds of jobs). `scripts/validate_gpu.sh`
  repeats the single-machine part on any NVIDIA machine.
- Performance and comparison: a docs page and README section with charts
  (`scripts/make_perf_charts.py`), the GPUlayer vs PyTorch Lightning harness
  (`benchmarks/lightning_comparison/`), `benchmarks/multi_machine.py`, raw results in
  `benchmarks/results/2026-10-30-gcp/`, and `benchmarks/cloud/gcp.sh` to rerun them on Google
  Cloud.
- NVIDIA CUDA independent jobs are now `hardware-validated` in the support matrix.

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
