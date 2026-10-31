# Changelog

All notable changes to GPUlayer. Versions follow [semantic versioning](https://semver.org);
before 1.0, minor versions may change public APIs, with migration notes here.

## Unreleased

- Topology-aware placement of distributed groups (ADR-0021). Workers report a `zone` (new
  `WorkerCapabilities` field) and an optional `gpulayer.dev/network-group` label: read from
  the GCE metadata server on Google Cloud, or set with `--zone`, `--region`
  (`$GPULAYER_ZONE`, `$GPULAYER_REGION`) and `--network-group` on `up`, `join` and
  `worker start`. New `PlacementPolicy.group_locality` (`--group-locality`): `prefer`
  (default) keeps a group in one network group, else zone, else region; `network_group`,
  `zone` or `region` require it; `any` ignores location. Workers without a location place as
  before. On a simulated fleet of 5 zones in 2 regions an 8-node group now stays in one zone
  instead of spanning both regions; `schedule()` costs 1 to 15% more
  (`benchmarks/placement.py`). Training speed-up not measured yet.
- Multi-GPU tasks get the best-connected free GPUs on a machine: workers read
  `nvidia-smi topo -m` into `AcceleratorDevice.links` and `numa_node`, and a task asking for
  several GPUs gets the set whose slowest link is fastest (an NVLink island, or GPUs behind one
  PCIe switch) instead of the lowest indices. Tested with recorded topology output; not yet run
  on a multi-GPU machine.
- Trainer on CUDA: `prefetch=True` (default) pins the next batch and copies it on a side stream
  while the current step runs; `benchmark=None` (default) turns on cuDNN autotuning unless
  `deterministic=True`. `Runtime.batches(loader)` exposes the prefetch for custom loops.
  On a Tesla T4: a compute-bound CNN step went from 12.67 to 10.93 ms (cuDNN autotuning about
  9%, prefetch about 5%); small MLPs unchanged within run-to-run variation. Against Lightning
  the Trainer is now 1.14x faster on that CNN (1.25 to 1.39x on the MLPs).
- Migration: the GCP adapter now reports `region="europe-west1"` and `zone="europe-west1-b"`
  (it used to put the zone in `region`). `placement.regions` matches either, so existing
  policies keep working.
- Trainer: `num_sanity_val_steps` (default 2, as in Lightning: a broken `validation_step`
  fails before training), `fast_dev_run`, `max_time`, `gradient_clip_algorithm="value"` and
  `deterministic`. Per-step time unchanged (about 102 us/step for a small MLP on one CPU
  thread, before and after).
- Examples run on an Apple M2 Pro: the Blender example now fails the job when the requested GPU
  device is missing (Blender exited 0 on a script error), the ffmpeg example uses VideoToolbox
  on a Mac and works with macOS `date`, and the inference service also uses MPS.

## 0.5.0 (2026-10-30)

- Two-level gradient sync (ADR-0020): with gloo and several processes per machine, the
  Trainer sums gradients inside each machine before all-reducing across machines
  (`Trainer(gradient_sync="auto" | "flat" | "two_level")`). On 8 GCP machines x 4 processes a
  DDP step went from 117 to 102 ms, and the strong-scaling job ran 11.9% faster than
  Lightning (50.5 s vs 57.3 s). NCCL jobs are unchanged.
- Workers report `jax.cuda` and `cupy.cuda` only when JAX or CuPy actually ran on the GPU (a
  probe in its own interpreter, run only when they are installed), as `torch.cuda` already
  was. Before, an installed CPU-only `jax` was enough for a `--accelerator cuda --framework
  jax` job to be placed there.
- Google Cloud GPU provisioning validated: a `--provider gcp --accelerator cuda` job made
  GPUlayer create a Spot T4 VM, which joined with its GPU and `torch.cuda` 78 s after boot
  started, ran a `Trainer` job on CUDA, and was deleted when it ended.
- Examples for remote NVIDIA GPUs: Blender Cycles rendering, ffmpeg NVENC transcoding, JAX and
  CuPy, and a PyTorch inference service. Not yet run on a GPU.

## 0.4.0 (2026-10-30)

- Long-running services (ADR-0019): `gpulayer submit --service`, `gl.service(...)` and
  `Job.endpoints()`. A service runs `replicas` copies on distinct workers, gives each a free
  `$PORT`, reports `host:port` once it accepts connections, restarts replicas that exit (back-off
  1 s to 30 s, `--restart`, `--max-restarts`), is placed again after a worker loss, and stops on
  cancel. New worker route `POST /v1/worker/tasks/{id}/endpoint` and `TaskStatus.endpoint`;
  the `tasks.endpoint` column is added to existing databases on start. No load balancer yet.
- Less server work per job: active-task queries use a partial index instead of scanning every
  task ever run (2.2 ms to 0.002 ms per scheduling pass at 13,000 tasks), the HTTP client
  shares kept-open connections between threads (a worker no longer opens a TLS connection per
  job), and worker capabilities are parsed once. Server CPU per job in
  `benchmarks/server_throughput.py`: 6.7-6.8 ms to 5.2-6.0 ms.
- `POST /v1/jobs/batch` and `Client.submit_batch`: many jobs in one request, one transaction
  and one scheduling pass (all or none); server CPU per job 5.6-6.2 to 4.1 ms.
- Site: sitemap.xml, robots.txt, canonical URLs, link previews and structured data.
- `Trainer(strategy="fsdp")` (experimental): FSDP2 sharding of parameters, gradients and
  optimizer state, full-state checkpoints; on CPU it trains exactly like DDP. Not yet run on
  CUDA GPUs.
- Multi-node training against PyTorch Lightning, both trainers on the same machines: the same
  speed within 2% on 1 to 8 machines (40.5 s vs 41.2 s on 8). The earlier comparison, which
  ran the trainers on different VMs, overstated GPUlayer's lead.
- On 8 GCP workers after the server changes, 2,000 tiny jobs ran at 83 jobs/s, or 93 with
  `submit_batch`, which submits all of them in 1.1 s. The workers now set the limit.
- Google Cloud provider adapter (`gpulayer[gcp]`, `gpulayer up --providers FILE`): when no worker
  fits a `--provider gcp` job, GPUlayer creates a Compute Engine VM from a configured offer, the
  VM installs GPUlayer and joins with a single-use token, the job runs, and the VM is deleted.
  Validated end to end on GCP with a CPU VM. `gpulayer join --instance-ref` for provisioned
  machines.
- Kubernetes: a Helm chart (`deploy/kubernetes/helm/gpulayer`) with the server as a
  StatefulSet and CPU and GPU worker pools; tokens and TLS generated into Secrets and kept on
  upgrade. Tested in CI on kind, including a distributed job across worker pods. Guide:
  Kubernetes.
- Releases push container images to `ghcr.io/antonellof/gpulayer` (`<version>` and
  `<version>-torch`).
- `gpulayer up` takes its tokens from `GPULAYER_UP_CLIENT_TOKEN` / `GPULAYER_UP_JOIN_TOKEN`;
  `--ca-file` defaults to `$GPULAYER_CA_FILE`; `gpulayer join` waits for a server that is not
  reachable yet instead of exiting.
- `fit(ckpt_path=...)` resumes launched (remote) runs: the local checkpoint is uploaded with
  the job. `launch(inputs={...})` ships local files to `$GPULAYER_INPUT_DIR`.

## 0.3.0 (2026-10-30)

First release on PyPI (`pip install gpulayer`), under the Apache-2.0 licence. Tested on 30
machines in two GCP regions, with a multi-machine training comparison against PyTorch
Lightning.

### Scale and speed

- Scheduling no longer slows down as the queue grows: a submit costs 0.5 ms at 2,000 queued
  jobs instead of 44 ms (`benchmarks/submit_queue.py`).
- Behind `gpulayer up`, idle workers wait until something changes instead of re-checking
  every 0.2 s, and a change wakes only the workers or job waiters it concerns. On 30 workers
  a 480-run training sweep finished in 45 s instead of 102 s.
- SQLite runs with `synchronous=NORMAL` (WAL): commits no longer wait for the disk (2.25 ms
  to 0.02 ms on a GCP standard disk). Consistent after a crash; a power loss can roll back
  the last commits.
- The Trainer does less work per step (0.60 to 0.585 ms on a small CPU model; a plain
  PyTorch loop is 0.55).
- Validated on 30 workers in two regions: distributed jobs on up to 16 machines (32
  processes), fail-fast across 8 machines in 4.9 s, retries after killing workers mid-run.
  One training job on 8 machines ran 3.4x faster than on one, and 7% sooner than PyTorch
  Lightning under torchrun (benchmarks/results/2026-10-30-gcp-fleet).
- Known limit: bursts of very short jobs reach about 25 jobs/s, bounded by the server using
  one CPU core (ADR-0016, item 6).

### Features and fixes

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
- A launched training run returns its metrics even with checkpointing off (`metrics.json` is
  written every epoch).
- Apache-2.0 licence (ADR-0012), PyPI package metadata, and a trusted-publishing release
  workflow (`.github/workflows/publish.yml`).
- Benchmarks: `benchmarks/cloud/gcp.sh` builds an image and starts a server plus a fleet of
  workers; `benchmarks/scale_test.py`; `benchmarks/lightning_comparison/scaling_run.py` and
  `multinode.py` compare multi-machine training with PyTorch Lightning.

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
