# Status and roadmap

Repository-specific plan derived from `prompt.md`, `CLAUDE.md` and `architecture_plan.md`.
Updated 2026-10-31 (v0.5.0 plus unreleased Trainer changes).

## At a glance

| Capability | Status |
|---|---|
| Local execution (SDK, CLI, Trainer), logs, results, artifacts, cancel, timeout | tested |
| Capability-aware scheduling with per-worker reasons, gang scheduling | tested |
| `gpulayer up` / `join` / `gl.connect`, TLS with a pinned certificate | tested (several processes, one host) |
| Workers on separate physical machines | validated: 30 GCP workers in two regions (2026-10-30) |
| Distributed DDP, 1 x 2 and 2 x 2 processes, gloo | tested on one host |
| Distributed across machines, gloo | validated: up to 16 machines x 2 processes; DDP training on 8 machines (3.6x faster than one, same speed as Lightning) |
| NCCL across GPUs or machines | not validated yet (GPU quota of one) |
| Apple MPS, independent jobs | hardware-validated |
| NVIDIA CUDA execution | hardware-validated on a Tesla T4 (single GPU) |
| AMD ROCm | detection only |
| Two-level gradient sync with gloo (ADR-0020) | validated on 8 GCP machines x 4 processes (11.9% faster than Lightning under torchrun) |
| Kubernetes: Helm chart (server StatefulSet, CPU and GPU worker pools) | tested in CI on kind: jobs and a distributed job across worker pods, upgrade, server restart; GPU worker pool not validated |
| Container images | built and smoke-tested in CI; published to GitHub Container Registry with each release (slim and `-torch`; no CUDA image) |
| JAX and CuPy GPU detection (reported only when they ran on the GPU) | tested; not yet run on a GPU |
| Trainer FSDP (`strategy="fsdp"`) | experimental; tested on CPU, not validated on GPUs |
| Long-running services (ADR-0019): replicas, `$PORT`, endpoints, restarts | tested on a local worker and on Kubernetes (kind); no load balancer |
| Cloud provisioning through provider adapters | Google Cloud: validated with a CPU VM and a Spot T4 VM (create, join, run a CUDA training job, delete); other clouds: none yet; `FakeProvider` for tests |
| Documentation site (`site/`) | builds as a static export |

## Milestone 2 (in progress): PyTorch and ease of use

| Item | Status |
|---|---|
| `gpulayer.pytorch`: `Module`, `Trainer`, `Runtime`, `launch()` (ADR-0014) | done, tested |
| `@gl.remote` with `.run`, `.map`, `.options`, distributed functions | done, tested |
| `gpulayer up`, `gpulayer join`, `gl.connect`, certificate pinning | done, tested |
| Examples: regression, CNN, char-level transformer, custom loop, DDP script, sweep | done, run on Apple Silicon |
| Documentation site (Next.js, Mermaid diagrams) | done |
| Gradient accumulation (with DDP `no_sync`) | done, tested |
| Event-driven dispatch: long-poll acquire and job waits, async log shipping (ADR-0016) | done, tested, benchmarked on one machine |
| Web dashboard (jobs, logs, workers, activity charts), Prometheus `/metrics`, Grafana dashboard | done, tested; charts checked in a browser |
| Worker telemetry: load, memory, task processes, NVIDIA GPU utilisation | done, tested (NVIDIA path not hardware-validated) |
| Callbacks, `EarlyStopping`, `ModelCheckpoint` (consistent stop across DDP ranks) | done, tested |
| Callbacks, sanity validation, `fast_dev_run`, `max_time`, gradient clipping by value, `deterministic` | done, tested (unreleased) |
| Validation on NVIDIA hardware and across machines | done for one T4 and up to 16 machines with gloo; several GPUs and NCCL wait on GPU quota |

## Milestone 1 steps (prompt.md §8)

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | Contracts and runtime interfaces | done | `contracts/models.py`, API `gpulayer.dev/v1alpha1`, protocol `1` |
| 2 | Local execution via SDK and CLI | done, tested | command + function jobs, logs, results, artifacts, cancel, timeout |
| 3 | Worker registry and scheduler | done, tested | heartbeats, leases, deterministic explainable placement |
| 4 | Remote-worker path | done, tested on one host | authenticated HTTP; separate process + state dir. Gap: no second-machine test |
| 5 | Distributed foundation | done, beyond the planned scope | contracts, gang scheduling, ranks, rendezvous, fail-fast, and a real torchrun run (2 nodes × 2 procs, one host) |
| 6 | NVIDIA foundation | done (unvalidated) | detection + hardware-gated smoke test; never run on NVIDIA hardware |
| 7 | MPS discovery | done, hardware-validated | mocked CI tests + Apple Silicon smoke test |
| 8 | Provider foundation | done, simulated | `ProviderAdapter`, `FakeProvider`, full provisioning lifecycle |
| 9 | Docs and ADRs | done | `docs/`, 13 ADRs (licence and website stack still Proposed) |

Definition of done (prompt.md §14): items 1 to 10 are met, with the classifications above.

## Where the implementation differs from `architecture_plan.md` (proposed plan edits)

These need the maintainer's approval before `architecture_plan.md` is edited. It currently has
uncommitted local changes, so it was deliberately left untouched.

1. Job lifecycle: The plan has `submitted → validated → queued → preparing → running →
   finalizing → succeeded`. The implementation uses `queued → [provisioning] → scheduled →
   running → terminal`. Validation is synchronous at submit time, so `submitted`/`validated`
   are never observable. Preparation and finalization are task-level phases reported as events.
   *Proposal:* adopt the implemented states and add `provisioning` to §8.
2. `--target distributed`: `prompt.md` §9 lists it, but `CLAUDE.md` §2.2 makes distributed a
   workload type, not a target. The CLI uses `gpulayer submit --distributed --nodes N` combined
   with `--target local|remote|cloud`. *Proposal:* update prompt §9.
3. Repository layout: The plan's tree has `distributed/`, `control_plane/{jobs,workers,...}/`,
   `observability/` and `accounting/`. The code is flatter, following the "no empty modules"
   rule: `control_plane/{service,scheduler,provisioning}.py`, `runtime/distributed.py`, events
   and logs in the store, and `CostRecord` in the contracts. A top-level `gpulayer.distributed`
   package would also shadow the public `gpulayer.distributed()` builder.
4. Worker states: `registering` and `unhealthy` are defined but not produced yet. Workers
   register atomically, and health checks beyond heartbeats don't exist.
5. Support vocabulary: An `implemented` level was added (execution path exists, not
   hardware-validated) so CUDA isn't labelled `tested` on the strength of mocks alone.
6. Open questions answered provisionally (ADRs Accepted for milestone 1): HTTP+JSON with
   worker polling; bearer tokens (client / join / per-worker session / single-use instance);
   source bundle + JSON function packaging; torchrun static rendezvous; SQLite + local CAS.
   Still open: licence, first real provider, website stack, shared DB / object store, isolation
   tiers.

## Known gaps, ordered for the next iterations

### Iteration 2: harden what exists

- Done: provider API calls run outside metadata transactions (intent, call, outcome).
- Done: worker artifact reads scoped to their tasks; `admin` and `submitter` client roles.
- Done: named, expiring, revocable tokens (`gpulayer token`); rate limiting of failed logins.
- Done: provisioning retries on `CapacityUnavailable`, retried terminations, orphan sweep.
- Done: retention and garbage collection (`gpulayer gc`, admin endpoint `POST /v1/admin/gc`).
- Done: CLI flag `--gpu-memory-kind`; runtime profiles (`profile="pytorch-cuda"`,
  `--profile mlx`) as a shorthand for accelerator and framework requirements.

### Performance (ADR-0016), runs alongside the iterations

- Done: long-poll dispatch and job waits, event-driven process exit, background log shipping,
  no per-step device sync in the Trainer (1.6x faster steps on MPS), bundles skip
  GPUlayer state. Idle-job latency 626 ms to 31 ms, burst throughput 27 to 87 jobs/s (one Mac).
- Done: several tasks per worker, each with its own reserved GPUs (tested with simulated
  multi-GPU workers and real concurrent CPU processes).
- Done: per-rank log streams (every distributed log line carries its global rank).
- Done: faster distributed start-up. The default `direct` launcher starts ranks without
  torchrun: a 2 x 2 job takes 0.93 s instead of about 2.1 s (`benchmarks/distributed_startup.py`,
  one Mac, CPU). GPUlayer itself adds under 0.01 s.
- Done: two-level gradient sync for gloo (ADR-0020): a DDP step on 8 machines x 4 processes
  went from 117 to 102 ms.
- Next: topology-aware group placement. Validating FSDP and `torch.compile` needs CUDA
  hardware.

### Distribution (ADR-0017)

- Done: `install.sh` (curl one-liner, private virtualenv, uv or pip, no sudo; tested from a
  local checkout with and without uv), `Dockerfile` (CPU or CUDA base, unprivileged user,
  `/data` volume) and `deploy/docker/docker-compose.yml`. CI builds the image and runs a server,
  a worker and a job across containers.
- Done: Apache-2.0 licence (ADR-0012), public repository, PyPI releases through trusted
  publishing (`pip install gpulayer`).
- Done: images on GitHub Container Registry with each release (`<version>`, `<version>-torch`).
- Done: Helm chart (`deploy/kubernetes/helm/gpulayer`), installed on kind in CI.
- Next: a CUDA image, possibly a Homebrew formula.

### Iteration 3: validate across machines

- Done: workers on separate machines (30 GCP workers, two regions) and distributed gloo jobs on
  up to 16 machines.
- Done: `tests/hardware` on one Tesla T4; CUDA is hardware-validated for a single GPU.
- Next: several GPUs per machine and NCCL across machines (blocked on GPU quota).

### Iteration 4: first real cloud provider

- Done: Google Cloud (Compute Engine) adapter (ADR-0010), validated with a CPU VM and a Spot T4
  VM: create, join, run a CUDA training job, delete.
- Next: a second provider.

### Later

- Shared database and object storage (multi-node control plane), push-based assignment,
  container packaging, hosted control-plane design, isolation review for multi-tenant use.
