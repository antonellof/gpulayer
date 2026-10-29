# Status and roadmap

Repository-specific plan derived from `prompt.md`, `CLAUDE.md` and `architecture_plan.md`.
Updated 2026-10-29 at the end of the first iteration.

## At a glance

| Capability | Status |
|---|---|
| Local execution (SDK, CLI, Trainer), logs, results, artifacts, cancel, timeout | tested |
| Capability-aware scheduling with per-worker reasons, gang scheduling | tested |
| `gpulayer up` / `join` / `gl.connect`, TLS with a pinned certificate | tested (several processes, one host) |
| Workers on separate physical machines | not validated yet |
| Distributed DDP, 1 x 2 and 2 x 2 processes, gloo | tested on one host |
| Distributed across machines, NCCL | not validated yet |
| Apple MPS, independent jobs | hardware-validated |
| NVIDIA CUDA execution | implemented, not hardware-validated |
| AMD ROCm | detection only |
| Cloud provisioning through provider adapters | simulated (`FakeProvider`); cloud VMs can join with `gpulayer join` |
| Documentation site (`site/`) | builds as a static export |

## Milestone 2 (in progress): PyTorch and ease of use

| Item | Status |
|---|---|
| `gpulayer.pytorch`: `Module`, `Trainer`, `Runtime`, `launch()` (ADR-0014) | done, tested |
| `@gl.remote` with `.run`, `.map`, `.options`, distributed functions | done, tested |
| `gpulayer up`, `gpulayer join`, `gl.connect`, certificate pinning | done, tested |
| Examples: regression, CNN, char-level transformer, custom loop, DDP script, sweep | done, run on Apple Silicon |
| Documentation site (Next.js, Mermaid diagrams) | done |
| Callbacks, early stopping, gradient accumulation | planned |
| Validation on NVIDIA hardware and across machines | planned |

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

### Iteration 3: validate across machines

- A two-machine remote-worker test (documented environment), then multi-machine torchrun with
  `--advertise-address` and interface selection.
- Run `tests/hardware` on an NVIDIA host (single GPU, then NCCL multi-GPU), and promote CUDA
  from `implemented` to `hardware-validated` only after it passes.

### Iteration 4: first real cloud provider (needs an ADR-0010 decision)

- Choose a provider using the ADR's criteria. Implement offers, pricing source, provisioning
  with bootstrap via user-data, cleanup, and failure tests against a real test account.

### Later

- Shared database and object storage (multi-node control plane), push-based assignment,
  container packaging, hosted control-plane design, isolation review for multi-tenant use.
