# Implementation plan and status

Repository-specific plan derived from `prompt.md`, `CLAUDE.md` and `architecture_plan.md`.
Updated 2026-10-29 at the end of the first iteration.

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

- Move provider API calls out of the metadata transaction (two-phase: record intent → call
  provider → record outcome). Today a slow provider blocks the control plane.
- Scope artifact reads to the requesting worker's tasks; add client roles (submitter vs admin).
- Make join tokens expiring and rotatable; add basic rate limiting.
- Retries and an orphan sweep for provisioning (`CapacityUnavailable`, failed terminate).
- Retention and garbage collection for artifacts, logs and events.
- A CLI flag for `gpu_memory_kind`; `runtime profiles` as an optional compatibility aid.

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
