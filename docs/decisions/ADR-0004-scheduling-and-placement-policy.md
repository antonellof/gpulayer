# ADR-0004: Scheduling and placement policy

- **Status:** Accepted (milestone 1)
- **Date:** 2026-10-29

## Context

Placement must use declared requirements and reported capabilities, must reject rather
than weaken an incompatible requirement, and must be explainable and deterministic so it
can be unit-tested. Distributed jobs need a whole compatible group, never a partial one.

## Decision

- `schedule(spec, workers, job_id)` in `src/gpulayer/control_plane/scheduler.py` is a
  **pure function** over `WorkerView` (capabilities plus current state and load). It
  returns a `SchedulingDecision` that lists every candidate with its rejection reasons.
  The decision is persisted on the job and shown by `gpulayer job status`.
- `compatibility_reasons` checks the target (worker origin → local/remote/cloud), the
  provider, region and labels, the reservation label, the OS and CPU arch, CPU, RAM and
  disk, the Python and framework PEP 440 specifiers, the support matrix (ADR-0005),
  device count, memory (including memory kind), CUDA compute capability, and the
  distributed prerequisites (launcher, comm backend, torch, `advertise_address`, procs ≤
  cores). An **unknown** value never satisfies a requirement; for example, unknown RAM
  fails `memory_gb`.
- **Capability vs availability:** incompatible workers are rejected with reasons.
  Compatible workers that are busy, offline, draining or unhealthy are noted separately.
  `ever_schedulable` distinguishes "wait" from "will never fit", which drives
  `on_unschedulable: queue | fail`.
- **Policies:** `first_fit` (ordered by worker id) and `most_free_memory`, with
  worker id as the final tiebreak. Pinning goes through `placement.worker_ids` and
  `worker_labels`.
- **Targets:** `target` plus an explicit ordered `fallback_targets` list. A local job
  never runs remotely unless a fallback is configured.
- **Gang scheduling:** a `DistributedJob` needs `num_workers` distinct eligible workers,
  all reporting the same torch version. Otherwise nothing is placed. Node ranks follow
  the selected order (`runtime/distributed.py: plan_group`).
- **Reservations:** workers labelled `gpulayer.dev/reserved-for=<job>`, which the
  provisioning flow sets, only serve that job.
- Jobs are considered in FIFO order each pass. Within one pass, placed tasks reduce
  free slots (`max_concurrent_tasks`, default 1).

## Alternatives considered

- **Scoring and bin-packing optimizers:** premature without reliable cost and topology
  data.
- **Per-rank independent scheduling for distributed jobs:** can deadlock or partially
  allocate. Rejected.
- **Provider instance names in the scheduler:** this would leak vendors into the core.
  Offers are expressed as capability templates instead (ADR-0010).

## Consequences and trade-offs

- Placement is reproducible and easy to explain; tests live in
  `tests/unit/test_scheduler.py`.
- FIFO has head-of-line behavior only per job, not per queue. There are no priorities,
  quotas or fairness, and no preemption.
- Devices are not individually reserved: a worker with `max_concurrent_tasks>1` and GPUs
  could oversubscribe them. The default of 1 avoids this.
- There is no topology or interconnect awareness, and data locality is limited to
  labels.

## Security and operational implications

- Rejection reasons may reveal worker attributes (hostnames, versions) to clients. This
  is acceptable for single-team deployments; review it for multi-tenant use.

## Follow-up

- Per-device allocation, priorities and quotas, and topology-aware groups (same rack,
  interconnect).
- Cost-aware ordering once reliable pricing exists (ADR-0011).
