# ADR-0011: Cost accounting and retry semantics

- **Status:** Accepted (milestone 1)
- **Date:** 2026-10-29

## Context

Users need to know what a job might cost and what happens when things fail. Both areas
invite overclaiming: estimates presented as bills, and "exactly once" promises that
distributed systems cannot keep.

## Decision: cost

- `CostRecord` (`contracts/models.py`) has an explicit `kind`: `estimate`,
  `provider_usage`, `provider_charge` or `internal`. It also carries `amount`, `currency`,
  `unit` (`per_hour` or `total`), `source`, `recorded_at` and `note`.
- Today only `estimate` records are produced, when an offer carries a `PriceQuote`
  (`control_plane/provisioning.py`). The `FakeProvider` source string says
  "simulated". Records attach to `JobResult.costs`.
- An unknown price stays unknown. With `max_hourly_cost` set, unpriced offers are
  rejected rather than assumed cheap. GpuLayer contains no hardcoded prices.

## Decision: delivery, leases and retries

- Delivery is at-least-once at best, never exactly-once. A task whose lease expires (no
  heartbeat for `lease_ttl_seconds`, default 20 s) becomes `lost`. The worker may still
  be running it. If the worker reconnects, its heartbeat tells it to kill lost tasks, and
  its late outcome report is recorded as `task.late_report` and ignored.
- **Retries are opt-in:** `RetryPolicy(max_attempts=1, retry_on=["worker_lost"])`. With
  `max_attempts>1`, an independent job is requeued after `worker_lost` (and after
  `failed` if listed), as a new task with `attempt+1`. The requeue event states that side
  effects may repeat.
- Distributed jobs are never retried automatically (validation rejects it). One node failure
  fails the group (ADR-0007).
- **Timeouts:** the worker enforces `timeout_seconds` by killing the process group. The
  control plane adds a backstop after `timeout_grace_seconds` for workers that stop
  reporting.
- Cancellation is idempotent. Unstarted tasks are cancelled immediately; running
  ones are cancelled through heartbeat delivery.

## Alternatives considered

- **Automatic retry by default:** surprising duplicate side effects. Rejected.
- **Fencing tokens or epochs passed to workloads:** a useful future addition for
  idempotent sinks.

## Consequences and trade-offs

- Users must make side-effecting jobs idempotent before enabling retries.
- Actual spend is not tracked. Provider usage and charge ingestion are future work.

## Security and operational implications

- Cost records do not contain credentials. Estimates must be labelled as estimates in
  every UI.

## Follow-up

- Ingest provider usage and charges where the provider APIs offer them.
- Expose `GPULAYER_ATTEMPT` (already set) in docs as an idempotency key, and consider
  fencing.
