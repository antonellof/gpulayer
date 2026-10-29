# ADR-0010: Provider adapters and provisioning

- **Status:** Accepted (milestone 1) for the adapter interface and provisioning flow, **tested only against `FakeProvider` (simulated)**. The choice of a first real provider is Proposed.
- **Date:** 2026-10-29

## Context

Cloud execution must be a first-class target without coupling the core to one vendor or
inventing provider capabilities, prices or APIs. Provisioning an instance does not prove
that its GPU runtime works.

## Decision

- `ProviderAdapter` (`src/gpulayer/providers/base.py`) declares `features()` and
  implements `list_offers`, `provision(offer_id, bootstrap)`, `get_instance` and an
  idempotent `terminate`. Errors are `ProviderError`, `CapacityUnavailable` (retryable)
  and `QuotaExceeded`. `simulated=True` marks fakes in events.
- An **offer** is a `WorkerCapabilities` template plus an optional `PriceQuote` with
  currency, unit, source and timestamp. The provider-specific `instance_type` is metadata
  only, and the scheduler never sees it.
- `ProvisioningManager` (`control_plane/provisioning.py`) handles jobs that have
  `placement.provisioning` and no eligible worker:
  1. It filters offers with the same `compatibility_reasons` used for workers, plus
     `max_hourly_cost`. An offer without a price is rejected when a cap is set.
  2. It picks the cheapest known price deterministically and records a `CostRecord(kind=
     "estimate")`.
  3. It provisions `num_nodes` instances. Each gets a single-use instance token bound
     to an internal `inst-…` reference and a `reserved-for=<job>` label, passed in
     `WorkerBootstrap`.
  4. The job waits in `provisioning`. The booted worker registers with that token, and
     identity fields (provider, region, reservation label) are taken from the control
     plane's record.
  5. The job is placed only when the worker's *reported* capabilities are compatible.
  6. Provisioning failure or timeout fails the job and terminates the instances. When
     the job reaches a terminal state, instances with
     `release=terminate_on_completion` are terminated and their workers revoked.
- `FakeProvider` (`providers/fake.py`) is deterministic and in-memory, with
  capacity limits, failure injection and an `on_provision` hook that plays the
  machine's boot script in tests.

## Candidate evaluation criteria for the first real provider (no choice made)

- A documented, stable API for create/describe/terminate, with idempotency tokens.
- Bootstrap mechanism (user-data or cloud-init) and secret delivery that keeps the
  instance token out of logs and metadata.
- GPU types available, with a reliable machine-readable pricing source.
- Scoped credentials (least privilege), and tag-based discovery of orphaned instances.
- Egress model and reachability back to a self-hosted control plane.
- Availability of a low-cost test environment for CI validation.

## Alternatives considered

- **Provider SDKs inside the scheduler:** rejected, because they leak vendor concepts.
- **Pre-provisioned pools only:** supported already as `remote` workers, but not
  sufficient for elastic cloud use.

## Consequences and trade-offs

- **No real cloud provisioning exists.** All cloud-path tests are simulated.
- **Known gaps:** provider calls run while the DB transaction lock is held (ADR-0001).
  There is no orphan reconciliation (scanning the provider for tagged instances we lost
  track of), and no retry on `CapacityUnavailable`. Termination failure is logged as an
  event ("may be orphaned") but not retried.

## Security and operational implications

- Instance tokens are single-use, hashed at rest, and unusable after registration or
  for another instance. Provider credentials belong to the adapter configuration, never
  to job specs or logs.

## Follow-up

- An ADR selecting the first real provider using the criteria above, then an adapter
  with integration tests against a real account.
- Move provider I/O out of transactions, orphan sweeps and retry with backoff.
