# Cloud execution and provider adapters

## Interface

`providers/base.py` defines `ProviderAdapter`:

| Method | Notes |
|---|---|
| `features()` | which operations the adapter really supports |
| `list_offers()` | `InstanceOffer`s with a **`WorkerCapabilities` template** and an optional `PriceQuote` |
| `provision(offer_id, bootstrap)` | starts an instance; `WorkerBootstrap` carries the control-plane URL, an instance reference and a single-use join token |
| `get_instance(id)` / `terminate(id)` | state inspection; idempotent termination |

Errors: `ProviderError`, `CapacityUnavailable` (retryable), `QuotaExceeded`. Provider-specific
instance names are metadata only; the scheduler sees capabilities.

## Lifecycle (`control_plane/provisioning.py`)

1. A cloud job with `placement.provisioning` finds no eligible provider worker.
2. Offers are filtered with **the same `compatibility_reasons()` the scheduler uses**, plus
   `max_hourly_cost` (an offer without a reliable price is rejected when a cap is set). The
   cheapest known price wins; ties break on offer id.
3. One instance per node is requested. Each gets a fresh single-use token, bound to that
   instance record; only its SHA-256 is stored.
4. The job moves to `provisioning`; an estimated cost record is attached
   (`kind="estimate"`, currency, unit, source, timestamp).
5. The booted worker registers with origin `provider` and its token. Identity fields
   (provider, region, reservation label) come from GpuLayer's record, not the worker's claims.
6. The worker is reserved for that job. The scheduler places the job only after the worker
   **reports** compatible capabilities — provisioning success is not proof of GPU readiness.
7. On a terminal job state, `release="terminate_on_completion"` instances are terminated and
   their workers revoked. Failed terminations are logged as possible orphans.
8. Provider failure or `provisioning_timeout_seconds` fails the job and releases instances.

## Status

- `FakeProvider` (in-memory, deterministic, failure injection) — **simulated**; it provisions
  nothing. All lifecycle tests use it.
- **No real provider adapter exists.** Choosing the first one is an open decision (ADR-0010):
  it requires validating current APIs, auth, regions/instance types, lifecycle and cleanup
  behaviour, and pricing sources. GpuLayer will not ship invented prices or capacities.

## Existing vs managed workers

A private GPU box joining with a join token is an *existing* worker (`target=remote`). A machine
created by an adapter is a *managed* worker (`target=cloud`). GpuLayer orchestrates both; it
provisions only the latter.
