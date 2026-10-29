# Scheduling and cost accounting

## Scheduler (`control_plane/scheduler.py`)

A pure function `schedule(spec, workers, job_id) -> SchedulingDecision`. No I/O, no provider
calls, no mutation — so it is deterministic and unit-testable.

1. For each allowed target in order (`placement.target`, then explicit `fallback_targets`):
2. For each worker (sorted by id): compute `compatibility_reasons()` — target/origin, provider,
   region, labels, reservation, OS, arch, CPU, RAM, disk, Python, frameworks, support matrix,
   accelerator devices/memory/compute capability, and for distributed jobs launcher, backend,
   torch and `advertise_address`.
3. Compatible workers are then checked for **availability** (state and free task slots).
   *Capability* and *availability* are separate: "compatible but busy" queues; "incompatible"
   can fail fast with `on_unschedulable="fail"`.
4. Rank eligible workers by policy — `first_fit` (worker id) or `most_free_memory` — and pick
   one worker, or for distributed jobs a whole group sharing one torch version.

Every candidate's reasons are persisted on the job (`gpulayer job status` prints them).

## What is not scheduled on (yet)

- `cpu_cores` / `memory_gb` are placement constraints, **not enforced limits** at runtime.
- Each worker runs `max_concurrent_tasks` tasks (default 1); devices are not partitioned
  between concurrent tasks.
- Data locality, topology/interconnect and cost optimization are planned once reliable
  capability and pricing data exist.

## Cost accounting

`CostRecord` keeps estimates and charges distinct:

| `kind` | Meaning |
|---|---|
| `estimate` | from a provider `PriceQuote` at provisioning time |
| `provider_usage` / `provider_charge` | reserved for real adapters that report usage/billing |
| `internal` | GpuLayer-calculated accounting |

Every record has `amount`, `currency`, `unit` (`per_hour` or `total`), `source` and
`recorded_at`. Unknown prices stay unknown: with `max_hourly_cost` set, an unpriced offer is
rejected. Today only `FakeProvider` produces estimates, labelled as simulated.
