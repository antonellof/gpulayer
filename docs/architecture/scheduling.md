# Scheduling and cost accounting

## Scheduler (`control_plane/scheduler.py`)

The scheduler is a pure function, `schedule(spec, workers, job_id) -> SchedulingDecision`. It
does no I/O, makes no provider calls and changes no state, which keeps it deterministic and easy
to unit-test.

1. For each allowed target in order (`placement.target`, then explicit `fallback_targets`):
2. For each worker, sorted by id, it computes `compatibility_reasons()`. The checks cover
   target and origin, provider, region, labels, reservation, OS, CPU architecture, CPU, RAM,
   disk, Python, frameworks, the support matrix, and accelerator devices, memory and compute
   capability. Distributed jobs also check the launcher, backend, torch and `advertise_address`.
3. Compatible workers are then checked for availability (state and free task slots). A job whose
   compatible workers are all busy waits in the queue. A job with no compatible worker can fail
   immediately if `on_unschedulable="fail"`.
4. Eligible workers are ranked by policy, either `first_fit` (by worker id) or
   `most_free_memory`. The scheduler picks one worker, or for a distributed job a whole group
   that shares one torch version.

The reasons for every candidate are stored on the job, and `gpulayer job status` prints them.

## What is not scheduled on (yet)

- `cpu_cores` and `memory_gb` only constrain placement; nothing enforces them at runtime.
- Each worker runs `max_concurrent_tasks` tasks (default 1); devices are not partitioned
  between concurrent tasks.
- Data locality, topology/interconnect and cost optimization are planned once reliable
  capability and pricing data exist.

## Cost accounting

`CostRecord` keeps estimates and charges apart:

| `kind` | Meaning |
|---|---|
| `estimate` | from a provider `PriceQuote` at provisioning time |
| `provider_usage` / `provider_charge` | reserved for real adapters that report usage/billing |
| `internal` | GpuLayer-calculated accounting |

Every record has `amount`, `currency`, `unit` (`per_hour` or `total`), `source` and
`recorded_at`. A missing price is recorded as unknown, and when `max_hourly_cost` is set, an
offer without a price is rejected. Today only `FakeProvider` produces estimates, and they are
labelled as simulated.
