# Scheduling

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
   that shares one torch version, kept as close together as the workers allow (below).

The reasons for every candidate are stored on the job, and `gpulayer job status` prints them.

## Distributed groups and network locality

A distributed job runs its collectives at the speed of its slowest link, so the scheduler
keeps the group close together ([ADR-0021](../decisions/ADR-0021-topology-aware-group-placement.md)).
Workers report:

| Level | Where it comes from |
|---|---|
| network group | the label `gpulayer.dev/network-group` (`--network-group rack-1`): a rack, LAN or switch you define |
| zone | GCE metadata on Google Cloud, the provider's record for workers GPUlayer created, or `--zone` / `$GPULAYER_ZONE` |
| region | the same sources, or `--region` / `$GPULAYER_REGION` |

`placement.group_locality` chooses how strict to be:

| Value | Behaviour |
|---|---|
| `prefer` (default) | the tightest group available now: one network group, else one zone, else one region, else any |
| `network_group`, `zone`, `region` | required; the job waits (or fails with `on_unschedulable="fail"` when it never could) until enough eligible workers share one |
| `any` | location ignored |

Within a level the policy's ranking still applies: workers are taken in rank order and the
first location to have enough of them wins. The choice is shown in `gpulayer job status`, for
example `placed on worker group [...] (same zone europe-west1-b, ...)`. Workers that report
no location are placed as before.

```bash
gpulayer join https://server:8470 --token ... --zone dc1-row3 --network-group rack-7
gpulayer submit --distributed --nodes 8 --group-locality zone -- ...
```

From Python: `launch(..., group_locality="zone")`, `Trainer(..., launch_options={"group_locality": "zone"})`,
or `@gl.remote(..., group_locality="zone")`.

## What is not scheduled on (yet)

- `cpu_cores` and `memory_gb` only constrain placement; nothing enforces them at runtime.
- Each worker runs up to `max_concurrent_tasks` tasks at once (default: one per usable
  NVIDIA or AMD GPU, at least 1; `--concurrency N` overrides it). Each task is given its own
  GPUs, which the worker exposes through `CUDA_VISIBLE_DEVICES`. CPU jobs take a slot but no
  GPU. Apple Silicon Macs have one GPU, so they run one task at a time by default.
- GPU interconnect inside a machine (NVLink, PCIe switches): a multi-GPU task gets the
  lowest-numbered free GPUs that meet its requirements.
- Data locality and cost optimization are planned once reliable capability and pricing data
  exist.

Cost data is described in [Cost accounting](cost-accounting.md).
