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

Measured on Google Cloud, a 4-machine gloo training job split across two regions ran 16 to
24% slower than in one region, while two zones of one region were as fast as one zone
([Performance and comparison](../get-started/performance-and-comparison.md#one-distributed-job-in-one-region)).

```bash
gpulayer join https://server:8470 --token ... --zone dc1-row3 --network-group rack-7
gpulayer submit --distributed --nodes 8 --group-locality zone -- ...
```

From Python: `launch(..., group_locality="zone")`, `Trainer(..., launch_options={"group_locality": "zone"})`,
or `@gl.remote(..., group_locality="zone")`.

## GPUs inside a machine

On machines with several NVIDIA GPUs, workers read the interconnect between them from
`nvidia-smi topo -m`: NVLink (`NV<n>`, n bonded links), one PCIe switch (`PIX`), several
switches (`PXB`), the CPU's host bridge (`PHB`), one NUMA node (`NODE`) or across CPU sockets
(`SYS`). A task that asks for several GPUs gets the free ones whose slowest link is fastest,
so a 4-GPU job on an 8-GPU NVLink machine takes one NVLink island, and a 2-GPU job on a
two-socket PCIe box takes two GPUs behind one switch. Single-GPU tasks, and machines where the
topology is unknown, get the lowest-numbered free GPUs as before. This is tested with recorded
`nvidia-smi topo -m` output; it has not run on a multi-GPU machine yet.

## What is not scheduled on (yet)

- `cpu_cores` and `memory_gb` only constrain placement; nothing enforces them at runtime.
- Each worker runs up to `max_concurrent_tasks` tasks at once (default: one per usable
  NVIDIA or AMD GPU, at least 1; `--concurrency N` overrides it). Each task is given its own
  GPUs, which the worker exposes through `CUDA_VISIBLE_DEVICES`. CPU jobs take a slot but no
  GPU. Apple Silicon Macs have one GPU, so they run one task at a time by default.
- Single-GPU tasks are not placed to keep NVLink islands free for later multi-GPU tasks.
- Distributed groups do not yet consider GPU interconnect inside each machine when choosing
  machines.
- Data locality and cost optimization are planned once reliable capability and pricing data
  exist.

Cost data is described in [Cost accounting](cost-accounting.md).
