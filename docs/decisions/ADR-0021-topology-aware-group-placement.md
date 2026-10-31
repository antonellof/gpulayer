# ADR-0021: Topology-aware placement of distributed groups

- **Status:** Accepted (network locality between machines; GPU interconnect inside a machine)
- **Date:** 2026-10-31

## Context

A distributed job's collectives run at the speed of the slowest link in the group. Until now
the scheduler chose a group by load and worker id (ADR-0004), with one network rule: every
node has the same torch version. On a fleet spread over zones or regions, such as the 30
GCP workers in two regions of the 2026-10-30 runs, an 8-machine job could take workers from
both regions, so every all-reduce crossed a long link. Workers did not report a zone, and
the GCP adapter stored its zone in `region`.

## Decision

1. Workers report where they are:
   - `WorkerCapabilities.zone` (new, optional) next to `region`.
   - An optional label `gpulayer.dev/network-group` for anything finer than a zone that only
     the operator knows: a rack, a LAN, a switch, one office.
   - Discovery fills `region` and `zone` from the GCE metadata server, asked only when the
     machine identifies itself as Google Compute Engine (`/sys/class/dmi/id/product_name`),
     with a 1 s timeout. Elsewhere: `--region`, `--zone` (or `$GPULAYER_REGION`,
     `$GPULAYER_ZONE`) and `--network-group` on `gpulayer up`, `join` and `worker start`.
   - Provider-created workers get both from GPUlayer's record of the offer, not from the
     worker's claims. The GCP adapter now reports the region (`europe-west1`) as `region` and
     the zone (`europe-west1-b`) as `zone`.
2. `PlacementPolicy.group_locality` decides how close a distributed group must be:
   - `prefer` (default): the tightest group available, trying one network group, then one
     zone, then one region, then any. A job is never delayed to wait for a tighter group.
   - `network_group`, `zone`, `region`: required. Workers with no value at that level never
     qualify. When no level has enough eligible workers, the job waits; when not enough
     compatible workers share one at all, `on_unschedulable="fail"` fails it.
   - `any`: location is ignored (the previous behaviour).
3. Within a level, candidate groups are buckets of (torch version, location). Workers are
   taken in the policy's rank order (`spread`, `first_fit`, `most_free_memory`), and the first
   bucket to fill wins, so load balancing still decides between equally close groups.
4. The decision summary names the locality: `placed on worker group [...] (same zone
   europe-west1-b, ...)`, or why a required one could not be met.
5. `placement.regions` matches a worker's region or zone, so policies written with GCP zone
   names keep working.

Code: `control_plane/scheduler.py` (`locality`, `_pick_group`, `ever_schedulable`),
`gpu/discovery.py` (`detect_location`), `providers/gcp.py`, `cli/main.py`.

## Alternatives considered

- **Measuring latency between workers** and placing by it: more accurate, but needs probes
  between every pair of workers and a way to keep them fresh. Zones and operator labels are
  free and cover the cases seen so far; measurement can feed `network-group` later.
- **A scoring optimizer** over all groups: the number of groups grows combinatorially, and
  with levels tried tightest first the greedy bucket fill already finds a group in one
  domain whenever one exists.
- **Requiring one zone by default:** would make jobs wait or fail on fleets that never
  report a zone. `prefer` keeps every placement that worked before.

## Consequences and trade-offs

- Workers that report no location are placed exactly as before.
- `prefer` can choose a less-loaded group in another zone over a tighter one only when the
  tighter one does not have enough free workers right now; it does not wait.
- Cost per `schedule()` call (`benchmarks/placement.py`, Apple M2 Pro): 55 to 60 us before
  and 60 to 64 us after with 30 workers, 210 to 219 us with 120, and 925 to 939 us with 500;
  the per-worker compatibility checks dominate. On the benchmark's fleet (5 zones in 2
  regions, half the workers busy) an 8-node group spanned 5 zones in 2 regions before, and 1
  zone (3 zones in 1 region with only 15 free workers) after. Raw output:
  `benchmarks/results/2026-10-31-placement/`.
- Measured on GCP (`benchmarks/lightning_comparison/locality_run.py`, 4 machines x 4
  processes, gloo, all Skylake, median of five): a job split across `europe-west1` and
  `europe-west4` took 56.2 s, against 45.0 to 48.6 s in one region. Two zones of one region
  were as fast as one zone. So the region level carries the measured benefit; preferring one
  zone inside a region costs nothing and is kept for providers where zones are farther apart.
  Logs: `benchmarks/results/2026-10-31-gcp-locality/`.

## Addendum: GPUs inside a machine

- Discovery parses `nvidia-smi topo -m` on machines with more than one NVIDIA GPU into
  `AcceleratorDevice.links` (peer index to `NV<n>`, `PIX`, `PXB`, `PHB`, `NODE` or `SYS`) and
  `numa_node`. Indices match `CUDA_VISIBLE_DEVICES`, which the worker sets with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- `pick_devices` gives a task that asks for several GPUs the free set whose slowest link is
  fastest, then the most total link speed, then the lowest indices (greedy from every
  starting GPU). Without link data, or for one GPU, it keeps the lowest free indices.
- `schedule()` only checks that enough GPUs are free (`has_free_devices`); the choice runs
  once, when a task is assigned: 44 to 66 us on an 8-GPU NVLink layout, against about 3 us
  for the lowest indices. On that layout a 4-GPU task with GPU 0 busy now gets the other
  NVLink island (4 to 7) instead of 1 to 4, which crosses the CPU sockets.
- Tested with recorded matrices (NVLink cube mesh, two-socket PCIe with NIC columns). Not run
  on a multi-GPU machine: the GCP project's GPU quota is one.

## Security and operational implications

- Zones and network groups appear in placement explanations, like hostnames already do.
- A worker that joined with a join token can claim any zone or network group and so attract
  or avoid distributed jobs. This is the same trust as labels today. Provider-created
  workers cannot: their location comes from GPUlayer's record.

## Follow-up

- Measure one training job in one zone against the same job across regions.
- Validate the GPU choice on a multi-GPU machine and measure NCCL all-reduce on the chosen
  set against the lowest indices.
- Keep NVLink islands free: place single-GPU tasks on GPUs outside them when possible.
- AWS and Azure metadata, once those providers exist.
