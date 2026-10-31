# ADR-0021: Topology-aware placement of distributed groups

- **Status:** Accepted (first slice: network locality between machines)
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
- The training-time benefit has not been measured: that needs a run of the same job on one
  zone and across regions on real machines.
- GPU topology inside a machine (NVLink, PCIe switches) is not part of this decision.

## Security and operational implications

- Zones and network groups appear in placement explanations, like hostnames already do.
- A worker that joined with a join token can claim any zone or network group and so attract
  or avoid distributed jobs. This is the same trust as labels today. Provider-created
  workers cannot: their location comes from GPUlayer's record.

## Follow-up

- Measure one training job in one zone against the same job across regions.
- Pick GPUs inside a machine by interconnect (`nvidia-smi topo -m`) for multi-GPU tasks.
- AWS and Azure metadata, once those providers exist.
