# ADR-0020: Two-level gradient sync for several processes per machine

- **Status:** Accepted
- **Date:** 2026-10-30

## Context

DDP averages gradients with one all-reduce over every process. With gloo and several
processes per machine (CPU training, one process per core group), that all-reduce is a ring
over all N x P processes, and most of its steps cross the network. For small models the
per-step cost is latency, not bandwidth, so it grows with the number of processes.

A profile on 8 GCP machines x 4 processes (n1-standard-4, gloo, the 141k-parameter CNN of
the strong-scaling benchmark) measured, per step on the slowest process: 83 ms of compute,
25.5 ms for a flat all-reduce of the gradients, 11.0 ms for a two-level one, and a full DDP
step of 117 ms flat against 102 ms with the two-level hook
(`benchmarks/lightning_comparison/profile_step.py`).

## Decision

1. `gpulayer.pytorch.hierarchical.HierarchicalAllReduce` sums inside each machine
   (`reduce` to local rank 0), all-reduces among one process per machine, and broadcasts
   back inside each machine. It is registered as a DDP communication hook.
2. `Trainer(gradient_sync="auto")` (the default) and `Runtime(gradient_sync=...)` use it when
   the backend is gloo, there is more than one machine, and more than one process per
   machine. `"flat"` keeps DDP's own all-reduce; `"two_level"` forces the hook.
3. NCCL is left alone: it builds topology-aware rings and trees itself, and the hook would
   serialize communication that NCCL overlaps with backward.
4. Ranks must be numbered machine by machine (`rank = machine * P + local rank`), which the
   GPUlayer launcher and torchrun do; `LOCAL_WORLD_SIZE` gives P.

## Consequences

- The sum is the same up to floating-point rounding order; training results match flat DDP
  within that (tested on 2 x 2 CPU processes).
- The hook runs synchronously inside backward, so it gives up DDP's overlap of the last
  buckets with computation. For the small models where gloo latency dominates this costs
  less than it saves; for large models on CPU, measure with `gradient_sync="flat"`.
- The Trainer log says when it is active.
