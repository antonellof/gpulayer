# Distributed execution

A `DistributedJob` is one logical workload across `num_workers` nodes with
`procs_per_worker` processes each (`world_size = num_workers × procs_per_worker`).
GpuLayer owns allocation and lifecycle; **PyTorch owns communication** (torchrun + `gloo` or
`nccl`). There is no custom collective layer. See ADR-0007.

## Flow

1. **Validate.** Distributed MPS, `nccl` without CUDA, `procs_per_worker > gpu_count`, and
   automatic retries are rejected at submission.
2. **Gang-schedule.** The scheduler selects `num_workers` *distinct, available, compatible*
   workers at once (same torch version, `torchrun` launcher, the requested backend, an
   `advertise_address` when multi-node). Otherwise nothing is placed and the job stays queued.
3. **Assign ranks.** Node rank `r` hosts global ranks `[r·P, (r+1)·P)` (`runtime/distributed.py`).
4. **Rendezvous.** The rank-0 worker picks a port (or uses `rendezvous.port`) and publishes
   `advertise_address:port`. Other nodes' assignments are withheld until it is published.
5. **Launch.** Every node runs
   `python -m torch.distributed.run --nnodes N --nproc-per-node P --node-rank r
   --master-addr A --master-port p --max-restarts=0 <entrypoint> <args>`.
6. **Observe.** Per-node logs (tagged `node_rank`), per-task state, events.
7. **Fail fast.** The first failed/lost/timed-out node fails the job; running siblings are
   told to stop on their next heartbeat; unstarted ones are cancelled.
8. **Collect.** Each node's `GPULAYER_OUTPUT_DIR` is uploaded as `node<r>/<path>`.

## Networking requirements (multi-machine)

- Every node must reach rank 0's `advertise_address:master_port` (TCP).
- `gloo`/`nccl` open additional ports between all nodes; firewalls must allow node-to-node
  traffic, and the right interface may need `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME`
  (pass with `--env`).
- Workers need the same torch version (enforced) and compatible code (the same source bundle).

## Checkpoints and retries

No automatic retry of a partially executed group. Checkpointing is the workload's
responsibility: write checkpoints to `GPULAYER_OUTPUT_DIR` (returned as artifacts) or to
storage your workload controls, and resubmit with an input pointing at them.

## Validation status

| Scenario | Status |
|---|---|
| Rank planning, gang scheduling, rendezvous gating, fail-fast, worker loss | tested (simulated workers) |
| 2 nodes × 2 procs, gloo, all_reduce + DDP, real processes, one host over loopback | **tested** |
| Multiple machines | **planned** — not yet validated |
| NCCL multi-GPU | implemented path, **not hardware-validated** |
| MPS | unsupported |
