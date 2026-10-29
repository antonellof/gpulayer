# Distributed execution

A `DistributedJob` is one logical workload across `num_workers` nodes with
`procs_per_worker` processes each (`world_size = num_workers × procs_per_worker`).
GpuLayer handles allocation and lifecycle. PyTorch handles communication through torchrun and
`gloo` or `nccl`, and GpuLayer has no collective layer of its own. See ADR-0007.

## Flow

1. Submission rejects distributed MPS, `nccl` without CUDA, `procs_per_worker > gpu_count` and
   automatic retries.
2. The scheduler selects `num_workers` distinct, available and compatible workers in one step.
   They must share a torch version and have the `torchrun` launcher and the requested backend,
   and multi-node jobs need an `advertise_address` on each worker. If no such group exists,
   nothing is placed and the job stays queued.
3. Node rank `r` hosts global ranks `[r·P, (r+1)·P)` (`runtime/distributed.py`).
4. The rank-0 worker picks a port (or uses `rendezvous.port`) and publishes
   `advertise_address:port`. The other nodes receive their assignments only after that.
5. Every node runs
   `python -m torch.distributed.run --nnodes N --nproc-per-node P --node-rank r
   --master-addr A --master-port p --max-restarts=0 <entrypoint> <args>`.
6. Logs are tagged with `node_rank`, and each task's state and events are recorded.
7. The first node that fails, is lost or times out fails the job. Running nodes are told to stop
   on their next heartbeat, and nodes that have not started are cancelled.
8. Each node's `GPULAYER_OUTPUT_DIR` is uploaded as `node<r>/<path>`.

## Networking requirements (multi-machine)

- Every node must reach rank 0's `advertise_address:master_port` (TCP).
- `gloo` and `nccl` open more ports between all nodes, so firewalls must allow node-to-node
  traffic. Selecting the right network interface may need `GLOO_SOCKET_IFNAME` or
  `NCCL_SOCKET_IFNAME`, passed with `--env`.
- Workers need the same torch version, which the scheduler enforces, and they all run the same
  source bundle.

## Checkpoints and retries

GpuLayer never retries a partially executed group automatically. Checkpointing is up to the
workload: write checkpoints to `GPULAYER_OUTPUT_DIR` (returned as artifacts) or to storage your
workload controls, then resubmit with an input that points at them.

## Processes, threads and logs

- Each worker starts `procs_per_worker` processes through torchrun with `OMP_NUM_THREADS=1`
  per process unless the job sets it. Sharing the cores between ranks was measured and was not
  faster (ADR-0016).
- All local ranks write to the same output stream today. A program that writes one line in
  several calls (unbuffered `print` writes the text and the newline separately) can have lines
  from two ranks interleave. Write each line in one call, or log from rank 0 only. Separate
  per-rank streams are on the roadmap (ADR-0016).

## Validation status

| Scenario | Status |
|---|---|
| Rank planning, gang scheduling, rendezvous gating, fail-fast, worker loss | tested (simulated workers) |
| 2 nodes × 2 procs, gloo, all_reduce + DDP, real processes, one host over loopback | tested |
| Multiple machines | planned, not yet validated |
| NCCL multi-GPU | implemented, not hardware-validated |
| MPS | unsupported |
