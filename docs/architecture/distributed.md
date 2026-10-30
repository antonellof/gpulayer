# Distributed execution

A `DistributedJob` is one logical workload across `num_workers` nodes with
`procs_per_worker` processes each (`world_size = num_workers × procs_per_worker`).
GPUlayer handles allocation, process launch and lifecycle. PyTorch handles communication through
`gloo` or `nccl`, and GPUlayer has no collective layer of its own. See ADR-0007.

## Flow

1. Submission rejects distributed MPS, `nccl` without CUDA, `procs_per_worker > gpu_count` and
   automatic retries.
2. The scheduler selects `num_workers` distinct, available and compatible workers in one step.
   They must share a torch version and support the job's launcher and backend,
   and multi-node jobs need an `advertise_address` on each worker. If no such group exists,
   nothing is placed and the job stays queued.
3. Node rank `r` hosts global ranks `[r·P, (r+1)·P)` (`runtime/distributed.py`).
4. The rank-0 worker picks a port (or uses `rendezvous.port`) and publishes
   `advertise_address:port`. The other nodes receive their assignments only after that.
5. Every node starts its `P` ranks with the job's `launcher`:
   - `direct` (default): the worker's `rank_launcher.py` starts `python <entrypoint> <args>`
     once per rank with the torchrun environment (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
     `LOCAL_WORLD_SIZE`, `GROUP_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `TORCHELASTIC_RUN_ID`,
     ...). Rank 0 hosts the c10d store at `MASTER_ADDR:MASTER_PORT`. The first rank that exits
     with an error stops the others on that node, and its exit code becomes the task's.
   - `torchrun`: `python -m torch.distributed.run --nnodes N --nproc-per-node P --node-rank r
     --master-addr A --master-port p --max-restarts=0 <entrypoint> <args>`.

   Both run the same code with the same variables. `direct` skips torchrun's elastic agent,
   which saves about 1.1 s per job (see [Performance](performance.md)). Use `torchrun` for
   code that needs the agent itself, such as `torch.distributed.elastic` error files.
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

GPUlayer never retries a partially executed group automatically. Checkpointing is up to the
workload: write checkpoints to `GPULAYER_OUTPUT_DIR` (returned as artifacts) or to storage your
workload controls, then resubmit with an input that points at them.

## Processes, threads and logs

- Each worker starts `procs_per_worker` processes with `OMP_NUM_THREADS=1`
  per process unless the job sets it. Sharing the cores between ranks was measured and was not
  faster (ADR-0016).
- Each rank's output is captured separately (by the direct launcher, or torchrun `--tee`) and
  stored line by line with
  the global rank that wrote it, so lines from different ranks never mix. The CLI and
  `launch()` print them as `[rank3] ...`, the dashboard as `[rank 3] ...`, and
  `LogLine.rank` carries the number.

## Validation status

| Scenario | Status |
|---|---|
| Rank planning, gang scheduling, rendezvous gating, fail-fast, worker loss | tested (simulated workers) |
| 2 nodes × 2 procs, gloo, all_reduce + DDP, real processes, one host over loopback | tested |
| Multiple machines | validated with gloo on two GCP VMs, both launchers |
| NCCL multi-GPU | implemented, not hardware-validated |
| MPS | unsupported |
