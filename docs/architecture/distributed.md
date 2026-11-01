# Distributed execution

A `DistributedJob` runs across `num_workers` nodes with `procs_per_worker` processes each
(`world_size = num_workers × procs_per_worker`). GPUlayer allocates nodes and launches and
supervises processes. PyTorch communicates through `gloo` or `nccl`; GPUlayer has no collective
layer of its own. See ADR-0007.

## Flow

1. Submission rejects distributed MPS, `nccl` without CUDA, `procs_per_worker > gpu_count` and
   automatic retries.
2. The scheduler picks `num_workers` distinct, available, compatible workers in one step. They
   must share a torch version and support the job's launcher and backend. Multi-node jobs need
   an `advertise_address` on each worker. If no such group exists, the job stays queued.
3. Node rank `r` hosts global ranks `[r·P, (r+1)·P)` (`runtime/distributed.py`).
4. The rank-0 worker picks a port (or uses `rendezvous.port`) and publishes
   `advertise_address:port`. Other nodes get their assignments only after that.
5. Every node starts its `P` ranks with the job's `launcher`:
   - `direct` (default): the worker's `rank_launcher.py` starts `python <entrypoint> <args>`
     once per rank with the torchrun environment (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
     `LOCAL_WORLD_SIZE`, `GROUP_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `TORCHELASTIC_RUN_ID`,
     ...). Rank 0 hosts the c10d store at `MASTER_ADDR:MASTER_PORT`. The first rank to exit
     with an error stops the others on that node, and its exit code becomes the task's.
   - `torchrun`: `python -m torch.distributed.run --nnodes N --nproc-per-node P --node-rank r
     --master-addr A --master-port p --max-restarts=0 <entrypoint> <args>`.

   Both set the same variables. `direct` skips torchrun's elastic agent, saving about 1.1 s per
   job (see [Performance](performance.md)). Use `torchrun` when code needs the agent, such as
   `torch.distributed.elastic` error files.
6. Logs are tagged with `node_rank`; each task's state and events are recorded.
7. The first node that fails, is lost or times out fails the job. Running nodes stop on their
   next heartbeat; nodes that have not started are cancelled.
8. Each node's `GPULAYER_OUTPUT_DIR` is uploaded as `node<r>/<path>`.

## Networking requirements (multi-machine)

- Every node must reach rank 0's `advertise_address:master_port` (TCP).
- `gloo` and `nccl` open more ports between nodes, so firewalls must allow node-to-node
  traffic. Pick the interface with `GLOO_SOCKET_IFNAME` or `NCCL_SOCKET_IFNAME` via `--env`.
- All workers run the same source bundle and torch version (enforced by the scheduler).
- The scheduler keeps a group inside one network group, zone or region when it can
  ([ADR-0021](../decisions/ADR-0021-topology-aware-group-placement.md)). Report a zone with
  `--zone` (automatic on Google Cloud) and a rack or LAN with `--network-group`.

## Checkpoints and retries

A partially executed group is never retried automatically. Checkpointing is up to the
workload: write checkpoints to `GPULAYER_OUTPUT_DIR` (returned as artifacts) or to your own
storage, then resubmit with an input pointing at them.

## Processes, threads and logs

- Each process gets `OMP_NUM_THREADS=1` unless the job sets it. Sharing cores between ranks was
  measured and was not faster (ADR-0016).
- Each rank's output is captured separately (direct launcher, or torchrun `--tee`) and stored
  per line with its global rank, so ranks never mix. The CLI and `launch()` print
  `[rank3] ...`, the dashboard `[rank 3] ...`, and `LogLine.rank` carries the number.

## Validation status

| Scenario | Status |
|---|---|
| Rank planning, gang scheduling, rendezvous gating, fail-fast, worker loss | tested (simulated workers) |
| 2 nodes × 2 procs, gloo, all_reduce + DDP, real processes, one host over loopback | tested |
| Multiple machines | validated with gloo on up to 16 GCP VMs, both launchers |
| NCCL, several GPUs in one machine | validated on 4× RTX 3070 (PCIe) and 2× A100 SXM4 (NVLink), 2026-10-31 |
| NCCL across machines | validated on 2, 4 and 8 AWS machines with one Tesla T4 each (DDP, FSDP) |
| MPS | unsupported |
