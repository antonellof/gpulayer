# ADR-0007: Distributed launch and rendezvous

- **Status:** Accepted (milestone 1) for torchrun static rendezvous, validated multi-process on one host. Multi-machine operation is Proposed (not validated).
- **Date:** 2026-10-29

## Context

A distributed job is one logical workload across several nodes and processes. GpuLayer
should own allocation, rank assignment, rendezvous configuration and lifecycle. It should
not own collective communication, which PyTorch already provides.

## Decision

- `DistributedJob` (`contracts/models.py`) declares `num_workers`, `procs_per_worker`,
  `launcher="torchrun"`, `comm_backend` (`gloo` or `nccl`; `nccl` requires CUDA), the
  `entrypoint` script inside the source bundle, and `rendezvous` (`static`, optional
  port, join timeout). `failure_policy` is `fail_fast`.
- **Allocation:** gang scheduling of a compatible worker group (ADR-0004). One task is
  created per node with `node_rank` 0..N-1. `plan_group`
  (`runtime/distributed.py`) defines the global rank layout: node r hosts ranks
  [r·P, (r+1)·P).
- **Rendezvous:** when the rank-0 worker acquires its task, it picks the port (from the
  spec, or a free one) and publishes `advertise_address:port` through
  `publish_rendezvous`. The control plane withholds rank>0 assignments until that
  endpoint exists. Only node rank 0 may publish, and a conflicting re-publish is
  rejected.
- **Launch:** each worker runs `python -m torch.distributed.run --nnodes N
  --nproc-per-node P --node-rank r --master-addr A --master-port p --max-restarts=0
  <entrypoint>` (`torchrun_argv`) in its own process group. The script reads
  `GPULAYER_COMM_BACKEND` and calls `init_process_group`.
- **Coordination:** the job succeeds only when every node succeeds. When any node
  fails, is lost or times out, the job fails immediately, and running siblings get
  `cancel_requested` (delivered by heartbeat) while unstarted ones are cancelled.
  Distributed jobs are never retried automatically (`JobSpec` validation rejects
  `max_attempts>1`).
- Logs are tagged with `node_rank`. Outputs are uploaded per node under `node<r>/`.

## Validation

- `tests/distributed/test_torchrun_multiprocess.py` runs two worker agents as two
  "nodes" on a single host, each launching torchrun with 2 processes (world size 4), and
  verifies a gloo `all_reduce`. A second test shows that one failing rank fails the whole
  job and stops the sibling.
- **Not validated:** separate machines, real networks, firewalls and interface
  selection, NCCL, or CUDA multi-GPU. Single-host loopback does not establish multi-node
  support.

## Alternatives considered

- **c10d/etcd elastic rendezvous:** supports elasticity, but it is not needed for
  fail-fast semantics. It remains a later option behind the same `RendezvousSpec`.
- **A custom collective layer:** forbidden by CLAUDE.md.
- **Ray Train or DeepSpeed launchers:** possible additional `launcher` values later.

## Consequences and trade-offs

- Rank-0's advertised address must be reachable from every peer. GpuLayer does not
  verify reachability before launch, so failures surface as torchrun rendezvous
  timeouts.
- `free_port()` has a small race: the port could be taken before torchrun binds it.
- Partial execution can leave side effects such as checkpoints. There is no
  checkpoint/restart support.

## Security and operational implications

- The rendezvous port is unauthenticated (a torch limitation). Run distributed jobs only
  on trusted private networks.

## Follow-up

- A multi-machine validation environment (two hosts, documented network) and a test.
- A reachability pre-check, NCCL/CUDA validation, and an explicit network-interface
  selection option.
- Checkpoint-aware restart as an opt-in policy.
