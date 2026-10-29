# Examples

All examples need PyTorch in the interpreter that runs the workload (the worker's
interpreter, `--python` on `gpulayer worker start`; by default the one running GpuLayer).

## `pytorch_train/`: single-device training and JSON inference (SDK)

```bash
python examples/pytorch_train/run.py                     # CPU
python examples/pytorch_train/run.py --accelerator mps   # Apple Silicon (independent jobs only)
python examples/pytorch_train/run.py --accelerator cuda  # NVIDIA
```

What happens:

1. `client.package(...)` uploads a deterministic, content-addressed bundle of the folder.
2. A `CommandJob` (`python train.py`) is scheduled onto a worker whose *reported* capabilities
   satisfy `accelerator` and `torch>=2.0`. If none can, it fails immediately and prints why
   each worker was rejected (e.g. on a Mac: `needs 1 usable cuda device(s), worker has 0`).
3. `train.py` picks its device from `GPULAYER_ACCELERATOR` and writes `model.pt` and
   `metrics.json` to `GPULAYER_OUTPUT_DIR`; they come back as output artifacts and are
   downloaded to `examples/pytorch_train/outputs/`.
4. A `FunctionJob` calls `predict:predict` with JSON arguments and gets JSON back, without pickling.

Verified on Apple Silicon (MPS) and CPU. The CUDA path uses the same mechanism but has not
been run on NVIDIA hardware in this repository yet.

## `pytorch_ddp/`: DistributedDataParallel through torchrun (CLI)

```bash
# one node, two processes
gpulayer submit --distributed --source examples/pytorch_ddp --entrypoint train_ddp.py \
    --nodes 1 --procs-per-node 2 --framework 'torch>=2.0'

# two nodes x two processes: two workers, one logical job (world size 4)
gpulayer worker start --name node-a &
gpulayer worker start --name node-b &
gpulayer submit --detach --distributed --source examples/pytorch_ddp \
    --entrypoint train_ddp.py --nodes 2 --procs-per-node 2
gpulayer job logs <job-id>
gpulayer job result <job-id> --download ./out          # node0/ddp_model.pt
```

GpuLayer gang-schedules the worker group, the rank-0 worker publishes the rendezvous endpoint,
and every node launches `torchrun --nnodes N --node-rank r --master-addr ... --max-restarts=0`.
PyTorch handles the collectives (`gloo` on CPU; `--backend nccl --accelerator cuda` for NVIDIA).

Verified with two worker processes on one machine over loopback. Across machines, each worker
needs an `--advertise-address` its peers can reach and open rendezvous and collective ports (see
[docs/architecture/distributed.md](../docs/architecture/distributed.md)). Multi-machine runs
have not been validated yet.
