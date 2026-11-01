# Performance

Speed, parallelism and job distribution are primary design goals
([ADR-0016](../decisions/ADR-0016-performance-parallelism-and-distribution.md)).

## Where time goes

| Stage | What happens | How GPUlayer keeps it short |
|---|---|---|
| Submit | the spec is validated and stored, the scheduler places it | one database transaction; placement runs at once |
| Dispatch | a worker learns it has a task | workers hold a long poll open; the control plane wakes them on placement |
| Prepare | the worker fetches source and inputs, creates a workspace | artifacts are content-addressed and cached per worker |
| Run | the process starts and runs | exit is reported by a waiter thread, not by polling |
| Logs | output lines reach the control plane | a background sender batches them every 250 ms, so the job never waits on the network |
| Finish | the outcome and outputs are stored | waiting clients are woken through a long poll on the job |

## Dispatch overhead

`benchmarks/dispatch.py` measures the time added around a job that does nothing
(`python -c pass`):

```bash
python benchmarks/dispatch.py --jobs 40 --workers 4
```

Apple Silicon Mac (10 CPU cores), Python 3.12, embedded control plane:

| | |
|---|---|
| one job on an idle worker, submit until `wait()` returns | 31 ms |
| burst of 40 jobs on 4 local workers | 87.1 jobs/s |

Starting the Python interpreter takes about 30 ms on the same machine, so a single job's time is
almost all its own start-up. These are single-machine numbers; over HTTP, add one round trip
each for acquire, start, logs and finish.

## HTTP API

Clients and workers keep one connection open per thread, so a request is one round trip without
a new TCP and TLS handshake. The server sets `TCP_NODELAY` (no delayed-ACK stalls on Linux) and
runs TLS handshakes in each connection's thread, not the accept loop. Idle connections close
after 120 seconds and clients reconnect transparently.

`benchmarks/http_requests.py` times `GET /v1/jobs` over TLS on loopback:

| | Linux (GCP e2-standard-8) | macOS (M2 Pro) |
|---|---|---|
| one request, p50 | 0.60 ms | 0.24 ms |
| one thread | 1555 req/s | 4114 req/s |
| 8 threads | 853 req/s | 2781 req/s |

On two GCP VMs (server and two workers on one, a third worker on the other; 15 slots), 500 jobs
of 0.02 s each took 9.0 s (55.6 jobs/s).

## Distributed job start-up

`benchmarks/distributed_startup.py` runs a trivial 2 nodes x 2 processes job (import torch,
`init_process_group`, barrier) on local workers. Apple M2 Pro, macOS, Python 3.12, torch 2.15,
gloo, 4 runs per launcher, seconds:

| Stage | `direct` (default) | `torchrun` |
|---|---|---|
| GPUlayer dispatch (submit to tasks started) | 0.00 | 0.00 |
| launcher start-up (task start to script running) | 0.06 to 0.07 | 0.73 to 0.74 |
| `import torch` in each rank | 0.60 to 0.63 | 0.57 to 0.67 |
| process-group init (gloo) | 0.03 to 0.06 | 0.02 to 0.07 |
| shutdown after the script ends | 0.21 | 0.64 to 0.75 |
| GPUlayer completion (task finished to client sees it) | 0.00 | 0.00 |
| **total** | **0.93 to 0.94** | **2.05 to 2.21** |

The direct launcher replaces torchrun's elastic agent with a small standard-library process.
The rest is PyTorch import and exit. These are CPU results on one machine; NVIDIA and
multi-machine start-up have not been measured.

## Training loop

What the PyTorch `Trainer` does to keep steps fast:

- `self.log()` values stay on the device and are read only when printed or at epoch end, so the
  CPU does not wait for the GPU each step.
- `accumulate_grad_batches=N` steps the optimizer every N batches, and DDP exchanges gradients
  only on those batches.
- `prefetch=True` on CUDA loads and pins upcoming batches in a background thread and copies each
  to the GPU on a side stream during the current step. Off by default: on a T4 and an RTX 3070
  it helped compute-heavy CNNs by about 5% and slowed short MLP steps by up to 36%, because the
  thread competes with the training loop for the Python interpreter.
- cuDNN autotuning is on for CUDA unless `deterministic=True` (`benchmark=None`).

On a Tesla T4 the last two made a compute-bound CNN 1.16x faster per step (12.67 to 10.93 ms;
autotuning about 9%, copy-ahead about 5%) and made no difference beyond run-to-run variation on
small MLPs. Measured with `benchmarks/lightning_comparison` and
`TRAINERS="gl gl-base gl-noprefetch gl-nobench pl"`; logs in
`benchmarks/results/2026-10-31-gcp-t4/`. The comparison with PyTorch Lightning on a T4 is in
[Performance and comparison](../get-started/performance-and-comparison.md).

Distributed CPU jobs keep one thread per rank (torchrun's default). Sharing the cores per rank
was slower for small models; set `OMP_NUM_THREADS` in the job's environment to change it.

## Tips for fast jobs

- Keep the script's directory small. Bundles skip `.git`, virtual environments,
  `gpulayer_runs/` and any GPUlayer home, but large datasets next to the script are uploaded on
  every launch. Pass data as inputs or read it from storage the workers can reach.
- Use `DataLoader(num_workers=..., pin_memory=True)` on GPU jobs.
- Prefer `bf16-mixed` precision on GPUs that support it.
- Spread many small independent jobs over several workers: `gpulayer worker start` on each
  machine, or `client.ensure_local_workers(n)` locally.

## Known limits

- Tasks share a worker only up to its slot count (one per NVIDIA or AMD GPU by default), and
  each gets whole GPUs. Fractional GPU sharing is not supported.
- Each waiting worker holds one server thread during its long poll; the built-in server suits
  tens to low hundreds of workers.
- Distributed groups stay in one network group, zone or region when possible
  ([Scheduling](scheduling.md#distributed-groups-and-network-locality)); multi-GPU tasks get the
  best-connected free GPUs on a machine.
