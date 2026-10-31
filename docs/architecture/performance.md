# Performance

Speed, parallelism and job distribution are primary design goals ([ADR-0016](../decisions/ADR-0016-performance-parallelism-and-distribution.md)).
This page explains where time goes when GPUlayer runs a job, what has been done about it, and
how to measure it yourself.

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

`benchmarks/dispatch.py` measures the time GPUlayer adds around a job that does nothing
(`python -c pass`):

```bash
python benchmarks/dispatch.py --jobs 40 --workers 4
```

Measured on an Apple Silicon Mac (10 CPU cores), Python 3.12, embedded control plane:

| | |
|---|---|
| one job on an idle worker, submit until `wait()` returns | 31 ms |
| burst of 40 jobs on 4 local workers | 87.1 jobs/s |

Starting the Python interpreter takes about 30 ms on the same machine, so almost all of the
remaining time for a single job is the job's own start-up.

These are single-machine numbers. Over HTTP, add one network round trip for each of acquire,
start, logs and finish.

## HTTP API

Clients and remote workers keep one connection open per thread, so a request costs one round
trip, not a new TCP and TLS handshake. The server sets `TCP_NODELAY`, so Linux does not hold
responses back waiting for a delayed ACK, and runs TLS handshakes in each connection's thread
instead of the accept loop. Idle connections close after
120 seconds and clients reconnect transparently.

`benchmarks/http_requests.py` times `GET /v1/jobs` over TLS on loopback:

| | Linux (GCP e2-standard-8) | macOS (M2 Pro) |
|---|---|---|
| one request, p50 | 0.60 ms | 0.24 ms |
| one thread | 1555 req/s | 4114 req/s |
| 8 threads | 853 req/s | 2781 req/s |

On two GCP VMs (a server with two workers on one, a third worker on the other; 15 slots),
500 jobs of 0.02 s each took 9.0 s (55.6 jobs/s).

## Distributed job start-up

`benchmarks/distributed_startup.py` runs a trivial 2 nodes x 2 processes job (import torch,
`init_process_group`, barrier) on local workers and splits its wall time. Apple M2 Pro, macOS,
Python 3.12, torch 2.15, gloo, 4 runs per launcher:

| Stage | `direct` (default) | `torchrun` |
|---|---|---|
| GPUlayer dispatch (submit to tasks started) | 0.00 | 0.00 |
| launcher start-up (task start to script running) | 0.06 to 0.07 | 0.73 to 0.74 |
| `import torch` in each rank | 0.60 to 0.63 | 0.57 to 0.67 |
| process-group init (gloo) | 0.03 to 0.06 | 0.02 to 0.07 |
| shutdown after the script ends | 0.21 | 0.64 to 0.75 |
| GPUlayer completion (task finished to client sees it) | 0.00 | 0.00 |
| **total** | **0.93 to 0.94** | **2.05 to 2.21** |

The direct launcher starts the ranks from a small standard-library process instead of
torchrun's elastic agent. What is left is PyTorch itself: importing torch in each rank and the
rank's own exit. These are CPU results on one machine; NVIDIA and multi-machine start-up have
not been measured.

## Training loop

The PyTorch `Trainer` avoids work that slows each step:

- values passed to `self.log()` stay on the device and are read only when printed or at the
  end of an epoch, so the CPU does not wait for the GPU on every step;
- `accumulate_grad_batches=N` applies the optimizer every N batches, and DDP exchanges
  gradients only on those batches.

How this compares with PyTorch Lightning on an NVIDIA T4 is in
[Performance and comparison](../get-started/performance-and-comparison.md).

Distributed CPU jobs keep one thread per rank (torchrun's default). Giving each rank a share of
the cores was measured and was slower for small models; set `OMP_NUM_THREADS` in the job's
environment to change it.

## Tips for fast jobs

- Keep the script's directory small. Bundles skip `.git`, virtual environments,
  `gpulayer_runs/` and any GPUlayer home, but large datasets next to the script are uploaded on
  every launch. Pass data as inputs or read it from storage the workers can reach.
- Use `DataLoader(num_workers=..., pin_memory=True)` on GPU jobs so data loading overlaps with
  compute.
- Prefer `bf16-mixed` precision on GPUs that support it.
- Run many small independent jobs on several workers: `gpulayer worker start` on each machine,
  or `client.ensure_local_workers(n)` locally.

## Known limits

- Several tasks share a worker only up to its slot count (one per NVIDIA or AMD GPU by
  default), and each gets whole GPUs. Fractional GPU sharing is not supported.
- Each waiting worker holds one server thread during its long poll; the built-in server suits
  tens to low hundreds of workers.
- Distributed groups are kept in one network group, zone or region when possible
  ([Scheduling](scheduling.md#distributed-groups-and-network-locality)); multi-GPU tasks get the
  best-connected free GPUs on a machine.
