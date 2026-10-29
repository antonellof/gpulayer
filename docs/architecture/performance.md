# Performance

Speed, parallelism and job distribution are primary design goals ([ADR-0016](../decisions/ADR-0016-performance-parallelism-and-distribution.md)).
This page explains where time goes when GpuLayer runs a job, what has been done about it, and
how to measure it yourself.

## Where time goes

| Stage | What happens | How GpuLayer keeps it short |
|---|---|---|
| Submit | the spec is validated and stored, the scheduler places it | one database transaction; placement runs at once |
| Dispatch | a worker learns it has a task | workers hold a long poll open; the control plane wakes them on placement |
| Prepare | the worker fetches source and inputs, creates a workspace | artifacts are content-addressed and cached per worker |
| Run | the process starts and runs | exit is reported by a waiter thread, not by polling |
| Logs | output lines reach the control plane | a background sender batches them every 250 ms, so the job never waits on the network |
| Finish | the outcome and outputs are stored | waiting clients are woken through a long poll on the job |

## Dispatch overhead

`benchmarks/dispatch.py` measures the time GpuLayer adds around a job that does nothing
(`python -c pass`):

```bash
python benchmarks/dispatch.py --jobs 40 --workers 4
```

Measured on an Apple Silicon Mac (10 CPU cores), Python 3.12, embedded control plane:

| | before ADR-0016 | after |
|---|---|---|
| one job on an idle worker, submit until `wait()` returns | 626 ms | 31 ms |
| burst of 40 jobs on 4 local workers | 26.8 jobs/s | 87.1 jobs/s |

Starting the Python interpreter takes about 30 ms on the same machine, so almost all of the
remaining time for a single job is the job's own start-up.

These are single-machine numbers. Over HTTP, add one network round trip for each of acquire,
start, logs and finish. Multi-machine numbers will be published once they are measured.

## Training loop

The PyTorch `Trainer` avoids work that slows each step:

- values passed to `self.log()` stay on the device and are read only when printed or at the
  end of an epoch, so the CPU does not wait for the GPU on every step;
- `accumulate_grad_batches=N` applies the optimizer every N batches, and DDP exchanges
  gradients only on those batches.

Removing the per-step read made each step 1.6 times faster on an Apple GPU (MPS) for a
three-layer MLP: 0.89 to 0.57 ms at width 256, and 1.37 to 0.83 ms at width 2048. On CPU the
change is within noise, since there is no device to wait for. The same effect is expected on
NVIDIA GPUs but has not been measured yet.

Distributed CPU jobs keep one thread per rank (torchrun's default). Giving each rank a share of
the cores was measured and was slower for small models; set `OMP_NUM_THREADS` in the job's
environment to change it.

## Tips for fast jobs

- Keep the script's directory small. Bundles skip `.git`, virtual environments,
  `gpulayer_runs/` and any GpuLayer home, but large datasets next to the script are uploaded on
  every launch. Pass data as inputs or read it from storage the workers can reach.
- Use `DataLoader(num_workers=..., pin_memory=True)` on GPU jobs so data loading overlaps with
  compute.
- Prefer `bf16-mixed` precision on GPUs that support it.
- Run many small independent jobs on several workers: `gpulayer worker start` on each machine,
  or `client.ensure_local_workers(n)` locally.

## Known limits

- A worker runs one task at a time. A machine with several GPUs runs one single-GPU job at a
  time until concurrent tasks per worker land (next on the roadmap).
- Each waiting worker holds one server thread during its long poll; the built-in server suits
  tens to low hundreds of workers.
- Distributed placement does not yet consider network topology.
