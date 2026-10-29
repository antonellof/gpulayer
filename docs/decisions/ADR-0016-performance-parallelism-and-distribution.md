# ADR-0016: Speed, parallelism and job distribution as primary goals

- **Status:** Accepted (milestone 2)
- **Date:** 2026-10-29

## Context

GPUlayer sits between a user's code and the GPUs that run it. Every millisecond it adds to
starting a job, every idle GPU it fails to fill, and every synchronisation it forces inside a
training step is paid on every run. Users choose a layer like this to finish work sooner, so
speed has to be a design input and not a later optimisation.

A measurement pass on 2026-10-29 found that most of GPUlayer's own overhead came from fixed
sleeps rather than real work:

- idle workers polled for work every 500 ms;
- the process runner checked for exit every 100 ms;
- `Job.wait()` and `follow_logs()` polled every 200 to 300 ms;
- log lines were sent to the control plane on the thread that reads the job's output, so a
  slow control plane could fill the pipe and stall the job;
- the Trainer read the loss with `.item()` on every step, which makes the CPU wait for the GPU;
- source bundles could include a GPUlayer home or previous run outputs next to the script.

## Decision

1. **Performance is a requirement.** Changes to a hot path (job dispatch, the worker loop,
   process supervision, log and artifact transfer, the training loop, group placement) come
   with a benchmark result or a test that guards the behaviour. Benchmarks live in
   `benchmarks/` and print the hardware-independent method next to the numbers.
2. **No fixed sleeps on hot paths.** Waiting is event-driven:
   - `POST /v1/worker/acquire` accepts `{"wait": seconds}` and is held open until a task is
     assigned to the worker or the wait ends (at most 30 s). The control plane wakes waiters
     through an in-process condition signalled on every job state change and rendezvous
     publication, and re-checks the database every 200 ms for changes made by other processes
     (`ControlPlane.acquire_task`, `ControlPlane.signal`).
   - `GET /v1/jobs/{id}?wait=seconds` returns as soon as the job is terminal
     (`ControlPlane.wait_job`); `Job.wait()` and `Job.follow_logs()` use it.
   - The process runner learns about exit from a waiter thread; its 100 ms loop only checks
     cancellation and timeouts (`runtime/process.py`).
   - Log lines are shipped by a background sender every 250 ms or every 500 lines, so reading
     the job's output never blocks on the network (`worker/agent.py`, `_LogBuffer`).
   Clients and workers back off to the old polling interval when a server answers at once, so
   old servers keep working.
3. **The training loop never synchronises with the device unless it must.** Logged values stay
   on the device and are read when printed or at the end of an epoch. Gradient accumulation
   (`accumulate_grad_batches`) uses DDP `no_sync()` so gradients are all-reduced only on the
   step that applies them.
4. **Keep what measures faster, revert what does not.** Sharing CPU cores between local ranks
   (`OMP_NUM_THREADS = cores // procs_per_worker`) was tried and measured: 2-rank CPU DDP got
   48% slower per step on a small model and 2% faster on a larger one, so ranks keep torchrun's
   default of one thread. Jobs can set `OMP_NUM_THREADS` themselves.
5. **Ship only source.** Bundles skip any GPUlayer home (a directory holding `metadata.db`) and
   `gpulayer_runs/`.
6. **Parallelism and distribution are scheduled, not hoped for.** The roadmap below is ordered by
   expected gain per unit of work; each item needs a benchmark before and after.
7. **Claims follow measurements.** Published numbers state the hardware, software versions and
   method, and are marked measured, simulated or hardware-validated as in CLAUDE.md. GPUlayer is
   not described as "the fastest" anything without a reproducible comparison.

## Results on the development machine

Apple Silicon Mac, 10 CPU cores, Python 3.12, torch 2.15.0, embedded control plane,
`benchmarks/dispatch.py --jobs 40 --workers 4`, job = `python -c pass` (about 30 ms to start):

| | before | after |
|---|---|---|
| one job on an idle worker, submit until `wait()` returns | 626 ms | 31 ms |
| burst of 40 jobs on 4 local workers | 26.8 jobs/s | 87.1 jobs/s |

Training loop, same machine, per-step time with the per-step sync removed (medians of three
runs, 100 and 600 steps, three-layer MLP, batch 64):

| | before | after |
|---|---|---|
| Apple GPU (MPS), width 256 | 0.89 ms | 0.57 ms |
| Apple GPU (MPS), width 2048 | 1.37 ms | 0.83 ms |
| CPU, one process | unchanged within noise | |

These are CPU and Apple GPU measurements on one machine. They do not show remote (HTTP) or multi-node
behaviour, which still needs a measured run on separate machines.

## Alternatives considered

- **Shorter poll intervals.** Cheap to write, but every idle worker would query the database
  several times a second, and latency would still be bounded by the interval.
- **A message broker (Redis, NATS) for dispatch.** Lower latency across machines, but a new
  required service for every deployment. Long polling over the existing HTTPS API gives most of
  the gain with no new dependency; a push channel can be added behind the same worker protocol
  later.
- **Server-sent events or WebSockets for workers.** Stream-friendly, but harder to run through
  proxies and harder to authenticate per request than plain long polls. Revisit for live logs.

## Consequences and trade-offs

- Each waiting worker or client holds one server thread while it waits. `ThreadingHTTPServer`
  handles this for tens to low hundreds of workers; larger fleets need an async server.
- Changes made by another process (a CLI submit against the same embedded home) are seen within
  200 ms rather than at once.
- Logged training values are now tensors until read; user code that inspects
  `_gl_step_logs` directly sees tensors. `callback_metrics` and `history` are still floats.

## Security and operational implications

Long polls are capped at 30 seconds and use the same authentication as other calls. A client
can hold at most one server thread per open request; the failed-login limiter still applies.

## Follow-up (roadmap, in order)

1. **Several tasks per worker.** Done: a worker runs up to `max_concurrent_tasks` tasks at
   once (default one per discrete GPU), and the control plane reserves specific GPUs per task
   (`TaskAssignment.devices`). Tested with simulated 4-GPU workers and with real concurrent
   processes on CPU; not yet run on a multi-GPU NVIDIA machine.
2. **Per-rank log streams.** Done: torchrun `--tee` captures each rank separately, the worker
   tags every line with its global rank (`LogLine.rank`), and output from different ranks no
   longer interleaves.
3. **Topology-aware group placement.** Prefer GPUs on one host, then hosts on one fast network,
   using reported interconnect data (NVLink, InfiniBand) when it exists.
4. **Faster start-up of distributed jobs.** Measure and reduce bundle, extract and torchrun
   start times; reuse warm worker environments between jobs.
5. **Sharded data parallelism (FSDP)** for models larger than one GPU, and a `torch.compile`
   option, validated on CUDA hardware.
6. **Async server** if long-poll thread counts become the limit.

Validating items 3 and 5, and any multi-node claim, requires NVIDIA hardware.
