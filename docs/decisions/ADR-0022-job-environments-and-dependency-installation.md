# ADR-0022: Job environments and dependency installation

- **Status:** Accepted (points 1 to 3 implemented; 4 and 5 planned)
- **Date:** 2026-10-31

## Context

Installing Python packages takes seconds to minutes (PyTorch with CUDA is over 2 GB). If it
happens inside a job, it is paid on every run and mixed into the job's time. Today GPUlayer
installs nothing per job: a job runs with the Python environment of the worker that picks it
up, and the scheduler matches the job's `frameworks` against what that environment reports
(ADR-0005). The cost is moved to whoever sets up the worker, and every job that needs a
package the worker lacks is rejected instead of installed.

How others handle it:

| Tool | Where dependencies come from | When they are installed |
|---|---|---|
| PyTorch Lightning (Trainer), torchrun, Accelerate | the environment you start them in | by you, before running |
| Lightning AI Studios | a persistent environment per Studio | once, then reused by every run and job |
| Ray | the cluster image, or a per-job `runtime_env` (pip, conda, uv) | `runtime_env`: on first use on each node, then cached by content and reused |
| SkyPilot | a task's `setup` commands, or a Docker image | `setup` runs once when the cluster is launched; later `sky exec` runs skip it |
| Modal | an image defined in code | built once, cached in layers, reused on every container start |
| dstack, Kubernetes, Slurm | a container image (Slurm: modules or conda) | pulled once per node and cached |

All of them keep installation out of the hot path: it happens once per machine (or image)
and is reused, and where it is per job (Ray) it is cached by the content of the request.

## Decision (proposed)

1. **Keep worker environments as the default.** A worker's Python environment is prepared
   once, when the worker is set up (`gpulayer join --python /path/to/venv/bin/python`, the
   Docker image, or a provider offer's `packages`), and jobs are matched against it. Job
   time never includes installation.
2. **Add per-job requirements with cached environments.** A job may declare
   `environment.requirements` (a list, or a `requirements.txt` in its source bundle). The
   worker builds a virtual environment keyed by the hash of (requirements, file content,
   interpreter path and version), and reuses it for later jobs with the same request.
   Implemented in `worker/environments.py`: the environment is created with `uv venv` (or
   `venv`), the worker interpreter's own site-packages are linked in with an
   `site.addsitedir` .pth line (so a worker running in a virtualenv, the usual case, keeps
   PyTorch and the CUDA libraries, including editable installs), and packages are installed
   with that environment's pip. pip, not uv: uv does not count packages reached through a
   .pth file as installed and would install torch again; pip resolves against the whole
   `sys.path`. Builds are locked per key (threads and processes) and counted as ready only
   once marked, so a failed build is retried, never reused.
3. **Report preparation separately.** Building or reusing an environment is a `task.prepare`
   phase with its own duration in the job's events and in `gpulayer job status`, never part
   of the run time. Benchmarks run with warm environments and report preparation apart.
4. **Warm workers ahead of time.** `gpulayer env prepare REQUIREMENTS` builds an environment
   on every matching worker before jobs arrive, the equivalent of SkyPilot's `setup`.
5. **Container images later.** For workloads that need system packages, an `image` field run
   by the worker with Docker or Podman, pulled once per worker and cached, as Kubernetes and
   dstack do.

## Alternatives considered

- **Install into the worker's environment per job (`pip install` before each run):** simple,
  but slow on every run, and concurrent jobs would modify one shared environment.
- **Conda environments:** cover non-Python libraries, but solving and installing is much
  slower than uv, and CUDA libraries now ship as pip wheels.
- **Images only:** the most reproducible, but building and pushing an image for every code
  change is slow for interactive work; kept for later (point 5).

## Consequences

- Benchmarks and job timings stay comparable: no run includes installation.
- Per-job requirements need disk space per distinct environment on each worker; the cache
  needs a size limit and cleanup with `gpulayer gc`, as Ray's does.
- Jobs that need packages a worker cannot build (no network, no matching wheels) fail in
  `task.prepare` with the installer's message, before any of the workload runs.

## Measured

On an M2 Pro (worker virtualenv with torch 2.15.0): `tqdm` built in 1.56 s and reused in
0.00 s; `timm` built in 6.62 s and reused in 0.00 s, without installing torch again
(`benchmarks/results/2026-10-31-environments/`).

## Follow-up

- Measure a cold and a warm preparation on a GPU worker.
- Point 4 (`gpulayer env prepare`), point 5 (images), and cleanup of unused environments in
  `gpulayer gc`.
