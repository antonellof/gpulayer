---
name: gpulayer
description: Run commands, scripts and training jobs on GPUs (NVIDIA CUDA or Apple Silicon) or other machines with the GPUlayer CLI. Use when a task needs a GPU, more compute than the current machine, several runs in parallel, or a job on a specific server; also to check job status, logs and outputs of GPUlayer jobs.
---

# Running work on GPUs with GPUlayer

GPUlayer sends a command to a machine that has what it needs (GPU type and memory,
frameworks such as torch or mlx), runs it, streams its output and returns files it writes.
It works on this machine alone, or against a GPUlayer server when `GPULAYER_URL` and
`GPULAYER_TOKEN` are set in the environment.

## Before submitting

1. See what is available: `gpulayer resources list --json`. Each worker lists its
   accelerators (`cuda`, `mps`), GPU memory, installed frameworks and free task slots.
2. Ask only for what the work needs. CPU work does not need `--accelerator`. Use
   `--accelerator cuda` for NVIDIA and `--accelerator mps` for Apple GPUs, `--gpus N` for more
   than one, and `--framework 'torch>=2.1'` when the code needs a package on the worker.

## Run a command

```bash
# Ship the current project directory and run a script in it on an NVIDIA GPU
gpulayer submit --name train --source . --accelerator cuda --framework 'torch>=2.1' \
  -- python train.py --epochs 3

# Stage an input file (available in $GPULAYER_INPUT_DIR on the worker)
gpulayer submit --source . --input data.csv -- python analyse.py

# Long job: return at once and check later
gpulayer submit --detach --source . --accelerator cuda -- python train.py
```

- `python` as the first word means the worker's own Python interpreter.
- Without `--detach` the command streams the job's output and exits with the job's exit code.
- Files the job writes to `$GPULAYER_OUTPUT_DIR` come back as outputs.
- In local mode without `--detach`, the CLI runs the job on this machine itself; a detached
  local job waits for `gpulayer worker start`.

## Follow a job

```bash
gpulayer job status <job-id> --json      # state, placement, exit codes, errors
gpulayer job logs <job-id> -f            # stream output until it finishes
gpulayer job result <job-id> --download ./outputs   # result JSON and output files
gpulayer job cancel <job-id>
gpulayer job list --json
```

## When a job is rejected

If no worker qualifies, the job fails with `unschedulable` and `gpulayer job status` lists
every worker with the reason, for example `torch on this worker cannot use CUDA` or
`needs 1 usable cuda device(s), worker has 0`. Change the request (a different accelerator,
fewer GPUs, a framework that is installed) instead of resubmitting the same job. Use
`--queue-if-unschedulable` only when a suitable worker is expected to join.

## Rules

- Submitting a job runs code on another machine. Only submit what the user asked for.
- Never put secrets in commands or environment variables of a job; use
  `--secret NAME=KEY` with a key the worker already has.
- Do not use admin tokens. A `submitter` token is enough to submit, follow and cancel jobs.
- Report the job id to the user so they can follow it on the dashboard (`/ui/` on the
  server, or `gpulayer dashboard` locally).
