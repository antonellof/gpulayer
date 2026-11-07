# ADR-0025: Keeping a job's newest checkpoint and restarting groups from it

- **Status:** Accepted
- **Date:** 2026-11-07
- **Amends:** [ADR-0007](ADR-0007-distributed-launch-and-rendezvous.md) and
  [ADR-0011](ADR-0011-cost-accounting-and-retry-semantics.md): distributed jobs may now retry,
  when they have a resume policy

## Context

A Spot VM can disappear at any time. Until 0.9.2, a retry started the job from the beginning:
outputs were uploaded when a task ended, so a checkpoint on a lost VM was lost with it, and the
next attempt was not given one. Distributed jobs could not retry at all, because a partially
executed group is not safe to replay. Our migration guide said SLURM's auto-requeue was "not
needed", which overstated what GPUlayer did. Lightning relies on a cluster for this.

## Decision

1. **`JobSpec.resume`** (`ResumePolicy`, with `path`, by default `checkpoints/last.ckpt`) asks
   for the job's newest checkpoint to be kept on the control plane while the job runs.
   `launch(max_attempts=N)` and `Trainer(launch_options={"max_attempts": N})` set it.
2. **The job never waits.** After writing `last.ckpt`, the Trainer prints a marker line. The
   worker, which reads the job's output anyway, uploads the file on its own thread; while one
   uploads, only the newest announced after it is sent next. There is no polling, and the line
   stays out of the job's log. When the process ends the worker finishes what was announced,
   for up to 120 s, so the checkpoint saved on SIGTERM reaches the next attempt.
3. **One checkpoint per job.** A newer one replaces it and the old artifact is deleted. A report
   from an attempt that is no longer current is ignored. After a success the checkpoint is
   deleted unless it is also an output; if the attempts run out it is returned as an output,
   `resume/last.ckpt`.
4. **The next attempt gets it as an input**, at `$GPULAYER_INPUT_DIR/gpulayer-resume.ckpt`, in
   the spec of its assignment, so no new assignment field was needed. In a later attempt `fit`
   resumes from it, over any `ckpt_path` in the script: the kept checkpoint is newer.
5. **A distributed group restarts as a whole**, and only with a resume policy. A failed rank
   stops the others, and the new attempt is placed only after every task of the old one has
   ended, so the two never share GPUs. This is a restart from a checkpoint, not a replay of part
   of the group, so ADR-0007's reason for forbidding retries does not apply.
6. **`preempted` is a retry reason of its own**: a task that ended with SIGTERM's exit code
   (143) without being cancelled. `launch(max_attempts=N)` retries on `worker_lost` and
   `preempted`, not on `failed`: a crash in the script would crash again.

## Consequences

`JobSpec` gained an optional field, left out of serialized specs when unset because workers
reject unknown fields: a job without `resume` runs on any worker. A job with `resume` needs
0.10 workers and a 0.10 control plane; on an older control plane the worker's report gets a 404
and the job runs on without keeping checkpoints. The control plane needs disk space for one
checkpoint per running job, up to the 2 GB limit for one artifact.

Work since the last checkpoint is repeated, and a retry may repeat side effects: ADR-0011's "no
exactly-once" stands. A job loses at most the time since its last saved `last.ckpt`, which is
one epoch by default.

## Limits

- Sharded checkpoints (folders) and a remote `default_root_dir` are not streamed. A run writing
  to an object store keeps its checkpoints there; `ckpt_path="last"` finds them.
- Tested with real processes on one machine, where the preemption is a SIGTERM the script sends
  itself, and on the control plane with simulated workers. **Not validated on a reclaimed cloud
  VM.**

## Security and operational implications

The new call is authenticated like the others, and a worker can report only for a task assigned
to it. `resume.path` must stay inside the job's output folder. A checkpoint holds model weights
and sits in the artifact store with the job's other outputs.

## Follow-up

Run it on a real Spot VM reclaimed by GCP, and record the time lost. Stream large checkpoints in
parts instead of reading them into memory.
