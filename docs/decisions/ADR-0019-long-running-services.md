# ADR-0019: Long-running services

- **Status:** Accepted (first slice implemented)
- **Date:** 2026-10-30

## Context

GPUlayer ran only jobs that finish: commands, functions and distributed training. Users also
run things that must stay up and answer requests: an inference server, a notebook, a vector
database next to a GPU. Today they keep those out of GPUlayer, so the same GPUs are managed
twice. Kubernetes, Ray Serve, dstack and Modal support services; Slurm does not.

## Decision

1. **A third workload type, `ServiceJob`** (`kind: "service"`, workload type `service`): a
   command (and optional source bundle) that is expected to keep running. `replicas` (1 to
   64) copies are placed together on distinct workers, like a distributed group, and each
   holds a task slot (and its GPUs) for as long as it runs.
2. **Ports.** With `port: true` (the default) the worker picks a free TCP port for each
   replica and passes it as `$PORT`. When the port accepts connections, the worker reports
   the replica's endpoint (`<worker advertise address>:<port>`) to the control plane; it
   appears in `JobStatus.tasks[].endpoint` and `gpulayer job status`.
3. **Restarts on the worker.** When a replica's process exits, the worker starts it again
   (`restart: "always"`, or `"on_failure"` for a non-zero exit only) after a back-off of 1 s
   doubling to 30 s, up to `max_restarts` times (unlimited by default). A replica that exits
   for good ends its task; any replica ending fails or completes the service (like a
   distributed group).
4. **Worker loss.** If a worker is lost, the whole service is placed again, up to
   `retry.max_attempts` (services default to 10 attempts on worker loss). Healthy replicas
   restart with it; per-replica replacement is a later step.
5. **Stopping.** Cancelling the job stops every replica (`gpulayer job cancel`). Services have
   no timeout unless one is set.
6. **Not in this slice:** a load balancer or public ingress, rolling updates, autoscaling,
   and health checks beyond "the port accepts connections". Clients connect to the reported
   endpoints directly, so they must reach the workers' advertise addresses.

## Consequences

- One GPU pool serves batch jobs and services; the scheduler's capability matching and
  device reservation apply to both.
- Endpoints are worker addresses: fine inside a cluster or private network, not a public
  entry point.
- The control plane and worker protocol gain a task endpoint report
  (`POST /v1/worker/tasks/{id}/endpoint`) and a `tasks.endpoint` column, added in place.
