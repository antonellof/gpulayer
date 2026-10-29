# ADR-0001: Control plane and deployment model

- **Status:** Accepted (milestone 1) for the embedded and self-hosted modes. The hosted mode is Proposed.
- **Date:** 2026-10-29

## Context

GPUlayer must work on a laptop with no account and no service, and must also coordinate
many remote, cloud and distributed workers. CLAUDE.md asks for logical boundaries without
premature microservices. It also asks that a "hosted" control plane never imply that user
data is sent to a GPUlayer-operated service.

## Decision

- There is a single `ControlPlane` class (`src/gpulayer/control_plane/service.py`). It
  owns the job lifecycle, worker registry, leases, scheduling passes, distributed
  coordination and provisioning (through `control_plane/provisioning.py`). Scheduling is a
  separate pure module (`control_plane/scheduler.py`; see ADR-0004).
- **Embedded mode.** The SDK (`sdk/client.py`) and CLI open `ControlPlane` in-process
  over `$GPULAYER_HOME/metadata.db` (`sdk/home.py`). Local workers talk to it through
  `EmbeddedChannel` (`worker/channel.py`). Several local processes, such as the CLI and
  `gpulayer worker start`, share the SQLite file. No network listener is opened.
- **Self-hosted mode.** `gpulayer server start` wraps the same object in
  `ControlPlaneServer` (`api/server.py`, stdlib HTTP). Clients use
  `HttpControlPlaneClient` and workers use `HttpWorkerChannel` (`api/client.py`).
- The worker agent (`worker/agent.py`) is identical in both modes; only the channel differs.
- Reconciliation, meaning lease and heartbeat expiry, deadlines and provisioning, runs
  opportunistically on API calls and every second in a server background thread.
- **Hosted mode (Proposed).** This would be the same contracts operated as a managed
  service. Nothing hosted exists. No telemetry is sent anywhere.

## Alternatives considered

- **Separate services** (API, scheduler, registry, coordinator): this adds deployment and
  consistency complexity before any scale requires it.
- **Local-only runner first, server later:** this risks two divergent code paths, which
  CLAUDE.md forbids.
- **A Kubernetes/Ray/Slurm-based control plane:** this is a heavy dependency for the
  local case. It may become a provider or worker backend later.

## Consequences and trade-offs

- Local, remote and cloud jobs share one code path, and the tests exercise it in both modes.
- The single-process server is a single point of failure, and availability equals that
  of one host. There is no HA.
- SQLite limits write concurrency. All writes serialize on one lock per process
  (`storage/sqlite.py`), which is adequate for tens of workers but not for thousands.
- **Known gap:** some provider calls (`provision`, `terminate`) run while the SQLite
  transaction lock is held: `schedule_pending` calls `ProvisioningManager.request`, and
  `_set_job_state` calls `_finalize`, which calls `release_for_job`, all inside `store.tx()`. A slow
  provider API would stall the control plane. This must change before a real provider
  ships (ADR-0010).

## Security and operational implications

- Embedded mode trusts the OS user: anyone who can write `$GPULAYER_HOME` controls jobs.
  The directory is created with mode 0700.
- Served mode authenticates every route except `/v1/health` (ADR-0003).
- What reaches the control plane: job specs, logs, events, artifacts (source bundles,
  inputs, outputs) and worker capabilities. Secret *values* never do (ADR-0009).

## Follow-up

- Move provider I/O outside DB transactions (outbox or queued commands).
- Choose a shared database for multi-replica deployments (ADR-0008).
- Write the hosted-mode ADR, covering tenancy, data residency and isolation (ADR-0009).
