# Architecture overview

GPUlayer is a modular monolith: one Python package that runs in one process (embedded mode) or
as a served control plane with remote workers. See
[ADR-0001](../decisions/ADR-0001-control-plane-and-deployment-model.md).

```
src/gpulayer/
├── contracts/models.py      versioned pydantic contracts (API_VERSION gpulayer.dev/v1alpha1)
├── control_plane/
│   ├── service.py           job lifecycle, worker registry, leases, coordination, reconcile
│   ├── scheduler.py         pure, deterministic, explainable placement
│   └── provisioning.py      provider-backed capacity: offers, bootstrap tokens, release
├── worker/
│   ├── agent.py             heartbeat / acquire / prepare / execute / report / clean up
│   └── channel.py           WorkerChannel protocol + in-process implementation
├── runtime/
│   ├── process.py           process-group execution, timeout, cancel, log streaming
│   ├── function_runner.py   stdlib-only runner for FunctionJob (copied into the workspace)
│   └── distributed.py       rank planning, direct and torchrun launch
├── gpu/discovery.py         nvidia-smi, Apple Silicon, rocm-smi; GPU frameworks; torch probe
├── gpu/support.py           explicit support matrix consulted by the scheduler
├── providers/               ProviderAdapter interface, FakeProvider (simulated)
├── storage/sqlite.py        metadata store (jobs, tasks, workers, events, logs, instances, costs)
├── artifacts/store.py       content-addressed artifact store, deterministic source bundles
├── api/                     stdlib HTTP server, credentials, HTTP clients
├── sdk/                     Client, Job, builders, embedded control plane
└── cli/main.py              `gpulayer` CLI
```

The Trainer, `launch()`, `@gl.remote` and the CLI all go through the SDK `Client`, which talks
to the control plane in-process (embedded) or over HTTPS (server). Workers poll the control
plane, so they can sit behind NAT. The scheduler is a pure function with no I/O, which keeps
placement deterministic and testable.

## Data model

- **Job**: one `JobSpec` submission. States:
  `queued → [provisioning] → scheduled → running → succeeded | failed | cancelled | timed_out`.
  Validation is synchronous at submission; a retry moves the job back to `queued`. Transitions
  not in `JOB_TRANSITIONS` (`contracts/models.py`) are rejected.
- **Task**: one unit of work for one worker. An independent job has one task per attempt; a
  distributed job has one per node, all created in one transaction at placement. States:
  `assigned → running → succeeded | failed | cancelled | timed_out | lost`.
- **Worker**: reported `WorkerCapabilities`, a heartbeat-derived state (`idle`, `busy`,
  `draining`, `offline`) and a revocable session credential. Worker state is separate from job
  state, so a worker can recover after its job failed.
- **Events and logs**: append-only, sequence-numbered, queryable per job.
- **Artifacts**: immutable bytes addressed by SHA-256 (`artifact://sha256/<hex>`).

## Request flow (independent job)

1. `submit` validates the spec, checks referenced artifacts exist and persists the job (`queued`).
2. A scheduling pass builds `WorkerView`s (capabilities + running tasks) and calls
   `scheduler.schedule()`. The decision, with per-worker reasons, is stored.
3. Placement creates a task with a lease; the worker's next `acquire` returns it.
4. The worker calls `start`, stages the source bundle and inputs (verifying SHA-256), builds an
   allow-listed environment, resolves secret references locally and runs the process group.
5. Logs are batched to the control plane with secrets redacted. Heartbeats renew leases and
   deliver cancellation.
6. Outputs are uploaded as artifacts, `finish` reports the outcome and a `JobResult` is persisted.

`reconcile()` (every second on the server, opportunistically in embedded mode) expires worker
heartbeats and task leases, enforces a deadline backstop and advances provisioning.

## Deployment shapes

| Shape | Control plane | Workers | Auth |
|---|---|---|---|
| Embedded (default) | in the SDK/CLI process, SQLite in `$GPULAYER_HOME` | in-process thread or `gpulayer worker start` sharing the same home | OS user owning `$GPULAYER_HOME` |
| Self-hosted | `gpulayer server start` (HTTP/HTTPS) | `gpulayer --url ... worker start` anywhere reachable | client token, join token, per-worker session token |
| Kubernetes | the [Helm chart](../guides/kubernetes.md): server StatefulSet with a persistent volume | CPU and NVIDIA GPU worker Deployments | same as self-hosted; tokens and TLS certificate in Secrets |
| Hosted | planned | | |

In self-hosted mode the network carries job specs, logs, events, artifacts and worker
capabilities. Job specs include `env` values, so pass secrets as secret references; their values
stay on the workers. Nothing goes to a GPUlayer-operated service, and there is no telemetry.
