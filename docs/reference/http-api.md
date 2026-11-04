# HTTP API

JSON over HTTP(S). Every request except `GET /v1/health` needs `Authorization: Bearer <token>`
and `X-GpuLayer-Protocol: 1`. Errors are `{"error": {"code": "...", "message": "..."}}` with
codes `invalid_request` (400), `unauthorized` (401), `forbidden` (403), `not_found` (404),
`conflict` (409), `rate_limited` (429) and `protocol_mismatch` (400).

Client endpoints accept `admin` and `submitter` tokens. Cancelling needs an admin token or the
submitting token; revoking a worker needs admin. Workers can download only artifacts of their
current tasks. Models: `src/gpulayer/contracts/models.py`.

## Dashboard and metrics

| Method and path | Token | Returns |
|---|---|---|
| `GET /ui/`, `GET /` | none (the page asks for one) | web dashboard |
| `GET /metrics` | `metrics`, `submitter` or `admin`; no protocol header | Prometheus text |
| `GET /v1/whoami` | client | `{principal, role}` |
| `GET /v1/metrics/history?minutes=` | client | worker resource samples and running/waiting jobs over time (max 24 h) |

## Client endpoints (client token)

| Method and path | Body / query | Returns |
|---|---|---|
| `POST /v1/jobs` | `JobSpec` | `JobStatus` |
| `POST /v1/jobs/batch` | `{"jobs": [JobSpec, ...]}` (1 to 1000) | `{"job_ids": [...]}`; one transaction, all or none |
| `GET /v1/jobs?limit=&state=` | | list of `JobStatus` |
| `GET /v1/jobs/{id}?wait=` | `wait` (seconds, max 30): hold until terminal | `JobStatus` |
| `POST /v1/jobs/{id}/cancel` | | `JobStatus` |
| `GET /v1/jobs/{id}/result` | | `JobResult` |
| `GET /v1/jobs/{id}/logs?after=` | | list of `LogLine` |
| `GET /v1/jobs/{id}/events?after=` | | list of `ExecutionEvent` |
| `GET /v1/workers` | | list of `WorkerStatus` |
| `POST /v1/workers/{id}/revoke` | | (admin) |
| `POST /v1/admin/gc` | `{older_than_seconds, delete_jobs, dry_run}` | counts removed (admin) |
| `PUT /v1/artifacts` | raw bytes; headers `X-Artifact-Name`, `X-Artifact-Kind` | `ArtifactReference` |
| `GET /v1/artifacts/{sha256}` | | raw bytes |

## Worker endpoints

All but `register` use the session token, which can also upload and download artifacts.

| Method and path | Purpose |
|---|---|
| `POST /v1/workers/register` | join token (or single-use instance token); returns worker id and session token |
| `POST /v1/worker/heartbeat` | `{draining, metrics?}`: renews leases, stores an optional `WorkerMetrics` sample; returns tasks to cancel and revocation |
| `POST /v1/worker/acquire` | `{wait}` (seconds, max 30): long poll for the next assignment, or `null` |
| `POST /v1/worker/tasks/{id}/start` | mark running (before artifacts are fetched and the environment is built) |
| `POST /v1/worker/tasks/{id}/executing` | the process has started: the control plane's timeout backstop counts `timeout_seconds` from here (added after 0.9.1; older workers do not call it and get the longer backstop) |
| `POST /v1/worker/tasks/{id}/rendezvous` | node rank 0 publishes `{addr, port}` |
| `POST /v1/worker/tasks/{id}/logs` | `{lines: [[stream, line], ...]}`; optional third element: global rank |
| `POST /v1/worker/tasks/{id}/finish` | `TaskOutcome` |
| `POST /v1/worker/deregister` | graceful exit |
