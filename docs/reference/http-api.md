# HTTP API

The server speaks JSON over HTTP(S). Every request except `GET /v1/health` needs
`Authorization: Bearer <token>` and `X-GpuLayer-Protocol: 1`. Errors look like
`{"error": {"code": "...", "message": "..."}}` with codes `invalid_request` (400),
`unauthorized` (401), `forbidden` (403), `not_found` (404), `conflict` (409) and
`protocol_mismatch` (400).

## Client endpoints (client token)

| Method and path | Body / query | Returns |
|---|---|---|
| `POST /v1/jobs` | `JobSpec` | `JobStatus` |
| `GET /v1/jobs?limit=&state=` | | list of `JobStatus` |
| `GET /v1/jobs/{id}` | | `JobStatus` |
| `POST /v1/jobs/{id}/cancel` | | `JobStatus` |
| `GET /v1/jobs/{id}/result` | | `JobResult` |
| `GET /v1/jobs/{id}/logs?after=` | | list of `LogLine` |
| `GET /v1/jobs/{id}/events?after=` | | list of `ExecutionEvent` |
| `GET /v1/workers` | | list of `WorkerStatus` |
| `POST /v1/workers/{id}/revoke` | | |
| `PUT /v1/artifacts` | raw bytes; headers `X-Artifact-Name`, `X-Artifact-Kind` | `ArtifactReference` |
| `GET /v1/artifacts/{sha256}` | | raw bytes |

## Worker endpoints

| Method and path | Token | Purpose |
|---|---|---|
| `POST /v1/workers/register` | join token (or single-use instance token) | returns worker id and session token |
| `POST /v1/worker/heartbeat` | session | renews leases; returns tasks to cancel and revocation |
| `POST /v1/worker/acquire` | session | next assignment or `null` |
| `POST /v1/worker/tasks/{id}/start` | session | mark running |
| `POST /v1/worker/tasks/{id}/rendezvous` | session | node rank 0 publishes `{addr, port}` |
| `POST /v1/worker/tasks/{id}/logs` | session | `{lines: [[stream, line], ...]}` |
| `POST /v1/worker/tasks/{id}/finish` | session | `TaskOutcome` |
| `POST /v1/worker/deregister` | session | graceful exit |

Workers can also upload and download artifacts with their session token. The models are
defined in `src/gpulayer/contracts/models.py`.
