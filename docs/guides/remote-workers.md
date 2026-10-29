# Remote workers and private networking

For most setups, [`gpulayer up` and `gpulayer join`](self-hosting.md) are all you need. This
page describes the lower-level commands and the protocol underneath them.

## Setup

```bash
# control-plane host
gpulayer server init                      # writes token HASHES to $GPULAYER_HOME/server-credentials.json
gpulayer server start --host 0.0.0.0 --port 8470 --tls-cert cert.pem --tls-key key.pem

# each worker machine (needs network reachability to the control plane only)
export GPULAYER_JOIN_TOKEN=glj_...
gpulayer --url https://cp.internal:8470 --ca-file ca.pem worker start \
    --name gpu-01 --label pool=a100 --python /opt/envs/torch/bin/python \
    --advertise-address 10.0.0.21         # only needed for multi-node distributed jobs
```

`server start` refuses plaintext HTTP on a non-loopback address unless
`--allow-insecure-remote` is passed (e.g. behind a TLS-terminating proxy you control).

## Protocol (v1, `X-GpuLayer-Protocol: 1`)

Workers poll the control plane, which never opens a connection to a worker. Workers can
therefore sit behind NAT.

| Call | Purpose |
|---|---|
| `POST /v1/workers/register` | join token → worker id + per-worker session token |
| `POST /v1/worker/heartbeat` | renew leases; response carries `cancel_task_ids`, `revoked` |
| `POST /v1/worker/acquire` | next assignment (idempotent until started) |
| `POST /v1/worker/tasks/{id}/start · /logs · /rendezvous · /finish` | task lifecycle, ownership-checked |
| `GET/PUT /v1/artifacts` | fetch inputs / upload outputs (SHA-256 verified) |
| `POST /v1/worker/deregister` | graceful exit |

Client endpoints (`/v1/jobs...`, `/v1/workers`) require the client token. A worker session
token cannot call client endpoints, and the join token can only register.

## Leases, loss and duplicates

- Heartbeat interval 2 s, lease TTL 20 s (defaults, `ControlPlaneConfig`).
- A task whose lease expires becomes `lost`; the worker is marked `offline`.
- A lost task may still be running. If the worker reconnects, its next heartbeat tells it to
  kill the task, and a late result is recorded as an event and otherwise ignored.
- Independent jobs are retried only if `retry.max_attempts > 1` and the reason is listed in
  `retry.retry_on` (default `worker_lost`). A retry can repeat side effects, because GPUlayer
  gives at-least-once execution at best and never exactly-once.

## Artifacts across machines

Workers never read the submitter's filesystem. Source bundles and inputs are uploaded to the
control plane and fetched by digest, and outputs come back the same way, so workers need no
shared filesystem.

## Validation status

The HTTP path is tested with a worker in a separate OS process and separate state directory on
the same host (`tests/integration/test_http_remote.py`). A worker on a second physical machine
runs the same code, but no automated test has validated that setup yet.
