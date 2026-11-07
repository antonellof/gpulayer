# Remote workers and private networking

Most setups only need [`gpulayer up` and `gpulayer join`](self-hosting.md). This page covers the
lower-level commands and the worker protocol.

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

`server start` refuses plain HTTP on a non-loopback address unless given
`--allow-insecure-remote` (for example behind your own TLS-terminating proxy).

## Protocol (v1, `X-GpuLayer-Protocol: 1`)

Workers poll the control plane, which never connects to them, so workers can sit behind NAT.
The join token can only register, and the per-worker session token it returns cannot call
client endpoints. Calls: [HTTP API](../reference/http-api.md#worker-endpoints).

## Leases, loss and duplicates

- Defaults (`ControlPlaneConfig`): heartbeat every 2 s, lease TTL 20 s. A heartbeat gives up
  after 5 s and the next uses a new connection, so a server restarting at the same address
  keeps its workers' leases. A long-poll acquire gives up 10 s after its wait.
- A task whose lease expires becomes `lost` and its worker `offline`. If the worker
  reconnects, it is told to kill the task; a late result is recorded as an event only.
- Jobs retry only if `retry.max_attempts > 1` and the reason is in `retry.retry_on` (default
  `worker_lost`; also `preempted` and `failed`). A distributed job also needs `resume`, and
  restarts as a group from its newest checkpoint
  ([ADR-0025](../decisions/ADR-0025-keeping-checkpoints-and-restarting-groups.md)). Execution is
  at-least-once at best, so retries can repeat side effects.

## Artifacts across machines

Source bundles, inputs and outputs go through the control plane by SHA-256 digest. Workers
never read the submitter's filesystem and need no shared filesystem.

## Validation status

Tested with a worker in a separate OS process and state directory on the same host
(`tests/integration/test_http_remote.py`). A worker on a second physical machine has not been
validated by an automated test.
