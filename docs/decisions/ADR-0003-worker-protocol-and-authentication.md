# ADR-0003: Worker protocol and authentication

- **Status:** Accepted (milestone 1). mTLS, OIDC/SSO and scoped RBAC are Proposed.
- **Date:** 2026-10-29

## Context

Workers may sit behind NAT on other networks, so they must be able to reach the control
plane without the control plane reaching them. Remote execution is a security boundary.
A credential that lets someone submit jobs must not also let them impersonate a worker,
and the reverse also holds.

## Decision

- **Transport:** HTTP/1.1 + JSON, stdlib only (`api/server.py`, `api/client.py`).
  Workers poll: they send heartbeats (default every 2 s) and `acquire` calls. All connections are
  outbound from the worker.
- **Protocol version:** every authenticated request carries `X-GpuLayer-Protocol: 1`, and
  a mismatch returns `protocol_mismatch`.
- **Credentials** (`api/auth.py`). They are never interchangeable, and only SHA-256 hashes
  are stored (`server-credentials.json`, mode 0600):
  - *client token* (`glc_…`): jobs, workers list/revoke, artifacts.
  - *join token* (`glj_…`): `POST /v1/workers/register` only.
  - *worker session token*: minted per worker at registration and stored hashed in the
    `workers` table. It is valid only for that worker's heartbeat, acquire, task
    start/logs/finish/rendezvous and deregister. Task ownership is enforced in
    `ControlPlane._owned_task`.
  - *instance token*: single-use and bound to one provisioned instance (ADR-0010).
    Workers that claim `origin=provider` must present it; a static join token is
    rejected for them.
- Token comparison uses `hmac.compare_digest` over hashes.
- **TLS:** `ControlPlaneServer` refuses to bind a non-loopback address without a
  cert/key, unless `--allow-insecure-remote` is given. TLS 1.2 is the minimum. Clients
  accept `--ca-file`.
- **Leases:** each assignment carries `lease_expires_at`. Heartbeats renew the leases of
  the worker's active tasks. Expiry marks the task `lost` and the worker `offline`
  (ADR-0011). Heartbeat responses carry `cancel_task_ids` (cancellations, lost tasks) and
  `revoked`.
- Revocation (`gpulayer worker revoke`) flips `revoked`. Subsequent calls fail with 403,
  and active tasks become `lost`.

## Alternatives considered

- **gRPC bidirectional streams:** lower latency push, but a heavier dependency and harder
  debugging. This can be revisited for log streaming at scale.
- **WebSocket push:** same trade-off. Polling was chosen for simplicity and
  NAT-friendliness.
- **mTLS per worker:** stronger identity, but needs a PKI and enrollment flow. Proposed
  as a hardening step.
- **A single shared API key:** rejected, because it collapses the client/worker
  boundary.

## Consequences and trade-offs

- Cancellation and assignment latency are bounded by the poll and heartbeat interval
  (seconds).
- Polling load grows linearly with the number of workers.
- **Known gaps:**
  - The join token is static and shared by every non-provider worker. Anyone who has it
    can register a worker and receive jobs. There is no per-worker enrollment or approval
    step yet.
  - `GET /v1/artifacts/<sha>` is allowed for *any* authenticated worker, not only one
    whose task references that artifact. Possession of a digest acts as a capability.
  - There is no rate limiting or request throttling. Body size caps exist (16 MiB JSON,
    2 GiB artifacts).
  - There is no audit log beyond lifecycle events, and tokens have no expiry or rotation
    API (`server init --rotate` replaces all of them).
  - One client role: any client token can cancel any job and revoke any worker.

## Security and operational implications

- Tokens are printed once at `server init`. Request logging never includes headers.
- Plaintext HTTP is only permitted on loopback by default.

## Follow-up

- Per-worker enrollment (one-time join codes, admin approval), token expiry and rotation.
- Scope artifact reads to the artifacts referenced by the worker's assigned tasks.
- Client roles and RBAC (submit, read, admin), rate limiting and audit events.
- Evaluate mTLS and OIDC for team and hosted deployments.
