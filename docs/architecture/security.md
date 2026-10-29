# Security and trust model

**Scope today: single user or a mutually trusting team.** Public or multi-tenant execution of
untrusted code is out of scope until a dedicated threat-model and isolation review (ADR-0009).

## Trust boundaries

| Boundary | Mechanism |
|---|---|
| Client → control plane | bearer *client token* (hash stored, constant-time compare); TLS required off-loopback |
| Worker → control plane (join) | shared *join token*, valid only for `POST /v1/workers/register` |
| Worker → control plane (session) | per-worker random session token; every task call checks ownership |
| Provisioned worker | single-use token bound to one instance record; identity fields taken from GpuLayer's record |
| Control plane → workload | job spec, source bundle, inputs; secrets as *references* only |
| Workload → host | OS process group under the worker's user — **not a sandbox** |
| Embedded mode | anyone who can write `$GPULAYER_HOME` (created 0700) controls it |

Tokens are generated with `secrets.token_urlsafe(32)`, printed once by `server init`, and stored
only as SHA-256. Revoking a worker (`gpulayer worker revoke`) invalidates its session and loses
its active tasks.

## Secrets

- `SecretRef(name, source="env", key)` is resolved **on the worker** from its environment.
  The value never enters the job record, the database or the wire.
- Secret values are redacted (`***`) from streamed logs and function error messages.
- Workloads receive an **allow-listed** environment (`PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`,
  `TMPDIR`, `TERM`) plus explicit `env`/`secrets` — worker credentials are not inherited.
- `env` values are *not* secret; they are stored and transmitted in the job spec.

## Validation

- Strict pydantic models (`extra="forbid"`), reserved env names, safe relative paths for
  inputs and entrypoints, zip-slip/symlink-safe bundle extraction, SHA-256 verification of
  every artifact, request size limits, protocol-version header.
- No telemetry. Logs from the HTTP server never include headers or tokens.

## Known gaps (tracked in the implementation plan)

- One client role: any client token can cancel any job and revoke any worker.
- The join token is shared and has no expiry; no rate limiting.
- Artifact downloads are allowed for any authenticated worker, not only for its tasks' artifacts.
- `cpu_cores`/`memory_gb` are not enforced as resource limits; timeouts are.
- No mTLS/OIDC, audit log export, artifact retention or garbage collection yet.
- Provider API calls currently run while the metadata transaction lock is held.

## Reporting vulnerabilities

Until a security policy is published, please report privately to the maintainers rather than
opening a public issue.
