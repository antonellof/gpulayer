# Security and trust model

GPUlayer targets a single user or a team whose members trust each other. Public or multi-tenant
execution of untrusted code is out of scope until a dedicated threat model and isolation review
exist (ADR-0009).

## Trust boundaries

| Boundary | Mechanism |
|---|---|
| Client → control plane | named bearer token with role `admin` or `submitter` (hash stored, constant-time compare); TLS required off-loopback |
| Worker → control plane (join) | named *join token*, valid only for `POST /v1/workers/register`; can expire |
| Worker → control plane (session) | per-worker random session token; every task call checks ownership |
| Provisioned worker | single-use token bound to one instance record; identity fields taken from GPUlayer's record |
| Control plane → workload | job spec, source bundle, inputs; secrets as *references* only |
| Workload → host | OS process group under the worker's user; this is not a sandbox |
| Embedded mode | anyone who can write `$GPULAYER_HOME` (created 0700) controls it |

Tokens come from `secrets.token_urlsafe(32)`, are printed once by `server init` and stored only
as SHA-256. `gpulayer worker revoke` invalidates a worker's session and loses its active tasks.

## Secrets

- The worker resolves `SecretRef(name, source="env", key)` from its own environment. The value
  never enters the job record or database and is never sent over the network.
- Secret values are redacted (`***`) from streamed logs and function error messages.
- Workloads get an allow-listed environment (`PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`,
  `TMPDIR`, `TERM`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, `CUDA_HOME`, `CUDA_PATH`,
  `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES`, `SSL_CERT_FILE`, `SSL_CERT_DIR`,
  `REQUESTS_CA_BUNDLE`) plus the job's `env` and `secrets`, not the worker's credentials. Proxy
  and package-index URLs are not passed, because they can embed credentials.
- `env` values are stored and sent in the job spec, so they are not secret.

## Validation

- Pydantic models reject unknown fields (`extra="forbid"`), reserved environment names and
  unsafe input or entrypoint paths.
- Bundle extraction refuses absolute paths, `..` and symlinks; every artifact is verified
  against its SHA-256.
- The API limits request sizes and requires the protocol-version header.
- No telemetry. HTTP server logs never include headers or tokens.

## Tokens and roles

| Role | Can |
|---|---|
| `admin` | everything a client can do, plus revoke workers, cancel any job and run maintenance |
| `submitter` | submit jobs, read all jobs and workers, cancel its own jobs |
| `join` | register workers, nothing else |
| `metrics` | read `GET /metrics`, nothing else (for Prometheus) |

```bash
gpulayer token create alice --role submitter
gpulayer token create ci-runner --role join --ttl 12h
gpulayer token list
gpulayer token revoke alice
```

These edit `$GPULAYER_HOME/server-credentials.json` on the server (hashes only); a running
server picks up changes on its next request. After 20 failed authentications in a minute, an
address gets HTTP 429 for the rest of that minute.

Workers can download only artifacts used by a task currently assigned to them (source bundle and
inputs). A revoked worker is rejected on its next heartbeat and its agent stops.

## Known gaps (tracked in the implementation plan)

- Any client token can read every job and its logs; there is no per-team isolation.
- `cpu_cores`/`memory_gb` are not enforced as resource limits; timeouts are.
- No mTLS/OIDC or audit log export yet.
- Rate limiting covers failed authentication only, per client address, in one server process.

## Reporting vulnerabilities

Until a security policy is published, report privately to the maintainers, not in a public issue.

## AI agents

Agents use `gpulayer mcp`, the CLI or the HTTP API with the permissions of their token (or the
local user in embedded mode). Give them expiring `submitter` tokens, or
`gpulayer mcp --read-only` for monitoring; see
[AI and coding agents](../guides/ai-agents.md#security).
