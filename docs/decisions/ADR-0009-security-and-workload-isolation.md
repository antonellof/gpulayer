# ADR-0009: Security and workload isolation

- **Status:** Accepted (milestone 1) for **trusted single-user or single-team deployments only**. Multi-tenant isolation requires a separate review and is not supported.
- **Date:** 2026-10-29

## Context

Running submitted code on someone else's machine is a security boundary. Subprocesses and
containers alone do not make hostile code safe. Secrets must not leak into job records or
logs.

## Decision

- **Execution model** (`runtime/process.py`, `worker/agent.py`): each task runs as a
  subprocess of the worker, **as the worker's OS user**, in its own session/process
  group, inside a per-task workspace under the worker's workdir. Cancellation and timeout
  kill the whole group (SIGTERM, a 10 s grace period, then SIGKILL). Leftover group
  members are killed after a normal exit.
- **Environment allowlist:** only `PATH, HOME, USER, LANG, LC_ALL, TMPDIR, SYSTEMROOT,
  TERM` are inherited from the worker. Everything else comes from `spec.env`, resolved
  secrets or `GPULAYER_*` variables. Credentials in the worker's environment do not leak
  into jobs by default.
- **Secrets:** `SecretRef(name, source="env", key)` is resolved **on the worker** from
  its own environment. Values never enter the job record, events or the control plane. A
  missing secret fails the task. Secret values are redacted (`***`) from log lines and
  function error text before upload.
- **Input validation:** strict pydantic models (`extra="forbid"`), relative-path checks,
  zip-slip-safe extraction, digest verification, and size caps on HTTP bodies.
- **Network:** see ADR-0003. Plaintext is refused off loopback by default, and
  credentials are role-separated.
- No telemetry exists, so none is sent.

## Explicitly not provided

- A sandbox: jobs can read whatever the worker user can read, use the network, and
  consume resources beyond the declared `cpu_cores` and `memory_gb`, which are scheduling
  hints, not enforced limits.
- Tenant separation between submitters, and per-job resource cgroups.
- Protection against a malicious worker, which sees code, inputs and resolved secrets.

## Alternatives considered

- **Containers (rootless Podman or Docker) per job:** better filesystem and resource
  isolation, but a heavy dependency and unavailable for MPS. Proposed as an opt-in
  executor.
- **gVisor, Firecracker or VMs:** required before any public multi-tenant offering.

## Consequences and trade-offs

- Simple and portable (Linux/macOS), with GPU access that just works.
- Operators must run workers under a dedicated low-privilege user and treat everyone who
  can submit as able to execute code there.

## Security and operational implications

- Recommended: a dedicated worker user, a private network, TLS, and per-team control
  planes.
- Redaction is substring-based. Transformed secrets (base64, split strings) are not
  caught.

## Follow-up

- Threat-model document and review before team or hosted multi-tenancy.
- Container executor, cgroup limits, and a no-network option.
- Secret sources beyond env (files, Vault or cloud secret managers) with scoped access.
