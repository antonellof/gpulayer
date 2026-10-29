# Architecture Decision Records

ADRs record the significant architectural decisions behind GpuLayer: what was decided,
why, which alternatives were rejected, and what the decision costs us.

**Status vocabulary**

- **Accepted (milestone 1):** implemented in the current code and covered by tests.
- **Proposed:** a direction under evaluation. It is not policy and must not be described
  as available.
- **Superseded by ADR-XXXX:** replaced; kept for history.

Several ADRs are split: the implemented part is accepted and named follow-ups stay
proposed. Each ADR says which part is which.

## Index

| # | Title | Status | Summary |
|---|-------|--------|---------|
| [0001](ADR-0001-control-plane-and-deployment-model.md) | Control plane and deployment model | Accepted (milestone 1); hosted mode Proposed | One modular-monolith `ControlPlane`, used embedded (SQLite file) or served over authenticated HTTP. |
| [0002](ADR-0002-job-contracts-and-packaging.md) | Job contracts and packaging | Accepted (milestone 1); container images Proposed | Versioned pydantic contracts; code ships as content-addressed source bundles; functions take JSON args, never pickle. |
| [0003](ADR-0003-worker-protocol-and-authentication.md) | Worker protocol and authentication | Accepted (milestone 1); mTLS/OIDC Proposed | HTTP+JSON polling; separate client, join, per-worker session and single-use instance tokens; TLS required off loopback. |
| [0004](ADR-0004-scheduling-and-placement-policy.md) | Scheduling and placement policy | Accepted (milestone 1) | A pure, deterministic, explainable scheduler with gang scheduling, explicit fallback and reservations. |
| [0005](ADR-0005-gpu-capabilities-and-runtime-profiles.md) | GPU capabilities and runtime profiles | Accepted (milestone 1); runtime profiles Proposed | Capability model, isolated torch probe, explicit support matrix. |
| [0006](ADR-0006-apple-metal-mps-boundaries.md) | Apple Metal / MPS boundaries | Accepted (milestone 1) | MPS supports independent jobs only; unified memory is explicit; distributed MPS is unsupported. |
| [0007](ADR-0007-distributed-launch-and-rendezvous.md) | Distributed launch and rendezvous | Accepted (milestone 1, single host); multi-machine Proposed | torchrun with static rendezvous published by the rank-0 worker; fail-fast; no distributed retry. |
| [0008](ADR-0008-storage-and-artifact-management.md) | Storage and artifact management | Accepted (milestone 1); shared DB/object store Proposed | SQLite (WAL) metadata plus a local content-addressed artifact store behind interfaces. |
| [0009](ADR-0009-security-and-workload-isolation.md) | Security and workload isolation | Accepted (milestone 1) for trusted single-team use | Process groups, an env allowlist and secret references with log redaction. This is not a sandbox. |
| [0010](ADR-0010-provider-adapters-and-provisioning.md) | Provider adapters and provisioning | Accepted (interface + FakeProvider); first real provider Proposed | Offers are expressed as capability templates; provisioning flow is tested only against a simulated provider. |
| [0011](ADR-0011-cost-accounting-and-retry-semantics.md) | Cost accounting and retry semantics | Accepted (milestone 1) | Estimates and charges are distinct record kinds; delivery is at-least-once; retries are opt-in. |
| [0012](ADR-0012-licensing-and-project-governance.md) | Licensing and project governance | Proposed | Compares Apache-2.0, MIT, MPL-2.0 and AGPL-3.0. Undecided. |
| [0013](ADR-0013-website-and-documentation-stack.md) | Website and documentation stack | Proposed | Compares MkDocs-Material, Docusaurus and Astro Starlight. Undecided. |

## Template

```markdown
# ADR-NNNN: Title

- **Status:** Proposed | Accepted (milestone N) | Superseded by ADR-XXXX
- **Date:** YYYY-MM-DD

## Context
The problem, the constraints and the forces involved.

## Decision
What we do. Name the modules that implement it.

## Alternatives considered
Each option and why it was not chosen.

## Consequences and trade-offs
What gets easier, what gets harder, and known gaps.

## Security and operational implications

## Follow-up
Concrete next steps, each with the condition that would change this decision.
```
