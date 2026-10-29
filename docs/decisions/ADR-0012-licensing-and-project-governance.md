# ADR-0012: Licensing and project governance

- **Status:** Proposed (no decision; `pyproject.toml` deliberately declares no license)
- **Date:** 2026-10-29

## Context

GPUlayer is intended to be open source. The license affects adoption by companies, cloud
providers' ability to offer it as a service, contributor expectations, and any future
commercial model (for example, a hosted control plane). No commercial model has been
decided, and CLAUDE.md asks not to pick one prematurely.

## Options

| License | Pros | Cons |
|---|---|---|
| Apache-2.0 | Permissive, explicit patent grant, standard in infra and ML (PyTorch is BSD, Kubernetes is Apache). | Allows proprietary forks and hosted resale without contribution back. |
| MIT | Simplest and most permissive. | No patent grant; same resale concern. |
| MPL-2.0 | File-level copyleft: modifications to GPUlayer files stay open while embedding stays easy. | Less familiar; some corporate policy friction. |
| AGPL-3.0 | Network copyleft discourages closed hosted forks. | Many companies ban it, which hurts adoption of a developer tool and its workers. |

Governance questions to settle alongside:

- Contributor terms: DCO sign-off vs a CLA (a CLA enables relicensing but adds friction).
- Maintainer model, decision process (ADRs), and security disclosure policy
  (`SECURITY.md`).
- Trademark policy for the name "GPUlayer".

## Decision

None yet. Until decided, the repository must not be published under an implied license.

## Consequences

- External contributions should wait for the license and contributor-terms decision.

## Follow-up

- Owner decision on commercial intent (hosted offering or not), then pick a license,
  add `LICENSE`, the contributor terms and `SECURITY.md`, and update `pyproject.toml`.
