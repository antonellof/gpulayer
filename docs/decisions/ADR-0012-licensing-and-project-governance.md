# ADR-0012: Licensing and project governance

- **Status:** Accepted (licence); contributor terms and governance Proposed
- **Date:** 2026-10-29, decided 2026-10-30

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

GPUlayer is licensed under the **Apache License 2.0** (owner decision, 2026-10-30). `LICENSE`
holds the text and `pyproject.toml` declares `license = "Apache-2.0"`. The package is
published on PyPI as `gpulayer` and the repository is public.

Apache-2.0 was chosen for adoption: it is the common licence for infrastructure that
companies run (Kubernetes, Ray, SkyPilot, PyTorch Lightning), and its explicit patent grant
matters for a project that may receive contributions from companies. It allows proprietary
forks and hosted resale; that is accepted.

## Consequences

- Anyone may use, modify and redistribute GPUlayer under Apache-2.0, including commercially.
- Contributions are accepted under the same licence (inbound = outbound) until contributor
  terms are decided.

## Follow-up

- Contributor terms (DCO sign-off or CLA), `SECURITY.md` with a disclosure contact, and a
  trademark policy for the name "GPUlayer" are still Proposed.
