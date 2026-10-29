# site/

This directory will hold GpuLayer's public website and documentation site.

- **Stack:** not chosen yet. See [ADR-0013](../docs/decisions/ADR-0013-website-and-documentation-stack.md)
  (Proposed), which compares MkDocs-Material, Docusaurus and Astro Starlight.
- **Content source:** Markdown under [`docs/`](../docs/). The site renders those files.
  It does not keep a second copy, so documentation is reviewed together with code changes.
- **Rule:** the site may only present capabilities that the docs classify as implemented,
  tested or hardware-validated. Simulated and planned features must be labelled as such.

## Planned page map

| Page | Source |
|---|---|
| Overview and architecture | `docs/architecture/overview.md` |
| Quickstart | `docs/getting-started.md` |
| Local execution (and execution modes) | `docs/architecture/execution-modes.md` |
| Remote workers and private networking | `docs/architecture/remote-workers.md` |
| Cloud execution and provider adapters | `docs/architecture/providers.md` |
| Distributed execution | `docs/architecture/distributed.md` |
| GPU compatibility and runtime profiles | `docs/architecture/gpu-compatibility.md` |
| Apple Metal / MPS boundaries | `docs/architecture/gpu-compatibility.md` + [ADR-0006](../docs/decisions/ADR-0006-apple-metal-mps-boundaries.md) |
| Security and trust model | `docs/architecture/security.md` |
| Cost accounting and scheduling | `docs/architecture/scheduling-and-cost.md` |
| Self-hosting and hosted control plane | `docs/architecture/remote-workers.md` + [ADR-0001](../docs/decisions/ADR-0001-control-plane-and-deployment-model.md) |
| Contributing and release process | `docs/development/contributing.md` |
| Architecture decisions | `docs/decisions/README.md` |

Nothing here is built or deployed yet.
