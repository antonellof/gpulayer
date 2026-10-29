# ADR-0013: Website and documentation stack

- **Status:** Proposed (no decision)
- **Date:** 2026-10-29

## Context

`site/` will host the public website and documentation. The source of truth for content
is Markdown in `docs/`, which already exists and is reviewed with code changes. The site
must only advertise validated capabilities (CLAUDE.md §15), so it should render the same
files rather than maintain a parallel copy.

## Options

| Stack | Pros | Cons |
|---|---|---|
| MkDocs + Material | Python-native, fits the project toolchain, renders `docs/` with little change, good search, versioning via `mike`. API reference via mkdocstrings. | Less flexible for marketing-style landing pages. |
| Docusaurus | Mature versioning, MDX components, large ecosystem. | Node/React toolchain alongside Python; MDX may diverge from plain Markdown. |
| Astro Starlight | Fast static output, good DX, flexible landing pages, Markdown-first. | Node toolchain; younger ecosystem; Python API reference needs extra tooling. |

Evaluation criteria:

- Renders `docs/` unchanged, or nearly.
- Versioned docs per release.
- Generated Python SDK and contract reference.
- Static hosting without server-side code, accessibility, offline search.
- Contributor friction (one toolchain is better than two).

## Decision

None yet. `site/README.md` documents the planned page map so content can be written now
in `docs/` regardless of the stack.

## Follow-up

- Prototype the page map with MkDocs-Material and one Node option, then decide.
- Add a CI check that site pages build from `docs/`.
