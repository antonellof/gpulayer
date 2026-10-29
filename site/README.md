# GPUlayer website and docs

The public website and documentation for GPUlayer: a landing page at `/` and the docs at
`/docs`. It is a Next.js 14 static site (App Router, TypeScript, Tailwind) that renders the
Markdown in [`../docs`](../docs) at build time. There is no second copy of the content: edit
the files in `docs/` and rebuild.

## Develop

```bash
cd site
npm install
npm run dev          # http://localhost:3000
```

## Build

```bash
npm run build        # static export to site/out/
npm start            # serve site/out locally
npm run lint
```

`npm run build` writes a static export (`output: "export"`, `trailingSlash: true`), so
`site/out/` is a plain folder of HTML, CSS and JS. `npm run dev` runs a normal Next.js server in
its own folder (`.next-dev`), so a build never breaks a running dev server. Host it on any static host (GitHub Pages, Netlify, Cloudflare
Pages, S3) or deploy the `site/` directory to Vercel.

## How content is wired

- **Navigation** comes only from [`docs/nav.json`](../docs/nav.json): sections of pages with a
  `slug` and `title`. Slug `index` is `/docs/`; `get-started/quickstart` is
  `/docs/get-started/quickstart/` (file `docs/get-started/quickstart.md`); a slug ending in
  `/README` maps to its directory (`decisions/README` is `/docs/decisions/`). Previous/next
  links follow the order in `nav.json`.
- **ADRs**: every `docs/decisions/ADR-*.md` is published at `/docs/decisions/<file-stem>/`
  without a sidebar entry; the decisions index links to them.
- **Missing pages**: a `nav.json` entry whose Markdown file does not exist yet renders a
  "This page is being written" placeholder instead of failing the build.
- **Links**: relative links between Markdown files (`../pytorch/trainer.md`,
  `ADR-0006-apple-metal-mps-boundaries.md`, `./concepts.md#anchor`) become site routes. Links
  to anything outside the generated docs (examples, source files, the README) point to the
  file on GitHub.
- **Diagrams**: fenced ` ```mermaid ` blocks render client-side with Mermaid and follow the
  light/dark theme. GitHub renders the same blocks natively.
- **Code**: other fenced blocks are highlighted with `prism-react-renderer` (plus a small
  shell grammar) and get a copy button.
- **Headings**: `h2`/`h3` feed the "On this page" list; ids match `rehype-slug`.

Source layout:

```text
site/src/
├── app/                    layout, landing page, docs routes, 404
├── components/docs/        markdown renderer, code block, mermaid, sidebar, TOC
├── components/landing/     terminal, quickstart, logo strip, layer stack, architecture diagram
├── components/site/        header, footer, logo, theme toggle
└── lib/docs.ts             nav.json loading, link resolution, heading extraction
```

The site has no analytics or telemetry. Text uses Geist Sans (the `geist` npm package) and code
uses JetBrains Mono, fetched from
Google Fonts at build time by `next/font` and self-hosted in the export.
