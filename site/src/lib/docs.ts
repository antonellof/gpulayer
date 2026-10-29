/**
 * Docs content loader. Reads Markdown from ../docs at build time; navigation comes only
 * from docs/nav.json. Everything here runs on the server during `next build`.
 */
import fs from "node:fs";
import path from "node:path";
import GithubSlugger from "github-slugger";

export const REPO = "https://github.com/antonellof/gpulayer";
const DOCS_DIR = path.resolve(process.cwd(), "..", "docs");

export interface NavPage {
  slug: string;
  title: string;
}

export interface NavSection {
  title: string;
  pages: NavPage[];
}

export interface DocEntry {
  /** Route segments under /docs ([] for /docs itself). */
  route: string[];
  /** Path relative to docs/, e.g. "get-started/quickstart.md". */
  file: string;
  title: string;
  inNav: boolean;
}

export interface Heading {
  depth: number;
  text: string;
  id: string;
}

export function readNav(): NavSection[] {
  const raw = fs.readFileSync(path.join(DOCS_DIR, "nav.json"), "utf8");
  const data = JSON.parse(raw) as { sections: NavSection[] };
  return data.sections;
}

/** "index" -> [], "decisions/README" -> ["decisions"], "a/b" -> ["a", "b"]. */
export function slugToRoute(slug: string): string[] {
  if (slug === "index") return [];
  const parts = slug.split("/");
  if (parts[parts.length - 1] === "README") parts.pop();
  return parts;
}

export function routeHref(route: string[]): string {
  return route.length ? `/docs/${route.join("/")}/` : "/docs/";
}

function firstH1(file: string): string | null {
  const full = path.join(DOCS_DIR, file);
  if (!fs.existsSync(full)) return null;
  const m = fs.readFileSync(full, "utf8").match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : null;
}

let cache: DocEntry[] | null = null;

/** Every page the site generates: nav pages plus all ADRs under docs/decisions/. */
export function allDocs(): DocEntry[] {
  if (cache) return cache;
  const entries: DocEntry[] = [];
  for (const section of readNav()) {
    for (const page of section.pages) {
      entries.push({
        route: slugToRoute(page.slug),
        file: `${page.slug}.md`,
        title: page.title,
        inNav: true,
      });
    }
  }
  const decisions = path.join(DOCS_DIR, "decisions");
  if (fs.existsSync(decisions)) {
    for (const f of fs.readdirSync(decisions).sort()) {
      if (!/^ADR-.*\.md$/.test(f)) continue;
      const stem = f.replace(/\.md$/, "");
      entries.push({
        route: ["decisions", stem],
        file: `decisions/${f}`,
        title: firstH1(`decisions/${f}`) ?? stem,
        inNav: false,
      });
    }
  }
  cache = entries;
  return entries;
}

export function findDoc(route: string[]): DocEntry | undefined {
  const key = route.join("/");
  return allDocs().find((d) => d.route.join("/") === key);
}

export function navNeighbors(entry: DocEntry): { prev?: DocEntry; next?: DocEntry } {
  const nav = allDocs().filter((d) => d.inNav);
  const i = nav.findIndex((d) => d.file === entry.file);
  if (i < 0) return {};
  return { prev: nav[i - 1], next: nav[i + 1] };
}

export function readDoc(entry: DocEntry): string | null {
  const full = path.join(DOCS_DIR, entry.file);
  return fs.existsSync(full) ? fs.readFileSync(full, "utf8") : null;
}

/** Normalise a repo-relative path ("docs/../examples/x" -> "examples/x"). */
function normalise(p: string): string {
  const out: string[] = [];
  for (const part of p.split("/")) {
    if (part === "..") out.pop();
    else if (part && part !== ".") out.push(part);
  }
  return out.join("/");
}

/**
 * Map a link found in docs/<currentFile> to a site route or a GitHub URL.
 * Absolute URLs, mailto: and in-page anchors are returned unchanged.
 */
export function resolveHref(href: string, currentFile: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("#") || href.startsWith("//")) {
    return href;
  }
  const [rawPath, anchor] = href.split("#");
  const hash = anchor ? `#${anchor}` : "";
  const repoPath = rawPath.startsWith("/")
    ? normalise(rawPath.slice(1))
    : normalise(path.posix.join("docs", path.posix.dirname(currentFile), rawPath));
  if (repoPath.startsWith("docs/") && repoPath.endsWith(".md")) {
    const file = repoPath.slice("docs/".length);
    const entry = allDocs().find((d) => d.file === file);
    if (entry) return routeHref(entry.route) + hash;
  }
  if ((repoPath === "docs" || repoPath.startsWith("docs/")) && !path.posix.extname(repoPath)) {
    // A link to a docs directory resolves to its README.md / index.md page, if generated.
    const dir = repoPath.slice("docs/".length);
    const candidates = dir ? [`${dir}/README.md`, `${dir}/index.md`] : ["index.md"];
    const entry = allDocs().find((d) => candidates.includes(d.file));
    if (entry) return routeHref(entry.route) + hash;
  }
  const kind = path.posix.extname(repoPath) ? "blob" : "tree";
  return `${REPO}/${kind}/main/${repoPath}${hash}`;
}

/** Plain text of a Markdown heading, matching what rehype-slug sees. */
function headingText(md: string): string {
  return md
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .trim();
}

/** h2/h3 headings for the "On this page" list, with ids identical to rehype-slug's. */
export function extractHeadings(md: string): Heading[] {
  const slugger = new GithubSlugger();
  const out: Heading[] = [];
  let fence: string | null = null;
  for (const line of md.split("\n")) {
    const f = line.match(/^\s*(```+|~~~+)/);
    if (f) {
      if (!fence) fence = f[1];
      else if (f[1].startsWith(fence[0]) && f[1].length >= fence.length) fence = null;
      continue;
    }
    if (fence) continue;
    const m = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (!m) continue;
    const text = headingText(m[2]);
    const id = slugger.slug(text);
    const depth = m[1].length;
    if (depth === 2 || depth === 3) out.push({ depth, text, id });
  }
  return out;
}
