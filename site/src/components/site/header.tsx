import Link from "next/link";
import { Github } from "lucide-react";
import { Logo } from "@/components/site/logo";
import { ThemeToggle } from "@/components/site/theme-toggle";
import { REPO } from "@/lib/docs";

const NAV = [
  { href: "/docs/", label: "Docs" },
  { href: "/docs/get-started/quickstart/", label: "Quickstart" },
  { href: "/docs/workloads/commands/", label: "Workloads" },
  { href: "/docs/pytorch/trainer/", label: "PyTorch" },
  { href: "/docs/reference/python-api/", label: "API" },
  { href: "/compare/", label: "Compare" },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="mx-auto flex h-14 max-w-[90rem] items-center justify-between gap-4 px-4 md:px-6">
        <div className="flex items-center gap-3">
          <Logo />
          <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
            pre-alpha
          </span>
        </div>
        <nav className="flex items-center gap-1 sm:gap-4">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className="hidden text-sm text-muted-foreground transition-colors hover:text-foreground sm:inline"
            >
              {n.label}
            </Link>
          ))}
          <Link href="/docs/" className="text-sm text-muted-foreground hover:text-foreground sm:hidden">
            Docs
          </Link>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            <Github className="h-4 w-4" />
          </a>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
