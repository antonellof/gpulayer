import Link from "next/link";
import { REPO } from "@/lib/docs";
import { VERSION } from "@/lib/version";

export function SiteFooter() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-[90rem] flex-col gap-3 px-4 py-8 text-sm text-muted-foreground md:flex-row md:items-center md:justify-between md:px-6">
        <p>GPUlayer v{VERSION}: one layer between your code and your GPUs. No telemetry.</p>
        <div className="flex gap-5">
          <Link href="/docs/" className="hover:text-foreground">Docs</Link>
          <Link href="/docs/project/status/" className="hover:text-foreground">Status</Link>
          <Link href="/docs/decisions/" className="hover:text-foreground">Decisions</Link>
          <a href={REPO} target="_blank" rel="noreferrer" className="hover:text-foreground">GitHub</a>
        </div>
      </div>
    </footer>
  );
}
