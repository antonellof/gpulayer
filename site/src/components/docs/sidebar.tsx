"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, Search, X } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SidebarSection {
  title: string;
  pages: { title: string; href: string }[];
}

function normalise(p: string): string {
  return p.endsWith("/") ? p : `${p}/`;
}

function SidebarNav({ sections, onNavigate }: { sections: SidebarSection[]; onNavigate?: () => void }) {
  const pathname = normalise(usePathname() || "/");
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const filtered = sections
    .map((s) => ({
      ...s,
      pages: q
        ? s.pages.filter((p) => p.title.toLowerCase().includes(q) || s.title.toLowerCase().includes(q))
        : s.pages,
    }))
    .filter((s) => s.pages.length);

  return (
    <nav aria-label="Documentation" className="space-y-6">
      <label className="relative block">
        <Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter pages"
          className="w-full rounded-md border border-border bg-background py-1.5 pl-8 pr-2 text-sm outline-none placeholder:text-muted-foreground focus:border-primary"
        />
      </label>
      {filtered.map((section) => (
        <div key={section.title}>
          <p className="mb-2 px-2 text-xs font-semibold uppercase tracking-wider text-foreground/70">
            {section.title}
          </p>
          <ul className="space-y-0.5">
            {section.pages.map((page) => {
              const active =
                pathname === page.href ||
                (page.href === "/docs/decisions/" && pathname.startsWith("/docs/decisions/"));
              return (
                <li key={page.href}>
                  <Link
                    href={page.href}
                    onClick={onNavigate}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "block rounded-md px-2 py-1.5 text-sm transition-colors",
                      active
                        ? "bg-primary/10 font-medium text-primary"
                        : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                    )}
                  >
                    {page.title}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
      {!filtered.length && <p className="px-2 text-sm text-muted-foreground">No matching pages.</p>}
    </nav>
  );
}

export function DocsSidebar({ sections }: { sections: SidebarSection[] }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <aside className="sticky top-14 hidden h-[calc(100vh-3.5rem)] w-64 shrink-0 overflow-y-auto border-r border-border px-4 py-6 lg:block">
        <SidebarNav sections={sections} />
      </aside>
      <div className="sticky top-14 z-30 border-b border-border bg-background/95 px-4 py-2 backdrop-blur lg:hidden">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <Menu className="h-4 w-4" /> Menu
        </button>
      </div>
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label="Close menu"
            className="absolute inset-0 bg-background/70 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-72 overflow-y-auto border-r border-border bg-background px-4 py-5 shadow-xl">
            <div className="mb-4 flex justify-end">
              <button type="button" onClick={() => setOpen(false)} aria-label="Close menu">
                <X className="h-5 w-5 text-muted-foreground" />
              </button>
            </div>
            <SidebarNav sections={sections} onNavigate={() => setOpen(false)} />
          </div>
        </div>
      )}
    </>
  );
}
