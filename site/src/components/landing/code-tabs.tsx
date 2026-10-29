"use client";

import { useState } from "react";
import { CodeBlock } from "@/components/docs/code-block";
import { cn } from "@/lib/utils";

export interface CodeTab {
  label: string;
  language: string;
  code: string;
}

export function CodeTabs({ tabs }: { tabs: CodeTab[] }) {
  const [active, setActive] = useState(0);
  const tab = tabs[active];
  return (
    <div className="min-w-0">
      <div role="tablist" aria-label="Examples" className="flex gap-1 border-b border-border">
        {tabs.map((t, i) => (
          <button
            key={t.label}
            type="button"
            role="tab"
            aria-selected={i === active}
            onClick={() => setActive(i)}
            className={cn(
              "-mb-px border-b-2 px-3 py-2 text-sm transition-colors",
              i === active
                ? "border-primary font-medium text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel">
        <CodeBlock code={tab.code} language={tab.language} className="mt-3 shadow-sm" />
      </div>
    </div>
  );
}
