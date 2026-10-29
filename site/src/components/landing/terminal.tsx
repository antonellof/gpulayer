"use client";

import { useState } from "react";
import { Highlight, themes } from "prism-react-renderer";
import { cn } from "@/lib/utils";

type Line =
  | { t: "cmd"; s: string }
  | { t: "out" | "ok" | "err" | "dim"; s: string }
  | { t: "code"; s: string };

export interface TerminalTab {
  label: string;
  title: string;
  lines: Line[];
}

function CodeLines({ code }: { code: string }) {
  return (
    <Highlight code={code} language="python" theme={themes.oneDark}>
      {({ tokens, getLineProps, getTokenProps }) => (
        <>
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })} style={undefined}>
              {line.map((token, k) => (
                <span key={k} {...getTokenProps({ token })} />
              ))}
              {line.length === 1 && line[0].empty ? " " : null}
            </div>
          ))}
        </>
      )}
    </Highlight>
  );
}

const TONE: Record<string, string> = {
  out: "text-neutral-300",
  dim: "text-neutral-500",
  ok: "text-emerald-400",
  err: "text-rose-400",
};

export function Terminal({ tabs }: { tabs: TerminalTab[] }) {
  const [active, setActive] = useState(0);
  const tab = tabs[active];
  return (
    <div className="min-w-0 overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950 shadow-frame">
      <div className="flex items-center gap-4 border-b border-neutral-800 px-4 py-2.5">
        <div className="flex gap-1.5" aria-hidden>
          <span className="h-2.5 w-2.5 rounded-full bg-neutral-700" />
          <span className="h-2.5 w-2.5 rounded-full bg-neutral-700" />
          <span className="h-2.5 w-2.5 rounded-full bg-neutral-700" />
        </div>
        <div role="tablist" aria-label="Examples" className="flex flex-1 gap-1 overflow-x-auto">
          {tabs.map((t, i) => (
            <button
              key={t.label}
              type="button"
              role="tab"
              aria-selected={i === active}
              onClick={() => setActive(i)}
              className={cn(
                "whitespace-nowrap rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                i === active
                  ? "bg-neutral-800 text-neutral-100"
                  : "text-neutral-500 hover:text-neutral-200",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <div className="border-b border-neutral-900 px-4 py-1.5 font-mono text-[11px] text-neutral-600">
        {tab.title}
      </div>
      <div
        role="tabpanel"
        className="h-[332px] overflow-auto px-5 py-5 font-mono text-[13px] leading-relaxed"
      >
        {tab.lines.map((line, i) =>
          line.t === "code" ? (
            <div key={i} className="mb-4 rounded-lg border border-neutral-800 bg-neutral-900/60 p-3.5">
              <CodeLines code={line.s} />
            </div>
          ) : line.t === "cmd" ? (
            <div key={i} className="whitespace-pre-wrap break-words text-neutral-100">
              <span className="select-none text-neutral-500">$ </span>
              {line.s}
            </div>
          ) : (
            <div key={i} className={cn("whitespace-pre-wrap break-words", TONE[line.t])}>
              {line.s}
            </div>
          ),
        )}
        <div className="mt-1 text-neutral-100">
          <span className="text-neutral-500">$ </span>
          <span className="inline-block h-4 w-2 translate-y-0.5 animate-pulse bg-neutral-400" />
        </div>
      </div>
    </div>
  );
}
