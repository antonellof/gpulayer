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
    <Highlight code={code} language="python" theme={themes.vsDark}>
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
  out: "text-slate-300",
  dim: "text-slate-500",
  ok: "text-emerald-400",
  err: "text-rose-400",
};

export function Terminal({ tabs }: { tabs: TerminalTab[] }) {
  const [active, setActive] = useState(0);
  const tab = tabs[active];
  return (
    <div className="min-w-0 overflow-hidden rounded-xl border border-white/10 bg-[#0b0f17] shadow-[0_30px_80px_-30px_rgba(16,185,129,0.35)] ring-1 ring-white/5">
      <div className="flex items-center gap-3 border-b border-white/10 bg-white/[0.03] px-4 py-2.5">
        <div className="flex gap-1.5" aria-hidden>
          <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
          <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
          <span className="h-3 w-3 rounded-full bg-[#28c840]" />
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
                "whitespace-nowrap rounded-md px-2.5 py-1 font-mono text-xs transition-colors",
                i === active
                  ? "bg-white/10 text-white"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <div className="border-b border-white/5 px-4 py-1.5 font-mono text-[11px] text-slate-500">
        {tab.title}
      </div>
      <div
        role="tabpanel"
        className="h-[340px] overflow-auto px-4 py-4 font-mono text-[12.5px] leading-relaxed"
      >
        {tab.lines.map((line, i) =>
          line.t === "code" ? (
            <div key={i} className="mb-3 rounded-lg border border-white/5 bg-white/[0.02] p-3">
              <CodeLines code={line.s} />
            </div>
          ) : line.t === "cmd" ? (
            <div key={i} className="whitespace-pre-wrap break-words text-white">
              <span className="select-none text-emerald-400">$ </span>
              {line.s}
            </div>
          ) : (
            <div key={i} className={cn("whitespace-pre-wrap break-words", TONE[line.t])}>
              {line.s}
            </div>
          ),
        )}
        <div className="mt-1 text-white">
          <span className="text-emerald-400">$ </span>
          <span className="inline-block h-4 w-2 translate-y-0.5 animate-pulse bg-slate-300" />
        </div>
      </div>
    </div>
  );
}
