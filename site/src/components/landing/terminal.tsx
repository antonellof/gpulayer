"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Highlight, themes } from "prism-react-renderer";
import { RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";

// A replay of GPUlayer runs: commands are typed, output streams in with the pauses a real run
// has, and the GPU strip shows which devices are training. Output lines follow the formats
// GPUlayer prints; the runs themselves are illustrative, not measurements.

type Line =
  | { t: "cmd"; s: string; wait?: number }
  | { t: "out" | "ok" | "err" | "dim"; s: string; wait?: number }
  | { t: "code"; s: string; wait?: number }
  | { t: "busy"; gpus: string[] | "all" | "none"; wait?: number };

export interface Gpu {
  id: string;
  host: string;
  name: string;
}

export interface TerminalTab {
  label: string;
  title: string;
  gpus: Gpu[];
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
              {line.length === 1 && line[0].empty ? " " : null}
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

const TYPE_MS = 18; // per character of a typed command

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(m.matches);
    const on = () => setReduced(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return reduced;
}

function GpuStrip({ gpus, busy }: { gpus: Gpu[]; busy: Set<string> }) {
  const [tick, setTick] = useState(0);
  const active = busy.size > 0;
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setTick((n) => n + 1), 700);
    return () => window.clearInterval(id);
  }, [active]);
  const hosts = [...new Set(gpus.map((g) => g.host))];
  return (
    <div className="grid gap-3 border-t border-neutral-800 px-4 py-3" style={{ gridTemplateColumns: `repeat(${hosts.length}, minmax(0, 1fr))` }}>
      {hosts.map((host) => (
        <div key={host} className="min-w-0">
          <div className="truncate font-mono text-[10.5px] text-neutral-500">{host}</div>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {gpus
              .filter((g) => g.host === host)
              .map((g, i) => {
                const on = busy.has(g.id);
                // a steady-looking utilisation that moves a little, per GPU
                const util = on ? 88 + ((tick * 7 + i * 13 + g.id.length * 5) % 11) : 0;
                return (
                  <div
                    key={g.id}
                    title={`${g.name}: ${on ? `${util}% busy` : "idle"}`}
                    className={cn(
                      "w-9 overflow-hidden rounded-[3px] border bg-neutral-900",
                      on ? "border-emerald-500/70" : "border-neutral-800",
                    )}
                  >
                    <div className="relative h-5 w-full">
                      <span className="absolute inset-0 z-10 flex items-center justify-center font-mono text-[9px] text-neutral-200">
                        {on ? `${util}%` : ""}
                      </span>
                      <div
                        className={cn(
                          "h-full transition-[width] duration-700 ease-out",
                          on ? "bg-emerald-500/80" : "bg-transparent",
                        )}
                        style={{ width: `${util}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
          <div className="mt-1 truncate font-mono text-[10px] text-neutral-600">
            {gpus.find((g) => g.host === host)?.name}
          </div>
        </div>
      ))}
    </div>
  );
}

export function Terminal({ tabs }: { tabs: TerminalTab[] }) {
  const [active, setActive] = useState(0);
  const [shown, setShown] = useState(0); // lines fully shown
  const [typed, setTyped] = useState(0); // characters of the line being typed
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [visible, setVisible] = useState(false);
  const [run, setRun] = useState(0); // bumps to replay
  const reduced = usePrefersReducedMotion();
  const box = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const tab = tabs[active];

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const applyBusy = useCallback(
    (line: Line) => {
      if (line.t !== "busy") return;
      const ids = line.gpus === "all" ? tab.gpus.map((g) => g.id) : line.gpus === "none" ? [] : line.gpus;
      setBusy(new Set(ids));
    },
    [tab],
  );

  // Reset when the tab or a replay starts.
  useEffect(() => {
    setShown(0);
    setTyped(0);
    setBusy(new Set());
  }, [active, run]);

  // Reduced motion: everything at once, with the final GPU state.
  useEffect(() => {
    if (!reduced) return;
    setShown(tab.lines.length);
    setBusy(new Set());
  }, [reduced, tab, run]);

  // Playback.
  useEffect(() => {
    if (reduced || !visible || shown >= tab.lines.length) return;
    const line = tab.lines[shown];
    if (line.t === "cmd" && typed < line.s.length) {
      const id = window.setTimeout(() => setTyped((n) => n + 1), typed === 0 ? (line.wait ?? 500) : TYPE_MS);
      return () => window.clearTimeout(id);
    }
    const wait = line.t === "cmd" ? 250 : (line.wait ?? 220);
    const id = window.setTimeout(() => {
      applyBusy(line);
      setShown((n) => n + 1);
      setTyped(0);
    }, wait);
    return () => window.clearTimeout(id);
  }, [reduced, visible, shown, typed, tab, applyBusy]);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown, typed]);

  const done = shown >= tab.lines.length;
  const current = !done && tab.lines[shown].t === "cmd" ? tab.lines[shown] : null;

  return (
    <div ref={box} className="min-w-0 overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950 shadow-frame">
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
                i === active ? "bg-neutral-800 text-neutral-100" : "text-neutral-500 hover:text-neutral-200",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setRun((n) => n + 1)}
          aria-label="Replay"
          title="Replay"
          className="rounded-md p-1 text-neutral-500 transition-colors hover:text-neutral-200"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="flex items-center justify-between border-b border-neutral-900 px-4 py-1.5 font-mono text-[11px] text-neutral-600">
        <span className="truncate">{tab.title}</span>
        <span className="shrink-0 pl-3">simulated replay</span>
      </div>
      <div
        ref={scroller}
        role="tabpanel"
        aria-live="off"
        className="h-[300px] overflow-auto px-5 py-4 font-mono text-[12.5px] leading-relaxed"
      >
        {tab.lines.slice(0, shown).map((line, i) =>
          line.t === "busy" ? null : line.t === "code" ? (
            <div key={i} className="mb-3 rounded-lg border border-neutral-800 bg-neutral-900/60 p-3 text-[#abb2bf]">
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
        {done || current ? (
          // The prompt only between commands: while a job runs, its output has the screen.
          <div className="text-neutral-100">
            <span className="select-none text-neutral-500">$ </span>
            {current ? current.s.slice(0, typed) : null}
            <span className="inline-block h-4 w-2 translate-y-0.5 animate-pulse bg-neutral-400" />
          </div>
        ) : null}
      </div>
      <GpuStrip gpus={tab.gpus} busy={busy} />
    </div>
  );
}
