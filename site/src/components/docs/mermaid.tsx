"use client";

import { useEffect, useId, useState } from "react";
import { useIsDark } from "@/lib/use-dark";

/** Renders a ```mermaid block client-side; falls back to the source on error. */
export function Mermaid({ chart }: { chart: string }) {
  const dark = useIsDark();
  const id = "m" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const [svg, setSvg] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: dark ? "dark" : "neutral",
          fontFamily: "var(--font-sans), system-ui, sans-serif",
        });
        const { svg } = await mermaid.render(`${id}-${dark ? "d" : "l"}`, chart);
        if (!cancelled) {
          setSvg(svg);
          setFailed(false);
        }
      } catch {
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [chart, dark, id]);

  if (failed) {
    return (
      <pre className="my-6 overflow-x-auto rounded-lg border border-border bg-card p-4 font-mono text-[13px]">
        {chart}
      </pre>
    );
  }
  return (
    <figure className="my-8 flex justify-center overflow-x-auto rounded-lg border border-border bg-card p-4">
      {svg ? (
        <div className="max-w-full [&_svg]:h-auto [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />
      ) : (
        <div className="py-8 text-xs text-muted-foreground">Loading diagram…</div>
      )}
    </figure>
  );
}
