"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";

/** A one-line shell command with a copy button, for hero and CTA areas. */
export function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(command);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard unavailable */
        }
      }}
      className="group inline-flex h-10 items-center gap-3 rounded-md border border-border bg-card px-4 font-mono text-[13px] transition-colors hover:bg-secondary"
      aria-label={`Copy ${command}`}
    >
      <span className="text-muted-foreground">$</span>
      <span>{command}</span>
      {copied ? (
        <Check className="h-3.5 w-3.5 text-muted-foreground" />
      ) : (
        <Copy className="h-3.5 w-3.5 text-muted-foreground transition-colors group-hover:text-foreground" />
      )}
    </button>
  );
}
