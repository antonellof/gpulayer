"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { Highlight, Prism, themes } from "prism-react-renderer";
import { useIsDark } from "@/lib/use-dark";
import { cn } from "@/lib/utils";

// prism-react-renderer ships without a shell grammar; register a small one.
const shell = {
  comment: { pattern: /(^|[^"{\\$])#.*/, lookbehind: true },
  string: { pattern: /(["'])(?:\\[\s\S]|(?!\1)[^\\])*\1/, greedy: true },
  variable: /\$\{?[\w#?*!@$]+\}?/,
  parameter: { pattern: /(^|\s)--?[\w][\w-]*/, lookbehind: true, alias: "attr-name" },
  operator: /&&|\|\||[|;&<>]/,
  function: { pattern: /(^\s*(?:\$\s+)?)[\w./-]+/m, lookbehind: true },
};
for (const alias of ["bash", "sh", "shell", "zsh", "console"]) {
  if (!Prism.languages[alias]) Prism.languages[alias] = shell;
}

const LABELS: Record<string, string> = {
  py: "python",
  sh: "bash",
  shell: "bash",
  console: "bash",
  ts: "typescript",
  js: "javascript",
  yml: "yaml",
};

export function CodeBlock({
  code,
  language,
  className,
}: {
  code: string;
  language?: string;
  className?: string;
}) {
  const dark = useIsDark();
  const [copied, setCopied] = useState(false);
  const lang = (language || "text").toLowerCase();
  const grammar = Prism.languages[lang] ? lang : "text";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (e.g. insecure context) */
    }
  };

  return (
    <div className={cn("group relative my-6 overflow-hidden rounded-lg border border-border", className)}>
      <div className="flex items-center justify-between border-b border-border bg-secondary/60 px-4 py-1.5">
        <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          {LABELS[lang] ?? lang}
        </span>
        <button
          type="button"
          onClick={copy}
          aria-label="Copy code"
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-background hover:text-foreground"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <Highlight code={code.replace(/\n$/, "")} language={grammar} theme={dark ? themes.vsDark : themes.github}>
        {({ className: cls, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={cn(cls, "overflow-x-auto p-4 font-mono text-[13px] leading-relaxed")}
            style={{ ...style, background: "hsl(var(--card))", margin: 0 }}
          >
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, j) => (
                  <span key={j} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
}
