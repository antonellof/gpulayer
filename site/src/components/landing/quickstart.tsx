"use client";

import { useState } from "react";
import { Check, Copy, Laptop, Network } from "lucide-react";
import { cn } from "@/lib/utils";

type Step = { title: string; body: string; command: string; output?: string[] };
type Track = { id: string; label: string; icon: typeof Laptop; blurb: string; steps: Step[] };

// Outputs are copied from real runs on an Apple Silicon Mac.
const TRACKS: Track[] = [
  {
    id: "local",
    label: "On this machine",
    icon: Laptop,
    blurb: "No server, no account. GpuLayer runs a control plane and a worker inside your process.",
    steps: [
      {
        title: "Install",
        body: "Python 3.10 or newer. PyTorch is optional for the core.",
        command:
          "git clone https://github.com/antonellof/gpulayer && cd gpulayer\nuv venv && source .venv/bin/activate && uv pip install -e '.[dev,examples]'",
      },
      {
        title: "See what this machine offers",
        body: "GPUs come from the driver or platform; frameworks from the Python that runs jobs.",
        command: "gpulayer resources detect",
        output: [
          "host      my-mac (darwin/arm64), 10 cores, 32.0 GB RAM",
          "device    [0] mps Apple arm64 GPU (Metal) (32.0 GB unified)",
        ],
      },
      {
        title: "Run any command as a job",
        body: "Inputs are staged, output files come back, logs stream to your terminal.",
        command:
          "gpulayer submit --source examples/shell_job \\\n  --input examples/shell_job/sales.csv:sales.csv -- bash report.sh",
        output: ["running on my-mac, accelerator=cpu", "east,99.99", "job-88eeb3960503 succeeded"],
      },
      {
        title: "Train PyTorch, then scale the same file",
        body: "One process, then DDP on two processes. Only the flags change.",
        command:
          "python examples/trainer_regression/train.py\npython examples/trainer_regression/train.py \\\n  --accelerator cpu --devices 2",
        output: [
          "[gpulayer] training on mps, precision 32-true, 5 epoch(s)",
          "[gpulayer] job-5cc477d4ffad succeeded; 2 output file(s) in gpulayer_runs/job-5cc477d4ffad",
        ],
      },
    ],
  },
  {
    id: "fleet",
    label: "Across your machines",
    icon: Network,
    blurb: "A small self-hosted server. Machines join it; your laptop sends work to it.",
    steps: [
      {
        title: "Start a server",
        body: "On any machine. TLS with a self-signed certificate is set up for you, and the machine runs jobs too.",
        command: "gpulayer up",
        output: [
          "GpuLayer is up at https://192.168.1.56:8470  (TLS, self-signed certificate pinned by fingerprint)",
          "Add a machine by running on it:",
          "  gpulayer join https://192.168.1.56:8470 --token glj_… --fingerprint sha256:…",
        ],
      },
      {
        title: "Join each GPU machine or cloud VM",
        body: "Paste the line printed by `gpulayer up`. The worker reports its GPUs and frameworks.",
        command:
          "gpulayer join https://192.168.1.56:8470 \\\n  --token glj_… --fingerprint sha256:…",
        output: ["worker wkr-4ad94eaeee80 registered (existing; mps:Apple arm64 GPU (Metal))"],
      },
      {
        title: "Point your laptop at the server",
        body: "Also printed by `gpulayer up`. The SDK, CLI and PyTorch API all pick it up.",
        command:
          "export GPULAYER_URL=https://192.168.1.56:8470 \\\n  GPULAYER_TOKEN=glc_… \\\n  GPULAYER_FINGERPRINT=sha256:…",
      },
      {
        title: "Run anything there",
        body: "A command, a Python function or a PyTorch Trainer. Jobs land where they fit.",
        command:
          "gpulayer submit --accelerator cuda -- python infer.py\npython examples/trainer_regression/train.py \\\n  --accelerator cpu --nodes 2",
        output: [
          "[gpulayer] job-03d2e9fb3b51: train.py on 2 node(s) x 1 cpu process(es), target remote",
          "[gpulayer] job-03d2e9fb3b51 succeeded; 2 output file(s) in gpulayer_runs/job-03d2e9fb3b51",
        ],
      },
    ],
  },
];

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard unavailable */
        }
      }}
      className="absolute right-2 top-2 inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-slate-300 opacity-0 transition hover:bg-white/10 group-hover:opacity-100"
      aria-label="Copy command"
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function Inline({ text }: { text: string }) {
  return (
    <>
      {text.split(/(`[^`]+`)/g).map((p, i) =>
        p.startsWith("`") ? (
          <code key={i} className="rounded bg-secondary px-1 py-0.5 font-mono text-[0.85em] text-foreground">
            {p.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}

export function Quickstart() {
  const [active, setActive] = useState(0);
  const track = TRACKS[active];
  return (
    <div>
      <div role="tablist" aria-label="Quickstart" className="inline-flex rounded-xl border border-border bg-card p-1">
        {TRACKS.map((t, i) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={i === active}
            onClick={() => setActive(i)}
            className={cn(
              "inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition",
              i === active
                ? "bg-background text-foreground shadow-soft"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <t.icon className="h-4 w-4" />
            {t.label}
          </button>
        ))}
      </div>
      <p className="mt-4 max-w-2xl text-muted-foreground">{track.blurb}</p>

      <ol role="tabpanel" className="relative mt-10 space-y-10">
        <span
          className="absolute bottom-6 left-[19px] top-6 w-px bg-gradient-to-b from-primary/60 via-border to-transparent"
          aria-hidden
        />
        {track.steps.map((step, i) => (
          <li key={step.title} className="relative grid gap-4 pl-14 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] lg:gap-10">
            <span className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-full border border-primary/40 bg-background font-mono text-sm font-semibold text-primary shadow-soft">
              {i + 1}
            </span>
            <div className="pt-1.5">
              <h3 className="font-display text-lg font-semibold">{step.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                <Inline text={step.body} />
              </p>
            </div>
            <div className="group relative min-w-0 overflow-hidden rounded-xl border border-white/10 bg-[#0b0f17] font-mono text-[12.5px] leading-relaxed shadow-soft">
              <CopyButton text={step.command} />
              <div className="px-4 py-3 pr-20 text-slate-100">
                {step.command.split("\n").map((line, k) => (
                  <div key={k} className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
                    <span className="select-none text-emerald-400">{line.startsWith("  ") ? "  " : "$ "}</span>
                    {line.startsWith("  ") ? line.slice(2) : line}
                  </div>
                ))}
              </div>
              {step.output && (
                <div className="border-t border-white/5 bg-white/[0.02] px-4 py-3 text-slate-400">
                  {step.output.map((line) => (
                    <div
                      key={line}
                      className={cn(
                        "whitespace-pre-wrap break-words [overflow-wrap:anywhere]",
                        /succeeded|registered|is up/.test(line) && "text-emerald-400",
                      )}
                    >
                      {line}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
