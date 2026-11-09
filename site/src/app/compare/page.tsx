import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, Check, CircleDashed, X } from "lucide-react";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";
import { OG_IMAGE } from "@/lib/site";

export const metadata: Metadata = {
  title: "GPUlayer compared | PyTorch Lightning, Accelerate, Ray, SkyPilot, dstack, Modal",
  description:
    "How the GPUlayer Trainer compares with PyTorch Lightning, Hugging Face Accelerate, Ray, SkyPilot, dstack, Modal, Slurm and Kubernetes for training PyTorch on many GPUs.",
  alternates: { canonical: "/compare/" },
  openGraph: { url: "/compare/", images: [OG_IMAGE] },
};

type Mark = "yes" | "partial" | "no";

const TOOLS = ["GPUlayer", "Lightning", "Accelerate", "Ray", "SkyPilot", "dstack", "Modal", "Slurm / K8s"] as const;

// Order of marks follows TOOLS.
const MATRIX: { feature: string; marks: Mark[]; note?: string }[] = [
  {
    feature: "Lightning-style Trainer (hooks, callbacks, loggers, CLI)",
    marks: ["yes", "yes", "no", "partial", "no", "no", "no", "no"],
    note: "GPUlayer keeps Lightning's names; Ray Train runs Lightning code inside Ray",
  },
  {
    feature: "DDP, mixed precision, FSDP",
    marks: ["yes", "yes", "yes", "yes", "no", "no", "no", "partial"],
    note: "GPUlayer's FSDP options validated on T4s; Kubernetes through training operators",
  },
  {
    feature: "DeepSpeed",
    marks: ["yes", "yes", "yes", "yes", "no", "no", "no", "partial"],
    note: "GPUlayer: ZeRO 1 to 3 and CPU offload validated on Tesla T4s",
  },
  {
    feature: "Starts every node of a multi-machine run from one call",
    marks: ["yes", "no", "no", "yes", "yes", "yes", "partial", "yes"],
    note: "Lightning and Accelerate train across nodes once a launcher is started on each",
  },
  {
    feature: "Picks machines that can train the run",
    marks: ["yes", "no", "no", "yes", "yes", "yes", "yes", "yes"],
  },
  {
    feature: "NVIDIA servers, Macs and cloud VMs in one pool",
    marks: ["yes", "no", "no", "partial", "partial", "partial", "no", "partial"],
  },
  {
    feature: "Creates cloud GPUs for a run",
    marks: ["partial", "no", "no", "yes", "yes", "yes", "yes", "partial"],
    note: "GPUlayer creates them on Google Cloud; AWS and Vast.ai GPUs were tested by joining them",
  },
  {
    feature: "No cluster to operate",
    marks: ["yes", "yes", "yes", "partial", "yes", "partial", "yes", "no"],
  },
  {
    feature: "Runs on your own hardware",
    marks: ["yes", "yes", "yes", "yes", "partial", "yes", "no", "yes"],
  },
];

const WHAT: [string, string][] = [
  ["GPUlayer", "A PyTorch Trainer with Lightning's API that also places each run on the machines you can reach and starts every process."],
  ["PyTorch Lightning", "The reference Trainer; runs on the machines it is started on (Slurm, torchrun or its own platform)."],
  ["Accelerate", "Makes an existing training loop portable across devices and precision; you launch it on each machine."],
  ["Ray", "A distributed computing framework; Ray Train runs training on a Ray cluster you operate."],
  ["SkyPilot, dstack", "Run jobs on cloud VMs, Kubernetes or registered servers from a spec; training code is up to you."],
  ["Modal", "Serverless GPUs: decorated Python functions run on Modal's hardware."],
  ["Slurm, Kubernetes", "Cluster schedulers for shared infrastructure you administer."],
];

const ELSEWHERE = [
  "You want VMs created for you outside Google Cloud, or autoscaling pools: SkyPilot, dstack or Ray cover more (AWS or Vast.ai machines you start can join GPUlayer).",
  "You train on TPUs today: TPU support in the GPUlayer Trainer is implemented but not yet run on a TPU.",
  "Untrusted users share the machines: GPUlayer does not sandbox jobs.",
  "One GPU on the machine in front of you: a plain training loop is simpler.",
];

function MarkIcon({ mark }: { mark: Mark }) {
  const base = "mx-auto flex h-6 w-6 items-center justify-center rounded-full";
  if (mark === "yes")
    return (
      <span className={`${base} bg-emerald-500/15 text-emerald-600 dark:text-emerald-400`} title="Yes">
        <Check className="h-3.5 w-3.5" strokeWidth={3} />
      </span>
    );
  if (mark === "partial")
    return (
      <span className={`${base} bg-amber-500/15 text-amber-600 dark:text-amber-400`} title="Partly">
        <CircleDashed className="h-3.5 w-3.5" strokeWidth={2.5} />
      </span>
    );
  return (
    <span className={`${base} bg-muted text-muted-foreground/60`} title="No">
      <X className="h-3.5 w-3.5" strokeWidth={2.5} />
    </span>
  );
}

export default function ComparePage() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 pb-14 pt-20 lg:pt-24">
            <p className="eyebrow">Compare</p>
            <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-[-0.04em] sm:text-5xl">
              GPUlayer and the alternatives
            </h1>
            <p className="lede mt-5 max-w-3xl">
              Training frameworks shape how a model trains on the machines they are started on; orchestrators find
              and start machines. The GPUlayer Trainer does both for PyTorch, with Lightning&apos;s API.
            </p>
          </div>
        </section>

        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <h2 className="section-title">For training PyTorch on many GPUs</h2>
            <div className="mt-8 overflow-x-auto rounded-xl border border-border">
              <table className="w-full min-w-[860px] text-sm">
                <thead>
                  <tr className="border-b border-border bg-card text-xs text-muted-foreground">
                    <th className="px-4 py-3 text-left font-medium" />
                    {TOOLS.map((t) => (
                      <th key={t} className={`px-2 py-3 font-medium ${t === "GPUlayer" ? "text-foreground" : ""}`}>
                        {t}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {MATRIX.map((row) => (
                    <tr key={row.feature} className="border-b border-border last:border-0">
                      <td className="px-4 py-3">
                        <div className="font-medium">{row.feature}</div>
                        {row.note && <div className="mt-0.5 text-xs text-muted-foreground">{row.note}</div>}
                      </td>
                      {row.marks.map((m, i) => (
                        <td key={TOOLS[i]} className={`px-2 py-3 ${i === 0 ? "bg-card" : ""}`}>
                          <MarkIcon mark={m} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <dl className="mt-10 grid gap-x-10 gap-y-4 sm:grid-cols-2">
              {WHAT.map(([name, text]) => (
                <div key={name}>
                  <dt className="text-sm font-medium">{name}</dt>
                  <dd className="mt-1 text-sm text-muted-foreground">{text}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <section className="border-b border-border">
          <div className="mx-auto grid max-w-6xl gap-12 px-6 py-16 lg:grid-cols-2">
            <div>
              <h2 className="section-title">Speed against Lightning</h2>
              <p className="mt-4 text-muted-foreground">
                Same model, data and optimizer, each trainer&apos;s defaults (Lightning 2.6.6). The GPUlayer Trainer
                was faster in 37 of 43 GPU cases and every CPU case: 1.08 to 1.38x on a Tesla T4, 1.26 to 1.60x on 2
                and 4 RTX 3070s, 1.04 to 1.68x on 1 to 4 L4s, 1.10 to 2.50x on 1 and 2 H200s, 17.8% on 8 CPU machines.
                Lightning was faster on a CNN on one GPU of one host (1.12x), on a CNN across 2 to 8 machines over
                NCCL (1.01 to 1.09x), on ResNet-50 on an H200 (1.01x) and per step on a GPT-2-size model on 8 GPUs
                (1.15x). With realistic models on an L4 the two are within 4%.
              </p>
              <Link href="/docs/get-started/performance-and-comparison/" className="link-arrow mt-6">
                Charts, method and logs <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div>
              <h2 className="section-title">Choose something else when</h2>
              <ul className="mt-4 space-y-3 text-muted-foreground">
                {ELSEWHERE.map((t) => (
                  <li key={t} className="flex gap-3">
                    <ArrowRight className="mt-1 h-4 w-4 shrink-0" />
                    <span>{t}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section>
          <div className="mx-auto flex max-w-6xl flex-wrap gap-3 px-6 py-16">
            <Link href="/docs/get-started/quickstart/" className="btn-primary">
              Try the quickstart <ArrowRight className="h-4 w-4" />
            </Link>
            <Link href="/docs/pytorch/from-lightning/" className="btn-secondary">
              Moving from Lightning
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
