import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  Check,
  Cloud,
  FileCode2,
  Flame,
  Library,
  Minus,
  ScrollText,
} from "lucide-react";
import { ArchitectureDiagram } from "@/components/landing/architecture-diagram";
import { CopyCommand } from "@/components/landing/copy-command";
import { VERSION } from "@/lib/version";
import { WorksWith } from "@/components/landing/logos";
import { Quickstart } from "@/components/landing/quickstart";
import { Terminal, type TerminalTab } from "@/components/landing/terminal";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";
import { REPO } from "@/lib/docs";
import { SITE_DESCRIPTION, SITE_URL } from "@/lib/site";

// Structured data for search engines: GPUlayer as free, open-source software.
const JSON_LD = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "GPUlayer",
  url: SITE_URL,
  description: SITE_DESCRIPTION,
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Linux, macOS",
  softwareVersion: VERSION,
  license: "https://www.apache.org/licenses/LICENSE-2.0",
  downloadUrl: "https://pypi.org/project/gpulayer/",
  codeRepository: REPO,
  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
};

// Output below is copied from real runs on an Apple Silicon Mac (see the quickstart).
const TERMINAL_TABS: TerminalTab[] = [
  {
    label: "PyTorch",
    title: "train.py",
    lines: [
      {
        t: "code",
        s: `trainer = glt.Trainer(max_epochs=5, accelerator=args.accelerator,
                      devices=args.devices, num_nodes=args.nodes)
trainer.fit(model, train_loader, val_loader)`,
      },
      { t: "cmd", s: "python train.py --accelerator cpu --nodes 2 --devices 2" },
      { t: "dim", s: "[gpulayer] job-4f5c1e1bc659: train.py on 2 node(s) x 2 cpu process(es)" },
      { t: "out", s: "[node0] epoch 4 done: train_loss=1.6501 val_loss=1.3001" },
      { t: "ok", s: "[gpulayer] job-4f5c1e1bc659 succeeded; checkpoint downloaded" },
    ],
  },
  {
    label: "Placement",
    title: "asking for hardware this machine does not have",
    lines: [
      { t: "cmd", s: "python train.py --accelerator cuda --target local" },
      { t: "dim", s: "[gpulayer] job-df9c63e7c4d8: train.py on 1 node(s) x 1 cuda process(es)" },
      { t: "err", s: "[gpulayer] job-df9c63e7c4d8 failed: unschedulable: no eligible worker" },
      {
        t: "err",
        s: "  wkr-f50e620bfaef: torch on this worker cannot use CUDA, needs 1 usable cuda device(s), worker has 0",
      },
    ],
  },
];

const GPU_RESULTS: { gpu: string; result: string }[] = [
  { gpu: "Tesla T4", result: "1.14 to 1.39x per step; fp32, bf16, fp16 and torch.compile" },
  { gpu: "4x RTX 3070 (PCIe)", result: "1.11 to 1.76x on 2 and 4 GPUs; 1.14x slower on one GPU with a CNN" },
  { gpu: "2x A100 SXM4 (NVLink)", result: "DDP over NCCL and NVLink; CUDA checks passed" },
  { gpu: "Apple M2 Pro", result: "1.05 to 1.75x on the GPU and the CPU" },
];

const FEATURES = [
  {
    icon: Flame,
    title: "One Trainer, any scale",
    body: "The same script runs in your process, as DDP on every GPU of a machine, or across machines with Trainer(devices=4, num_nodes=2). Mixed precision, checkpoints, callbacks and experimental FSDP.",
  },
  {
    icon: Boxes,
    title: "Placed where it trains fastest",
    body: "Multi-GPU runs get the best-connected GPUs of a machine. Multi-machine runs stay in one rack, zone or region when possible.",
  },
  {
    icon: Library,
    title: "Every GPU you can reach",
    body: "Your laptop, workstations, servers that join with one command, and cloud VMs. NVIDIA CUDA and Apple Metal are validated; AMD ROCm is implemented and not yet validated.",
  },
  {
    icon: Cloud,
    title: "A cloud GPU for one run",
    body: "When no machine fits, GPUlayer can create a Google Cloud VM from offers you list and delete it when training ends.",
  },
  {
    icon: ScrollText,
    title: "Logs, checkpoints, metrics",
    body: "Logs stream while training runs, checkpoints and metrics come back when it ends, and a dashboard shows every run.",
  },
  {
    icon: FileCode2,
    title: "Your own training loop",
    body: "Already have a DDP script? Two launch() lines make it run anywhere. Runtime handles device, DDP and precision in a custom loop.",
  },
];

const COMPARISON: { situation: string; without: string; with: string }[] = [
  {
    situation: "Code on a laptop, GPUs elsewhere",
    without: "ssh, copy code, rebuild the environment, run in tmux, copy checkpoints back",
    with: "`python train.py --target remote`",
  },
  {
    situation: "Several GPUs in one machine",
    without: "torchrun with the right process count and visible devices",
    with: "`Trainer(devices=4)`",
  },
  {
    situation: "Several machines",
    without: "torchrun on every node with matching ranks, address and port",
    with: "`Trainer(num_nodes=2, devices=4)`",
  },
  {
    situation: "NVIDIA servers and Apple Silicon Macs",
    without: "separate setup per machine",
    with: "`accelerator=\"cuda\"` or `\"mps\"`, same script",
  },
  {
    situation: "A learning-rate sweep",
    without: "loops and ad hoc scripts",
    with: "`train.map([...])` in parallel across free GPUs",
  },
];

const STATUS: { label: string; dot: string; items: string }[] = [
  {
    label: "Hardware-validated",
    dot: "bg-emerald-500",
    items:
      "Trainer on NVIDIA CUDA (Tesla T4, RTX 3070, A100) and Apple Metal · DDP over NCCL on up to 4 GPUs of a machine · distributed training with gloo on up to 16 machines · Google Cloud VMs created per job",
  },
  {
    label: "Implemented, not validated",
    dot: "bg-amber-500",
    items: "NCCL across machines · AMD ROCm · FSDP (experimental) · NVIDIA GPU workers on Kubernetes",
  },
  {
    label: "Planned",
    dot: "bg-neutral-400",
    items: "Other cloud providers · hosted control plane",
  },
];

function Inline({ text }: { text: string }) {
  return (
    <>
      {text.split(/(`[^`]+`)/g).map((part, i) =>
        part.startsWith("`") && part.endsWith("`") ? (
          <code key={i} className="rounded border border-border bg-secondary px-1 py-0.5 font-mono text-[0.82em]">
            {part.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function SectionHeader({ eyebrow, title, lede }: { eyebrow: string; title: string; lede?: string }) {
  return (
    <div className="max-w-2xl">
      <p className="eyebrow">{eyebrow}</p>
      <h2 className="section-title mt-3">{title}</h2>
      {lede && <p className="lede mt-4">{lede}</p>}
    </div>
  );
}

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }} />
      <main className="flex-1">
        {/* Hero */}
        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 pb-16 pt-20 lg:pt-28">
            <Link
              href="/docs/project/status/"
              className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              v{VERSION} · validated on NVIDIA CUDA, Apple Metal and CPU
              <ArrowRight className="h-3 w-3" />
            </Link>
            <h1 className="mt-6 max-w-4xl text-5xl font-semibold leading-[1.02] tracking-[-0.045em] sm:text-6xl lg:text-7xl">
              Train PyTorch on every GPU you can reach.
            </h1>
            <p className="lede mt-6 max-w-2xl sm:text-xl">
              One Trainer, from your laptop to many machines. GPUlayer picks the machines and GPUs, starts every
              process and brings back logs, checkpoints and metrics.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link href="/docs/get-started/quickstart/" className="btn-primary">
                Get started <ArrowRight className="h-4 w-4" />
              </Link>
              <Link href="/docs/" className="btn-secondary">
                Documentation
              </Link>
              <CopyCommand command="gpulayer up" />
            </div>
            <div className="mt-16">
              <Terminal tabs={TERMINAL_TABS} />
            </div>
          </div>
        </section>

        <section className="border-b border-border bg-card">
          <WorksWith />
        </section>

        {/* GPU benchmarks */}
        <section id="gpu-benchmarks" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader
              eyebrow="Benchmarks"
              title="GPUlayer Trainer against PyTorch Lightning"
              lede="Same model, data and optimizer, each trainer with its defaults. The GPUlayer Trainer was faster in 10 of 11 GPU cases and every CPU case; each number has a script and a log in the repository."
            />
            <div className="mt-12 grid gap-6 lg:grid-cols-2">
              {[
                ["training-step", "Training step time on one Tesla T4: GPUlayer and PyTorch Lightning"],
                ["multigpu-step", "Step time on 1, 2 and 4 RTX 3070 GPUs: GPUlayer and PyTorch Lightning"],
                ["apple-cpu-step", "Step time on an Apple M2 Pro GPU and CPU: GPUlayer and PyTorch Lightning"],
              ].map(([name, alt]) => (
                <div key={name}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/docs-assets/perf/${name}-light.svg`}
                    alt={alt}
                    className="w-full rounded-xl border border-border dark:hidden"
                    width={760}
                    height={352}
                  />
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/docs-assets/perf/${name}-dark.svg`}
                    alt=""
                    aria-hidden
                    className="hidden w-full rounded-xl border border-border dark:block"
                    width={760}
                    height={352}
                  />
                </div>
              ))}
            </div>
            <dl className="mt-10 grid gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
              {GPU_RESULTS.map((g) => (
                <div key={g.gpu} className="bg-background p-6">
                  <dt className="text-sm font-medium">{g.gpu}</dt>
                  <dd className="mt-2 text-sm text-muted-foreground">{g.result}</dd>
                </div>
              ))}
            </dl>
            <Link href="/docs/get-started/performance-and-comparison/" className="link-arrow mt-8">
              All results, method and logs <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* Quickstart */}
        <section id="quickstart" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader
              eyebrow="Quickstart"
              title="Start on your laptop, add machines later"
            />
            <div className="mt-12">
              <Quickstart />
            </div>
            <Link href="/docs/get-started/quickstart/" className="link-arrow mt-12">
              Full quickstart, with the output of every step <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* Features */}
        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader eyebrow="Capabilities" title="What the Trainer handles" />
            <div className="mt-14 grid overflow-hidden rounded-xl border border-border sm:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map(({ icon: Icon, title, body }) => (
                <div key={title} className="-mb-px -mr-px border-b border-r border-border p-8">
                  <Icon className="h-5 w-5 text-muted-foreground" strokeWidth={1.75} />
                  <h3 className="mt-5 font-medium tracking-[-0.01em]">{title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Placement spotlight */}
        <section className="border-b border-border bg-card">
          <div className="mx-auto grid max-w-6xl items-center gap-12 px-6 py-24 lg:grid-cols-2">
            <div>
              <SectionHeader
                eyebrow="Placement"
                title="A run that cannot start fails at once, with the reason"
                lede="Workers report their GPUs, memory, drivers and PyTorch build. A run goes only to machines that can train it."
              />
              <Link href="/docs/architecture/scheduling/" className="link-arrow mt-8">
                How scheduling works <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-950 shadow-frame">
              <div className="border-b border-neutral-800 px-4 py-2.5 font-mono text-[11px] text-neutral-500">
                gpulayer job status job-1ebd35a459b9
              </div>
              <pre className="overflow-x-auto p-5 font-mono text-[12.5px] leading-[1.8] text-neutral-300">
                <span className="text-neutral-500">scheduling: </span>
                <span className="text-rose-400">no eligible worker</span>
                {"\n  gpu-box-1    needs 2 usable cuda device(s), worker has 1"}
                {"\n  mac-studio   needs cuda, worker has mps"}
                {"\n  laptop       torch on this worker cannot use CUDA"}
                {"\n  "}
                <span className="text-neutral-100">gpu-box-2    eligible</span>
              </pre>
            </div>
          </div>
        </section>

        {/* With and without */}
        <section id="comparison" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader eyebrow="Before and after" title="Training with and without GPUlayer" />
            <div className="mt-14 overflow-x-auto rounded-xl border border-border bg-background">
              <table className="w-full min-w-[720px] text-left text-sm">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground">
                    <th className="w-[28%] px-6 py-3 font-medium" />
                    <th className="px-6 py-3 font-medium">Without</th>
                    <th className="px-6 py-3 font-medium text-foreground">With GPUlayer</th>
                  </tr>
                </thead>
                <tbody>
                  {COMPARISON.map((row) => (
                    <tr key={row.situation} className="border-b border-border align-top last:border-0">
                      <td className="px-6 py-4 font-medium">{row.situation}</td>
                      <td className="px-6 py-4 text-muted-foreground">
                        <span className="flex gap-2.5">
                          <Minus className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground/60" />
                          {row.without}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className="flex gap-2.5">
                          <Check className="mt-0.5 h-4 w-4 shrink-0" />
                          <span>
                            <Inline text={row.with} />
                          </span>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-6 max-w-3xl text-sm text-muted-foreground">
              With one GPU on the machine you are using, a plain training loop is simpler. Jobs are not sandboxed, so
              GPUlayer is for people who trust each other.{" "}
              <Link href="/docs/#when-not-to-use-it" className="link-arrow">
                When to use it
              </Link>
            </p>
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader
              eyebrow="Architecture"
              title="How a run gets to the GPUs"
              lede="The control plane runs inside your process or as a small server you host. A worker on each machine polls it, so machines can sit behind NAT, and starts the ranks of each run."
            />
            <div className="mt-14 rounded-xl border border-border bg-card p-4 md:p-10">
              <ArchitectureDiagram />
            </div>
            <Link href="/docs/architecture/overview/" className="link-arrow mt-8">
              Architecture overview <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* Status */}
        <section className="border-b border-border bg-card">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader
              eyebrow="Status"
              title="What works today"
            />
            <dl className="mt-12 divide-y divide-border rounded-xl border border-border bg-background">
              {STATUS.map((s) => (
                <div key={s.label} className="grid gap-2 px-6 py-4 sm:grid-cols-[240px_1fr]">
                  <dt className="flex items-center gap-2.5 text-sm font-medium">
                    <span className={`h-2 w-2 rounded-full ${s.dot}`} />
                    {s.label}
                  </dt>
                  <dd className="text-sm text-muted-foreground">{s.items}</dd>
                </div>
              ))}
            </dl>
            <Link href="/docs/project/status/" className="link-arrow mt-8">
              Status and roadmap <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* CTA */}
        <section>
          <div className="mx-auto flex max-w-6xl flex-col items-start gap-8 px-6 py-24 md:flex-row md:items-end md:justify-between">
            <div className="max-w-xl">
              <h2 className="section-title">Try it on your own machines</h2>
              <p className="lede mt-4">
                The quickstart trains one PyTorch file in your process, with DDP, on two nodes and on a GPU, in
                about ten minutes.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link href="/docs/get-started/quickstart/" className="btn-primary">
                Start the quickstart <ArrowRight className="h-4 w-4" />
              </Link>
              <a href={REPO} target="_blank" rel="noreferrer" className="btn-secondary">
                View on GitHub
              </a>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
