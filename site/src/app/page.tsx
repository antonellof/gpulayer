import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  Check,
  FileCode2,
  Flame,
  Library,
  Minus,
  Network,
  ScrollText,
} from "lucide-react";
import { ArchitectureDiagram } from "@/components/landing/architecture-diagram";
import { CopyCommand } from "@/components/landing/copy-command";
import { LayerStack } from "@/components/landing/layer-stack";
import { VERSION } from "@/lib/version";
import { WorksWith } from "@/components/landing/logos";
import { Quickstart } from "@/components/landing/quickstart";
import { Terminal, type TerminalTab } from "@/components/landing/terminal";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";
import { REPO } from "@/lib/docs";

// Output below is copied from real runs on an Apple Silicon Mac (see the quickstart).
const TERMINAL_TABS: TerminalTab[] = [
  {
    label: "Command",
    title: "bash",
    lines: [
      {
        t: "cmd",
        s: "gpulayer submit --source examples/shell_job --input sales.csv -- bash report.sh",
      },
      { t: "dim", s: "submitted job-88eeb3960503 (independent, target local)" },
      { t: "out", s: "running on my-mac, accelerator=cpu" },
      { t: "out", s: "east,99.99\nnorth,162.75\nsouth,90.00" },
      { t: "ok", s: "job-88eeb3960503 succeeded" },
    ],
  },
  {
    label: "Python function",
    title: "matmul.py · MLX on the Apple GPU",
    lines: [
      {
        t: "code",
        s: `@gl.remote(accelerator="mps", frameworks={"mlx": ">=0.10"})
def matmul_benchmark(n: int) -> dict:
    import mlx.core as mx
    ...

matmul_benchmark.map([1024, 2048, 4096])`,
      },
      { t: "cmd", s: "python examples/mlx_metal/matmul.py" },
      { t: "out", s: "{'n': 1024, 'gpu_s': 0.0013, 'gpu_gflops': 1716.0}" },
      { t: "out", s: "{'n': 2048, 'gpu_s': 0.0042, 'gpu_gflops': 4118.0}" },
      { t: "out", s: "{'n': 4096, 'gpu_s': 0.0316, 'gpu_gflops': 4349.9}" },
    ],
  },
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

const FEATURES = [
  {
    icon: FileCode2,
    title: "Any command",
    body: "Scripts and binaries in any language run as jobs, with inputs staged and output files returned.",
  },
  {
    icon: Library,
    title: "Any library",
    body: "PyTorch, JAX, MLX, CuPy and NumPy are detected on each worker and required per job.",
  },
  {
    icon: Boxes,
    title: "Any machine you can reach",
    body: "Your laptop, servers that join with one command, cloud VMs. NVIDIA CUDA and Apple Metal.",
  },
  {
    icon: Network,
    title: "Distributed when needed",
    body: "Worker groups are placed together, with ranks and rendezvous set up. Ranks start with the standard torchrun environment, without torchrun's start-up cost.",
  },
  {
    icon: ScrollText,
    title: "Results come back",
    body: "Logs stream while a job runs. Output files and checkpoints download when it ends.",
  },
  {
    icon: Flame,
    title: "PyTorch integration",
    body: "A small Trainer, launch() for existing scripts, and helpers for devices, DDP and precision.",
  },
];

const COMPARISON: { situation: string; without: string; with: string }[] = [
  {
    situation: "Code on a laptop, GPU somewhere else",
    without: "ssh, copy code, rebuild the environment, run in tmux, copy results back",
    with: "`gpulayer submit --target remote -- <cmd>`",
  },
  {
    situation: "Several GPU machines",
    without: "track free machines by hand; jobs crash on the wrong GPU, driver or library",
    with: "one queue; each job lands where it fits, or is rejected with the reason",
  },
  {
    situation: "NVIDIA boxes and Apple Silicon Macs",
    without: "separate scripts and setup per machine",
    with: "ask for `cuda` or `mps`; one interface for both",
  },
  {
    situation: "A batch of independent tasks",
    without: "loops and ad-hoc scripts",
    with: "`fn.map([...])` in parallel across free workers",
  },
  {
    situation: "Multi-node training",
    without: "torchrun on every node with matching ranks, address and port",
    with: "`Trainer(num_nodes=2, devices=4)`",
  },
];

const STATUS: { label: string; dot: string; items: string }[] = [
  {
    label: "Tested",
    dot: "bg-emerald-500",
    items:
      "Local execution, SDK and CLI · scheduling · servers and workers over HTTPS (one host) · DDP, 2 nodes × 2 processes (one host) · MLX, NumPy and shell jobs",
  },
  { label: "Hardware-validated", dot: "bg-emerald-500", items: "Apple Metal (MPS) · NVIDIA CUDA on one GPU (Tesla T4) · workers on separate machines" },
  { label: "Implemented, not validated", dot: "bg-amber-500", items: "Several GPUs per machine · NCCL across machines" },
  { label: "Simulated", dot: "bg-amber-500", items: "Cloud provisioning (FakeProvider)" },
  {
    label: "Planned",
    dot: "bg-neutral-400",
    items: "A real cloud provider · long-running services · hosted control plane",
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
      <main className="flex-1">
        {/* Hero */}
        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 pb-16 pt-20 lg:pt-28">
            <Link
              href="/docs/project/status/"
              className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              v{VERSION} · tested on CPU, Apple Silicon and NVIDIA
              <ArrowRight className="h-3 w-3" />
            </Link>
            <h1 className="mt-6 max-w-4xl text-5xl font-semibold leading-[1.02] tracking-[-0.045em] sm:text-6xl lg:text-7xl">
              Run any workload on every GPU and CPU you can reach.
            </h1>
            <p className="lede mt-6 max-w-2xl sm:text-xl">
              GPUlayer sits between your code and your GPUs and CPUs: your laptop, remote servers, clusters, data
              centers and cloud VMs. Commands, Python functions and PyTorch training go to a machine that can run them,
              and the results come back.
            </p>
            <p className="mt-4 max-w-2xl text-sm text-muted-foreground">
              Its PyTorch Trainer ran up to 1.38x faster per step than PyTorch Lightning on an NVIDIA T4, and started
              sooner.{" "}
              <Link href="/docs/get-started/performance-and-comparison/" className="link-arrow">
                Benchmarks and method <ArrowRight className="h-3 w-3" />
              </Link>
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

        {/* Quickstart */}
        <section id="quickstart" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader
              eyebrow="Quickstart"
              title="From one machine to all of your GPUs"
              lede="Start on your laptop with nothing to set up. Add machines when you need them."
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
            <SectionHeader eyebrow="Capabilities" title="What GPUlayer handles for you" />
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
                title="Every decision comes with a reason"
                lede="Workers report their GPUs, memory, drivers and installed frameworks. Each job goes to a worker that has what it asks for. When none does, it is rejected before it starts, and you see why for every worker."
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
                {"\n  mac-studio   framework 'jax' not installed"}
                {"\n  laptop       torch on this worker cannot use CUDA"}
                {"\n  "}
                <span className="text-neutral-100">gpu-box-2    eligible</span>
              </pre>
            </div>
          </div>
        </section>

        {/* Positioning */}
        <section className="border-b border-border">
          <div className="mx-auto grid max-w-6xl gap-14 px-6 py-24 lg:grid-cols-[1fr_1.35fr]">
            <div>
              <SectionHeader
                eyebrow="Positioning"
                title="The layer underneath your framework"
                lede="Frameworks decide how a computation uses the hardware they are given. GPUlayer decides which hardware runs it, gets code and data there, starts it, and returns the results."
              />
              <Link href="/compare/" className="link-arrow mt-8">
                Compare with PyTorch Lightning, Ray, SkyPilot and others{" "}
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <LayerStack />
          </div>
          <div className="mx-auto max-w-6xl px-6 pb-24">
            {/* docs/assets, generated by scripts/make_diagram.py; the site toggles themes with a class, so
                both variants are present and CSS shows the matching one. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/docs-assets/architecture-light.svg"
              alt="Your workloads (commands, Python functions, PyTorch training, rendering) go through GPUlayer to the machines you can reach: this machine, servers that joined, cloud VMs."
              className="w-full rounded-2xl border border-border dark:hidden"
              width={980}
              height={460}
            />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/docs-assets/architecture-dark.svg"
              alt=""
              aria-hidden
              className="hidden w-full rounded-2xl border border-border dark:block"
              width={980}
              height={460}
            />
          </div>
        </section>

        {/* With and without */}
        <section id="comparison" className="border-b border-border bg-card">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader eyebrow="Before and after" title="With and without GPUlayer" />
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
              On a single machine with a single GPU, running your code directly is simpler. GPUlayer runs jobs that
              finish (long-running services are planned) and is meant for people who trust each other.{" "}
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
              title="One job model and one scheduler for every environment"
              lede="Workers poll the control plane, so they can sit behind NAT. Code travels as source bundles and functions are called by name. The control plane runs inside your process or as a small self-hosted server."
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
              lede="Each capability is labelled by how it has been tested so far."
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
                The quickstart takes about fifteen minutes: a shell job, a NumPy job, then one PyTorch file trained
                in several places.
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
