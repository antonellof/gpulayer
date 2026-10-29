import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  CircleCheck,
  CircleX,
  FileCode2,
  Flame,
  Library,
  Network,
  ScrollText,
  SearchCheck,
} from "lucide-react";
import { ArchitectureDiagram } from "@/components/landing/architecture-diagram";
import { LayerStack } from "@/components/landing/layer-stack";
import { Quickstart } from "@/components/landing/quickstart";
import { WorksWith } from "@/components/landing/logos";
import { Terminal, type TerminalTab } from "@/components/landing/terminal";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";
import { REPO } from "@/lib/docs";

// Output below is copied from real runs on an Apple Silicon Mac (see the quickstart).
const TERMINAL_TABS: TerminalTab[] = [
  {
    label: "any command",
    title: "~/gpulayer · bash",
    lines: [
      { t: "cmd", s: "gpulayer submit --source examples/shell_job --input sales.csv -- bash report.sh" },
      { t: "dim", s: "submitted job-88eeb3960503 (independent, target local)" },
      { t: "out", s: "running on my-mac, accelerator=cpu" },
      { t: "out", s: "east,99.99\nnorth,162.75\nsouth,90.00" },
      { t: "ok", s: "job-88eeb3960503 succeeded" },
    ],
  },
  {
    label: "any library",
    title: "~/gpulayer · matmul.py (MLX on the Apple GPU)",
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
      { t: "ok", s: "{'n': 4096, 'gpu_s': 0.0316, 'gpu_gflops': 4349.9}" },
    ],
  },
  {
    label: "pytorch",
    title: "~/gpulayer · train.py",
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
      { t: "out", s: "model is trained in this process too" },
    ],
  },
  {
    label: "placement",
    title: "~/gpulayer · asking for hardware this machine lacks",
    lines: [
      { t: "cmd", s: "python train.py --accelerator cuda --target local" },
      { t: "dim", s: "[gpulayer] job-e8061b7e6ccc: train.py on 1 node(s) x 1 cuda process(es)" },
      { t: "err", s: "[gpulayer] job-e8061b7e6ccc failed: unschedulable: no eligible worker" },
      { t: "err", s: "  wkr-10c5089a6044: needs 1 usable cuda device(s), worker has 0" },
      { t: "dim", s: "# rejected before it starts, with a reason per worker" },
    ],
  },
];

const FEATURES = [
  {
    icon: FileCode2,
    title: "Any workload",
    body: "Commands, scripts and binaries in any language. Anything that runs in a shell can be a job.",
  },
  {
    icon: Library,
    title: "Any library",
    body: "PyTorch, JAX, MLX, CuPy, NumPy and others are detected per worker and required per job.",
  },
  {
    icon: Boxes,
    title: "Any GPU you can reach",
    body: "Your laptop, servers that join, cloud VMs. NVIDIA CUDA and Apple Metal; ROCm is detection only for now.",
  },
  {
    icon: SearchCheck,
    title: "Placement that explains itself",
    body: "Jobs go to workers whose GPUs, memory and frameworks fit. When none fits, the job is rejected with a reason for each worker.",
  },
  {
    icon: Network,
    title: "Distributed when you need it",
    body: "A group of workers is placed together, with ranks and rendezvous set up for you. Launched with torchrun.",
  },
  {
    icon: ScrollText,
    title: "Results come back",
    body: "Logs stream while the job runs, and output files and checkpoints come back when it ends. Jobs can be cancelled or time out.",
  },
  {
    icon: Flame,
    title: "PyTorch integration",
    body: "A small Trainer and Module, launch() for existing scripts, and Runtime helpers for devices, DDP and precision.",
  },
];

const COMPARISON: { situation: string; without: string; with: string }[] = [
  {
    situation: "Code on a laptop, GPU somewhere else",
    without: "ssh, copy code, rebuild the environment, run in tmux, copy results back",
    with: "`gpulayer submit --target remote -- <cmd>` or `python train.py --target remote`",
  },
  {
    situation: "Several GPU machines",
    without: "track free boxes by hand; jobs crash on the wrong GPU, driver or library version",
    with: "one queue; each job lands where its accelerator and frameworks are available, or is rejected with the reason",
  },
  {
    situation: "Mixed hardware (NVIDIA boxes and Apple Silicon Macs)",
    without: "separate scripts and setup per machine",
    with: "one interface; CUDA jobs go to NVIDIA workers, Metal jobs to Macs",
  },
  {
    situation: "Batch of independent tasks (sweeps, rendering, data processing)",
    without: "loops and ad-hoc scripts",
    with: "`fn.map([...])` in parallel across free workers",
  },
  {
    situation: "Multi-node training",
    without: "torchrun on every node with matching ranks, addresses and ports",
    with: "`Trainer(num_nodes=2, devices=4)` or `launch(nodes=2, ...)`",
  },
];

const CONS = [
  "One machine, one GPU, interactive work: plain `python train.py` is simpler; GpuLayer adds a control plane, workers and tokens.",
  "Pre-alpha: multi-machine and NVIDIA runs not validated yet; automatic cloud provisioning is simulated.",
  "Trusted users only: workloads run as normal processes on the workers, without a sandbox.",
  "`@gl.remote` arguments and results must be JSON; code ships as files.",
  "The Trainer is small: no callbacks, early stopping or FSDP yet.",
  "Long-running services (an inference server that must stay up) are not supported yet; GpuLayer runs batch jobs.",
];

const STATUS: { label: string; tone: string; items: string[] }[] = [
  {
    label: "Tested",
    tone: "text-emerald-600 dark:text-emerald-400",
    items: [
      "Local execution, SDK and CLI",
      "Capability-aware scheduling",
      "DDP: 2 nodes x 2 processes on one host",
      "Remote worker over HTTPS (one host)",
      "Non-PyTorch workloads (MLX on Apple GPU, NumPy, shell) on one host",
    ],
  },
  {
    label: "Hardware-validated",
    tone: "text-emerald-600 dark:text-emerald-400",
    items: ["Apple MPS, single-process jobs"],
  },
  {
    label: "Implemented, not hardware-validated",
    tone: "text-amber-600 dark:text-amber-400",
    items: ["NVIDIA CUDA and NCCL"],
  },
  {
    label: "Simulated",
    tone: "text-amber-600 dark:text-amber-400",
    items: ["Cloud provisioning (FakeProvider)"],
  },
  {
    label: "Planned",
    tone: "text-muted-foreground",
    items: ["Real cloud provider", "Multi-machine validation", "Hosted control plane", "Long-running services"],
  },
];

/** Render `inline code` spans inside short table/list strings. */
function Inline({ text }: { text: string }) {
  return (
    <>
      {text.split(/(`[^`]+`)/g).map((part, i) =>
        part.startsWith("`") && part.endsWith("`") ? (
          <code key={i} className="rounded bg-secondary px-1 py-0.5 font-mono text-[0.85em] text-foreground">
            {part.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex-1">
        {/* Hero: always dark */}
        <section className="relative overflow-hidden bg-[#070a10] text-white">
          <div className="hero-grid pointer-events-none absolute inset-0" aria-hidden />
          <div
            className="pointer-events-none absolute -left-40 -top-40 h-[520px] w-[520px] rounded-full bg-emerald-500/20 blur-[120px]"
            aria-hidden
          />
          <div
            className="pointer-events-none absolute -right-40 top-20 h-[480px] w-[480px] rounded-full bg-violet-500/20 blur-[120px]"
            aria-hidden
          />
          <div className="relative mx-auto grid max-w-6xl items-start gap-12 px-6 pb-20 pt-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] lg:pb-28 lg:pt-24">
            <div className="flex flex-col">
              <span className="mb-6 inline-flex w-fit items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                Pre-alpha · tested on CPU and Apple Silicon
              </span>
              <h1 className="font-display text-5xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
                Run any GPU workload on <span className="text-gradient">any GPU you can reach.</span>
              </h1>
              <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-300">
                One layer between your code and your GPUs: your laptop, workstations, servers and cloud VMs.
                Run commands in any language, Python functions with any library and distributed jobs. PyTorch
                gets a deeper integration on top.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-3">
                <Link
                  href="/docs/get-started/quickstart/"
                  className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-[#04120c] transition hover:bg-emerald-400"
                >
                  Get started <ArrowRight className="h-4 w-4" />
                </Link>
                <Link
                  href="/docs/"
                  className="inline-flex items-center rounded-lg border border-white/15 bg-white/5 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-white/10"
                >
                  Read the docs
                </Link>
                <a
                  href={REPO}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center px-3 py-2.5 text-sm font-medium text-slate-400 transition hover:text-white"
                >
                  GitHub
                </a>
              </div>
              <code className="mt-8 w-fit rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-xs text-slate-300">
                <span className="text-emerald-400">$</span> gpulayer up
                <span className="text-slate-500">{"   # then: gpulayer join <url> on each GPU machine"}</span>
              </code>
            </div>
            <Terminal tabs={TERMINAL_TABS} />
          </div>
          <div className="relative border-t border-white/10 bg-white/[0.02]">
            <WorksWith />
          </div>
        </section>

        {/* Quickstart */}
        <section id="quickstart" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-20">
            <p className="eyebrow">Quickstart</p>
            <h2 className="section-title mt-3">From one machine to all your GPUs</h2>
            <div className="mt-8">
              <Quickstart />
            </div>
            <Link
              href="/docs/get-started/quickstart/"
              className="mt-12 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
            >
              Full quickstart with every step and its output <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* Features */}
        <section className="border-b border-border bg-card/40">
          <div className="mx-auto max-w-6xl px-6 py-20">
            <p className="eyebrow">What you get</p>
            <h2 className="section-title mt-3">What GpuLayer handles</h2>
            <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map(({ icon: Icon, title, body }, i) => (
                <div
                  key={title}
                  className={`rounded-xl border border-border bg-background p-6 card-hover ${
                    i === 3 || i === 6 ? "lg:col-span-2" : ""
                  }`}
                >
                  <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-accent/20 text-primary">
                    <Icon className="h-5 w-5" />
                  </span>
                  <h3 className="mt-4 font-display text-lg font-semibold">{title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{body}</p>
                  {i === 3 && (
                    <pre className="mt-4 overflow-x-auto rounded-lg bg-[#0b0f17] p-3 font-mono text-[12px] leading-relaxed text-slate-300">
                      <span className="text-rose-400">scheduling: no eligible worker</span>
                      {"\n  gpu-box-1: needs 2 usable cuda device(s), worker has 1"}
                      {"\n  mac-studio: framework 'jax' not installed"}
                      {"\n  "}
                      <span className="text-emerald-400">gpu-box-2: eligible</span>
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Where it sits */}
        <section className="border-b border-border">
          <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 lg:grid-cols-[1fr_1.4fr]">
            <div>
              <p className="eyebrow">Positioning</p>
              <h2 className="section-title mt-3">The layer underneath your framework</h2>
              <p className="mt-4 leading-relaxed text-muted-foreground">
                Frameworks decide how a computation uses the hardware they are given. GpuLayer decides which
                hardware runs it, gets code and data there, starts it, and returns the results: for training,
                inference, rendering, simulation or data jobs, in PyTorch, JAX, MLX, CuPy, NumPy or no Python
                at all.
              </p>
              <Link
                href="/compare/"
                className="mt-6 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
              >
                Compare with PyTorch Lightning, Ray, SkyPilot and others <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
            <LayerStack />
          </div>
        </section>

        {/* With and without */}
        <section id="comparison" className="border-b border-border bg-card/40">
          <div className="mx-auto max-w-6xl px-6 py-20">
            <p className="eyebrow">Before and after</p>
            <h2 className="section-title mt-3">With and without GpuLayer</h2>
            <div className="mt-12 space-y-3">
              {COMPARISON.map((row) => (
                <div
                  key={row.situation}
                  className="grid overflow-hidden rounded-xl border border-border bg-background md:grid-cols-[220px_1fr_1fr]"
                >
                  <div className="border-b border-border p-4 font-display font-semibold md:border-b-0 md:border-r">
                    {row.situation}
                  </div>
                  <div className="flex gap-2.5 p-4 text-sm text-muted-foreground">
                    <CircleX className="mt-0.5 h-4 w-4 shrink-0 text-rose-500/70" />
                    <span>{row.without}</span>
                  </div>
                  <div className="flex gap-2.5 border-t border-border bg-primary/[0.05] p-4 text-sm md:border-l md:border-t-0">
                    <CircleCheck className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    <span>
                      <Inline text={row.with} />
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-12 rounded-xl border border-border bg-background p-6">
              <h3 className="font-display text-lg font-semibold">When not to use it</h3>
              <ul className="mt-4 grid gap-x-8 gap-y-2.5 text-sm text-muted-foreground md:grid-cols-2">
                {CONS.map((c) => (
                  <li key={c} className="flex gap-2">
                    <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-muted-foreground/60" />
                    <span>
                      <Inline text={c} />
                    </span>
                  </li>
                ))}
              </ul>
              <Link
                href="/docs/get-started/why-gpulayer/"
                className="mt-5 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
              >
                Why GpuLayer <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-20">
            <p className="eyebrow">Architecture</p>
            <h2 className="section-title mt-3">One job model, one scheduler, every environment</h2>
            <p className="mt-4 max-w-3xl leading-relaxed text-muted-foreground">
              Workers poll the control plane, so they can sit behind NAT. Code travels as source bundles and
              functions are called by name. The control plane runs inside your process or as a small
              self-hosted server.
            </p>
            <div className="mt-10 rounded-2xl border border-border bg-card p-4 shadow-soft md:p-8">
              <ArchitectureDiagram />
            </div>
            <Link
              href="/docs/architecture/overview/"
              className="mt-5 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
            >
              Architecture overview <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* Status */}
        <section className="border-b border-border bg-card/40">
          <div className="mx-auto max-w-6xl px-6 py-20">
            <p className="eyebrow">Status</p>
            <h2 className="section-title mt-3">What works today</h2>
            <p className="mt-4 max-w-3xl text-muted-foreground">
              Each capability is labelled by how it has been tested so far.
            </p>
            <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {STATUS.map((s) => (
                <div key={s.label} className="rounded-xl border border-border bg-background p-5">
                  <p className={`font-mono text-[11px] font-semibold uppercase tracking-[0.12em] ${s.tone}`}>
                    {s.label}
                  </p>
                  <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                    {s.items.map((it) => (
                      <li key={it}>{it}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="relative overflow-hidden bg-[#070a10] text-white">
          <div className="hero-grid pointer-events-none absolute inset-0" aria-hidden />
          <div
            className="pointer-events-none absolute left-1/2 top-0 h-[300px] w-[700px] -translate-x-1/2 rounded-full bg-emerald-500/15 blur-[100px]"
            aria-hidden
          />
          <div className="relative mx-auto flex max-w-6xl flex-col items-center px-6 py-20 text-center">
            <h2 className="font-display text-3xl font-semibold tracking-tight sm:text-4xl">
              Try it on your own machines
            </h2>
            <p className="mt-4 max-w-xl text-slate-300">
              The quickstart takes about fifteen minutes. You run a shell job and a NumPy job, then train one
              PyTorch file in several places.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <Link
                href="/docs/get-started/quickstart/"
                className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-[#04120c] transition hover:bg-emerald-400"
              >
                Start the quickstart <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href={REPO}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center rounded-lg border border-white/15 bg-white/5 px-5 py-2.5 text-sm font-medium transition hover:bg-white/10"
              >
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
