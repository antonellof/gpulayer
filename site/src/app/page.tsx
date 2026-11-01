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
import { HeroLandscape } from "@/components/landing/hero-landscape";
import { VERSION } from "@/lib/version";
import { WorksWith } from "@/components/landing/logos";
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
  // Run on one Mac: `gpulayer up`, a second worker joined from another terminal, and a client
  // in a third. The address is the one `gpulayer up` printed; tokens are elided.
  {
    label: "Server + client",
    title: "gpulayer up, a worker joins, your laptop trains on both",
    lines: [
      { t: "dim", s: "# machine 1: start the server; it runs jobs too" },
      { t: "cmd", s: "gpulayer up" },
      { t: "ok", s: "GPUlayer is up at https://192.168.1.56:8470  (TLS, self-signed certificate pinned by fingerprint)" },
      { t: "out", s: "  gpulayer join https://192.168.1.56:8470 --token glj_… --fingerprint sha256:…" },
      { t: "dim", s: "# machine 2: a GPU box or cloud VM pastes the join line" },
      { t: "cmd", s: "gpulayer join https://192.168.1.56:8470 --token glj_… --fingerprint sha256:…" },
      { t: "ok", s: "worker wkr-3d3577f2b58c registered" },
      { t: "dim", s: "# your laptop: the client" },
      { t: "cmd", s: "export GPULAYER_URL=https://192.168.1.56:8470 GPULAYER_TOKEN=glc_… GPULAYER_FINGERPRINT=sha256:…" },
      { t: "cmd", s: "python train.py --accelerator cpu --nodes 2" },
      { t: "dim", s: "[gpulayer] job-77b0944e7dba: train.py on 2 node(s) x 1 cpu process(es), target remote" },
      { t: "out", s: "[rank0] epoch 4 done in 0.047s: train_loss=0.5561 val_loss=0.4698" },
      { t: "ok", s: "[gpulayer] job-77b0944e7dba succeeded; 3 output file(s) in gpulayer_runs/job-77b0944e7dba" },
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
  { gpu: "Tesla T4", result: "1.08 to 1.38x per step; fp32, bf16, fp16 and torch.compile" },
  { gpu: "4x RTX 3070 (PCIe)", result: "1.45 to 1.60x on 2 and 4 GPUs; 1.12x slower on one GPU with a CNN" },
  { gpu: "8 machines, 1x T4 each (NCCL)", result: "MLP 1.08 to 1.29x faster; CNN 1.01 to 1.09x slower" },
  { gpu: "2x A100 SXM4 (NVLink)", result: "DDP over NCCL and NVLink; CUDA checks passed" },
  { gpu: "Apple M2 Pro", result: "1.02 to 1.80x on the GPU and the CPU" },
];

const FEATURES = [
  {
    icon: Library,
    title: "Lightning code, unchanged",
    body: "LightningModule, DataModule, callbacks, loggers, profilers and the CLI keep Lightning's names. Most code moves with an import change.",
  },
  {
    icon: Flame,
    title: "One Trainer, any scale",
    body: "The same script runs in your process, as DDP on every GPU of a machine, or across machines with Trainer(num_nodes=N). FSDP is experimental.",
  },
  {
    icon: Boxes,
    title: "Placed where it trains fastest",
    body: "A run goes only to machines that can train it, on their best-connected GPUs, in one rack or zone when possible. Otherwise it is rejected with a reason per machine.",
  },
  {
    icon: Cloud,
    title: "Every GPU you can reach",
    body: "Laptops, workstations and servers that join with one command, cloud GPUs on AWS and Vast.ai that join the same way, and Google Cloud VMs created for a run. NVIDIA CUDA and Apple Metal are validated.",
  },
  {
    icon: ScrollText,
    title: "Logs, checkpoints, metrics",
    body: "Logs stream while training runs. Checkpoints and metrics come back to your machine, and a dashboard lists every run.",
  },
  {
    icon: FileCode2,
    title: "Your own loop works too",
    body: "Any training script runs anywhere with two launch() lines; Runtime handles device, DDP and precision in a custom loop.",
  },
];

const COMPARISON: { situation: string; without: string; with: string }[] = [
  {
    situation: "Code on a laptop, GPUs elsewhere",
    without: "ssh, copy code, rebuild the environment, copy checkpoints back",
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
    situation: "A Lightning project",
    without: "a cluster scheduler or manual torchrun to leave one machine",
    with: "`import gpulayer.pytorch as glt`, same code",
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
        <section className="relative overflow-hidden border-b border-border">
          <HeroLandscape />
          <div className="relative mx-auto max-w-6xl px-6 pb-16 pt-20 lg:pt-28">
            <Link
              href="/docs/pytorch/from-lightning/"
              className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              PyTorch Lightning code moves over with one import change
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
              lede="Same model, data and optimizer, each trainer's defaults. Faster in 13 of 17 GPU cases and every CPU case; slower on a CNN on one GPU of one host and on a CNN across machines. Every number has a script and a log."
            />
            <div className="mt-12 grid gap-6 lg:grid-cols-2">
              {[
                ["multigpu-step", "Step time on 1, 2 and 4 RTX 3070 GPUs: GPUlayer and PyTorch Lightning"],
                ["nccl-step", "Step time with one T4 per machine on 2, 4 and 8 machines over NCCL: GPUlayer and PyTorch Lightning"],
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
            <dl className="mt-10 grid gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-2 lg:grid-cols-5">
              {GPU_RESULTS.map((g) => (
                <div key={g.gpu} className="bg-background p-6 sm:last:col-span-2 lg:last:col-span-1">
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

        {/* Features */}
        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader eyebrow="Capabilities" title="What you get" />
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

        {/* How it works */}
        <section id="how-it-works" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader
              eyebrow="How it works"
              title="Your script, the machines you can reach"
              lede="The Trainer asks the control plane for machines that can run it; workers on those machines start every process and send back logs, checkpoints and metrics. One command turns a machine into a server the others join."
            />
            <div className="mt-12 grid gap-10 lg:grid-cols-2">
              {[
                ["architecture", "Your training goes through GPUlayer to the machines you can reach", "Architecture", "/docs/architecture/overview/", 980, 460],
                ["server", "A GPUlayer server: your laptop and joined machines connect to the machine running gpulayer up over HTTPS with a pinned certificate", "Run a server (gpulayer up)", "/docs/guides/self-hosting/", 980, 490],
              ].map(([name, alt, label, href, w, h]) => (
                <figure key={name as string}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/docs-assets/${name}-light.svg`}
                    alt={alt as string}
                    className="w-full rounded-xl border border-border dark:hidden"
                    width={w as number}
                    height={h as number}
                  />
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/docs-assets/${name}-dark.svg`}
                    alt=""
                    aria-hidden
                    className="hidden w-full rounded-xl border border-border dark:block"
                    width={w as number}
                    height={h as number}
                  />
                  <figcaption className="mt-4">
                    <Link href={href as string} className="link-arrow">
                      {label} <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  </figcaption>
                </figure>
              ))}
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

        {/* CTA */}
        <section>
          <div className="mx-auto flex max-w-6xl flex-col items-start gap-8 px-6 py-24 md:flex-row md:items-end md:justify-between">
            <div className="max-w-xl">
              <h2 className="section-title">Try it on your own machines</h2>
              <p className="lede mt-4">
                Install with <code className="font-mono text-[0.9em]">curl -fsSL https://gpulayer.dev/install.sh | sh</code>{" "}
                and train one file in your process, with DDP and on two nodes, in about ten minutes.
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
