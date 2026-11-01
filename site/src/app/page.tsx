import Link from "next/link";
import {
  ArrowRight,
  Check,
  Cloud,
  Minus,
} from "lucide-react";
import { HeroLandscape } from "@/components/landing/hero-landscape";
import { VERSION } from "@/lib/version";
import { FeatureArt } from "@/components/landing/feature-art";
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

// Replays for the terminal: the commands and APIs are GPUlayer's, the output lines use the
// formats it prints, and the runs (models, timings, metrics) are illustrative.
const A100 = (host: string) =>
  [0, 1, 2, 3].map((i) => ({ id: `${host}:${i}`, host, name: "4x A100 80GB (NVLink)" }));
const RTX = [0, 1, 2, 3].map((i) => ({ id: `gpu-box:${i}`, host: "gpu-box", name: "4x RTX 4090" }));

const TERMINAL_TABS: TerminalTab[] = [
  {
    label: "Fine-tune on 8 GPUs",
    title: "ResNet-50 on 2 machines x 4 A100s, from a laptop",
    gpus: [...A100("a100-1"), ...A100("a100-2")],
    lines: [
      {
        t: "code",
        s: `model = ImageClassifier("resnet50", num_classes=100)
trainer = glt.Trainer(accelerator="cuda", devices=4, num_nodes=2,
                      precision="bf16-mixed", max_epochs=10)
trainer.fit(model, datamodule=ImageNet100())`,
        wait: 300,
      },
      { t: "cmd", s: "python finetune.py" },
      { t: "dim", s: "[gpulayer] job-7c1e04b2a9f3: finetune.py on 2 node(s) x 4 cuda process(es), target remote", wait: 700 },
      { t: "dim", s: "scheduling: placed on worker group [a100-1, a100-2] (same zone us-east-1b)", wait: 500 },
      { t: "busy", gpus: "all", wait: 600 },
      { t: "out", s: "[rank0] [gpulayer] training on cuda, rank 0/8, precision bf16-mixed, 10 epoch(s)", wait: 400 },
      { t: "out", s: "[rank0] epoch 0 step 50 train_loss=2.9134", wait: 1100 },
      { t: "out", s: "[rank0] epoch 0 step 100 train_loss=1.8420", wait: 900 },
      { t: "out", s: "[rank0] epoch 0 done in 63.2s: train_loss=1.6102 val_loss=1.1873 val_acc=0.7124", wait: 900 },
      { t: "out", s: "[rank0] epoch 1 done in 61.9s: train_loss=0.9877 val_loss=0.8215 val_acc=0.7981", wait: 1000 },
      { t: "dim", s: "...", wait: 700 },
      { t: "out", s: "[rank0] epoch 9 done in 61.8s: train_loss=0.3311 val_loss=0.5402 val_acc=0.8817", wait: 1100 },
      { t: "busy", gpus: "none", wait: 300 },
      { t: "ok", s: "[gpulayer] job-7c1e04b2a9f3 succeeded; 4 output file(s) in gpulayer_runs/job-7c1e04b2a9f3", wait: 500 },
      { t: "out", s: "trained weights loaded into model; checkpoint: gpulayer_runs/job-7c1e04b2a9f3/node0/checkpoints/last.ckpt" },
    ],
  },
  {
    label: "Your GPU fleet",
    title: "gpulayer up, two machines join, a BERT fine-tune lands on the free GPUs",
    gpus: [
      { id: "my-mac", host: "my-mac", name: "Apple M2 Pro" },
      ...RTX,
      { id: "cloud-l4", host: "cloud-l4", name: "1x L4 (cloud VM)" },
    ],
    lines: [
      { t: "dim", s: "# the server: any machine, here a Mac" },
      { t: "cmd", s: "gpulayer up" },
      { t: "ok", s: "GPUlayer is up at https://10.0.0.5:8470  (TLS, self-signed certificate pinned by fingerprint)", wait: 800 },
      { t: "out", s: "  gpulayer join https://10.0.0.5:8470 --token glj_... --fingerprint sha256:..." },
      { t: "dim", s: "# a workstation and a cloud VM paste the join line", wait: 600 },
      { t: "cmd", s: "gpulayer join https://10.0.0.5:8470 --token glj_... --name gpu-box" },
      { t: "ok", s: "worker wkr-3d3577f2b58c registered (cuda: 4x NVIDIA GeForce RTX 4090)", wait: 700 },
      { t: "cmd", s: "gpulayer join https://10.0.0.5:8470 --token glj_... --name cloud-l4" },
      { t: "ok", s: "worker wkr-91c0ae54d7e2 registered (cuda: NVIDIA L4)", wait: 700 },
      { t: "cmd", s: "gpulayer worker status" },
      { t: "out", s: "wkr-ed1b305e5b95   my-mac     idle   remote mps    tasks=0", wait: 300 },
      { t: "out", s: "wkr-3d3577f2b58c   gpu-box    idle   remote cuda   tasks=0", wait: 120 },
      { t: "out", s: "wkr-91c0ae54d7e2   cloud-l4   idle   remote cuda   tasks=0", wait: 120 },
      { t: "cmd", s: "python finetune_bert.py --devices 4", wait: 700 },
      { t: "dim", s: "[gpulayer] job-b42d9e17c0a6: finetune_bert.py on 1 node(s) x 4 cuda process(es), target remote", wait: 700 },
      { t: "dim", s: "scheduling: placed on gpu-box (4 free GPUs; my-mac: no CUDA device; cloud-l4: has 1 GPU, needs 4)", wait: 500 },
      { t: "busy", gpus: RTX.map((g) => g.id), wait: 600 },
      { t: "out", s: "[rank0] [gpulayer] training on cuda, rank 0/4, precision 16-mixed, 3 epoch(s)", wait: 400 },
      { t: "out", s: "[rank0] epoch 0 done in 88.4s: train_loss=0.3627 val_loss=0.2490 val_acc=0.9025", wait: 1300 },
      { t: "out", s: "[rank0] epoch 2 done in 87.9s: train_loss=0.1218 val_loss=0.2311 val_acc=0.9243", wait: 1300 },
      { t: "busy", gpus: "none", wait: 300 },
      { t: "ok", s: "[gpulayer] job-b42d9e17c0a6 succeeded; 3 output file(s) in gpulayer_runs/job-b42d9e17c0a6", wait: 500 },
    ],
  },
  {
    label: "Sweep",
    title: "a learning-rate sweep on every free GPU",
    gpus: [...RTX, { id: "cloud-l4", host: "cloud-l4", name: "1x L4 (cloud VM)" }],
    lines: [
      {
        t: "code",
        s: `@gl.remote(accelerator="cuda", gpus=1)
def train(lr: float) -> dict:
    ...  # fine-tune for 2 epochs, return the validation accuracy

results = train.map([1e-5, 2e-5, 3e-5, 5e-5, 1e-4])`,
        wait: 300,
      },
      { t: "cmd", s: "python sweep.py" },
      { t: "dim", s: "[gpulayer] 5 jobs submitted; placed on gpu-box (GPUs 0-3) and cloud-l4", wait: 800 },
      { t: "busy", gpus: "all", wait: 500 },
      { t: "out", s: "{'lr': 3e-05, 'val_acc': 0.9243, 'worker': 'gpu-box'}", wait: 2200 },
      { t: "out", s: "{'lr': 2e-05, 'val_acc': 0.9220, 'worker': 'gpu-box'}", wait: 300 },
      { t: "out", s: "{'lr': 5e-05, 'val_acc': 0.9186, 'worker': 'gpu-box'}", wait: 400 },
      { t: "out", s: "{'lr': 1e-05, 'val_acc': 0.9071, 'worker': 'gpu-box'}", wait: 300 },
      { t: "busy", gpus: ["cloud-l4"], wait: 200 },
      { t: "out", s: "{'lr': 0.0001, 'val_acc': 0.8794, 'worker': 'cloud-l4'}", wait: 1500 },
      { t: "busy", gpus: "none", wait: 200 },
      { t: "ok", s: "best: {'lr': 3e-05, 'val_acc': 0.9243}", wait: 400 },
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
    art: "api" as const,
    title: "A complete training API",
    body: "Module, DataModule, hooks, callbacks, loggers, profilers and a CLI, with DDP, FSDP, DeepSpeed and tensor parallelism behind one Trainer.",
  },
  {
    art: "scale" as const,
    title: "One Trainer, any scale",
    body: "The same script runs in your process, as DDP on every GPU of a machine, or across machines with Trainer(num_nodes=N). DDP, FSDP and DeepSpeed are validated on GPUs.",
  },
  {
    art: "placement" as const,
    title: "Placed where it trains fastest",
    body: "A run goes only to machines that can train it, on their best-connected GPUs, in one rack or zone when possible. Otherwise it is rejected with a reason per machine.",
  },
  {
    art: "gpus" as const,
    title: "Every GPU you can reach",
    body: "Laptops, workstations and servers that join with one command, cloud GPUs on AWS and Vast.ai that join the same way, and Google Cloud VMs created for a run. NVIDIA CUDA and Apple Metal are validated.",
  },
  {
    art: "results" as const,
    title: "Logs, checkpoints, metrics",
    body: "Logs stream while training runs. Checkpoints and metrics come back to your machine, and a dashboard lists every run.",
  },
  {
    art: "loop" as const,
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
    situation: "An existing PyTorch DDP script",
    without: "a cluster scheduler or manual torchrun to leave one machine",
    with: "`glt.launch()` at the top, the rest unchanged",
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
              href="/docs/get-started/quickstart/"
              className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              One Trainer. Every machine.
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
              {FEATURES.map(({ art, title, body }) => (
                <div key={title} className="-mb-px -mr-px border-b border-r border-border p-8">
                  <div className="-mx-2 overflow-hidden rounded-lg border border-border">
                    <FeatureArt kind={art} />
                  </div>
                  <h3 className="mt-6 font-medium tracking-[-0.01em]">{title}</h3>
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

        {/* Dashboard */}
        <section id="dashboard" className="border-b border-border bg-card">
          <div className="mx-auto max-w-6xl px-6 py-24">
            <SectionHeader
              eyebrow="Dashboard"
              title="Every run and every machine, in one place"
              lede="gpulayer up serves a dashboard: jobs and their placement, per-rank logs while training runs, the workers with their GPUs, and the reason when a job fails."
            />
            <div className="mt-12 overflow-hidden rounded-xl border border-border shadow-frame">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/docs-assets/dashboard/overview-light.png"
                alt="The GPUlayer dashboard: running and finished jobs, workers and their GPUs"
                className="w-full dark:hidden"
                width={1440}
                height={1180}
                loading="lazy"
              />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/docs-assets/dashboard/overview-dark.png"
                alt=""
                aria-hidden
                className="hidden w-full dark:block"
                width={1440}
                height={1180}
                loading="lazy"
              />
            </div>
            <div className="mt-6 grid gap-6 sm:grid-cols-2">
              {[
                ["job-ddp-dark", "A distributed job: its placement and every rank's logs"],
                ["workers-dark", "The workers: their GPUs, frameworks, load and memory"],
              ].map(([name, caption]) => (
                <figure key={name}>
                  <div className="overflow-hidden rounded-xl border border-border">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`/docs-assets/dashboard/${name}.png`}
                      alt={caption}
                      className="w-full"
                      width={1440}
                      height={1000}
                      loading="lazy"
                    />
                  </div>
                  <figcaption className="mt-3 text-sm text-muted-foreground">{caption}</figcaption>
                </figure>
              ))}
            </div>
            <Link href="/docs/guides/monitoring/" className="link-arrow mt-8">
              Monitoring and the dashboard <ArrowRight className="h-3.5 w-3.5" />
            </Link>
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
