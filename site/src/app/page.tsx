import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  Cpu,
  FileCode2,
  Network,
  ScrollText,
  SearchCheck,
} from "lucide-react";
import { CodeBlock } from "@/components/docs/code-block";
import { ArchitectureDiagram } from "@/components/landing/architecture-diagram";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";
import { REPO } from "@/lib/docs";

const TRAINER = `import torch
from torch import nn
import gpulayer.pytorch as glt


class Regressor(glt.Module):
    def __init__(self, lr: float = 1e-2) -> None:
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


trainer = glt.Trainer(max_epochs=5, accelerator="auto")   # this process, best device
# trainer = glt.Trainer(accelerator="cpu", devices=2)      # 2-process DDP
# trainer = glt.Trainer(accelerator="cuda", devices=4, num_nodes=2, target="remote")
trainer.fit(Regressor(), train_loader, val_loader)`;

const STEPS = [
  {
    title: "Start a server on any machine",
    code: "gpulayer up",
  },
  {
    title: "Add your GPU machines",
    code: "gpulayer join <url> --token ... --fingerprint ...",
  },
  {
    title: "Train there from your laptop",
    code: "python train.py --target remote",
  },
];

const LOCAL_VARIANT =
  "git clone https://github.com/antonellof/gpulayer && cd gpulayer\nuv venv && uv pip install -e '.[dev]' torch\npython examples/trainer_regression/train.py";

const COMPARISON: { situation: string; without: string; with: string }[] = [
  {
    situation: "Laptop for code, GPU on another machine (office server or cloud VM)",
    without: "ssh in, copy code, rebuild the environment, run in tmux, copy checkpoints back",
    with: "`python train.py --target remote`: code shipped, logs streamed, checkpoint downloaded, trained weights back in your local model",
  },
  {
    situation: "Several GPU machines (team, lab, rented VMs)",
    without: "track by hand which box is free; jobs crash on the wrong GPU, CUDA build or memory",
    with: "one queue; each job goes to a machine that fits, or is rejected before it starts with the reason",
  },
  {
    situation: "Multi-node training",
    without:
      "on every node: same code and env, torchrun with matching node rank, master address and port; start together; clean up when one dies",
    with: "`Trainer(num_nodes=2, devices=4)`",
  },
  {
    situation: "Hyperparameter sweep",
    without: "sequential loop, or your own scripts to spread runs",
    with: "`train.map([...])` in parallel across free workers",
  },
  {
    situation: "Same code on Mac (MPS), CPU and CUDA",
    without: "device logic, DDP and sampler boilerplate, different launch commands",
    with: "`accelerator=\"auto\"`; DDP, sharding and precision handled",
  },
];

const CONS = [
  "One machine, one GPU, interactive work: plain `python train.py` is simpler; GpuLayer adds a control plane, workers and tokens.",
  "Pre-alpha: multi-machine and NVIDIA runs not validated yet; automatic cloud provisioning is simulated.",
  "Trusted users only: workloads run as normal processes on workers, not in a sandbox.",
  "`@gl.remote` arguments and results must be JSON; code ships as files.",
  "The Trainer is small: no callbacks, early stopping or FSDP yet.",
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

const FEATURES = [
  {
    icon: FileCode2,
    title: "One file, any target",
    body: "The same training script runs in-process, on another worker, or across nodes. Only the Trainer arguments change.",
  },
  {
    icon: SearchCheck,
    title: "Placement you can read",
    body: "Jobs land only on workers whose reported GPU, memory, Python and torch versions fit. Rejections list the reason for each worker.",
  },
  {
    icon: Network,
    title: "Distributed with torchrun",
    body: "GpuLayer allocates the worker group, assigns ranks and publishes the rendezvous. PyTorch does the collectives.",
  },
  {
    icon: Boxes,
    title: "Local, remote, cloud",
    body: "An embedded control plane needs no account. A self-hosted server adds remote workers; provider adapters add cloud capacity.",
  },
  {
    icon: ScrollText,
    title: "Logs, artifacts, checkpoints",
    body: "Logs stream back while the job runs. Output files and checkpoints return as content-addressed artifacts.",
  },
  {
    icon: Cpu,
    title: "CUDA and MPS aware",
    body: "Apple MPS and NVIDIA CUDA are separate backends with their own rules. Unified memory is reported as unified memory.",
  },
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
    items: ["Real cloud provider", "Multi-machine validation", "Hosted control plane"],
  },
];

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />
      <main className="flex-1">
        {/* Hero */}
        <section className="border-b border-border">
          <div className="mx-auto grid max-w-6xl gap-12 px-6 py-16 lg:grid-cols-[1fr_1.1fr] lg:py-24">
            <div className="flex flex-col justify-center">
              <span className="mb-4 w-fit rounded-full border border-border px-3 py-1 text-xs text-muted-foreground">
                Open source · pre-alpha
              </span>
              <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
                Run PyTorch on any GPU you can reach.
              </h1>
              <p className="mt-5 text-lg text-muted-foreground">
                GpuLayer places your training job on a compatible machine (this laptop, a GPU box on your network,
                or a group of nodes) and brings back logs, metrics and checkpoints. Write the training code once;
                choose where it runs with an argument.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  href="/docs/get-started/quickstart/"
                  className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
                >
                  Quickstart <ArrowRight className="h-4 w-4" />
                </Link>
                <Link
                  href="/docs/"
                  className="inline-flex items-center rounded-md border border-border px-4 py-2 text-sm font-medium hover:bg-secondary"
                >
                  Read the docs
                </Link>
                <a
                  href={REPO}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center rounded-md px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground"
                >
                  GitHub
                </a>
              </div>
            </div>
            <div className="min-w-0">
              <CodeBlock code={TRAINER} language="python" className="my-0 shadow-sm" />
            </div>
          </div>
        </section>

        {/* Quickstart */}
        <section id="quickstart" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <h2 className="text-2xl font-semibold tracking-tight">Quickstart in three steps</h2>
            <p className="mt-2 text-muted-foreground">
              Run a server on any machine, join your GPU boxes to it, and send training there from your laptop.
            </p>
            <div className="mt-8 grid gap-6 lg:grid-cols-3">
              {STEPS.map((s, i) => (
                <div key={s.title} className="min-w-0">
                  <p className="text-sm font-medium">
                    <span className="mr-2 inline-flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs text-primary">
                      {i + 1}
                    </span>
                    {s.title}
                  </p>
                  <CodeBlock code={s.code} language="bash" className="mt-3" />
                </div>
              ))}
            </div>
            <div className="mt-4 rounded-lg border border-border bg-card p-5">
              <p className="text-sm font-medium">No second machine? Zero-setup local run</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Without a server, GpuLayer runs an embedded control plane and a worker inside your process.
              </p>
              <CodeBlock code={LOCAL_VARIANT} language="bash" className="mb-0" />
            </div>
            <Link
              href="/docs/get-started/quickstart/"
              className="mt-4 inline-flex items-center gap-1 text-sm text-primary hover:underline"
            >
              Full step-by-step quickstart <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* With and without */}
        <section id="comparison" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <h2 className="text-2xl font-semibold tracking-tight">With and without GpuLayer</h2>
            <div className="mt-8 overflow-x-auto rounded-lg border border-border">
              <table className="w-full min-w-[640px] text-left text-sm">
                <thead className="bg-secondary text-xs uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="w-1/4 px-4 py-3 font-semibold">Situation</th>
                    <th className="px-4 py-3 font-semibold">Without GpuLayer</th>
                    <th className="px-4 py-3 font-semibold text-primary">With GpuLayer</th>
                  </tr>
                </thead>
                <tbody>
                  {COMPARISON.map((row) => (
                    <tr key={row.situation} className="border-t border-border align-top">
                      <td className="px-4 py-3 font-medium">{row.situation}</td>
                      <td className="px-4 py-3 text-muted-foreground">{row.without}</td>
                      <td className="bg-primary/5 px-4 py-3">
                        <Inline text={row.with} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <h3 className="mt-10 text-lg font-semibold">When not to use it</h3>
            <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-muted-foreground">
              {CONS.map((c) => (
                <li key={c}>
                  <Inline text={c} />
                </li>
              ))}
            </ul>
            <Link
              href="/docs/get-started/why-gpulayer/"
              className="mt-4 inline-flex items-center gap-1 text-sm text-primary hover:underline"
            >
              Why GpuLayer <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* Features */}
        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <h2 className="text-2xl font-semibold tracking-tight">What GpuLayer adds to PyTorch</h2>
            <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map(({ icon: Icon, title, body }) => (
                <div key={title} className="rounded-lg border border-border bg-card p-5">
                  <Icon className="h-5 w-5 text-primary" />
                  <h3 className="mt-3 font-medium">{title}</h3>
                  <p className="mt-1.5 text-sm text-muted-foreground">{body}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <h2 className="text-2xl font-semibold tracking-tight">How it fits together</h2>
            <p className="mt-2 max-w-3xl text-muted-foreground">
              One job model, one capability system and one scheduler for every environment. Workers poll the control
              plane, so they can sit behind NAT; code travels as source bundles, never as pickled objects.
            </p>
            <div className="mt-8 rounded-xl border border-border bg-background p-4 md:p-8">
              <ArchitectureDiagram />
            </div>
            <Link
              href="/docs/architecture/overview/"
              className="mt-4 inline-flex items-center gap-1 text-sm text-primary hover:underline"
            >
              Architecture overview <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>

        {/* Status */}
        <section>
          <div className="mx-auto max-w-6xl px-6 py-16">
            <h2 className="text-2xl font-semibold tracking-tight">What works today</h2>
            <p className="mt-2 text-muted-foreground">
              GpuLayer labels every capability by how it has been verified. Nothing below is claimed beyond its
              label.
            </p>
            <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              {STATUS.map((s) => (
                <div key={s.label} className="rounded-lg border border-border p-4">
                  <p className={`text-xs font-semibold uppercase tracking-wider ${s.tone}`}>{s.label}</p>
                  <ul className="mt-3 space-y-1.5 text-sm text-muted-foreground">
                    {s.items.map((it) => (
                      <li key={it}>{it}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <Link
              href="/docs/project/status/"
              className="mt-4 inline-flex items-center gap-1 text-sm text-primary hover:underline"
            >
              Status and roadmap <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
