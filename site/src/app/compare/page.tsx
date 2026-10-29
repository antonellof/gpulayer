import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";

export const metadata: Metadata = {
  title: "GpuLayer compared | PyTorch Lightning, Ray, SkyPilot, dstack, Modal and more",
  description:
    "How GpuLayer relates to PyTorch Lightning, Hugging Face Accelerate, Ray, SkyPilot, dstack, Modal, Slurm and Kubernetes: what each one does, where they overlap, and when to use which.",
};

type Row = {
  name: string;
  what: string;
  runsOn: string;
  machines: string;
  workloads: string;
  hosting: string;
  highlight?: boolean;
};

const ROWS: Row[] = [
  {
    name: "GpuLayer",
    what: "Places workloads on machines you can reach, starts them, returns logs and results",
    runsOn: "This machine, servers that join, cloud VMs that join",
    machines: "Picks a compatible machine per job (GPU, memory, frameworks); cloud provisioning is simulated today",
    workloads: "Any command, Python functions with any library, torchrun groups, PyTorch training",
    hosting: "Self-hosted; embedded mode needs no server",
    highlight: true,
  },
  {
    name: "PyTorch Lightning",
    what: "Organizes PyTorch training code (LightningModule, Trainer) and handles devices, precision and parallel strategies",
    runsOn: "The machines it is launched on",
    machines: "No; you start it on each node (for example with Slurm or torchrun)",
    workloads: "PyTorch training, fine-tuning, evaluation",
    hosting: "Open-source library (a hosted platform is sold separately)",
  },
  {
    name: "Hugging Face Accelerate",
    what: "Adapts a PyTorch training loop to multi-GPU, multi-node and mixed precision with little code change",
    runsOn: "The machines it is launched on",
    machines: "No; `accelerate launch` runs on machines you provide",
    workloads: "PyTorch training and inference",
    hosting: "Open-source library",
  },
  {
    name: "Ray",
    what: "General distributed computing framework (tasks, actors) with libraries for training, tuning, serving and data",
    runsOn: "A Ray cluster (head node plus workers) on VMs or Kubernetes",
    machines: "Schedules tasks inside the cluster; an autoscaler can add cloud nodes",
    workloads: "Python programs written against Ray's API; functions and objects are serialized",
    hosting: "Open source, self-hosted; managed offerings exist",
  },
  {
    name: "SkyPilot",
    what: "Runs jobs and clusters across many clouds and Kubernetes from a YAML or Python spec",
    runsOn: "Cloud VMs and Kubernetes it provisions",
    machines: "Yes; provisions VMs, chooses regions and instance types, can recover spot jobs",
    workloads: "Any command in a task spec",
    hosting: "Open source, runs from your machine or a shared API server",
  },
  {
    name: "dstack",
    what: "Orchestrates AI dev environments, tasks and services",
    runsOn: "Cloud providers, Kubernetes, and on-prem servers added over SSH",
    machines: "Yes; provisions cloud capacity or uses fleets you register",
    workloads: "Containers running any command; long-running services",
    hosting: "Open source, self-hosted server; a hosted version exists",
  },
  {
    name: "Modal",
    what: "Serverless platform: decorate Python functions and run them on Modal's GPUs",
    runsOn: "Modal's infrastructure",
    machines: "Modal allocates them; you do not bring your own GPUs",
    workloads: "Python functions, batch jobs, web endpoints, containers defined in code",
    hosting: "Hosted service, pay per use",
  },
  {
    name: "Slurm / Kubernetes",
    what: "Cluster schedulers for shared HPC or container infrastructure",
    runsOn: "A cluster you operate",
    machines: "Yes, within the cluster; queues, quotas, priorities",
    workloads: "Batch jobs (Slurm); containers and services (Kubernetes, plus operators for training)",
    hosting: "Open source; needs cluster administration",
  },
];

type Note = { name: string; summary: string; choose: string; together: string };

const NOTES: Note[] = [
  {
    name: "PyTorch Lightning and Hugging Face Accelerate",
    summary:
      "Both decide how a PyTorch model trains inside the machines they are started on: device placement, precision, DDP, FSDP or DeepSpeed, callbacks, logging. They do not choose those machines, copy code to them, or start the processes on each node. GpuLayer works one level below: it decides where a workload runs and starts it there. Its own PyTorch Trainer is deliberately small and has no FSDP, DeepSpeed or callbacks yet.",
    choose:
      "Choose them for training features. Lightning has a far richer Trainer than GpuLayer's, and Accelerate makes an existing loop portable.",
    together:
      "Keep your training code and add `gpulayer.pytorch.launch(...)` at the top of the script. GpuLayer then places it on workers and starts torchrun with the standard environment that both libraries read. This combination follows from the torchrun environment but has not been tested in this repository yet.",
  },
  {
    name: "Ray",
    summary:
      "Ray is a programming model as much as a scheduler: you write tasks and actors against its API, run a Ray cluster, and use its libraries for training, tuning and serving. Functions and arguments are serialized and shipped between processes. GpuLayer ships source code and calls functions by name with JSON arguments, has no distributed object store, and runs plain commands written in any language.",
    choose:
      "Choose Ray for fine-grained distributed Python (many small tasks, actors, shared objects), Ray Tune sweeps, or Ray Serve.",
    together:
      "A Ray program can be a GpuLayer job on one machine; a multi-node Ray cluster is better managed by Ray itself or KubeRay.",
  },
  {
    name: "SkyPilot and dstack",
    summary:
      "Both provision and manage cloud capacity: they choose instance types and regions, start VMs or pods, and run your commands there. dstack also runs services and dev environments and can use on-prem servers. GpuLayer's cloud provisioning exists only as a simulated adapter today. Its strengths are a lightweight join for any machine you already have, capability-aware placement with explanations, and the Python and PyTorch APIs.",
    choose:
      "Choose SkyPilot or dstack when you mainly need cloud capacity across providers, spot recovery, or long-running services today.",
    together:
      "VMs they start can run `gpulayer join` and become GpuLayer workers.",
  },
  {
    name: "Modal",
    summary:
      "Modal's developer experience (decorate a function, call `.remote()` or `.map()`) is close to GpuLayer's `@gl.remote`, but it runs on Modal's own infrastructure. GpuLayer runs on hardware you control: your laptop, office servers and VMs you rent, including Apple Silicon Macs.",
    choose:
      "Choose Modal if you want hosted, autoscaled GPUs with no machines to manage and are fine paying per use.",
    together: "They solve the same problem for different hardware, so teams usually pick one.",
  },
  {
    name: "Slurm and Kubernetes",
    summary:
      "Cluster schedulers with queues, quotas, priorities and multi-tenant isolation, run by an operations team. GpuLayer is much lighter, with one command to start a server and one to join a machine, and it is meant for groups that trust each other, not for shared multi-tenant clusters.",
    choose: "Choose them for shared, policy-heavy clusters with many users.",
    together:
      "Integrations (submitting to Slurm or Kubernetes as a GpuLayer worker pool) are not built yet.",
  },
];

const BEHIND = [
  "No real cloud provisioning or autoscaling yet (simulated adapter only); SkyPilot, dstack and Ray already do this.",
  "No long-running services yet; dstack, Ray Serve, Modal and Kubernetes handle them.",
  "A small PyTorch Trainer: no FSDP, DeepSpeed, callbacks or logger integrations, unlike PyTorch Lightning.",
  "Distributed jobs use torchrun only; no JAX multi-host or MPI launcher yet.",
  "Pre-alpha: NVIDIA GPUs and multi-machine runs are implemented but not validated yet. Not a sandbox for untrusted users.",
];

const FITS = [
  "Your GPUs are spread over a laptop, a few servers and some rented VMs, and you want one queue for all of them.",
  "Your team mixes NVIDIA boxes and Apple Silicon Macs and wants jobs to land on hardware that can run them.",
  "Your workloads are not only training: rendering, inference batches, simulations, data jobs, in any language.",
  "You want placement to explain itself: per-worker reasons when a job cannot run.",
  "You want to start small: `gpulayer up` on one machine, `gpulayer join` on the others, no cluster to operate.",
];

function Code({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`)/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("`") ? (
          <code key={i} className="rounded bg-secondary px-1 py-0.5 font-mono text-[0.85em]">
            {p.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}

export default function ComparePage() {
  return (
    <>
      <SiteHeader />
      <main>
        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <p className="text-sm font-medium text-primary">Compare</p>
            <h1 className="mt-2 text-4xl font-semibold tracking-tight">
              GpuLayer and the alternatives
            </h1>
            <p className="mt-4 max-w-3xl text-lg text-muted-foreground">
              Training frameworks decide <em>how</em> a model trains on the machines they are
              started on. Cloud orchestrators decide <em>which</em> cloud capacity to rent.
              GpuLayer decides where any GPU workload runs among the machines you can reach,
              starts it there, and brings the results back. Several of these tools work well
              together.
            </p>
          </div>
        </section>

        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl px-6 py-16">
            <h2 className="text-2xl font-semibold tracking-tight">At a glance</h2>
            <div className="mt-8 overflow-x-auto rounded-lg border border-border">
              <table className="w-full min-w-[1000px] text-left text-sm">
                <thead className="bg-secondary text-xs uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-4 py-3 font-semibold">Tool</th>
                    <th className="px-4 py-3 font-semibold">What it does</th>
                    <th className="px-4 py-3 font-semibold">Runs on</th>
                    <th className="px-4 py-3 font-semibold">Chooses machines?</th>
                    <th className="px-4 py-3 font-semibold">Workloads</th>
                    <th className="px-4 py-3 font-semibold">Hosting</th>
                  </tr>
                </thead>
                <tbody>
                  {ROWS.map((r) => (
                    <tr
                      key={r.name}
                      className={`border-t border-border align-top ${r.highlight ? "bg-primary/5" : ""}`}
                    >
                      <td className="px-4 py-3 font-medium">{r.name}</td>
                      <td className="px-4 py-3 text-muted-foreground">{r.what}</td>
                      <td className="px-4 py-3 text-muted-foreground">{r.runsOn}</td>
                      <td className="px-4 py-3 text-muted-foreground">
                        <Code text={r.machines} />
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">{r.workloads}</td>
                      <td className="px-4 py-3 text-muted-foreground">{r.hosting}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Descriptions of other projects are summaries of their public documentation at the
              time of writing; check each project for current details.
            </p>
          </div>
        </section>

        <section className="border-b border-border">
          <div className="mx-auto max-w-6xl space-y-10 px-6 py-16">
            {NOTES.map((n) => (
              <div key={n.name}>
                <h2 className="text-xl font-semibold tracking-tight">GpuLayer and {n.name}</h2>
                <p className="mt-3 max-w-3xl text-muted-foreground">
                  <Code text={n.summary} />
                </p>
                <dl className="mt-4 grid max-w-3xl gap-3 text-sm sm:grid-cols-2">
                  <div className="rounded-lg border border-border p-4">
                    <dt className="font-medium">When to choose it</dt>
                    <dd className="mt-1 text-muted-foreground">
                      <Code text={n.choose} />
                    </dd>
                  </div>
                  <div className="rounded-lg border border-border p-4">
                    <dt className="font-medium">Using both</dt>
                    <dd className="mt-1 text-muted-foreground">
                      <Code text={n.together} />
                    </dd>
                  </div>
                </dl>
              </div>
            ))}
          </div>
        </section>

        <section className="border-b border-border">
          <div className="mx-auto grid max-w-6xl gap-10 px-6 py-16 md:grid-cols-2">
            <div>
              <h2 className="text-xl font-semibold tracking-tight">GpuLayer fits when</h2>
              <ul className="mt-4 list-disc space-y-2 pl-5 text-sm text-muted-foreground">
                {FITS.map((f) => (
                  <li key={f}>
                    <Code text={f} />
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h2 className="text-xl font-semibold tracking-tight">Where GpuLayer is behind today</h2>
              <ul className="mt-4 list-disc space-y-2 pl-5 text-sm text-muted-foreground">
                {BEHIND.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section>
          <div className="mx-auto flex max-w-6xl flex-wrap gap-3 px-6 py-16">
            <Link
              href="/docs/get-started/quickstart/"
              className="inline-flex items-center gap-1 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
            >
              Try the quickstart <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/docs/get-started/why-gpulayer/"
              className="inline-flex items-center gap-1 rounded-md border border-border px-4 py-2 text-sm font-medium hover:bg-secondary"
            >
              Why GpuLayer
            </Link>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
