import { Cloud, Cpu, Network, Server, Sparkles, type LucideIcon } from "lucide-react";

type Chip = { name: string; strong?: boolean; faded?: boolean };

const LAYERS: { title: string; icon: LucideIcon; tools: Chip[] }[] = [
  {
    title: "Training code and libraries",
    icon: Sparkles,
    tools: [
      { name: "PyTorch Lightning" },
      { name: "Accelerate" },
      { name: "JAX / MLX / CuPy code" },
      { name: "GpuLayer Trainer (small)", faded: true },
    ],
  },
  {
    title: "Place and launch work across machines",
    icon: Network,
    tools: [{ name: "GpuLayer", strong: true }, { name: "Ray" }, { name: "Slurm / Kubernetes" }],
  },
  {
    title: "Provision cloud capacity",
    icon: Cloud,
    tools: [
      { name: "SkyPilot" },
      { name: "dstack" },
      { name: "Ray autoscaler" },
      { name: "GpuLayer (planned)", faded: true },
    ],
  },
  {
    title: "Hardware",
    icon: Cpu,
    tools: [
      { name: "Laptop / Mac" },
      { name: "Workstations and servers" },
      { name: "Cloud VMs" },
      { name: "Clusters" },
    ],
  },
];

/** Where GpuLayer sits relative to other tools. Shared by the landing and compare pages. */
export function LayerStack() {
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_210px]">
      <div className="space-y-3">
        {LAYERS.map((layer, i) => (
          <div
            key={layer.title}
            className={`rounded-xl border p-4 ${
              i === 1
                ? "border-primary/50 bg-primary/[0.06] shadow-glow"
                : "border-border bg-card"
            }`}
          >
            <div className="flex items-center gap-2 font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              <layer.icon className={`h-4 w-4 ${i === 1 ? "text-primary" : ""}`} />
              {layer.title}
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {layer.tools.map((t) => (
                <span
                  key={t.name}
                  className={`rounded-lg border px-3 py-1.5 text-sm ${
                    t.strong
                      ? "border-primary bg-primary font-semibold text-primary-foreground"
                      : t.faded
                        ? "border-dashed border-border text-muted-foreground"
                        : "border-border bg-background"
                  }`}
                >
                  {t.name}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="flex flex-col justify-center rounded-xl border border-dashed border-border bg-card/50 p-4 text-center">
        <Server className="mx-auto h-5 w-5 text-muted-foreground" />
        <div className="mt-2 font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
          Hosted, all layers
        </div>
        <span className="mx-auto mt-3 rounded-lg border border-border bg-background px-3 py-1.5 text-sm">
          Modal
        </span>
        <p className="mt-3 text-xs text-muted-foreground">Runs on its own GPUs, not yours</p>
      </div>
    </div>
  );
}
