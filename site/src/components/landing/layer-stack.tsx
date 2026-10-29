import { Cloud, Cpu, Network, Server, Sparkles, type LucideIcon } from "lucide-react";
import {
  siApple,
  siHuggingface,
  siKubernetes,
  siLightning,
  siModal,
  siNvidia,
  siPytorch,
  siRay,
} from "simple-icons";

import { Mark, type Item } from "@/components/landing/logos";
import { LogoMark } from "@/components/site/logo";

type Chip = Item & { strong?: boolean; faded?: boolean; gpulayer?: boolean };

const LAYERS: { title: string; icon: LucideIcon; tools: Chip[] }[] = [
  {
    title: "Training code and libraries",
    icon: Sparkles,
    tools: [
      { label: "PyTorch", icon: siPytorch },
      { label: "PyTorch Lightning", icon: siLightning },
      { label: "Accelerate", icon: siHuggingface },
      { label: "JAX", monogram: { text: "JX", bg: "#4285F4" } },
      { label: "MLX", monogram: { text: "MX", bg: "#525252" } },
      { label: "CuPy", monogram: { text: "Cp", bg: "#2E9E5B" } },
      { label: "GPUlayer Trainer (small)", gpulayer: true, faded: true },
    ],
  },
  {
    title: "Place and launch work across machines",
    icon: Network,
    tools: [
      { label: "GPUlayer", gpulayer: true, strong: true },
      { label: "Ray", icon: siRay },
      { label: "Slurm", monogram: { text: "Sl", bg: "#3B6EA8" } },
      { label: "Kubernetes", icon: siKubernetes },
    ],
  },
  {
    title: "Provision cloud capacity",
    icon: Cloud,
    tools: [
      { label: "SkyPilot", monogram: { text: "SP", bg: "#3D5AFE" } },
      { label: "dstack", monogram: { text: "ds", bg: "#7C3AED" } },
      { label: "Ray autoscaler", icon: siRay },
      { label: "GPUlayer (planned)", gpulayer: true, faded: true },
    ],
  },
  {
    title: "Hardware",
    icon: Cpu,
    tools: [
      { label: "Macs (Apple GPU)", icon: siApple },
      { label: "NVIDIA servers", icon: siNvidia },
      { label: "Cloud VMs", lucide: Cloud },
      { label: "Clusters", lucide: Server },
    ],
  },
];

function ChipView({ t }: { t: Chip }) {
  const base = "inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm";
  const style = t.strong
    ? "border-foreground bg-foreground font-medium text-background"
    : t.faded
      ? "border-dashed border-border text-muted-foreground"
      : "border-border bg-background text-foreground/80";
  return (
    <span className={`${base} ${style}`}>
      {t.gpulayer ? (
        <LogoMark className={`h-[18px] w-[18px] ${t.faded ? "opacity-60" : ""}`} />
      ) : (
        <Mark item={t} />
      )}
      {t.label}
    </span>
  );
}

/** Where GPUlayer sits relative to other tools. Shared by the landing and compare pages. */
export function LayerStack() {
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_210px]">
      <div className="space-y-3">
        {LAYERS.map((layer, i) => (
          <div
            key={layer.title}
            className={`rounded-xl border p-4 ${
              i === 1 ? "border-foreground/40 bg-background shadow-soft" : "border-border bg-card"
            }`}
          >
            <div className="flex items-center gap-2 font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              <layer.icon className={`h-4 w-4 ${i === 1 ? "text-foreground" : ""}`} strokeWidth={1.75} />
              {layer.title}
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {layer.tools.map((t) => (
                <ChipView key={t.label} t={t} />
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
        <span className="mx-auto mt-3 inline-flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5 text-sm">
          <Mark item={{ label: "Modal", icon: siModal }} />
          Modal
        </span>
        <p className="mt-3 text-xs text-muted-foreground">Uses Modal&apos;s own GPUs</p>
      </div>
    </div>
  );
}
