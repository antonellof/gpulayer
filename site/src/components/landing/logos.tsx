import { Cpu, Terminal } from "lucide-react";
import {
  siApple,
  siNumpy,
  siNvidia,
  siOnnx,
  siPytorch,
  siTensorflow,
  type SimpleIcon,
} from "simple-icons";

// Brand marks from simple-icons (CC0). They identify the technologies GpuLayer works with;
// no endorsement by their owners is implied.
type Item = { label: string; icon?: SimpleIcon; lucide?: typeof Cpu; color?: string };

const FRAMEWORKS: Item[] = [
  { label: "PyTorch", icon: siPytorch },
  { label: "JAX" },
  { label: "MLX" },
  { label: "CuPy" },
  { label: "NumPy", icon: siNumpy },
  { label: "TensorFlow", icon: siTensorflow },
  { label: "ONNX Runtime", icon: siOnnx },
  { label: "any binary", lucide: Terminal },
];

const HARDWARE: Item[] = [
  { label: "NVIDIA CUDA", icon: siNvidia, color: "#76B900" },
  { label: "Apple Metal", icon: siApple, color: "#ffffff" },
  { label: "CPU", lucide: Cpu, color: "#cbd5e1" },
];

function Mark({ item, size = 18 }: { item: Item; size?: number }) {
  if (item.icon) {
    return (
      <svg
        role="img"
        aria-hidden
        viewBox="0 0 24 24"
        width={size}
        height={size}
        fill={item.color ?? "currentColor"}
        className="shrink-0"
      >
        <path d={item.icon.path} />
      </svg>
    );
  }
  if (item.lucide) {
    const Icon = item.lucide;
    return <Icon className="h-[18px] w-[18px] shrink-0" style={item.color ? { color: item.color } : undefined} />;
  }
  return null;
}

function Chip({ item }: { item: Item }) {
  return (
    <span className="inline-flex items-center gap-2 text-slate-300 transition-colors hover:text-white">
      <Mark item={item} />
      <span>{item.label}</span>
    </span>
  );
}

export function WorksWith() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-6 text-sm lg:flex-row lg:items-center lg:gap-10">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-slate-500">Works with</span>
        {FRAMEWORKS.map((f) => (
          <Chip key={f.label} item={f} />
        ))}
      </div>
      <span className="hidden h-6 w-px bg-white/10 lg:block" aria-hidden />
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-slate-500">On</span>
        {HARDWARE.map((h) => (
          <span
            key={h.label}
            className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-1.5 font-medium text-white"
          >
            <Mark item={h} />
            {h.label}
          </span>
        ))}
      </div>
    </div>
  );
}
