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
  { label: "NVIDIA CUDA", icon: siNvidia },
  { label: "Apple Metal", icon: siApple },
  { label: "CPU", lucide: Cpu },
];

/** Relative luminance of a hex colour (0 = black, 1 = white). */
function luminance(hex: string): number {
  const [r, g, b] = [0, 2, 4].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function Mark({ item, size = 18 }: { item: Item; size?: number }) {
  if (item.icon) {
    const hex = item.icon.hex;
    // Near-black brand colours (Apple, NumPy) use the text colour in dark mode.
    const dark = luminance(hex) < 0.05;
    return (
      <svg
        role="img"
        aria-hidden
        viewBox="0 0 24 24"
        width={size}
        height={size}
        style={{ fill: `#${hex}` }}
        className={dark ? "shrink-0 dark:!fill-current" : "shrink-0"}
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
    <span className="inline-flex items-center gap-2 text-muted-foreground transition-colors hover:text-foreground">
      <Mark item={item} />
      <span>{item.label}</span>
    </span>
  );
}

export function WorksWith() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-5 px-6 py-8 text-sm">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="eyebrow w-24">Works with</span>
        {FRAMEWORKS.map((f) => (
          <Chip key={f.label} item={f} />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="eyebrow w-24">Runs on</span>
        {HARDWARE.map((h) => (
          <span
            key={h.label}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-3 py-1.5 font-medium"
          >
            <Mark item={h} />
            {h.label}
          </span>
        ))}
      </div>
    </div>
  );
}
