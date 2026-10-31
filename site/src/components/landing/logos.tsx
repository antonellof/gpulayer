import { Cpu, Terminal } from "lucide-react";
import {
  siApple,
  siNumpy,
  siNvidia,
  siPytorch,
  type SimpleIcon,
} from "simple-icons";

// Brand marks from simple-icons (CC0). They identify the technologies GPUlayer works with;
// no endorsement by their owners is implied. Projects without a simple-icons mark get a
// neutral two-letter monogram rather than a copy of their logo.
export type Item = {
  label: string;
  icon?: SimpleIcon;
  lucide?: typeof Cpu;
  color?: string;
  monogram?: { text: string; bg: string };
};

// Only what has been run as GPUlayer jobs (see the performance page and examples/). Other
// libraries (JAX, CuPy, TensorFlow, ...) are detected on workers but not yet validated on a GPU.
const FRAMEWORKS: Item[] = [
  { label: "PyTorch", icon: siPytorch },
  { label: "MLX", monogram: { text: "MX", bg: "#525252" } },
  { label: "NumPy", icon: siNumpy },
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

export function Mark({ item, size = 18 }: { item: Item; size?: number }) {
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
  if (item.monogram) {
    return (
      <span
        aria-hidden
        className="inline-flex shrink-0 items-center justify-center rounded-[4px] font-mono text-[9px] font-bold leading-none text-white"
        style={{ width: size, height: size, background: item.monogram.bg }}
      >
        {item.monogram.text}
      </span>
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
