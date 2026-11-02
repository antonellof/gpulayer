// Feature art: one still, flat illustration per capability card, drawn from what GPUlayer
// actually shows (its API, the scheduler's reasons, validated hardware). Colours follow the
// site theme and the docs diagrams (--fa-* in globals.css), so they switch with dark mode.

import type { CSSProperties, ReactNode } from "react";

const W = 260;
const H = 160;
const MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace";
const SANS = "Geist, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif";

const FG = "hsl(var(--foreground))";
const MUTED = "hsl(var(--muted-foreground))";
const BORDER = "hsl(var(--border))";
const BG = "hsl(var(--background))";
const ACCENT = "var(--fa-accent)";
const ACCENT_SOFT = "var(--fa-accent-soft)";
const OK = "var(--fa-ok)";
const BAD = "var(--fa-bad)";

type TextProps = {
  x: number;
  y: number;
  children: ReactNode;
  color?: string;
  size?: number;
  mono?: boolean;
  bold?: boolean;
  anchor?: "start" | "middle" | "end";
};

function T({ x, y, children, color = FG, size = 7.5, mono = true, bold = false, anchor = "start" }: TextProps) {
  return (
    <text
      x={x}
      y={y}
      fontSize={size}
      fontFamily={mono ? MONO : SANS}
      fontWeight={bold ? 600 : 400}
      textAnchor={anchor}
      style={{ fill: color, whiteSpace: "pre" }}
    >
      {children}
    </text>
  );
}

type RectProps = {
  x: number;
  y: number;
  w: number;
  h: number;
  r?: number;
  fill?: string;
  stroke?: string;
  dash?: boolean;
};

function R({ x, y, w, h, r = 4, fill = BG, stroke = BORDER, dash = false }: RectProps) {
  const style: CSSProperties = { fill, stroke, strokeWidth: 1 };
  if (dash) style.strokeDasharray = "3 2";
  return <rect x={x} y={y} width={w} height={h} rx={r} style={style} />;
}

function Check({ x, y }: { x: number; y: number }) {
  return (
    <path
      d={`M${x} ${y} l2.4 2.4 l4.6 -5`}
      style={{ fill: "none", stroke: OK, strokeWidth: 1.4, strokeLinecap: "round", strokeLinejoin: "round" }}
    />
  );
}

function Cross({ x, y }: { x: number; y: number }) {
  const s: CSSProperties = { stroke: BAD, strokeWidth: 1.4, strokeLinecap: "round" };
  return (
    <g>
      <line x1={x} y1={y - 2.6} x2={x + 5} y2={y + 2.4} style={s} />
      <line x1={x + 5} y1={y - 2.6} x2={x} y2={y + 2.4} style={s} />
    </g>
  );
}

/** A GPU tile: filled with the accent when in use, an outline when idle. */
function Gpu({ x, y, s = 12, on = true }: { x: number; y: number; s?: number; on?: boolean }) {
  return <R x={x} y={y} w={s} h={s} r={2} fill={on ? ACCENT : BG} stroke={on ? ACCENT : BORDER} />;
}

// ------------------------------------------------------------------- scenes

function Api() {
  // the Trainer's surface as a file: a Module with its hooks, a Trainer with a strategy
  const lines: [string, string][] = [
    ["class Net(glt.Module):", FG],
    ["  def training_step(self, batch, i)", MUTED],
    ["  def configure_optimizers(self)", MUTED],
    ["", FG],
    ["trainer = glt.Trainer(", FG],
    ['  strategy="fsdp", devices=4,', MUTED],
    ["  callbacks=[ModelCheckpoint()])", MUTED],
  ];
  const tags = ["DDP", "FSDP", "DeepSpeed", "TP"];
  let tx = 30;
  return (
    <>
      <R x={20} y={16} w={220} h={100} r={6} />
      <line x1={20} y1={32} x2={240} y2={32} style={{ stroke: BORDER }} />
      <T x={30} y={27} size={7} color={MUTED}>
        train.py
      </T>
      {lines.map(([s, c], i) => (
        <T key={i} x={30} y={46 + i * 10.5} size={7} color={c}>
          {s}
        </T>
      ))}
      {tags.map((t) => {
        const w = t.length * 4.6 + 12;
        const x = tx;
        tx += w + 6;
        return (
          <g key={t}>
            <R x={x} y={126} w={w} h={16} r={8} fill={ACCENT_SOFT} stroke={ACCENT} />
            <T x={x + w / 2} y={137} size={7} color={FG} anchor="middle">
              {t}
            </T>
          </g>
        );
      })}
    </>
  );
}

function Scale() {
  // the same script at three sizes: one GPU, every GPU of a machine, two machines
  const machine = (x: number, y: number, n: number) => (
    <g>
      <R x={x} y={y} w={62} h={22} r={4} />
      {[0, 1, 2, 3].map((i) => (
        <Gpu key={i} x={x + 6 + i * 13.5} y={y + 5} on={i < n} />
      ))}
    </g>
  );
  const arrow = (x: number) => (
    <path d={`M${x} 80 h10 m-3.5 -3.5 l3.5 3.5 l-3.5 3.5`} style={{ fill: "none", stroke: MUTED, strokeWidth: 1 }} />
  );
  return (
    <>
      {machine(10, 69, 1)}
      {arrow(78)}
      {machine(96, 69, 4)}
      {arrow(164)}
      {machine(184, 52, 4)}
      {machine(184, 86, 4)}
      <T x={41} y={108} size={7} color={MUTED} anchor="middle">
        devices=1
      </T>
      <T x={127} y={108} size={7} color={MUTED} anchor="middle">
        devices=4
      </T>
      <T x={215} y={125} size={7} color={MUTED} anchor="middle">
        num_nodes=2
      </T>
      <T x={130} y={30} size={8} color={FG} anchor="middle">
        trainer.fit(model)
      </T>
    </>
  );
}

function Placement() {
  // one machine is chosen; every other one says why not (the scheduler's own reasons)
  const rows: { host: string; ok: boolean; why: string }[] = [
    { host: "a100-box", ok: true, why: "placed: 2 GPUs over NVLink" },
    { host: "m2-pro", ok: false, why: "torch cannot use cuda" },
    { host: "t4-vm", ok: false, why: "needs 2 GPUs, has 1" },
    { host: "rtx-box", ok: false, why: "reserved for job-41c" },
  ];
  return (
    <>
      <T x={20} y={26} size={7} color={MUTED}>
        accelerator=cuda, devices=2
      </T>
      {rows.map((r, i) => {
        const y = 36 + i * 28;
        return (
          <g key={r.host}>
            <R x={20} y={y} w={220} h={22} r={4} fill={r.ok ? ACCENT_SOFT : BG} stroke={r.ok ? ACCENT : BORDER} />
            {r.ok ? <Check x={29} y={y + 11.5} /> : <Cross x={30} y={y + 11} />}
            <T x={44} y={y + 14} size={7} color={FG}>
              {r.host}
            </T>
            <T x={232} y={y + 14} size={7} color={r.ok ? FG : MUTED} anchor="end">
              {r.why}
            </T>
          </g>
        );
      })}
    </>
  );
}

function Gpus() {
  // the machines GPUlayer has run on, side by side behind one server
  const cards = [
    { where: "laptop", gpu: "Apple M2 Pro", backend: "mps" },
    { where: "workstation", gpu: "4x RTX 3070", backend: "cuda" },
    { where: "Google Cloud", gpu: "Tesla T4", backend: "cuda" },
    { where: "server", gpu: "2x A100 SXM4", backend: "cuda" },
  ];
  return (
    <>
      <R x={92} y={14} w={76} h={20} r={10} fill={ACCENT_SOFT} stroke={ACCENT} />
      <T x={130} y={27} size={7} color={FG} anchor="middle">
        gpulayer up
      </T>
      <path d="M130 34 V42 M71 42 H189 V50 M71 42 V50" style={{ fill: "none", stroke: BORDER, strokeWidth: 1 }} />
      {cards.map((c, i) => {
        const x = 16 + (i % 2) * 118;
        const y = 50 + Math.floor(i / 2) * 52;
        return (
          <g key={c.gpu}>
            <R x={x} y={y} w={110} h={42} r={5} />
            <Gpu x={x + 8} y={y + 9} s={10} />
            <T x={x + 24} y={y + 17} size={7.5} color={FG}>
              {c.gpu}
            </T>
            <T x={x + 8} y={y + 33} size={6.5} color={MUTED}>
              {c.where}
            </T>
            <T x={x + 102} y={y + 33} size={6.5} color={ACCENT} anchor="end">
              {c.backend}
            </T>
          </g>
        );
      })}
    </>
  );
}

function Results() {
  // a loss curve next to the files that come back to your machine
  const pts: string[] = [];
  for (let i = 0; i <= 30; i++) {
    const t = i / 30;
    const y = 112 - 62 * Math.exp(-4 * t) - 4 * Math.sin(i * 1.7) * (1 - t) * 0.6;
    pts.push(`${(26 + 104 * t).toFixed(1)} ${y.toFixed(1)}`);
  }
  const files = ["last.ckpt", "metrics.csv", "rank0.log", "rank1.log"];
  return (
    <>
      <R x={16} y={18} w={124} h={110} r={6} />
      <T x={26} y={34} size={7} color={MUTED}>
        train_loss
      </T>
      {[60, 80, 100].map((y) => (
        <line key={y} x1={26} y1={y} x2={130} y2={y} style={{ stroke: BORDER, strokeDasharray: "2 2" }} />
      ))}
      <path d={`M${pts.join(" L")}`} style={{ fill: "none", stroke: ACCENT, strokeWidth: 1.6, strokeLinejoin: "round" }} />
      <T x={152} y={34} size={7} color={MUTED}>
        outputs/
      </T>
      {files.map((f, i) => (
        <g key={f}>
          <R x={150} y={42 + i * 22} w={94} h={17} r={3} fill={i === 0 ? ACCENT_SOFT : BG} stroke={i === 0 ? ACCENT : BORDER} />
          <T x={158} y={53.5 + i * 22} size={7} color={FG}>
            {f}
          </T>
        </g>
      ))}
      <T x={16} y={146} size={7} color={MUTED}>
        streamed while it trains, saved to your machine
      </T>
    </>
  );
}

function Loop() {
  // a custom loop: two launch lines on top, Runtime calls inside the user's own loop
  const lines: { s: string; hi?: boolean; dim?: boolean }[] = [
    { s: 'glt.launch(devices=4, accelerator="cuda")', hi: true },
    { s: "rt = glt.Runtime().setup()", hi: true },
    { s: "model = rt.module(model)" },
    { s: "for xb, yb in rt.dataloader(loader):" },
    { s: "    with rt.autocast():", dim: true },
    { s: "        loss = loss_fn(model(xb), yb)", dim: true },
    { s: "    rt.backward(loss)" },
    { s: "    rt.step(optimizer)" },
  ];
  return (
    <>
      <R x={14} y={14} w={232} h={132} r={6} />
      {lines.map((l, i) => {
        const y = 30 + i * 14.5;
        return (
          <g key={i}>
            {l.hi && <rect x={15} y={y - 9} width={230} height={13} style={{ fill: ACCENT_SOFT }} />}
            {l.hi && <rect x={15} y={y - 9} width={2} height={13} style={{ fill: ACCENT }} />}
            <T x={24} y={y} size={6.8} color={l.dim ? MUTED : FG}>
              {l.s}
            </T>
          </g>
        );
      })}
    </>
  );
}

const ART = { api: Api, scale: Scale, placement: Placement, gpus: Gpus, results: Results, loop: Loop };

export function FeatureArt({ kind }: { kind: keyof typeof ART }) {
  const Art = ART[kind];
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="feature-art block h-28 w-full bg-card sm:h-48"
      aria-hidden
    >
      <Art />
    </svg>
  );
}
