// Feature art: fine isometric wireframes on a dark panel, one glowing accent, small monospace
// labels, and one simple looping symbol per card. Motion is CSS
// (.fa-* in globals.css) plus SMIL for movement along paths; with prefers-reduced-motion the
// scenes show a still frame.

import type { CSSProperties, ReactNode } from "react";

const W = 260;
const H = 160;
const S = 10; // one unit
const C30 = Math.cos(Math.PI / 6);
const LINE = "rgba(255,255,255,0.22)";
const LINE_SOFT = "rgba(255,255,255,0.09)";
const ACCENT = "#22d3ee";
const ACCENT_TOP = "#a5f3fc";
const MONO = "'JetBrains Mono', ui-monospace, monospace";

type Pt = [number, number];

function iso(o: Pt, x: number, y: number, z: number): Pt {
  return [o[0] + (x - y) * C30 * S, o[1] + (x + y) * 0.5 * S - z * S];
}

function poly(points: Pt[]): string {
  return points.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(" ");
}

function delay(seconds: number, extra?: CSSProperties): CSSProperties {
  return { animationDelay: `${seconds}s`, ...extra };
}

type BoxProps = {
  o: Pt;
  x: number;
  y: number;
  z?: number;
  w?: number;
  d?: number;
  h?: number;
  glow?: boolean;
  dim?: boolean;
};

/** A box: wire edges on a dark fill, or filled with the glowing accent. */
function Box({ o, x, y, z = 0, w = 1, d = 1, h = 1, glow = false, dim = false }: BoxProps) {
  const P = (a: number, b: number, c: number) => iso(o, a, b, c);
  const top = [P(x, y, z + h), P(x + w, y, z + h), P(x + w, y + d, z + h), P(x, y + d, z + h)];
  const left = [P(x, y + d, z), P(x + w, y + d, z), P(x + w, y + d, z + h), P(x, y + d, z + h)];
  const right = [P(x + w, y, z), P(x + w, y + d, z), P(x + w, y + d, z + h), P(x + w, y, z + h)];
  if (glow) {
    return (
      <g filter="url(#fa-glow)">
        <polygon points={poly(left)} fill={ACCENT} fillOpacity={0.55} stroke={ACCENT_TOP} strokeWidth={0.6} />
        <polygon points={poly(right)} fill={ACCENT} fillOpacity={0.35} stroke={ACCENT_TOP} strokeWidth={0.6} />
        <polygon points={poly(top)} fill={ACCENT_TOP} fillOpacity={0.85} stroke="#ecfeff" strokeWidth={0.6} />
      </g>
    );
  }
  return (
    <g fill="#0a0a0a" stroke={dim ? LINE_SOFT : LINE} strokeWidth={0.7}>
      <polygon points={poly(left)} />
      <polygon points={poly(right)} />
      <polygon points={poly(top)} />
    </g>
  );
}

/** A wire box with a glowing copy on top that fades in and out (class sets the timing). */
function LitBox({ cls, at, still = false, ...p }: BoxProps & { cls: string; at: number; still?: boolean }) {
  return (
    <g>
      <Box {...p} />
      <g className={`fa-lit ${cls}${still ? " fa-still" : ""}`} style={delay(at)}>
        <Box {...p} glow />
      </g>
    </g>
  );
}

function Grid({ o, n, m }: { o: Pt; n: number; m: number }) {
  const lines = [];
  for (let i = 0; i <= n; i++) {
    const [a, b] = [iso(o, i, 0, 0), iso(o, i, m, 0)];
    lines.push(<line key={`a${i}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />);
  }
  for (let j = 0; j <= m; j++) {
    const [a, b] = [iso(o, 0, j, 0), iso(o, n, j, 0)];
    lines.push(<line key={`b${j}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />);
  }
  return (
    <g stroke={LINE_SOFT} strokeWidth={0.6}>
      {lines}
    </g>
  );
}

function Label({ at, text, cls, start }: { at: Pt; text: string; cls?: string; start?: number }) {
  const w = text.length * 4.6 + 8;
  return (
    <g className={cls} style={start !== undefined ? delay(start) : undefined}>
      <rect x={at[0]} y={at[1] - 8} width={w} height={11} rx={1.5} fill="#0a0a0a" stroke={LINE} strokeWidth={0.6} />
      <text x={at[0] + 4} y={at[1] + 0.5} fill="rgba(255,255,255,0.6)" fontSize={6.5} fontFamily={MONO}>
        {text}
      </text>
    </g>
  );
}

/** A glowing dot that travels along `path` (SMIL), starting at `begin` seconds. */
function Packet({ path, dur, begin = 0 }: { path: string; dur: number; begin?: number }) {
  return (
    <circle r={2.2} fill={ACCENT_TOP} filter="url(#fa-glow)" className="fa-move" opacity={0}>
      <animateMotion dur={`${dur}s`} begin={`${begin}s`} repeatCount="indefinite" path={path} />
      <animate
        attributeName="opacity"
        values="0;1;1;0"
        keyTimes="0;0.1;0.85;1"
        dur={`${dur}s`}
        begin={`${begin}s`}
        repeatCount="indefinite"
      />
    </circle>
  );
}

/** Scales a scene about the panel centre. */
function Zoom({ k, dy = 0, children }: { k: number; dy?: number; children: ReactNode }) {
  return <g transform={`translate(${W / 2} ${H / 2 + dy}) scale(${k}) translate(${-W / 2} ${-H / 2})`}>{children}</g>;
}

// ------------------------------------------------------------------- scenes

function ApiStack() {
  // the plates breathe apart and together; the glowing layer travels up the stack
  const o: Pt = [130, 60];
  return (
    <>
      <Grid o={[130, 76]} n={7} m={7} />
      {[0, 1, 2, 3, 4].map((i) => (
        <g key={i} className="fa-breathe" style={delay(-i * 0.12, { ["--lift" as string]: `${-i * 3}px` })}>
          <LitBox cls="fa-climb" at={i * 0.8} still={i === 2} o={o} x={1} y={1} z={i * 1.1} w={5} d={5} h={0.35} />
        </g>
      ))}
      <Label at={[24, 40]} text="callbacks" />
      <Label at={[196, 100]} text="strategy" />
    </>
  );
}

function AnyScale() {
  // GPUs light up 1, then 2, then 4, then 8 (2 machines x 4), then the cycle starts again
  const o: Pt = [130, 44];
  const order: Record<string, number> = {};
  const steps = [["0,0"], ["0,1"], ["0,2", "0,3"], ["1,0", "1,1", "1,2", "1,3"]];
  steps.forEach((cells, s) => cells.forEach((c) => (order[c] = s)));
  const cells = [];
  for (let r = 0; r < 5; r++)
    for (let c = 0; c < 5; c++) {
      const step = order[`${r},${c}`];
      cells.push(
        step === undefined ? (
          <Box key={`${r}${c}`} o={o} x={r * 1.4} y={c * 1.4} w={1} d={1} h={0.6} dim />
        ) : (
          <LitBox key={`${r}${c}`} cls={`fa-scale-${step}`} at={0} still o={o} x={r * 1.4} y={c * 1.4} w={1} d={1} h={1.2} />
        ),
      );
    }
  return (
    <>
      <Grid o={[130, 50]} n={7} m={7} />
      {cells}
      <Label at={[22, 34]} text="num_nodes=2" cls="fa-scale-3" start={0} />
      <Label at={[186, 112]} text="devices=4" />
    </>
  );
}

function Placement() {
  // a job (glowing cube) drops into the one slot that fits; the slot lights when it lands
  const o: Pt = [130, 62];
  const slot = [1, 1];
  const tiles = [];
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) {
      const isSlot = r === slot[0] && c === slot[1];
      tiles.push(
        isSlot ? (
          <LitBox key={`${r}${c}`} cls="fa-chosen" at={0} still o={o} x={r * 1.3} y={c * 1.3} w={1} d={1} h={0.3} />
        ) : (
          <Box key={`${r}${c}`} o={o} x={r * 1.3} y={c * 1.3} w={1} d={1} h={0.3} dim={(r + c) % 2 === 1} />
        ),
      );
    }
  return (
    <>
      <Zoom k={1.6} dy={2}>
        <Grid o={[130, 64]} n={4} m={4} />
        {tiles}
        <g className="fa-drop">
          <Box o={o} x={slot[0] * 1.3 + 0.15} y={slot[1] * 1.3 + 0.15} z={0.3} w={0.7} d={0.7} h={0.7} glow />
        </g>
      </Zoom>
      <Label at={[186, 34]} text="best fit" cls="fa-chosen" start={0} />
    </>
  );
}

function Fanout() {
  // one server reaches a laptop, a workstation, a server and a cloud VM; pulses go out to each
  const o: Pt = [130, 42];
  const hub = iso(o, 4, 4, 1.6);
  const nodes = [
    { key: "workstation", x: 0.6, y: 0.6, z: 0, w: 1.1, d: 1.1, h: 1.2 },
    { key: "laptop", x: 0.4, y: 6.6, z: 0, w: 1.5, d: 1.1, h: 0.2 },
    { key: "server", x: 6.6, y: 0.6, z: 0, w: 1, d: 1.2, h: 2.2 },
    { key: "cloud", x: 6.6, y: 6.6, z: 1.2, w: 1.2, d: 1.2, h: 0.8 },
  ];
  const parts = nodes.map((n, i) => {
    const t = iso(o, n.x + n.w / 2, n.y + n.d / 2, n.z + n.h);
    const begin = i * 0.6;
    const path = `M${hub[0].toFixed(1)} ${hub[1].toFixed(1)} L${t[0].toFixed(1)} ${t[1].toFixed(1)}`;
    return {
      depth: n.x + n.y,
      el: (
        <g key={n.key}>
          <path d={path} stroke={ACCENT} strokeOpacity={0.45} strokeWidth={0.7} strokeDasharray="1.5 2" />
          <LitBox cls="fa-arrive" at={begin + 2.3} o={o} x={n.x} y={n.y} z={n.z} w={n.w} d={n.d} h={n.h} />
          <Packet path={path} dur={2.7} begin={begin} />
        </g>
      ),
    };
  });
  return (
    <>
      <Zoom k={1.2}>
        <Grid o={o} n={8} m={8} />
        {parts.filter((p) => p.depth < 8).map((p) => p.el)}
        <g className="fa-pulse-glow">
          <Box o={o} x={3.4} y={3.4} w={1.2} d={1.2} h={1.6} glow />
        </g>
        {parts.filter((p) => p.depth >= 8).map((p) => p.el)}
      </Zoom>
      <Label at={[20, 30]} text="gpulayer up" />
      <Label at={[204, 140]} text="join" />
    </>
  );
}

function Results() {
  // the loss curve draws on a chart card, then the checkpoint lands on your machine
  const o: Pt = [178, 92];
  const curve: Pt[] = [];
  for (let i = 0; i <= 24; i++) {
    const t = i / 24;
    curve.push([36 + 92 * t, 96 - 46 * Math.exp(-4 * t)]);
  }
  const d = `M${curve.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join(" L")}`;
  return (
    <>
      <Grid o={[178, 86]} n={5} m={5} />
      <rect x={26} y={30} width={112} height={74} rx={3} fill="#0a0a0a" stroke={LINE} strokeWidth={0.7} />
      {[50, 66, 82].map((y) => (
        <line key={y} x1={32} y1={y} x2={132} y2={y} stroke={LINE_SOFT} strokeWidth={0.6} />
      ))}
      <text x={32} y={40} fill="rgba(255,255,255,0.5)" fontSize={6} fontFamily={MONO}>
        train_loss
      </text>
      <path d={d} fill="none" stroke={ACCENT} strokeWidth={1.3} filter="url(#fa-glow)" pathLength={100} className="fa-draw" />
      <Box o={o} x={1} y={1} w={3} d={3} h={0.4} />
      <g className="fa-late">
        <Box o={o} x={1.8} y={1.8} z={0.4} w={1.4} d={1.4} h={1.4} glow />
      </g>
      <Label at={[190, 140]} text="last.ckpt" cls="fa-late" start={0} />
    </>
  );
}

function Loop() {
  // your own loop: a light runs round a ring on the floor, around a glowing core
  const o: Pt = [130, 46];
  const c = 3.5;
  const ring: Pt[] = [];
  for (let i = 0; i <= 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    ring.push(iso(o, c + 3 * Math.cos(a), c + 3 * Math.sin(a), 0));
  }
  const path = `M${ring.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join(" L")} Z`;
  const ticks = [0, 1, 2, 3].map((k) => {
    const a = (k / 4) * Math.PI * 2 + Math.PI / 4;
    const [p, q] = [iso(o, c + 2.6 * Math.cos(a), c + 2.6 * Math.sin(a), 0), iso(o, c + 3.4 * Math.cos(a), c + 3.4 * Math.sin(a), 0)];
    return <line key={k} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} stroke={LINE} strokeWidth={0.8} />;
  });
  return (
    <>
      <Zoom k={1.25} dy={-4}>
        <Grid o={[130, 46]} n={7} m={7} />
        <path d={path} fill="none" stroke={ACCENT} strokeOpacity={0.35} strokeWidth={0.9} />
        {ticks}
        <g className="fa-pulse-glow">
          <Box o={o} x={c - 0.7} y={c - 0.7} w={1.4} d={1.4} h={1.6} glow />
        </g>
        <circle r={2.4} fill={ACCENT_TOP} filter="url(#fa-glow)" className="fa-move">
          <animateMotion dur="4s" repeatCount="indefinite" path={path} />
        </circle>
      </Zoom>
      <Label at={[22, 128]} text="launch()" />
    </>
  );
}

const ART = { api: ApiStack, scale: AnyScale, placement: Placement, gpus: Fanout, results: Results, loop: Loop };

export function FeatureArt({ kind }: { kind: keyof typeof ART }) {
  const Art = ART[kind];
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid slice"
      className="feature-art block h-48 w-full"
      aria-hidden
    >
      <defs>
        <filter id="fa-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id="fa-bg" cx="50%" cy="55%" r="65%">
          <stop offset="0%" stopColor="#0f1f24" />
          <stop offset="100%" stopColor="#050505" />
        </radialGradient>
      </defs>
      <rect width={W} height={H} fill="url(#fa-bg)" />
      <Art />
    </svg>
  );
}
