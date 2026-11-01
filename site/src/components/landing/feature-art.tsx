// Feature art: fine isometric wireframes on a dark panel, one glowing accent, small monospace
// labels, and a looping animation that shows what the card describes. Motion is CSS
// (.fa-* in globals.css) plus SMIL for movement along paths; with prefers-reduced-motion the
// scenes show a still frame.

import type { CSSProperties } from "react";

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
  // a scan passes over the racks; the chosen pair lights up and their link flows
  const o: Pt = [130, 54];
  const cells: [number, number, boolean][] = [];
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 4; c++) cells.push([r, c, r === 1 && (c === 1 || c === 2)]);
  cells.sort((p, q) => p[0] * 2 + p[1] * 1.6 - (q[0] * 2 + q[1] * 1.6));
  const a = iso(o, 2.6, 2.1, 4.4);
  const b = iso(o, 2.6, 3.7, 4.4);
  const scanFrom = iso(o, -0.5, -0.5, 0);
  const scanTo = iso(o, 6.5, 6.5, 0);
  return (
    <>
      <Grid o={[130, 64]} n={6} m={7} />
      <line
        x1={scanFrom[0] - 70}
        y1={scanFrom[1] + 40}
        x2={scanFrom[0] + 70}
        y2={scanFrom[1] - 40}
        stroke={ACCENT}
        strokeOpacity={0.5}
        strokeWidth={1}
        className="fa-scan"
        style={{ ["--scan-x" as string]: `${scanTo[0] - scanFrom[0]}px`, ["--scan-y" as string]: `${scanTo[1] - scanFrom[1]}px` }}
      />
      {cells.map(([r, c, chosen]) =>
        chosen ? (
          <LitBox key={`${r}${c}`} cls="fa-chosen" at={0} still o={o} x={r * 2} y={c * 1.6} w={1.2} d={1.1} h={3.2} />
        ) : (
          <Box key={`${r}${c}`} o={o} x={r * 2} y={c * 1.6} w={1.2} d={1.1} h={2.2} dim={r === 0} />
        ),
      )}
      <path
        d={`M${a[0]} ${a[1]} Q${(a[0] + b[0]) / 2} ${a[1] - 16} ${b[0]} ${b[1]}`}
        fill="none"
        stroke={ACCENT}
        strokeWidth={0.9}
        strokeDasharray="2 2"
        className="fa-dash fa-chosen"
      />
      <Label at={[188, 30]} text="same zone" cls="fa-chosen" start={0} />
    </>
  );
}

function Fanout() {
  // pulses travel from the server to the workers; each worker lights up when one arrives
  const o: Pt = [130, 68];
  const hub = iso(o, 3.1, 3.1, 2.2);
  const tiles = [];
  const links = [];
  let n = 0;
  for (let r = 0; r < 5; r++)
    for (let c = 0; c < 5; c++) {
      if (r === 2 && c === 2) continue;
      const x = r * 1.4 - 0.2;
      const y = c * 1.4 - 0.2;
      const linked = (r + c) % 2 === 0;
      if (!linked) {
        tiles.push(<Box key={`${r}${c}`} o={o} x={x} y={y} w={0.8} d={0.8} h={0.3} dim />);
        continue;
      }
      const t = iso(o, x + 0.4, y + 0.4, 0.3);
      const begin = (n % 6) * 0.45;
      const path = `M${hub[0].toFixed(1)} ${hub[1].toFixed(1)} L${t[0].toFixed(1)} ${t[1].toFixed(1)}`;
      links.push(
        <g key={`l${r}${c}`}>
          <path d={path} stroke={ACCENT} strokeOpacity={0.22} strokeWidth={0.6} />
          <Packet path={path} dur={2.7} begin={begin} />
        </g>,
      );
      // the packet fades out at 85% of its 2.7 s trip: that is when it arrives
      tiles.push(<LitBox key={`${r}${c}`} cls="fa-arrive" at={begin + 2.3} o={o} x={x} y={y} w={0.8} d={0.8} h={0.3} />);
      n++;
    }
  return (
    <>
      <Grid o={[130, 66]} n={7} m={7} />
      {tiles}
      {links}
      <g className="fa-pulse-glow">
        <Box o={o} x={2.4} y={2.4} z={2.2} w={1.4} d={1.4} h={1.4} glow />
      </g>
      <Label at={[20, 30]} text="gpulayer up" />
      <Label at={[190, 128]} text="workers" />
    </>
  );
}

function Results() {
  // bars grow one after another, the loss curve draws itself, then the checkpoint appears
  const o: Pt = [96, 64];
  const bars = [0.8, 1.2, 1.6, 2.2, 2.8, 4.2];
  return (
    <>
      <Grid o={[96, 70]} n={7} m={3} />
      {bars.map((h, i) => (
        <g key={i} className="fa-grow" style={delay(i * 0.25)}>
          <Box o={o} x={i * 1.1} y={1} w={0.7} d={0.7} h={h} glow={i === bars.length - 1} dim={i < 2} />
        </g>
      ))}
      <g transform="translate(168 30)">
        <rect width={74} height={52} rx={2} fill="#0a0a0a" stroke={LINE} strokeWidth={0.7} />
        <path
          d="M6 12 C18 14 22 34 34 38 S56 44 68 45"
          fill="none"
          stroke={ACCENT}
          strokeWidth={1.1}
          filter="url(#fa-glow)"
          pathLength={100}
          className="fa-draw"
        />
        <text x={6} y={8} fill="rgba(255,255,255,0.5)" fontSize={5.5} fontFamily={MONO}>
          train_loss
        </text>
      </g>
      <Label at={[168, 104]} text="last.ckpt" cls="fa-late" start={0} />
    </>
  );
}

function Loop() {
  // a light runs round the ring; each tile lights up as it passes; the core pulses
  const o: Pt = [130, 50];
  const ring: Pt[] = [];
  for (let i = 0; i < 4; i++) ring.push([i, 0]);
  for (let i = 1; i < 4; i++) ring.push([3, i]);
  for (let i = 2; i >= 0; i--) ring.push([i, 3]);
  for (let i = 2; i >= 1; i--) ring.push([0, i]);
  const pts = ring.map(([x, y]) => iso(o, x * 1.5 + 0.5, y * 1.5 + 0.5, 0.6));
  const path = `M${pts.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join(" L")} Z`;
  const lap = 6; // seconds per lap
  return (
    <>
      <Grid o={[130, 50]} n={6} m={6} />
      {ring.map(([x, y], i) => (
        <LitBox key={i} cls="fa-pass" at={(i / ring.length) * lap} o={o} x={x * 1.5} y={y * 1.5} w={1} d={1} h={0.6} />
      ))}
      <path d={path} fill="none" stroke={ACCENT} strokeOpacity={0.3} strokeWidth={0.8} />
      <circle r={3} fill={ACCENT_TOP} filter="url(#fa-glow)" className="fa-move">
        <animateMotion dur={`${lap}s`} repeatCount="indefinite" path={path} />
      </circle>
      <g className="fa-pulse-glow">
        <Box o={o} x={2} y={2} w={1.6} d={1.6} h={1.2} glow />
      </g>
      <Label at={[24, 120]} text="launch()" />
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
