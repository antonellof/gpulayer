// Isometric illustrations for the "What you get" cards: plain SVG built from one primitive, an
// isometric box with three shaded faces. Colours are solid, so they read on light and dark.

type Hue = { top: string; left: string; right: string };

const HUES: Record<string, Hue> = {
  blue: { top: "#7cb4ff", left: "#3b82f6", right: "#1d4ed8" },
  violet: { top: "#c4a8ff", left: "#8b5cf6", right: "#6d28d9" },
  emerald: { top: "#6ee7b7", left: "#10b981", right: "#047857" },
  amber: { top: "#fcd34d", left: "#f59e0b", right: "#b45309" },
  rose: { top: "#fda4af", left: "#f43f5e", right: "#be123c" },
  cyan: { top: "#67e8f9", left: "#06b6d4", right: "#0e7490" },
  slate: { top: "#cbd5e1", left: "#94a3b8", right: "#64748b" },
};

const S = 9; // size of one unit
const C30 = Math.cos(Math.PI / 6);

// world (x, y, z) to screen: x goes down-right, y down-left, z up
function p(ox: number, oy: number, x: number, y: number, z: number): [number, number] {
  return [ox + (x - y) * C30 * S, oy + (x + y) * 0.5 * S - z * S];
}

function pts(list: [number, number][]): string {
  return list.map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join(" ");
}

function Box({
  o, x, y, z = 0, w = 1, d = 1, h = 1, hue, opacity = 1,
}: {
  o: [number, number]; x: number; y: number; z?: number; w?: number; d?: number; h?: number;
  hue: Hue; opacity?: number;
}) {
  const [ox, oy] = o;
  const P = (a: number, b: number, c: number) => p(ox, oy, a, b, c);
  const top = [P(x, y, z + h), P(x + w, y, z + h), P(x + w, y + d, z + h), P(x, y + d, z + h)];
  const left = [P(x, y + d, z), P(x + w, y + d, z), P(x + w, y + d, z + h), P(x, y + d, z + h)];
  const right = [P(x + w, y, z), P(x + w, y + d, z), P(x + w, y + d, z + h), P(x + w, y, z + h)];
  return (
    <g opacity={opacity}>
      <polygon points={pts(left)} fill={hue.left} />
      <polygon points={pts(right)} fill={hue.right} />
      <polygon points={pts(top)} fill={hue.top} />
    </g>
  );
}

function Floor({ o, w, d, hue = "slate" }: { o: [number, number]; w: number; d: number; hue?: string }) {
  const [ox, oy] = o;
  const P = (a: number, b: number) => p(ox, oy, a, b, 0);
  return (
    <polygon
      points={pts([P(-0.6, -0.6), P(w + 0.6, -0.6), P(w + 0.6, d + 0.6), P(-0.6, d + 0.6)])}
      fill={HUES[hue].top}
      opacity={0.18}
    />
  );
}

function Link({ a, b, color }: { a: [number, number]; b: [number, number]; color: string }) {
  return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={color} strokeWidth={1.6} strokeDasharray="3 3" />;
}

function ApiStack() {
  const o: [number, number] = [120, 58];
  const layers: [string, number][] = [["slate", 0], ["cyan", 1.3], ["violet", 2.6], ["blue", 3.9]];
  return (
    <>
      <Floor o={o} w={6} d={6} hue="blue" />
      {layers.map(([hue, z]) => (
        <Box key={z} o={o} x={0} y={0} z={z} w={6} d={6} h={1} hue={HUES[hue]} />
      ))}
      <Box o={o} x={2} y={2} z={5.2} w={2} d={2} h={1.6} hue={HUES.amber} />
    </>
  );
}

function AnyScale() {
  const o: [number, number] = [118, 40];
  return (
    <>
      <Floor o={o} w={9} d={5} hue="violet" />
      <Box o={o} x={0} y={2} h={1.4} hue={HUES.violet} />
      {[0, 1, 2, 3].map((i) => (
        <Box key={`r${i}`} o={o} x={3} y={i * 1.2} h={1.4} w={1} d={1} hue={HUES.violet} />
      ))}
      {[0, 1].map((r) =>
        [0, 1, 2, 3].map((c) => (
          <Box key={`g${r}${c}`} o={o} x={6 + r * 1.2} y={c * 1.2} h={1.4} hue={HUES.blue} />
        )),
      )}
    </>
  );
}

function Placement() {
  const o: [number, number] = [120, 46];
  const chosen = new Set(["1-1", "1-2"]);
  const servers = [];
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) {
      const key = `${r}-${c}`;
      servers.push(
        <Box
          key={key}
          o={o}
          x={r * 2}
          y={c * 2}
          w={1.4}
          d={1.4}
          h={chosen.has(key) ? 2.6 : 1.6}
          hue={chosen.has(key) ? HUES.emerald : HUES.slate}
          opacity={chosen.has(key) ? 1 : 0.75}
        />,
      );
    }
  const pin = p(o[0], o[1], 2.7, 3.4, 5.4);
  return (
    <>
      <Floor o={o} w={5.4} d={5.4} hue="emerald" />
      {servers}
      <g transform={`translate(${pin[0]} ${pin[1]})`}>
        <path d="M0 10 C-7 2 -7 -6 0 -6 C7 -6 7 2 0 10 Z" fill={HUES.emerald.left} />
        <circle cx={0} cy={-1} r={2.6} fill="#fff" />
      </g>
    </>
  );
}

function EveryGpu() {
  const o: [number, number] = [120, 52];
  const hub = p(o[0], o[1], 3, 3, 2.4);
  const laptop = p(o[0], o[1], 0.6, 6.2, 0.9);
  const server = p(o[0], o[1], 6.4, 1, 2.2);
  const cloud = p(o[0], o[1], 6.2, 6.4, 1.4);
  return (
    <>
      <Floor o={o} w={7} d={7} hue="amber" />
      <Link a={hub} b={laptop} color={HUES.amber.left} />
      <Link a={hub} b={server} color={HUES.amber.left} />
      <Link a={hub} b={cloud} color={HUES.amber.left} />
      {/* laptop: a base and a screen */}
      <Box o={o} x={-0.2} y={5.6} w={1.8} d={1.4} h={0.25} hue={HUES.slate} />
      <Box o={o} x={-0.2} y={5.4} w={1.8} d={0.2} h={1.3} hue={HUES.blue} />
      {/* server tower */}
      <Box o={o} x={6} y={0.2} w={1.2} d={1.6} h={3} hue={HUES.rose} />
      {/* cloud VM */}
      <Box o={o} x={5.6} y={5.8} w={1.4} d={1.4} h={1.3} hue={HUES.cyan} />
      <g transform={`translate(${cloud[0] + 4} ${cloud[1] - 20})`}>
        <path d="M14 10H3a5 5 0 1 1 1-9.9A6.5 6.5 0 0 1 16.5 2 4.6 4.6 0 0 1 14 10Z" fill={HUES.cyan.top} />
      </g>
      {/* the hub: GPUlayer */}
      <Box o={o} x={2.4} y={2.4} w={1.4} d={1.4} h={2} hue={HUES.amber} />
    </>
  );
}

function Results() {
  const o: [number, number] = [112, 52];
  const bars = [1.4, 2.4, 3.2, 4.4];
  const doc = p(o[0], o[1], 5.6, 0.2, 0);
  return (
    <>
      <Floor o={o} w={6.5} d={4} hue="rose" />
      {bars.map((h, i) => (
        <Box key={i} o={o} x={i * 1.3} y={2.2} w={1} d={1} h={h} hue={i === 3 ? HUES.rose : HUES.slate} />
      ))}
      <g transform={`translate(${doc[0]} ${doc[1] - 34})`}>
        <polygon points="0,8 22,-3 22,24 0,35" fill={HUES.rose.top} />
        {[0, 1, 2, 3].map((i) => (
          <line key={i} x1={4} y1={13 + i * 5} x2={18} y2={6 + i * 5} stroke={HUES.rose.right} strokeWidth={1.5} />
        ))}
      </g>
    </>
  );
}

function OwnLoop() {
  const o: [number, number] = [120, 50];
  const ring: [number, number][] = [
    [0, 0], [1.3, 0], [2.6, 0], [3.9, 0], [3.9, 1.3], [3.9, 2.6], [3.9, 3.9], [2.6, 3.9],
    [1.3, 3.9], [0, 3.9], [0, 2.6], [0, 1.3],
  ];
  return (
    <>
      <Floor o={o} w={5} d={5} hue="cyan" />
      {ring.map(([x, y], i) => (
        <Box key={i} o={o} x={x} y={y} w={1} d={1} h={0.9} hue={i % 3 === 0 ? HUES.cyan : HUES.slate} />
      ))}
      <Box o={o} x={1.5} y={1.5} w={2} d={2} h={1.8} hue={HUES.blue} />
    </>
  );
}

const ART: Record<string, () => JSX.Element> = {
  api: ApiStack,
  scale: AnyScale,
  placement: Placement,
  gpus: EveryGpu,
  results: Results,
  loop: OwnLoop,
};

export function FeatureArt({ kind, label }: { kind: keyof typeof ART; label: string }) {
  const Art = ART[kind];
  return (
    <svg viewBox="0 0 240 140" className="h-40 w-full" role="img" aria-label={label}>
      <g transform="translate(120 80) scale(1.25) translate(-120 -72)">
        <Art />
      </g>
    </svg>
  );
}
