// Monochrome wireframe art for the "What you get" cards: decorative line drawings in the text
// colour (currentColor), so they follow light and dark mode. Each card gets its own motif.

const W = 240;
const H = 140;

function Planes() {
  // stacked wire planes in perspective
  const rows = [0, 1, 2, 3, 4];
  return (
    <>
      {rows.map((i) => {
        const y = 34 + i * 18;
        const inset = i * 6;
        return (
          <g key={i} opacity={0.25 + i * 0.15}>
            <path d={`M${60 + inset} ${y} L${120} ${y - 16} L${180 - inset} ${y} L${120} ${y + 16} Z`} />
            {[0.25, 0.5, 0.75].map((t) => (
              <path
                key={t}
                d={`M${60 + inset + (60 - inset) * t} ${y - 16 * t} L${120 + (60 - inset) * t} ${y + 16 - 16 * t}`}
                opacity={0.5}
              />
            ))}
          </g>
        );
      })}
    </>
  );
}

function Rings() {
  // one ring, then a growing field of rings
  const out = [];
  for (let c = 0; c < 9; c++)
    for (let r = 0; r < 5; r++) {
      if (r > Math.floor(c / 2)) continue;
      out.push(
        <circle key={`${c}-${r}`} cx={36 + c * 21} cy={110 - r * 19} r={6.5} opacity={0.3 + (c / 9) * 0.7} />,
      );
    }
  return (
    <>
      <path d="M24 122 H216" opacity={0.3} />
      {out}
    </>
  );
}

function Pillars() {
  // a floor grid in perspective with wire pillars rising from it
  const vx = 120; // vanishing point
  const vy = -10;
  const floor = [];
  for (let i = 0; i <= 10; i++) {
    const x = 20 + i * 20;
    // lines towards the vanishing point, cut at the horizon
    const t = 0.55;
    floor.push(<path key={`r${i}`} d={`M${x} 128 L${x + (vx - x) * t} ${128 + (vy - 128) * t}`} opacity={0.18} />);
  }
  for (let j = 0; j < 6; j++) {
    const k = j / 6;
    const y = 128 + (vy - 128) * (0.55 * k);
    const half = 100 * (1 - 0.55 * k);
    floor.push(<path key={`h${j}`} d={`M${vx - half} ${y} L${vx + half} ${y}`} opacity={0.18} />);
  }
  const pillars = [
    [82, 104, 22, 0.35],
    [104, 96, 34, 0.5],
    [128, 92, 58, 1],
    [150, 96, 52, 1],
    [170, 104, 26, 0.45],
  ];
  return (
    <>
      {floor}
      {pillars.map(([x, y, h, o]) => (
        <g key={x} opacity={o}>
          <path d={`M${x - 6} ${y} L${x - 6} ${y - h} L${x + 6} ${y - h} L${x + 6} ${y}`} />
          <path d={`M${x - 6} ${y - h} L${x - 1} ${y - h - 4} L${x + 11} ${y - h - 4} L${x + 6} ${y - h}`} />
          <path d={`M${x + 6} ${y} L${x + 11} ${y - 4} L${x + 11} ${y - h - 4}`} />
        </g>
      ))}
    </>
  );
}

function Ripples() {
  // ripples spreading from one point, with nodes on the rings
  const cx = 120;
  const cy = 120;
  const rings = [14, 30, 48, 66, 84, 100];
  const nodes = [
    [1, -130],
    [2, -60],
    [3, -150],
    [3, -35],
    [4, -105],
    [5, -70],
    [5, -160],
    [5, -20],
  ];
  return (
    <>
      {rings.map((r, i) => (
        <path
          key={r}
          d={`M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          opacity={0.9 - i * 0.13}
        />
      ))}
      <path d="M16 120 H224" opacity={0.25} />
      <circle cx={cx} cy={cy} r={4} fill="currentColor" stroke="none" />
      {nodes.map(([ri, deg]) => {
        const r = rings[ri];
        const a = (deg * Math.PI) / 180;
        return (
          <circle
            key={`${ri}${deg}`}
            cx={cx + r * Math.cos(a)}
            cy={cy + r * Math.sin(a)}
            r={2.8}
            fill="currentColor"
            stroke="none"
            opacity={1 - ri * 0.1}
          />
        );
      })}
    </>
  );
}

function Waves() {
  // layered waves settling down, like a loss curve
  const out = [];
  for (let k = 0; k < 7; k++) {
    const amp = 26 - k * 3.2;
    const y0 = 40 + k * 11;
    let d = "";
    for (let x = 20; x <= 220; x += 4) {
      const t = (x - 20) / 200;
      const y = y0 + Math.sin(t * 9 + k * 0.6) * amp * Math.exp(-t * 2.2);
      d += `${d ? " L" : "M"}${x} ${y.toFixed(1)}`;
    }
    out.push(<path key={k} d={d} opacity={0.15 + k * 0.12} />);
  }
  return <>{out}</>;
}

function Twist() {
  // a twisting stack of rounded squares, each turned a little more
  const cx = 120;
  const cy = 70;
  const out = [];
  for (let i = 0; i < 7; i++) {
    const size = 112 - i * 14;
    out.push(
      <rect
        key={i}
        x={cx - size / 2}
        y={cy - size / 2}
        width={size}
        height={size}
        rx={size * 0.22}
        transform={`rotate(${i * 8} ${cx} ${cy})`}
        opacity={0.22 + i * 0.13}
      />,
    );
  }
  return (
    <>
      {out}
      <circle cx={cx} cy={cy} r={3.5} fill="currentColor" stroke="none" />
    </>
  );
}

const ART = { api: Planes, scale: Rings, placement: Pillars, gpus: Ripples, results: Waves, loop: Twist };

export function FeatureArt({ kind }: { kind: keyof typeof ART }) {
  const Art = ART[kind];
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-36 w-full text-foreground"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.1}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <Art />
    </svg>
  );
}
