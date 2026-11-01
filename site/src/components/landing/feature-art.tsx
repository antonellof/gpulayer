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

function Contours() {
  // a contour grid with one traced path through it
  const lines = [];
  for (let i = 0; i < 8; i++) {
    const y = 22 + i * 14;
    lines.push(
      <path
        key={`h${i}`}
        d={`M20 ${y} C80 ${y - 10 + (i % 3) * 6} 160 ${y + 10 - (i % 2) * 8} 220 ${y}`}
        opacity={0.22}
      />,
    );
  }
  for (let i = 0; i < 10; i++) {
    const x = 28 + i * 21;
    lines.push(<path key={`v${i}`} d={`M${x} 16 L${x} 124`} opacity={0.12} />);
  }
  return (
    <>
      {lines}
      <path d="M28 110 C70 108 80 60 112 62 S170 30 212 28" strokeWidth={1.8} />
      {[
        [28, 110],
        [112, 62],
        [212, 28],
      ].map(([x, y]) => (
        <circle key={x} cx={x} cy={y} r={3.5} fill="currentColor" stroke="none" />
      ))}
    </>
  );
}

function Globe() {
  // a wireframe globe with a few linked nodes on it
  const cx = 120;
  const cy = 70;
  const r = 52;
  const nodes = [
    [cx - 30, cy - 24],
    [cx + 34, cy - 6],
    [cx + 4, cy + 34],
    [cx - 40, cy + 18],
  ];
  return (
    <>
      <circle cx={cx} cy={cy} r={r} opacity={0.7} />
      {[-0.6, -0.3, 0, 0.3, 0.6].map((t) => (
        <ellipse
          key={`lat${t}`}
          cx={cx}
          cy={cy + t * r}
          rx={r * Math.sqrt(1 - t * t)}
          ry={r * 0.18 * Math.sqrt(1 - t * t)}
          opacity={0.3}
        />
      ))}
      {[0.2, 0.5, 0.8].map((t) => (
        <ellipse key={`lon${t}`} cx={cx} cy={cy} rx={r * t} ry={r} opacity={0.3} />
      ))}
      {nodes.map(([x, y], i) => {
        const [nx, ny] = nodes[(i + 1) % nodes.length];
        return (
          <g key={i}>
            <path d={`M${x} ${y} L${nx} ${ny}`} strokeDasharray="3 3" opacity={0.6} />
            <circle cx={x} cy={y} r={3.5} fill="currentColor" stroke="none" />
          </g>
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

function Orbit() {
  // an orbit of ellipses around a core
  const cx = 120;
  const cy = 70;
  return (
    <>
      {[0, 30, 60, 90, 120, 150].map((a, i) => (
        <ellipse
          key={a}
          cx={cx}
          cy={cy}
          rx={78}
          ry={24}
          transform={`rotate(${a} ${cx} ${cy})`}
          opacity={0.18 + i * 0.08}
        />
      ))}
      <circle cx={cx} cy={cy} r={12} />
      <circle cx={cx} cy={cy} r={4} fill="currentColor" stroke="none" />
      <circle cx={cx + 78} cy={cy} r={3.5} fill="currentColor" stroke="none" />
    </>
  );
}

const ART = { api: Planes, scale: Rings, placement: Contours, gpus: Globe, results: Waves, loop: Orbit };

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
