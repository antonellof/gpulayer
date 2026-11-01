"use client";

// A 3D infinity loop for the "Your own loop works too" card: a twisted wireframe ribbon on a
// figure-of-eight that turns slowly in space, with a light running along it. Drawn as SVG
// inside the card's <svg>; animates only while on screen and stays still with
// prefers-reduced-motion.

import { useEffect, useRef, useState } from "react";

const C30 = Math.cos(Math.PI / 6);
const ACCENT = "#22d3ee";
const ACCENT_TOP = "#a5f3fc";
const SAMPLES = 160;
const STRANDS = [-1, -0.5, 0, 0.5, 1];
const TURN_SECONDS = 18; // one turn of the loop around its vertical axis
const LAP_SECONDS = 3.6; // one lap of the light

type V3 = [number, number, number];

function curve(t: number): V3 {
  const a = 3.4;
  const d = 1 + Math.sin(t) ** 2;
  return [(a * Math.cos(t)) / d, (a * Math.sin(t) * Math.cos(t)) / d, 0.9 * Math.sin(t)];
}

/** Point on the ribbon: the curve, offset across its width by a direction that twists once per lap. */
function ribbon(t: number, across: number): V3 {
  const p = curve(t);
  const q = curve(t + 1e-3);
  let nx = -(q[1] - p[1]);
  let ny = q[0] - p[0];
  const len = Math.hypot(nx, ny) || 1;
  nx /= len;
  ny /= len;
  const w = 0.6 * across;
  return [p[0] + w * Math.cos(t) * nx, p[1] + w * Math.cos(t) * ny, p[2] + w * Math.sin(t)];
}

export function InfinityLoop({ cx, cy, scale }: { cx: number; cy: number; scale: number }) {
  const ref = useRef<SVGGElement>(null);
  const [time, setTime] = useState(1.2);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    let visible = false;
    const start = performance.now() - 1200;
    const tick = (now: number) => {
      raf = 0;
      if (!visible) return;
      setTime((now - start) / 1000);
      raf = requestAnimationFrame(tick);
    };
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible && !raf) raf = requestAnimationFrame(tick);
    });
    const target = ref.current?.ownerSVGElement;
    if (target) io.observe(target);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, []);

  const theta = (time / TURN_SECONDS) * Math.PI * 2;
  const [ct, st] = [Math.cos(theta), Math.sin(theta)];
  const k = 10 * scale;
  const project = ([x, y, z]: V3): [number, number, number] => {
    const xr = x * ct - y * st;
    const yr = x * st + y * ct;
    return [cx + (xr - yr) * C30 * k, cy + (xr + yr) * 0.5 * k - z * k, xr + yr];
  };
  const path = (pts: [number, number, number][], close: boolean) =>
    `M${pts.map(([a, b]) => `${a.toFixed(1)} ${b.toFixed(1)}`).join(" L")}${close ? " Z" : ""}`;

  const ts = Array.from({ length: SAMPLES }, (_, i) => (i / SAMPLES) * Math.PI * 2);
  const strands = STRANDS.map((s) => ts.map((t) => project(ribbon(t, s))));
  const rungs = ts
    .filter((_, i) => i % 4 === 0)
    .map((t) => {
      const [a, b] = [project(ribbon(t, -1)), project(ribbon(t, 1))];
      const near = Math.max(0, Math.min(1, (a[2] + 4) / 8));
      return { a, b, opacity: 0.08 + 0.3 * near };
    });

  const head = ((time % LAP_SECONDS) / LAP_SECONDS) * Math.PI * 2;
  const tail = Array.from({ length: 24 }, (_, i) => project(ribbon(head - (i / 23) * 1.1, 0)));
  const [hx, hy] = tail[0];

  return (
    <g ref={ref}>
      <ellipse cx={cx} cy={cy + 3.2 * k * 0.5 + 6} rx={3.6 * k * C30} ry={1.2 * k} fill={ACCENT} opacity={0.05} />
      {rungs.map(({ a, b, opacity }, i) => (
        <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#fff" strokeOpacity={opacity} strokeWidth={0.5} />
      ))}
      {strands.map((pts, i) => (
        <path
          key={i}
          d={path(pts, true)}
          fill="none"
          stroke={STRANDS[i] === 0 ? ACCENT : "#fff"}
          strokeOpacity={STRANDS[i] === 0 ? 0.35 : Math.abs(STRANDS[i]) === 1 ? 0.4 : 0.16}
          strokeWidth={Math.abs(STRANDS[i]) === 1 ? 0.8 : 0.6}
        />
      ))}
      <path d={path(tail, false)} fill="none" stroke={ACCENT} strokeWidth={2} strokeLinecap="round" filter="url(#fa-glow)" />
      <circle cx={hx} cy={hy} r={2.6} fill={ACCENT_TOP} filter="url(#fa-glow)" />
    </g>
  );
}
