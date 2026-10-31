"""Generate the performance charts in docs/assets/perf/ from a benchmark summary.

    python scripts/make_perf_charts.py [summary.json ...]   (default: the 2026-10-30 runs)

Writes <name>-light.svg and <name>-dark.svg for each chart. Grouped horizontal bars: GPUlayer
in blue, the baseline in orange (a palette validated for colour-vision deficiency in both
modes), values printed at the bar ends, so no reading depends on colour alone.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "perf"
DEFAULT = [
    ROOT / "benchmarks" / "results" / "2026-10-30-gcp" / "summary.json",
    ROOT / "benchmarks" / "results" / "2026-10-30-gcp-two-level" / "summary.json",
    ROOT / "benchmarks" / "results" / "2026-10-31-gcp-t4" / "summary.json",
    ROOT / "benchmarks" / "results" / "2026-10-31-vast-4x3070-xeon" / "summary.json",
]

THEMES = {
    "light": {
        "bg": "#ffffff",
        "text": "#1f2328",
        "muted": "#59636e",
        "grid": "#e6e8eb",
        "axis": "#c9ced4",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "bg": "#0d1117",
        "text": "#e6edf3",
        "muted": "#9198a1",
        "grid": "#21262d",
        "axis": "#3d444d",
        "series": ("#3987e5", "#d95926"),
    },
}
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
W, LABEL_W, PAD = 760, 210, 24
BAR, GAP, ROW_GAP = 14, 2, 18


def nice_step(maximum: float) -> float:
    raw = maximum / 5
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def text(x, y, s, t, *, size=12, color="text", weight=400, anchor="start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{t[color]}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def bar(x0: float, y: float, w: float, h: float, color: str) -> str:
    # Square at the baseline, 4px rounded at the data end.
    r = min(4.0, w / 2, h / 2)
    if w <= 0:
        return ""
    return (
        f'<path d="M{x0:.1f} {y:.1f}H{x0 + w - r:.1f}Q{x0 + w:.1f} {y:.1f} {x0 + w:.1f} '
        f"{y + r:.1f}V{y + h - r:.1f}Q{x0 + w:.1f} {y + h:.1f} {x0 + w - r:.1f} {y + h:.1f}"
        f'H{x0:.1f}Z" fill="{color}"/>'
    )


def grouped(
    theme: str,
    *,
    title: str,
    subtitle: str,
    series: tuple[str, str],
    rows: list[tuple[str, float, float]],
    unit: str,
    fmt: str,
    note: str,
) -> str:
    t = THEMES[theme]
    plot_x0, plot_x1 = PAD + LABEL_W, W - PAD - 64
    top = 104
    row_h = 2 * BAR + GAP + ROW_GAP
    h = top + len(rows) * row_h + 56
    vmax = max(max(a, b) for _, a, b in rows)
    step = nice_step(vmax)
    xmax = step * math.ceil(vmax / step)

    def sx(v: float) -> float:
        return plot_x0 + (plot_x1 - plot_x0) * v / xmax

    p = [f'<rect width="{W}" height="{h}" rx="12" fill="{t["bg"]}"/>']
    p.append(text(PAD, 34, title, t, size=16, weight=600))
    p.append(text(PAD, 56, subtitle, t, size=12.5, color="muted"))
    lx = PAD
    for name, color in zip(series, t["series"], strict=True):
        p.append(f'<rect x="{lx}" y="72" width="12" height="12" rx="3" fill="{color}"/>')
        p.append(text(lx + 18, 82.5, name, t, size=12.5))
        lx += 18 + 7.2 * len(name) + 28
    bottom = top + len(rows) * row_h - ROW_GAP + 6
    k = 0.0
    while k <= xmax + 1e-9:
        x = sx(k)
        p.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}" '
            f'stroke="{t["grid" if k else "axis"]}" stroke-width="1"/>'
        )
        label = f"{k:g}"
        p.append(text(x, bottom + 16, label, t, size=11, color="muted", anchor="middle"))
        k += step
    p.append(
        text(plot_x1, bottom + 34, f"{unit} · {note}", t, size=11, color="muted", anchor="end")
    )
    for i, (label, a, b) in enumerate(rows):
        y = top + i * row_h
        p.append(text(PAD, y + BAR + 5, label, t, size=12.5))
        for j, v in enumerate((a, b)):
            by = y + j * (BAR + GAP)
            p.append(bar(plot_x0, by, sx(v) - plot_x0, BAR, t["series"][j]))
            p.append(text(sx(v) + 6, by + BAR - 3, fmt.format(v), t, size=11.5))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
        f'height="{h}" role="img" aria-label="{escape(title)}: {escape(subtitle)}">'
        + "".join(p)
        + "</svg>\n"
    )


def charts(d: dict) -> dict[str, dict]:
    out = _base_charts(d) if "training_vs_lightning" in d else {}
    if "strong_scaling" in d:
        sc = d["strong_scaling"]
        out["strong-scaling"] = dict(
            title="One training job on more machines: GPUlayer and PyTorch Lightning",
            subtitle=sc["subtitle"],
            series=("GPUlayer Trainer(num_nodes=N)", "Lightning under torchrun"),
            rows=[
                (
                    f"{r['nodes']} machine{'s' if r['nodes'] > 1 else ''}, {r['procs']} processes",
                    r["gpulayer_s"],
                    r["lightning_s"],
                )
                for r in sc["rows"]
            ],
            unit="seconds to train",
            fmt="{:.1f} s",
            note="lower is better",
        )
    if "multi_gpu" in d:
        mg = d["multi_gpu"]
        out["multigpu-step"] = dict(
            title="Several GPUs in one machine: GPUlayer and PyTorch Lightning",
            subtitle=mg["subtitle"],
            series=("GPUlayer Trainer", "Lightning Trainer"),
            rows=[(r["case"], r["gpulayer_ms"], r["lightning_ms"]) for r in mg["rows"]],
            unit="milliseconds per step",
            fmt="{:.1f}",
            note="lower is better",
        )
        out["multigpu-startup"] = dict(
            title="Start-up on several GPUs: GPUlayer and PyTorch Lightning",
            subtitle="Same runs: Python start, imports, process start-up and fit() set-up",
            series=("GPUlayer Trainer", "Lightning Trainer"),
            rows=[
                (r["case"], r["gpulayer_startup_s"], r["lightning_startup_s"]) for r in mg["rows"]
            ],
            unit="seconds",
            fmt="{:.1f} s",
            note="lower is better",
        )
    return out


def _base_charts(d: dict) -> dict[str, dict]:
    tr = d["training_vs_lightning"]["cases"]
    ds = d["distributed_startup"]["rows"]
    mm = d["multi_machine"]["workloads"]
    return {
        "training-step": dict(
            title="Training step time: GPUlayer and PyTorch Lightning",
            subtitle="Same model, data and optimizer on one Tesla T4 (GCP), torch 2.15, "
            "Lightning 2.6.6",
            series=("GPUlayer Trainer", "Lightning Trainer"),
            rows=[(c["case"], c["gpulayer_ms"], c["lightning_ms"]) for c in tr],
            unit="milliseconds per step",
            fmt="{:.2f}",
            note="lower is better",
        ),
        "startup": dict(
            title="Start-up time: GPUlayer and PyTorch Lightning",
            subtitle="Python start, imports and the fixed cost of fit(), same runs as the "
            "step times",
            series=("GPUlayer Trainer", "Lightning Trainer"),
            rows=[(c["case"], c["gpulayer_startup_s"], c["lightning_startup_s"]) for c in tr],
            unit="seconds",
            fmt="{:.2f} s",
            note="lower is better",
        ),
        "distributed-startup": dict(
            title="Distributed job start-up: direct launcher and torchrun",
            subtitle="A trivial 2 nodes x 2 processes job, submit until the client sees the result",
            series=("Direct launcher (default)", "torchrun"),
            rows=[(r["setup"], r["direct_s"], r["torchrun_s"]) for r in ds],
            unit="seconds",
            fmt="{:.2f} s",
            note="lower is better",
        ),
        "scale": dict(
            title="Spreading jobs over two machines (3 workers, 15 slots)",
            subtitle="Wall-clock time with GPUlayer, and the same tasks' run time added up",
            series=("Wall clock with GPUlayer", "Summed task time"),
            rows=[(w["name"], w["wall_s"], w["task_s"]) for w in mm],
            unit="seconds",
            fmt="{:.1f} s",
            note="the gap is work done in parallel",
        ),
    }


def main() -> None:
    sources = [Path(a) for a in sys.argv[1:]] or DEFAULT
    d: dict = {}
    for src in sources:
        d.update(json.loads(src.read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in charts(d).items():
        for theme in THEMES:
            path = OUT / f"{name}-{theme}.svg"
            path.write_text(grouped(theme, **spec))
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
