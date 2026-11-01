"""Generate docs/assets/concepts-{light,dark}.svg: targets (where) x workload types (what).

    python scripts/make_concepts_diagram.py

A 3 x 2 grid: the columns are the targets (local, remote, cloud), the rows the workload types
(independent, distributed); every cell shows one machine layout and a real example, because
any combination is valid.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"

THEMES = {
    "light": {
        "bg": "#ffffff", "card": "#fafafa", "line": "#e5e5e5", "fg": "#171717",
        "muted": "#6b6b6b", "machine": "#d4d4d4", "machine_fill": "#ffffff",
        "accent": "#2a78d6", "accent_soft": "#e8f1fc", "rank": "#8250df", "rank_soft": "#f1ebfd",
        "link": "#8250df",
    },
    "dark": {
        "bg": "#0a0a0a", "card": "#141414", "line": "#2a2a2a", "fg": "#ededed",
        "muted": "#9a9a9a", "machine": "#3a3a3a", "machine_fill": "#1b1b1b",
        "accent": "#58a6ff", "accent_soft": "#0f2742", "rank": "#bc8cff", "rank_soft": "#2a1d47",
        "link": "#bc8cff",
    },
}  # fmt: skip

W, H = 980, 600
LEFT, TOP = 170, 118  # grid origin
CW, CH, GAP = 255, 196, 18  # cell size and gap
FONT = "Geist, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"


def text(x, y, s, t, *, size=14, color="fg", weight=400, anchor="start", mono=False):
    fam = MONO if mono else FONT
    return (
        f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
        f'fill="{t[color]}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def laptop(x, y, t, w=96):
    h = w * 0.62
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{t["machine_fill"]}" '
        f'stroke="{t["machine"]}" stroke-width="2"/>'
        f'<path d="M{x - 10} {y + h + 3} h{w + 20} l-8 9 h{-(w + 4)} Z" fill="{t["machine"]}"/>'
    )


def server(x, y, t, w=96, h=64):
    bars = "".join(
        f'<line x1="{x + 12}" y1="{y + 14 + i * 14}" x2="{x + 34}" y2="{y + 14 + i * 14}" '
        f'stroke="{t["machine"]}" stroke-width="3" stroke-linecap="round"/>'
        for i in range(3)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{t["machine_fill"]}" '
        f'stroke="{t["machine"]}" stroke-width="2"/>{bars}'
    )


def cloud_vm(x, y, t, w=96, h=64):
    # a server under a small cloud: a machine that a provider created
    c = (
        f'<g transform="translate({x + w - 30} {y - 18}) scale(1.4)">'
        f'<path d="M17.5 19H7a4.5 4.5 0 1 1 .9-8.9A6 6 0 0 1 19.4 9.6 4.75 4.75 0 0 1 17.5 19Z" '
        f'fill="{t["bg"]}" stroke="{t["muted"]}" stroke-width="1.6" stroke-linejoin="round"/></g>'
    )
    return server(x, y, t, w, h) + c


def proc(x, y, label, t, *, rank=False, w=58, h=26):
    stroke, fill = (t["rank"], t["rank_soft"]) if rank else (t["accent"], t["accent_soft"])
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.6"/>'
        + text(x + w / 2, y + 17, label, t, size=11.5, color="fg", anchor="middle", mono=True)
    )


def link(x1, y1, x2, y2, t):
    return (
        f'<path d="M{x1} {y1} C{(x1 + x2) / 2} {y1 - 28} {(x1 + x2) / 2} {y2 - 28} {x2} {y2}" '
        f'fill="none" stroke="{t["link"]}" stroke-width="1.8" stroke-dasharray="4 4"/>'
    )


MACHINE = {"local": laptop, "remote": server, "cloud": cloud_vm}

COLUMNS = [
    ("local", "Local", "workers on this machine"),
    ("remote", "Remote", "machines that joined your server"),
    ("cloud", "Cloud", "machines a provider creates for the job"),
]
ROWS = [
    ("independent", "Independent", ["one process tree", "on one worker"]),
    ("distributed", "Distributed", ["ranks on several", "processes or workers"]),
]
EXAMPLES = {
    ("independent", "local"): "python train.py",
    ("independent", "remote"): "a sweep job on a GPU box",
    ("independent", "cloud"): "one job on a fresh VM",
    ("distributed", "local"): "DDP on 2 processes",
    ("distributed", "remote"): "Trainer(num_nodes=2)",
    ("distributed", "cloud"): "2 VMs, rendezvous on rank 0",
}


def cell(col, row, t):
    x = LEFT + col * (CW + GAP)
    y = TOP + row * (CH + GAP)
    kind = COLUMNS[col][0]
    wtype = ROWS[row][0]
    out = [
        f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="14" fill="{t["card"]}" '
        f'stroke="{t["line"]}"/>'
    ]
    draw = MACHINE[kind]
    if wtype == "independent":
        mx, my = x + CW / 2 - 48, y + 46
        out.append(draw(mx, my, t))
        out.append(proc(mx + (19 if kind == "local" else 32), my + 18, "job", t, w=52))
    elif kind == "local":
        # one machine, two processes: ranks 0 and 1
        mx, my = x + CW / 2 - 64, y + 40
        out.append(draw(mx, my, t, w=128))
        out.append(proc(mx + 4, my + 26, "rank 0", t, rank=True))
        out.append(proc(mx + 66, my + 26, "rank 1", t, rank=True))
    else:
        # two machines, two ranks each, linked
        for k in range(2):
            mx, my = x + 18 + k * 122, y + 52
            out.append(draw(mx, my, t, w=96, h=64))
            out.append(proc(mx + 24, my + 20, f"ranks {2 * k}-{2 * k + 1}", t, rank=True, w=68))
        out.append(link(x + 18 + 48, y + 50, x + 140 + 48, y + 50, t))
    out.append(text(x + CW / 2, y + CH - 26, EXAMPLES[(wtype, kind)], t, size=12.5,
                    color="muted", anchor="middle", mono=True))  # fmt: skip
    return "".join(out)


def svg(t: dict[str, str]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="Targets (local, remote, cloud) by workload type (independent, '
        f'distributed): every combination is valid">',
        f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
        text(LEFT, 36, "Target: where it runs", t, size=13, color="muted", weight=500),
        text(28, TOP - 14, "Workload: what runs", t, size=13, color="muted", weight=500),
    ]
    for col, (_, name, sub) in enumerate(COLUMNS):
        cx = LEFT + col * (CW + GAP) + CW / 2
        parts.append(text(cx, 74, name, t, size=18, weight=600, anchor="middle"))
        parts.append(text(cx, 96, sub, t, size=12.5, color="muted", anchor="middle"))
    for row, (_, name, sub) in enumerate(ROWS):
        cy = TOP + row * (CH + GAP) + CH / 2
        parts.append(text(28, cy - 10, name, t, size=17, weight=600))
        for k, line in enumerate(sub):
            parts.append(text(28, cy + 12 + k * 17, line, t, size=12.5, color="muted"))
    for row in range(2):
        for col in range(3):
            parts.append(cell(col, row, t))
    note = (
        "Every combination is valid. A job moves to another target only when you list it in "
        "placement.fallback_targets."
    )
    parts.append(text(LEFT, H - 22, note, t, size=12.5, color="muted"))
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    for name, t in THEMES.items():
        path = OUT / f"concepts-{name}.svg"
        path.write_text(svg(t))
        print(path)


if __name__ == "__main__":
    main()
