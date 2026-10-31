"""Generate docs/assets/architecture-{light,dark}.svg.

    python scripts/make_diagram.py      (needs site/node_modules for the simple-icons paths)

Brand marks come from simple-icons (CC0); no endorsement by their owners is implied.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from make_logo import mark

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "site" / "node_modules" / "simple-icons" / "icons"
OUT = ROOT / "docs" / "assets"

THEMES = {
    "light": {
        "bg": "#ffffff",
        "card": "#fafafa",
        "line": "#e5e5e5",
        "fg": "#171717",
        "muted": "#6b6b6b",
        "accent": "#059669",
        "accent_soft": "#ecfdf5",
        "dark_icon": "#171717",
    },
    "dark": {
        "bg": "#0a0a0a",
        "card": "#141414",
        "line": "#2a2a2a",
        "fg": "#ededed",
        "muted": "#9a9a9a",
        "accent": "#34d399",
        "accent_soft": "#062e22",
        "dark_icon": "#ededed",
    },
}

BRAND = {
    "gnubash": "#4EAA25",
    "python": "#3776AB",
    "pytorch": "#EE4C2C",
    "blender": "#E87D0D",
    "ffmpeg": "#007808",
    "apple": None,
    "nvidia": "#76B900",
    "linux": "#FCC624",
    "numpy": None,
    "docker": "#2496ED",
}

CLOUD = (
    '<path d="M17.5 19H7a4.5 4.5 0 1 1 .9-8.9A6 6 0 0 1 19.4 9.6 4.75 4.75 0 0 1 17.5 19Z" '
    'fill="none" stroke="{c}" stroke-width="1.8" stroke-linejoin="round"/>'
)


def icon(slug: str, x: float, y: float, size: float, t: dict[str, str]) -> str:
    if slug == "cloud":
        return (
            f'<g transform="translate({x} {y}) scale({size / 24})">{CLOUD.format(c=t["muted"])}</g>'
        )
    d = re.search(r' d="([^"]+)"', (ICONS / f"{slug}.svg").read_text()).group(1)
    color = BRAND.get(slug) or t["dark_icon"]
    return (
        f'<g transform="translate({x} {y}) scale({size / 24})"><path fill="{color}" d="{d}"/></g>'
    )


def text(
    x: float,
    y: float,
    s: str,
    t: dict[str, str],
    *,
    size=14,
    weight=400,
    color="fg",
    anchor="start",
    mono=False,
) -> str:
    family = (
        "ui-monospace, SFMono-Regular, Menlo, monospace"
        if mono
        else "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
    )
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{t[color]}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def card(x, y, w, h, t, *, accent=False) -> str:
    fill = t["accent_soft"] if accent else t["card"]
    stroke = t["accent"] if accent else t["line"]
    width = 2 if accent else 1
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def arrow(x1, y1, x2, y2, t) -> str:
    mx = (x1 + x2) / 2
    return (
        f'<path d="M{x1} {y1} C{mx} {y1} {mx} {y2} {x2 - 6} {y2}" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6" stroke-opacity=".75"/>'
        f'<path d="M{x2 - 8} {y2 - 4} L{x2} {y2} L{x2 - 8} {y2 + 4}" fill="none" '
        f'stroke="{t["accent"]}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )


def render(theme: str) -> str:
    t = THEMES[theme]
    W, H = 980, 460
    parts = [f'<rect width="{W}" height="{H}" rx="16" fill="{t["bg"]}"/>']
    # Column headings
    for x, label in ((30, "YOUR TRAINING"), (370, "GPULAYER"), (690, "MACHINES YOU CAN REACH")):
        parts.append(text(x, 44, label, t, size=11, weight=600, color="muted", mono=True))

    # Left: workloads
    work = [
        (["pytorch"], "Trainer", "one GPU, many GPUs, many machines"),
        (["pytorch"], "Your DDP script", "launch(), Runtime"),
        (["pytorch"], "Sweeps", "@gl.remote(...).map([...])"),
        (["gnubash"], "CLI and AI agents", "gpulayer, MCP server"),
    ]
    left_y = []
    for i, (icons, title, sub) in enumerate(work):
        y = 64 + i * 94
        parts.append(card(30, y, 270, 76, t))
        for j, slug in enumerate(icons):
            parts.append(icon(slug, 48 + j * 26, y + 26, 22, t))
        tx = 48 + len(icons) * 26 + 8
        parts.append(text(tx, y + 34, title, t, size=15, weight=600))
        parts.append(text(tx, y + 54, sub, t, size=12.5, color="muted"))
        left_y.append(y + 38)

    # Middle: GPUlayer
    gx, gy, gw, gh = 370, 96, 250, 300
    parts.append(card(gx, gy, gw, gh, t, accent=True))
    parts.append(
        f'<svg x="{gx + 22}" y="{gy + 20}" width="44" height="44" viewBox="0 0 64 64">'
        f"{mark(f'dg-{theme}')}</svg>"
    )
    parts.append(
        f'<text x="{gx + 72}" y="{gy + 50}" font-family="-apple-system, \'Segoe UI\', '
        f'Helvetica, Arial, sans-serif" font-size="24" letter-spacing="-0.5">'
        f'<tspan font-weight="700" fill="{t["accent"]}">GPU</tspan>'
        f'<tspan font-weight="500" fill="{t["muted"]}">layer</tspan></text>'
    )
    features = [
        "Picks machines whose GPUs,",
        "memory and PyTorch fit",
        "Best-connected GPUs, one zone",
        "Ships code, starts every rank",
        "Sets up DDP rendezvous",
        "Streams logs while it trains",
        "Returns checkpoints, metrics",
    ]
    fy = gy + 96
    for k, line in enumerate(features):
        if k != 1:  # line 1 continues line 0
            parts.append(f'<circle cx="{gx + 30}" cy="{fy - 4}" r="3" fill="{t["accent"]}"/>')
        parts.append(text(gx + 42, fy, line, t, size=13))
        fy += 18 if k == 0 else 23
    parts.append(
        text(
            gx + gw / 2,
            gy + gh - 22,
            "one Trainer, any scale",
            t,
            size=12,
            color="muted",
            anchor="middle",
        )
    )

    # Right: machines
    machines = [
        (["apple", "nvidia"], "This machine", "CPU, Apple GPU, NVIDIA GPU"),
        (["linux", "nvidia"], "Servers that joined", "gpulayer join <url>"),
        (["cloud"], "Cloud VMs", "joined, or created per run"),
    ]
    right_y = []
    for i, (icons, title, sub) in enumerate(machines):
        y = 96 + i * 104
        parts.append(card(690, y, 260, 84, t))
        for j, slug in enumerate(icons):
            parts.append(icon(slug, 708 + j * 26, y + 30, 22, t))
        tx = 708 + len(icons) * 26 + 8
        parts.append(text(tx, y + 38, title, t, size=15, weight=600))
        parts.append(
            text(tx, y + 58, sub, t, size=12.5, color="muted", mono=sub.startswith("gpulayer"))
        )
        right_y.append(y + 42)

    for y in left_y:
        parts.append(arrow(300, y, gx, gy + gh / 2, t))
    for y in right_y:
        parts.append(arrow(gx + gw, gy + gh / 2, 690, y, t))

    body = "".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="GPUlayer architecture: your workloads go through '
        f'GPUlayer to the machines you can reach">{body}</svg>\n'
    )


def main() -> None:
    for theme in THEMES:
        path = OUT / f"architecture-{theme}.svg"
        path.write_text(render(theme))
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
