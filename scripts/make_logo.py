"""Generate the GPUlayer mark: a chip on top of a stack of three isometric layers.

    python scripts/make_logo.py

Writes docs/assets/logo-mark.svg and docs/assets/logo.svg, and copies the mark to the site
favicon (site/src/app/icon.svg) and the dashboard (src/gpulayer/api/ui/icon.svg).

Each layer is drawn as a flat rounded square and projected with an isometric matrix, so the
chip's die and pins stay parallel to the layer edges. Gaps between layers are masks, not
background-coloured strokes, so the mark works on any background.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

# Square face from -20 to 20, projected to 56 x 32 px (half-width 28, half-height 16).
ISO = "matrix(0.7 0.4 -0.7 0.4 {x} {y})"
CX, LAYER_Y = 32, (23.0, 32.0, 41.0)  # top, middle, bottom layer centres in a 64 x 64 box
FACE = '<rect x="-20" y="-20" width="40" height="40" rx="6"/>'
GAP = 2.4  # width of the gap between layers, in face units

GRADIENTS = {
    "top": ("#6b4dff", "#1ea7ff", "#16efc6"),
    "mid": ("#3b5cff", "#1690ff", "#1fc8f0"),
    "bot": ("#4a3fe0", "#2a5bf0", "#2f86f5"),
}


def chip_holes() -> str:
    """Die and pins, in face units: drawn black inside the top layer's mask."""
    parts = ['<rect x="-7" y="-7" width="14" height="14" rx="2.5"/>']
    for along in (-5.5, 0.0, 5.5):
        # One pin per side at each position: from the die (11) to near the face edge (16).
        parts.append(f'<rect x="{along - 1.25}" y="-16" width="2.5" height="5" rx="0.8"/>')
        parts.append(f'<rect x="{along - 1.25}" y="11" width="2.5" height="5" rx="0.8"/>')
        parts.append(f'<rect x="-16" y="{along - 1.25}" width="5" height="2.5" rx="0.8"/>')
        parts.append(f'<rect x="11" y="{along - 1.25}" width="5" height="2.5" rx="0.8"/>')
    return "".join(parts)


def mark(prefix: str = "gl") -> str:
    """The mark's SVG body for a 64 x 64 box. ``prefix`` keeps ids unique when it is inlined."""
    defs = []
    for name, (a, b, c) in GRADIENTS.items():
        defs.append(
            # Face units: from the left corner (-20, 20) to the right corner (20, -20), which
            # is left to right on screen after the projection.
            f'<linearGradient id="{prefix}-{name}" x1="-20" y1="20" x2="20" y2="-20" '
            f'gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="{a}"/>'
            f'<stop offset=".5" stop-color="{b}"/><stop offset="1" stop-color="{c}"/>'
            f"</linearGradient>"
        )
    box = 'maskUnits="userSpaceOnUse" x="0" y="0" width="64" height="64"'

    def above(y: float) -> str:
        # The layer above, grown by GAP, cut out of this one.
        return (
            f'<g transform="{ISO.format(x=CX, y=y)}" fill="#000" stroke="#000" '
            f'stroke-width="{GAP * 2}" stroke-linejoin="round">{FACE}</g>'
        )

    top, mid, bot = LAYER_Y
    white = '<rect width="64" height="64" fill="#fff"/>'
    defs.append(
        f'<mask id="{prefix}-m-top" {box}>{white}'
        f'<g transform="{ISO.format(x=CX, y=top)}" fill="#000">{chip_holes()}</g></mask>'
    )
    defs.append(f'<mask id="{prefix}-m-mid" {box}>{white}{above(top)}</mask>')
    defs.append(f'<mask id="{prefix}-m-bot" {box}>{white}{above(mid)}</mask>')

    def layer(name: str, y: float) -> str:
        return (
            f'<g mask="url(#{prefix}-m-{name})"><g transform="{ISO.format(x=CX, y=y)}" '
            f'fill="url(#{prefix}-{name})">{FACE}</g></g>'
        )

    body = layer("bot", bot) + layer("mid", mid) + layer("top", top)
    return f"<defs>{''.join(defs)}</defs>{body}"


def mark_svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" '
        f'aria-label="GPUlayer">{mark()}</svg>\n'
    )


def logo_svg() -> str:
    # Wordmark for the README: readable on GitHub's light and dark themes.
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 176 48" role="img" '
        'aria-label="GPUlayer">'
        f'<svg x="0" y="0" width="48" height="48" viewBox="0 0 64 64">{mark("wm")}</svg>'
        '<text x="56" y="33" font-family="Geist, Inter, -apple-system, \'Segoe UI\', '
        'Helvetica, Arial, sans-serif" font-size="26" letter-spacing="-0.5">'
        '<tspan font-weight="700" fill="#2f7cf6">GPU</tspan>'
        '<tspan font-weight="500" fill="#808080">layer</tspan></text></svg>\n'
    )


def main() -> None:
    (ASSETS / "logo-mark.svg").write_text(mark_svg())
    (ASSETS / "logo.svg").write_text(logo_svg())
    for dest in (ROOT / "site/src/app/icon.svg", ROOT / "src/gpulayer/api/ui/icon.svg"):
        shutil.copyfile(ASSETS / "logo-mark.svg", dest)
    print("docs/assets/logo-mark.svg, docs/assets/logo.svg, site and dashboard icons")


if __name__ == "__main__":
    main()
