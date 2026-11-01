"""Generate docs/assets/server-{light,dark}.svg: a GPUlayer server and the machines around it.

    python scripts/make_server_diagram.py

The machine running `gpulayer up` holds the control plane and a worker; other machines join
with `gpulayer join`, and your laptop submits. Every arrow points the way the connection is
opened (all towards the server), and ranks of a distributed job connect to each other directly.
Same look as scripts/make_concepts_diagram.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_concepts_diagram import FONT, OUT, THEMES, cloud_vm, laptop, server, text  # noqa: E402

W, H = 980, 490


def card(x, y, w, h, t, *, fill="card", stroke="line", rx=12, dash=False):
    d = ' stroke-dasharray="5 5"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{t[fill]}" '
        f'stroke="{t[stroke]}" stroke-width="1.5"{d}/>'
    )


def arrow(x1, y1, x2, y2, t, *, color="accent", dash=False, both=False):
    """A straight connection; the head marks the side that receives it."""
    d = ' stroke-dasharray="4 4"' if dash else ""
    start = f' marker-start="url(#head-{color})"' if both else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t[color]}" stroke-width="2"{d}'
        f' marker-end="url(#head-{color})"{start}/>'
    )


def svg(t: dict[str, str]) -> str:
    heads = "".join(
        f'<marker id="head-{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" '
        f'fill="{t[c]}"/></marker>'
        for c in ("accent", "rank")
    )
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'role="img" aria-label="A GPUlayer server: your laptop and joined machines connect to the '
        'machine running gpulayer up over HTTPS with a pinned certificate">',
        f"<defs>{heads}</defs>",
        f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
    ]
    # --- your laptop (left)
    p.append(card(30, 150, 200, 170, t))
    p.append(laptop(82, 178, t))
    p.append(text(130, 270, "Your laptop", t, size=16, weight=600, anchor="middle"))
    p.append(text(130, 292, "Trainer, launch(), CLI", t, size=12, color="muted", anchor="middle"))

    # --- the server machine (middle)
    p.append(card(330, 40, 330, 390, t, fill="bg", stroke="machine", rx=16))
    p.append(text(495, 70, "Machine running gpulayer up", t, size=15, weight=600, anchor="middle"))
    p.append(text(495, 90, "port 8470, TLS", t, size=12, color="muted", anchor="middle"))
    p.append(card(355, 110, 280, 120, t, fill="accent_soft", stroke="accent"))
    p.append(text(495, 145, "Control plane", t, size=16, weight=600, anchor="middle"))
    for i, line in enumerate(["queue and scheduler", "tokens, artifacts, logs"]):
        p.append(text(495, 170 + i * 20, line, t, size=12.5, color="muted", anchor="middle"))
    p.append(card(355, 290, 280, 115, t))
    p.append(server(375, 318, t, w=80, h=60))
    p.append(text(550, 342, "Worker", t, size=15, weight=600, anchor="middle"))
    p.append(text(550, 364, "this machine's GPUs", t, size=12, color="muted", anchor="middle"))
    p.append(arrow(495, 290, 495, 232, t))

    # --- joined machines (right)
    p.append(card(760, 60, 190, 150, t))
    p.append(server(807, 82, t))
    p.append(text(855, 172, "GPU server", t, size=15, weight=600, anchor="middle"))
    p.append(text(855, 192, "gpulayer join", t, size=12, color="muted", anchor="middle"))
    p.append(card(760, 265, 190, 150, t))
    p.append(cloud_vm(807, 292, t))
    p.append(text(855, 378, "Cloud VM", t, size=15, weight=600, anchor="middle"))
    p.append(text(855, 398, "gpulayer join", t, size=12, color="muted", anchor="middle"))

    # --- connections: all opened towards the server
    p.append(arrow(230, 200, 353, 170, t))
    p.append(arrow(760, 135, 637, 155, t))
    p.append(arrow(760, 340, 637, 205, t))
    # labels beside the lines, not on them
    p.append(text(280, 160, "HTTPS,", t, size=11.5, color="muted", anchor="middle"))
    p.append(text(280, 175, "pinned cert", t, size=11.5, color="muted", anchor="middle"))
    p.append(text(700, 118, "HTTPS,", t, size=11.5, color="muted", anchor="middle"))
    p.append(text(700, 133, "pinned cert", t, size=11.5, color="muted", anchor="middle"))
    # ranks of one distributed job talk directly
    p.append(arrow(905, 212, 905, 263, t, color="rank", dash=True, both=True))
    p.append(text(918, 242, "ranks", t, size=11.5, color="muted"))

    for i, note in enumerate(
        [
            "Every machine opens its own connection to the server, so workers can sit behind NAT.",
            "Ranks of one distributed job also connect to each other directly.",
        ]
    ):
        p.append(
            f'<text x="{W / 2}" y="{H - 30 + i * 18}" font-family="{FONT}" font-size="12.5" '
            f'fill="{t["muted"]}" text-anchor="middle">{note}</text>'
        )
    p.append("</svg>")
    return "".join(p)


def main() -> None:
    for name, t in THEMES.items():
        path = OUT / f"server-{name}.svg"
        path.write_text(svg(t))
        print(path)


if __name__ == "__main__":
    main()
