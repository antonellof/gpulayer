# ruff: noqa: E501  (drawing coordinates read better on one line)
"""Generate the architecture diagrams in docs/assets, light and dark:

- distributed-*.svg: one distributed job, from submission to outputs (docs/architecture/distributed.md)
- placement-*.svg: how the scheduler picks machines (docs/architecture/scheduling.md)
- lifecycle-*.svg: the states of a job (docs/get-started/concepts.md)

    python scripts/make_architecture_diagrams.py

Same look as scripts/make_concepts_diagram.py; every step drawn is one the docs describe.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_concepts_diagram import OUT, THEMES, laptop, proc, server, text  # noqa: E402

EXTRA = {
    "light": {"ok": "#1a7f37", "ok_soft": "#e6f4ea", "bad": "#cf222e", "bad_soft": "#fdecec"},
    "dark": {"ok": "#3fb950", "ok_soft": "#12261a", "bad": "#f85149", "bad_soft": "#2d1416"},
}


def theme(name: str) -> dict[str, str]:
    return {**THEMES[name], **EXTRA[name]}


def head(t: dict[str, str], w: int, h: int, label: str) -> list[str]:
    marks = "".join(
        f'<marker id="h-{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" '
        f'fill="{t[c]}"/></marker>'
        for c in ("accent", "rank", "muted", "ok", "bad")
    )
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" role="img" aria-label="{label}">',
        f"<defs>{marks}</defs>",
        f'<rect width="{w}" height="{h}" fill="{t["bg"]}"/>',
    ]


def box(x, y, w, h, t, fill="card", stroke="line", rx=12, dash=False):
    d = ' stroke-dasharray="5 5"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{t[fill]}" '
        f'stroke="{t[stroke]}" stroke-width="1.5"{d}/>'
    )


def line(x1, y1, x2, y2, t, color="accent", dash=False, both=False, curve=0):
    d = ' stroke-dasharray="4 4"' if dash else ""
    start = f' marker-start="url(#h-{color})"' if both else ""
    if curve:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + curve
        path = f"M{x1} {y1} Q{mx} {my} {x2} {y2}"
    else:
        path = f"M{x1} {y1} L{x2} {y2}"
    return (
        f'<path d="{path}" fill="none" stroke="{t[color]}" stroke-width="2"{d}'
        f' marker-end="url(#h-{color})"{start}/>'
    )


def step(x, y, n, t):
    """A numbered circle, matching the numbered steps in the docs."""
    return f'<circle cx="{x}" cy="{y}" r="11" fill="{t["accent"]}"/>' + text(
        x, y + 4.5, str(n), t, size=12, color="bg", weight=700, anchor="middle"
    )


def note(x, y, lines, t, size=12, anchor="start", color="muted"):
    return "".join(
        text(x, y + i * (size + 5), s, t, size=size, color=color, anchor=anchor)
        for i, s in enumerate(lines)
    )


# --------------------------------------------------------------------- distributed


def distributed(t: dict[str, str]) -> str:
    W, H = 980, 640
    p = head(t, W, H, "One distributed job: the control plane places a group of workers, node 0 "
             "publishes the rendezvous address, each node starts its ranks, ranks talk directly, "
             "logs and outputs come back per node")  # fmt: skip
    # submitter
    p.append(box(30, 40, 210, 150, t))
    p.append(laptop(87, 62, t))
    p.append(
        text(135, 150, "Trainer(num_nodes=2,", t, size=12, color="fg", anchor="middle", mono=True)
    )
    p.append(text(135, 168, "devices=2)", t, size=12, color="fg", anchor="middle", mono=True))
    # control plane
    p.append(box(380, 40, 300, 150, t, fill="accent_soft", stroke="accent"))
    p.append(text(530, 80, "Control plane", t, size=16, weight=600, anchor="middle"))
    p.append(note(530, 108, ["queue, scheduler, logs, artifacts"], t, anchor="middle"))
    p.append(line(240, 115, 378, 115, t))
    p.append(step(309, 100, 1, t))
    # workers
    for k, x in enumerate((210, 560)):
        p.append(box(x, 300, 300, 160, t, fill="bg", stroke="machine", rx=14))
        p.append(server(x + 18, 320, t, w=70, h=50))
        p.append(text(x + 105, 340, f"Node {k}", t, size=15, weight=600))
        p.append(
            text(
                x + 105,
                360,
                "worker" + (", hosts rank 0" if k == 0 else ""),
                t,
                size=12,
                color="muted",
            )
        )
        for j in range(2):
            p.append(proc(x + 22 + j * 140, 392, f"rank {2 * k + j}", t, rank=True, w=118, h=30))
        p.append(
            text(
                x + 150,
                446,
                "RANK, LOCAL_RANK, WORLD_SIZE=4",
                t,
                size=10.5,
                color="muted",
                anchor="middle",
                mono=True,
            )
        )
    p.append(line(470, 192, 380, 298, t))
    p.append(step(425, 245, 2, t))
    p.append(line(600, 192, 690, 298, t))
    p.append(step(645, 245, 3, t))
    p.append(line(512, 407, 578, 407, t, color="rank", dash=True, both=True))
    p.append(step(545, 385, 4, t))
    p.append(line(862, 300, 683, 150, t, color="muted", dash=True, curve=-30))
    p.append(step(805, 205, 5, t))
    # legend
    steps = [
        "The Trainer submits one DistributedJob: 2 nodes x 2 processes.",
        "The control plane places both workers at once (same torch, close together); node 0 picks a port and publishes address:port.",
        "Node 1 gets its assignment after that; each node starts its ranks with torchrun's environment.",
        "Ranks talk directly: NCCL on GPUs, gloo on CPUs, rendezvous at MASTER_ADDR on node 0.",
        "Logs come back tagged by node and outputs as node0/ and node1/; the first failing node stops the job.",
    ]
    for i, s in enumerate(steps):
        y = 500 + i * 26
        p.append(step(46, y - 4, i + 1, t))
        p.append(text(66, y, s, t, size=12.5))
    p.append("</svg>")
    return "".join(p)


# ----------------------------------------------------------------------- placement


def placement(t: dict[str, str]) -> str:
    W, H = 980, 470
    p = head(t, W, H, "How the scheduler places a job: compatible workers, then available ones, "
             "then the closest group, then the best-connected GPUs")  # fmt: skip
    cols = [
        ("Job asks for", 30),
        ("Compatible", 270),
        ("Available now", 500),
        ("Closest group", 730),
    ]
    for title, x in cols:
        p.append(text(x + 105, 40, title, t, size=15, weight=600, anchor="middle"))
    # job card
    p.append(box(30, 60, 210, 300, t, fill="accent_soft", stroke="accent"))
    for i, s in enumerate(["accelerator: cuda", "devices: 2 per node", "num_nodes: 2", "GPU memory >= 16 GB",
                           "torch, the job's packages", "group_locality: prefer"]):  # fmt: skip
        p.append(text(46, 96 + i * 40, s, t, size=12.5, color="fg", mono=True))
    workers = [
        ("a100-1", "zone b", True, True, True),
        ("a100-2", "zone b", True, True, True),
        ("t4-1", "zone c", True, False, False),
        ("mac", "home", False, False, False),
        ("a100-3", "zone c", True, True, False),
    ]
    reasons = {
        "mac": "no CUDA device",
        "t4-1": "busy: no free slot",
        "a100-3": "farther: other zone",
    }
    for i, (name, zone, compat, avail, chosen) in enumerate(workers):
        y = 64 + i * 58
        for c, (_, x) in enumerate(cols[1:], start=1):
            keep = {1: compat, 2: compat and avail, 3: chosen}[c]
            dropped_here = {
                1: not compat,
                2: compat and not avail,
                3: compat and avail and not chosen,
            }[c]
            if not keep and not dropped_here:
                continue
            fill, stroke = (
                ("ok_soft", "ok")
                if keep and c == 3
                else (("card", "line") if keep else ("bad_soft", "bad"))
            )
            p.append(box(x, y, 210, 46, t, fill=fill, stroke=stroke, rx=10))
            p.append(text(x + 14, y + 20, name, t, size=13, weight=600))
            p.append(text(x + 14, y + 37, zone, t, size=11, color="muted"))
            if dropped_here:
                p.append(
                    text(x + 196, y + 29, reasons[name], t, size=11, color="bad", anchor="end")
                )
    for x in (234, 464, 694):  # between the column titles, clear of the cards
        p.append(line(x, 35, x + 36, 35, t, color="muted"))
    p.append(note(30, 390, [
        "Every rejected worker gets its reason (gpulayer job status shows them). A job waits for busy workers",
        "and fails at once only when no worker could ever run it. Inside each chosen machine the 2 processes",
        "get the best-connected GPU pair (NVLink before PCIe), so DDP's all-reduce takes the fastest path.",
    ], t, size=12.5))  # fmt: skip
    p.append("</svg>")
    return "".join(p)


# ----------------------------------------------------------------------- lifecycle


def lifecycle(t: dict[str, str]) -> str:
    W, H = 980, 380
    p = head(t, W, H, "Job states: queued, provisioning, scheduled, running, then succeeded, "
             "failed, timed out or cancelled; a lost worker sends the job back to the queue")  # fmt: skip

    def state(x, y, label, kind="card", w=128):
        fill, stroke = {"card": ("card", "machine"), "ok": ("ok_soft", "ok"),
                        "bad": ("bad_soft", "bad"), "accent": ("accent_soft", "accent")}[kind]  # fmt: skip
        return box(x, y, w, 40, t, fill=fill, stroke=stroke, rx=20) + text(
            x + w / 2, y + 25, label, t, size=13, weight=600, anchor="middle", mono=True
        )

    # active states, in one dashed region (any of them can be cancelled)
    p.append(box(14, 30, 774, 228, t, fill="bg", stroke="line", rx=18, dash=True))
    p.append(state(30, 150, "queued", "accent"))
    p.append(state(190, 50, "provisioning"))
    p.append(state(370, 150, "scheduled"))
    p.append(state(610, 150, "running", "accent"))
    p.append(line(158, 170, 368, 170, t))
    p.append(text(263, 162, "placed on worker(s)", t, size=11.5, color="muted", anchor="middle"))
    p.append(line(120, 150, 200, 92, t, color="muted"))
    p.append(text(96, 106, "needs cloud", t, size=11.5, color="muted", anchor="middle"))
    p.append(text(96, 121, "capacity", t, size=11.5, color="muted", anchor="middle"))
    p.append(line(300, 92, 395, 150, t, color="muted"))
    p.append(text(372, 106, "a VM joined", t, size=11.5, color="muted"))
    p.append(line(498, 170, 608, 170, t))
    p.append(text(553, 162, "started", t, size=11.5, color="muted", anchor="middle"))
    # retry: running back to queued, below the row
    p.append(line(660, 191, 94, 191, t, color="muted", dash=True, curve=60))
    p.append(text(377, 242, "worker lost: back to the queue when retries are allowed", t,
                  size=11.5, color="muted", anchor="middle"))  # fmt: skip
    # outcomes
    p.append(state(830, 60, "succeeded", "ok"))
    p.append(state(830, 150, "failed", "bad"))
    p.append(state(830, 240, "timed_out", "bad"))
    p.append(line(738, 160, 828, 84, t, color="ok"))
    p.append(line(738, 170, 828, 170, t, color="bad"))
    p.append(line(738, 180, 828, 256, t, color="bad"))
    # unschedulable and cancelled
    p.append(line(60, 192, 60, 306, t, color="bad"))
    p.append(text(72, 324, "failed: no worker could ever run it", t, size=11.5, color="bad"))
    p.append(state(420, 312, "cancelled", "bad"))
    p.append(line(484, 260, 484, 310, t, color="bad"))
    p.append(
        text(496, 290, "gpulayer job cancel, from any state above", t, size=11.5, color="muted")
    )
    p.append("</svg>")
    return "".join(p)


def main() -> None:
    for name in THEMES:
        t = theme(name)
        for stem, fn in (
            ("distributed", distributed),
            ("placement", placement),
            ("lifecycle", lifecycle),
        ):
            path = OUT / f"{stem}-{name}.svg"
            path.write_text(fn(t))
            print(path)


if __name__ == "__main__":
    main()
