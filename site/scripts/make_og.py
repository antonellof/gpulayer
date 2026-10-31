"""Draw site/public/og.png, the 1200 x 630 image that link previews show.

    python site/scripts/make_og.py [--font PATH]

Needs Pillow. Uses a sans-serif system font (SF Pro on macOS, DejaVu Sans on Linux) unless
--font is given. Rerun it when the tagline changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "public" / "og.png"
FONTS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
BG, BLUE, GREY, WHITE, MUTED = "#0d1117", "#3987e5", "#9aa4b2", "#f0f3f6", "#9198a1"
TITLE = "Train PyTorch on every GPU\nyou can reach."
LEDE = (
    "One Trainer, from your laptop to many machines, faster than\n"
    "Lightning. NVIDIA CUDA, Apple Metal and CPU. Open source."
)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font")
    a = ap.parse_args()
    path = a.font or next(p for p in FONTS if Path(p).exists())
    img = Image.new("RGB", (1200, 630), BG)
    d = ImageDraw.Draw(img)
    logo = font(path, 62)
    d.text((80, 92), "GPU", font=logo, fill=BLUE)
    d.text((80 + d.textlength("GPU", font=logo), 92), "layer", font=logo, fill=GREY)
    d.multiline_text((80, 210), TITLE, font=font(path, 58), fill=WHITE, spacing=14)
    d.multiline_text((80, 385), LEDE, font=font(path, 28), fill=MUTED, spacing=12)
    d.text((80, 515), "pip install gpulayer", font=font(path, 28), fill=BLUE)
    img.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
