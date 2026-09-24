#!/usr/bin/env python3
"""Generate a PNG with Russian text — handy for demos and tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
DEFAULT_TEXT = "Привет, мир!\nРаспознавание русского текста"


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError(
        "No Cyrillic-capable TTF font found. Install fonts-dejavu-core."
    )


def make_image(text: str, out_path: Path, font_size: int = 48, padding: int = 30) -> Path:
    """Render multi-line text onto a white background and save as PNG."""
    lines = text.split("\n")
    font = _load_font(font_size)

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_sizes = [measure.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max((bbox[2] - bbox[0]) for bbox in line_sizes)
    line_height = max((bbox[3] - bbox[1]) for bbox in line_sizes) + 20

    width = text_width + 2 * padding
    height = line_height * len(lines) + 2 * padding

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill="black", font=font)
        y += line_height

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Russian-text sample image.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to render (\\n for new lines).")
    parser.add_argument("--out", default="sample.png", help="Output PNG path.")
    parser.add_argument("--font-size", type=int, default=48)
    args = parser.parse_args()

    out = make_image(args.text.replace("\\n", "\n"), Path(args.out), args.font_size)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
