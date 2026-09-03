#!/usr/bin/env python3
"""Build the figures the docs use, from a real run's output.

Reads what `slurm/gpu_smoke.sbatch` left in outputs/gpu_smoke/ and assembles
contact sheets into docs/img/. Committed because they are documentation, but
regenerable, so a change in what the model does is one command away from being
visible in the README rather than quietly contradicted by it.

    sbatch slurm/gpu_smoke.sbatch     # produces the frames
    python scripts/make_doc_images.py # assembles them
"""
from __future__ import annotations

import sys
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parent.parent
SRC = EXT_ROOT / "outputs" / "gpu_smoke"
OUT = EXT_ROOT / "docs" / "img"

BG = (17, 26, 43)
GAP = 8
MAX_WIDTH = 1200      # wide enough to read on GitHub, small enough to commit


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def _fitted_font(draw, text: str, width: int, start: int = 19):
    """Largest font from `start` down that fits `text` into `width`.

    Measured rather than assumed: a caption silently clipped at the right edge
    is worse than a smaller one, and which captions fit changes with the frame
    sizes a run happens to produce.
    """
    for size in range(start, 9, -1):
        font = _font(size)
        if draw.textlength(text, font=font) <= width - 12:
            return font
    return _font(10)


def sheet(name: str, stems: list[str], label: str = "",
          captions: list[str] | None = None) -> Path | None:
    from PIL import Image, ImageDraw

    frames = [SRC / f"{s}.png" for s in stems]
    missing = [f.name for f in frames if not f.is_file()]
    if missing:
        print(f"skip {name}: missing {missing}", flush=True)
        return None

    images = [Image.open(f).convert("RGB") for f in frames]
    height = max(i.height for i in images)
    width = sum(i.width for i in images) + GAP * (len(images) - 1)
    header = (32 if label else 0) + (30 if captions else 0)

    canvas = Image.new("RGB", (width, height + header), BG)
    draw = ImageDraw.Draw(canvas)
    if label:
        draw.text((6, 6), label, fill=(226, 232, 240), font=_fitted_font(draw, label, width))

    x = 0
    for i, image in enumerate(images):
        if captions:
            text = captions[i]
            font = _fitted_font(draw, text, image.width, start=20)
            offset = max(0, (image.width - int(draw.textlength(text, font=font))) // 2)
            draw.text((x + offset, (32 if label else 0) + 4), text,
                      fill=(148, 197, 255), font=font)
        canvas.paste(image, (x, header))
        x += image.width + GAP

    if canvas.width > MAX_WIDTH:
        canvas = canvas.resize(
            (MAX_WIDTH, round(canvas.height * MAX_WIDTH / canvas.width)), Image.LANCZOS
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    canvas.save(path, optimize=True)
    print(f"wrote {path.relative_to(EXT_ROOT)}  {canvas.width}x{canvas.height}  "
          f"{path.stat().st_size // 1024} KiB", flush=True)
    return path


def main() -> int:
    if not SRC.is_dir():
        print(f"No run output at {SRC}. Run slurm/gpu_smoke.sbatch first.", file=sys.stderr)
        return 1

    made = [
        sheet("trajectory", [f"01-original-{i}" for i in range(4)],
              "One trajectory, decoded at every level"),
        sheet("latents", [f"02-pca-{i}" for i in range(4)],
              "The same levels as raw token grids (PCA false colour)"),
        sheet("edit", ["01-original-3", "04-edited-3"],
              "A feature edit at level 2, with level 3 re-sampled from it",
              captions=["before", "after"]),
        sheet("regions", ["03-regions-0"],
              "Cosine regions at level 2"),
    ]
    return 0 if all(made) else 1


if __name__ == "__main__":
    sys.exit(main())
