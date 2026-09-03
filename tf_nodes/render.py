"""Turning latents, region maps and token selections into pictures.

Everything here produces uint8 HWC arrays; `to_image` is the single place they
become the float32 [B,H,W,C] tensor ComfyUI calls IMAGE. The grid overlay and
the selection highlight match `_overlay_points` in TrajectoryForcing's
`scripts/gradio_generate_and_edit.py`, so a token picked here looks like the same
token picked in the editing env.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import RegionMap, TokenSelection

GRID_LINE = np.array([90, 90, 90], dtype=np.uint8)
SELECTION = np.array([255, 80, 80], dtype=np.uint8)
BOUNDARY = np.array([255, 214, 64], dtype=np.uint8)


def to_image(frames) -> torch.Tensor:
    """uint8 HWC array, or a list of them, -> ComfyUI IMAGE [B,H,W,C] float32 in 0..1."""
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    stack = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)
    return torch.from_numpy(stack.astype(np.float32) / 255.0)


def from_mask(mask: torch.Tensor) -> np.ndarray:
    """ComfyUI MASK [B,H,W] (or [H,W]) -> a single 2D float array in 0..1."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a MASK of shape [H,W] or [B,H,W], got {arr.shape}")
    return arr


def upscale(tile: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour block upscale, so token boundaries stay hard edges."""
    factor = max(1, int(factor))
    return np.repeat(np.repeat(np.asarray(tile, dtype=np.uint8), factor, axis=0), factor, axis=1)


def fit_to_grid(image: np.ndarray, grid: tuple[int, int], target_px: int) -> np.ndarray:
    """Resize so every token gets a whole number of pixels and the result is ~target_px wide.

    Both PCA tiles (16x16) and decoded RGB (256x256) end up on the same canvas
    size, which is what lets one Painter mask serve either view.
    """
    arr = np.asarray(image, dtype=np.uint8)
    gh, gw = int(grid[0]), int(grid[1])
    cell = max(1, round(int(target_px) / max(gh, gw)))
    out_h, out_w = gh * cell, gw * cell
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr
    from PIL import Image

    resample = Image.NEAREST if arr.shape[0] <= gh else Image.LANCZOS
    return np.asarray(Image.fromarray(arr).resize((out_w, out_h), resample), dtype=np.uint8)


def _cells(image: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel boundaries of each token row / column."""
    gh, gw = int(grid[0]), int(grid[1])
    return (
        np.linspace(0, image.shape[0], gh + 1).round().astype(int),
        np.linspace(0, image.shape[1], gw + 1).round().astype(int),
    )


def draw_grid(image: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    out = np.array(image, dtype=np.uint8, copy=True)
    rows, cols = _cells(out, grid)
    for y in rows:
        out[min(y, out.shape[0] - 1)] = GRID_LINE
    for x in cols:
        out[:, min(x, out.shape[1] - 1)] = GRID_LINE
    return out


TICK_EVERY = 4
TICK_INK = (226, 232, 240)
TICK_SHADOW = (12, 18, 30)


def draw_ticks(image: np.ndarray, grid: tuple[int, int], every: int = TICK_EVERY) -> np.ndarray:
    """Number the token rows and columns along the top and left edges.

    Coordinates for the edit nodes are typed as `row,col` on this grid, and
    without labels the only way to find one is to count cells by eye. Every
    fourth line keeps it readable at 16 wide without becoming a ruler.
    """
    from PIL import Image, ImageDraw, ImageFont

    arr = np.asarray(image, dtype=np.uint8)
    gh, gw = int(grid[0]), int(grid[1])
    cell = arr.shape[0] / max(1, gh)
    size = max(9, min(15, int(cell * 0.42)))
    canvas = Image.fromarray(np.array(arr, copy=True))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        font = ImageFont.load_default(size)

    def label(text: str, x: int, y: int) -> None:
        # Drawn twice: the grids underneath range from near-white to near-black,
        # so a single colour is illegible on one or the other.
        for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            draw.text((x + dx, y + dy), text, fill=TICK_SHADOW, font=font)
        draw.text((x, y), text, fill=TICK_INK, font=font)

    rows, cols = _cells(arr, grid)
    for c in range(0, gw, every):
        label(str(c), cols[c] + 3, 2)
    for r in range(0, gh, every):
        if r:  # the 0,0 corner already carries the column label
            label(str(r), 3, rows[r] + 2)
    return np.asarray(canvas, dtype=np.uint8)


def draw_selection(image: np.ndarray, selection: TokenSelection) -> np.ndarray:
    """Outline and tint the selected tokens."""
    out = np.array(image, dtype=np.uint8, copy=True)
    rows, cols = _cells(out, selection.mask.shape)
    for r, c in np.argwhere(selection.mask):
        y0, y1 = rows[r], rows[r + 1]
        x0, x1 = cols[c], cols[c + 1]
        cell = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (0.72 * cell + 0.28 * SELECTION.astype(np.float32)).astype(np.uint8)
        border = max(1, (y1 - y0) // 12)
        out[y0:min(y1, y0 + border), x0:x1] = SELECTION
        out[max(y0, y1 - border):y1, x0:x1] = SELECTION
        out[y0:y1, x0:min(x1, x0 + border)] = SELECTION
        out[y0:y1, max(x0, x1 - border):x1] = SELECTION
    return out


def draw_region_boundaries(image: np.ndarray, regions: RegionMap) -> np.ndarray:
    """Draw the edges between differently-labelled neighbouring tokens."""
    out = np.array(image, dtype=np.uint8, copy=True)
    ids = regions.ids
    rows, cols = _cells(out, ids.shape)
    thickness = max(1, (rows[1] - rows[0]) // 10)
    for r in range(ids.shape[0]):
        for c in range(ids.shape[1]):
            if c + 1 < ids.shape[1] and ids[r, c] != ids[r, c + 1]:
                x = cols[c + 1]
                out[rows[r]:rows[r + 1], max(0, x - thickness):min(out.shape[1], x + thickness)] = BOUNDARY
            if r + 1 < ids.shape[0] and ids[r, c] != ids[r + 1, c]:
                y = rows[r + 1]
                out[max(0, y - thickness):min(out.shape[0], y + thickness), cols[c]:cols[c + 1]] = BOUNDARY
    return out


def region_colours(count: int) -> np.ndarray:
    """`count` visually distinct RGB colours, stable across runs.

    Hues walk the circle by the golden ratio so adjacent region ids -- which are
    often adjacent in space too -- never get near-identical colours.
    """
    import colorsys

    out = np.zeros((max(1, count), 3), dtype=np.uint8)
    for i in range(max(1, count)):
        h = (i * 0.61803398875) % 1.0
        s = 0.45 + 0.25 * ((i % 3) / 2.0)
        v = 0.70 + 0.25 * (i % 2)
        out[i] = np.array(colorsys.hsv_to_rgb(h, s, min(v, 1.0)), dtype=np.float32) * 255
    return out


def render_regions(regions: RegionMap, target_px: int) -> np.ndarray:
    """Flat-colour picture of the region map, one colour per region id."""
    colours = region_colours(regions.num_regions)
    tile = colours[np.clip(regions.ids, 0, len(colours) - 1)]
    return fit_to_grid(tile.astype(np.uint8), regions.ids.shape, target_px)


def caption(image: np.ndarray, text: str) -> np.ndarray:
    """Append a label strip below the image rather than drawing over it.

    Level previews arrive as a batch and ComfyUI's gallery shows no per-frame
    label, so without this a stack of four levels is four unlabelled squares.
    """
    from PIL import Image, ImageDraw, ImageFont

    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    strip = max(22, w // 16)
    out = np.empty((h + strip, w, 3), dtype=np.uint8)
    out[:h] = arr
    out[h:] = np.array([17, 26, 43], dtype=np.uint8)
    canvas = Image.fromarray(out)
    draw = ImageDraw.Draw(canvas)
    size = max(12, strip - 9)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default(size)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (max(2, (w - (box[2] - box[0])) // 2), h + max(1, (strip - (box[3] - box[1])) // 2 - 2)),
        text,
        fill=(203, 213, 225),
        font=font,
    )
    return np.asarray(canvas, dtype=np.uint8)


SHEET_GAP = 8
SHEET_BG = np.array([17, 26, 43], dtype=np.uint8)   # the caption strip's colour
# Past this many frames a single row stops being readable: twelve 384px arms in
# a row is 4600x408, an aspect ratio no screen and no page wants.
SHEET_MAX_ROW = 6


def contact_sheet(frames, max_row: int = SHEET_MAX_ROW) -> np.ndarray:
    """Lay equal-sized frames out as one image, in reading order.

    A batch is the right shape for saving one file per frame, and the wrong one
    for comparing: a sweep's arms mean nothing apart, and five frames sharing a
    320px node body compare nothing. So they get stitched -- a row while a row
    stays readable, a near-square grid past that.
    """
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    if not frames:
        raise ValueError("A contact sheet needs at least one frame.")
    if len(frames) == 1:
        return frames[0]
    h, w = frames[0].shape[:2]
    if any(f.shape[:2] != (h, w) for f in frames):
        raise ValueError(
            "Contact-sheet frames must all be the same size; got "
            f"{sorted({f.shape[:2] for f in frames})}."
        )
    cols = len(frames) if len(frames) <= max_row else int(np.ceil(np.sqrt(len(frames))))
    rows = int(np.ceil(len(frames) / cols))
    out = np.empty(
        (rows * h + (rows - 1) * SHEET_GAP, cols * w + (cols - 1) * SHEET_GAP, 3),
        dtype=np.uint8,
    )
    # Painted first, so a trailing part-row reads as empty space rather than
    # whatever numpy had lying around.
    out[:] = SHEET_BG
    for i, frame in enumerate(frames):
        row, col = divmod(i, cols)
        y, x = row * (h + SHEET_GAP), col * (w + SHEET_GAP)
        out[y:y + h, x:x + w] = frame
    return out


# ComfyUI draws a multi-image output **one frame at a time**, with a small
# "1/4" pager button in the corner:
#
#     if (!(l > 1)) return;
#     let C = (t.imageIndex ?? 0) + 1;
#     if (drawButton(f - 40, p + n - 40, 30, `${C}/${l}`)) { ... }
#
# So a node that returns four levels as a batch shows *level 0* and a control
# most people never spot. That is not a preview of a trajectory, it is a preview
# of its first pass -- and it was reported as exactly that ("I get level 0, it
# should give me all levels"). Stitching is therefore the default everywhere a
# node emits several frames meant to be seen together.
CONTACT_SHEET = "contact sheet"
SEPARATE_FRAMES = "separate frames"
SHEET_LAYOUTS = [CONTACT_SHEET, SEPARATE_FRAMES]


def lay_out(frames, layout: str = CONTACT_SHEET):
    """Frames as one stitched image, or as the batch ComfyUI pages through.

    The batch is still worth having: it is the only shape `SaveImage` can write
    as one file per frame.
    """
    frames = list(frames)
    if layout == SEPARATE_FRAMES or len(frames) < 2:
        return to_image(frames)
    return to_image(contact_sheet(frames))


LEVEL_NAMES = {0: "object/bg", 1: "parts", 2: "subparts"}


def level_caption(index: int, num_levels: int) -> str:
    name = "fine" if index == num_levels - 1 else LEVEL_NAMES.get(index)
    return f"Level {index} ({name})" if name else f"Level {index}"
