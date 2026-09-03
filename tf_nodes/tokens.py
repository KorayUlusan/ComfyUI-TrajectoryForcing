"""Region clustering and token selection -- the CPU half of the editing math.

Pure numpy, no ComfyUI and no JAX, so the edit semantics can be tested without a
GPU. `region_ids` is a port of `_region_id_map_from_local_cosine` from
TrajectoryForcing's `scripts/gradio_generate_and_edit.py` (which cannot be
imported here: that module pulls in gradio at import time). Keep the two in step
if the upstream clustering changes.
"""
from __future__ import annotations

import re

import numpy as np

from .data import RegionMap, TokenSelection


def region_ids(canvas: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of tokens whose 4-neighbour cosine similarity clears `threshold`.

    Port of scripts/gradio_generate_and_edit.py::_region_id_map_from_local_cosine.
    """
    arr = np.asarray(canvas, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a latent level [H,W,C], got {arr.shape}")
    h, w, c = arr.shape
    flat = np.nan_to_num(arr.reshape(h * w, c), nan=0.0, posinf=0.0, neginf=0.0)
    unit = (flat / np.clip(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8, None)).reshape(h, w, c)
    right = np.sum(unit[:, :-1, :] * unit[:, 1:, :], axis=-1) >= float(threshold)
    down = np.sum(unit[:-1, :, :] * unit[1:, :, :], axis=-1) >= float(threshold)

    parent = np.arange(h * w, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for r in range(h):
        for c0 in range(w - 1):
            if right[r, c0]:
                union(r * w + c0, r * w + c0 + 1)
    for r in range(h - 1):
        for c0 in range(w):
            if down[r, c0]:
                union(r * w + c0, (r + 1) * w + c0)

    roots = np.array([find(i) for i in range(h * w)], dtype=np.int32)
    _, inv = np.unique(roots, return_inverse=True)
    return inv.reshape(h, w).astype(np.int32)


def build_region_map(canvas: np.ndarray, level: int, threshold: float) -> RegionMap:
    return RegionMap(ids=region_ids(canvas, threshold), level=int(level), threshold=float(threshold))


# ---------------------------------------------------------------------------
# building selections
# ---------------------------------------------------------------------------
def mask_to_tokens(mask_hw: np.ndarray, grid: tuple[int, int], coverage: float) -> TokenSelection:
    """Downsample a painted mask to the token grid.

    A token is selected when at least `coverage` of the pixels covering it are
    painted, so a brush stroke that clips a token's corner does not silently
    replace that token's whole feature vector.
    """
    mask = np.asarray(mask_hw, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.shape}")
    gh, gw = int(grid[0]), int(grid[1])
    h, w = mask.shape
    if h < gh or w < gw:
        raise ValueError(f"Mask {h}x{w} is smaller than the {gh}x{gw} token grid.")
    # Average over each token's pixel footprint. Edges are split by index rather
    # than reshaped, so a mask whose size is not a multiple of the grid still works.
    rows = np.linspace(0, h, gh + 1).round().astype(int)
    cols = np.linspace(0, w, gw + 1).round().astype(int)
    out = np.zeros((gh, gw), dtype=bool)
    for r in range(gh):
        for c in range(gw):
            cell = mask[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
            out[r, c] = cell.size > 0 and float(cell.mean()) >= float(coverage)
    return TokenSelection(mask=out, note=f"painted mask >= {coverage:.2f} coverage")


_COORD = re.compile(r"(\d+)\s*,\s*(\d+)(?:\s*:\s*(\d+))?")


def parse_coords(text: str, grid: tuple[int, int]) -> TokenSelection:
    """Parse `"3,4  7,2  1,10:12"` into a selection.

    `r,c0:c1` is an inclusive column run on row r -- the same shorthand the
    TrajectoryForcing gradio app accepts, so notes taken there paste in here.
    """
    gh, gw = int(grid[0]), int(grid[1])
    out = np.zeros((gh, gw), dtype=bool)
    found = 0
    for row, col, col_end in _COORD.findall(text or ""):
        r, c0 = int(row), int(col)
        c1 = int(col_end) if col_end else c0
        if r < 0 or r >= gh:
            raise ValueError(f"Row {r} is outside the {gh}x{gw} token grid.")
        lo, hi = min(c0, c1), max(c0, c1)
        if lo < 0 or hi >= gw:
            raise ValueError(f"Columns {lo}..{hi} are outside the {gh}x{gw} token grid.")
        out[r, lo:hi + 1] = True
        found += 1
    if found == 0 and (text or "").strip():
        raise ValueError(
            f"No coordinates found in {text!r}. Use 'row,col' pairs, e.g. '3,4 7,2 1,10:12'."
        )
    return TokenSelection(mask=out, note=f"coords: {text.strip()}" if text else "empty")


def format_coords(mask: np.ndarray) -> str:
    """The inverse of `parse_coords`: a selection back to `"7,6:9 8,7"`.

    Runs are collapsed, because that is the notation someone would have typed
    and it is what ends up pasted into a writeup -- `7,6:9` rather than four
    separate pairs.

    This is the reference implementation for `web/tf_token_grid.js`, which has
    to produce byte-identical output: the clickable grid and the text field are
    two views of one value, and a round trip through the grid must not rewrite
    what the user typed into something merely equivalent.
    """
    mask = np.asarray(mask, dtype=bool)
    parts: list[str] = []
    for row in range(mask.shape[0]):
        col = 0
        while col < mask.shape[1]:
            if not mask[row, col]:
                col += 1
                continue
            start = col
            while col + 1 < mask.shape[1] and mask[row, col + 1]:
                col += 1
            parts.append(f"{row},{start}" if col == start else f"{row},{start}:{col}")
            col += 1
    return " ".join(parts)


def snap_to_regions(selection: TokenSelection, regions: RegionMap, min_overlap: float) -> TokenSelection:
    """Grow a selection to whole regions.

    Any region with at least `min_overlap` of its tokens selected is taken
    entirely; the rest are dropped. This is what turns a rough brush stroke into
    the paper's region-level edit, where R_tgt is a semantic part rather than an
    arbitrary set of tokens.
    """
    if selection.mask.shape != regions.ids.shape:
        raise ValueError(
            f"Selection {selection.mask.shape} does not match the region map {regions.ids.shape}."
        )
    out = np.zeros_like(selection.mask)
    kept = []
    for rid in np.unique(regions.ids[selection.mask]):
        member = regions.ids == rid
        if float(selection.mask[member].mean()) >= float(min_overlap):
            out |= member
            kept.append(int(rid))
    return TokenSelection(
        mask=out,
        note=f"snapped to regions {kept}" if kept else "snapped to regions (none cleared the overlap)",
        level=regions.level,
    )


def _combined_level(a: TokenSelection, b: TokenSelection) -> int | None:
    """The level a combination belongs to, or an error if the inputs disagree.

    Combining a selection snapped to level 1's regions with one snapped to
    level 2's produces something that describes neither.
    """
    if a.level is not None and b.level is not None and a.level != b.level:
        raise ValueError(
            f"Cannot combine a selection from level {a.level}'s regions with one from "
            f"level {b.level}'s -- point both region maps at the same level."
        )
    return a.level if a.level is not None else b.level


def combine(a: TokenSelection, b: TokenSelection | None, op: str) -> TokenSelection:
    if op == "invert":
        return TokenSelection(mask=~a.mask, note=f"invert({a.note})", level=a.level)
    if b is None:
        raise ValueError(f"Operation {op!r} needs a second selection.")
    if a.mask.shape != b.mask.shape:
        raise ValueError(f"Cannot combine {a.mask.shape} with {b.mask.shape}.")
    level = _combined_level(a, b)
    if op == "union":
        mask = a.mask | b.mask
    elif op == "intersection":
        mask = a.mask & b.mask
    elif op == "difference":
        mask = a.mask & ~b.mask
    elif op == "symmetric difference":
        mask = a.mask ^ b.mask
    else:
        raise ValueError(f"Unknown combine op {op!r}.")
    return TokenSelection(mask=mask, note=f"{op}({a.note}, {b.note})", level=level)


# ---------------------------------------------------------------------------
# the one edit primitive both edit nodes reduce to
# ---------------------------------------------------------------------------
def source_feature(canvas: np.ndarray, selection: TokenSelection, mode: str) -> np.ndarray:
    """The feature vector(s) written into the target tokens.

    `mean` is the paper's f_src: one vector, the mean over the source region.
    `cycle` reproduces the editing env's token-for-token copy, which preserves
    within-region variation when the two selections are the same shape.
    """
    picked = canvas[selection.mask]  # [n, C]
    if picked.shape[0] == 0:
        raise ValueError("Source selection is empty.")
    if mode == "region mean":
        return picked.mean(axis=0, keepdims=True)
    if mode == "token cycle":
        return picked
    raise ValueError(f"Unknown source mode {mode!r}.")


def write_feature(
    canvas: np.ndarray,
    target: TokenSelection,
    feature: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """z_i <- (1-s) z_i + s f_src for every selected token i.

    `feature` is broadcast over the target tokens, cycling if it is shorter --
    one source vector fills the whole target region, n source vectors repeat.
    """
    target.require_nonempty("Target selection")
    out = np.array(canvas, dtype=np.float32, copy=True)
    idx = np.argwhere(target.mask)
    feature = np.asarray(feature, dtype=np.float32).reshape(-1, out.shape[-1])
    s = float(np.clip(strength, 0.0, 1.0))
    for j, (r, c) in enumerate(idx):
        src = feature[j % feature.shape[0]]
        out[r, c] = (1.0 - s) * out[r, c] + s * src
    return out
