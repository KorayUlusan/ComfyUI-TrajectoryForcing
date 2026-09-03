"""How far apart two canvases are, in the space the edits act on.

Split out of the compare node because the sweep needs the same measure. A
number that means "changed" in one node and something slightly different in
another is worse than having no number at all: two rows of a results table stop
being comparable and nothing says so.

Everything here is pure numpy, so the measure can be tested without a GPU.
"""
from __future__ import annotations

import numpy as np

# Per-token change is scale-free: cosine distance between the two feature
# vectors, which is the same measure TF Region Map clusters with, so "changed"
# here means the same thing as "a different region" there.
CHANGED = 0.02


def per_token_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance per token between two [H,W,C] canvases, in 0..2."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    b = np.nan_to_num(np.asarray(b, dtype=np.float32))
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    # A zero vector has no direction, so cosine is undefined; call it "changed"
    # if exactly one side is zero and "identical" if both are, rather than
    # letting a 0/0 become a NaN that quietly poisons the mean below.
    both_zero = (na < 1e-8) & (nb < 1e-8)
    one_zero = ((na < 1e-8) ^ (nb < 1e-8))
    cosine = dot / np.clip(na * nb, 1e-8, None)
    distance = 1.0 - np.clip(cosine, -1.0, 1.0)
    distance[both_zero] = 0.0
    distance[one_zero] = 1.0
    return distance


def changed_tokens(a: np.ndarray, b: np.ndarray) -> int:
    return int((per_token_distance(a, b) > CHANGED).sum())


def mean_pairwise_distance(canvases: list[np.ndarray]) -> float:
    """Mean per-token cosine distance over every pair of canvases.

    The number a sweep is actually asking for: how much the axis being swept
    moves the outcome at all. Near zero means every arm landed in the same
    place, and the axis is not doing what the run assumed it was.
    """
    if len(canvases) < 2:
        return 0.0
    totals = [
        float(per_token_distance(canvases[i], canvases[j]).mean())
        for i in range(len(canvases))
        for j in range(i + 1, len(canvases))
    ]
    return float(np.mean(totals))


def heat(distance: np.ndarray) -> np.ndarray:
    """Per-token distance as a black-to-amber tile, scaled to its own maximum.

    Self-scaled on purpose: the absolute numbers are in the report, and what the
    picture is for is *where* the change is, which a fixed scale hides whenever
    an edit is subtle.
    """
    peak = float(distance.max())
    norm = distance / peak if peak > 1e-8 else np.zeros_like(distance)
    tile = np.zeros((*distance.shape, 3), dtype=np.float32)
    tile[..., 0] = norm                              # red first
    tile[..., 1] = np.clip(norm * 1.6 - 0.6, 0, 1)   # then green, giving amber
    tile[..., 2] = np.clip(norm * 3.0 - 2.0, 0, 1)   # white only at the very top
    return (tile * 255).astype(np.uint8)
