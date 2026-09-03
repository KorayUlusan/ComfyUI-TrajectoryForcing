"""The three payloads that travel along this extension's own sockets.

`TF_LEVELS` is the trajectory itself, `TF_REGIONS` the cosine clustering of one
of its levels, `TF_TOKENS` a selection on the token grid. None of them is a
ComfyUI tensor type: the token grid is 16x16 and lives in DINOv2 space, so
squeezing it through `LATENT` or `MASK` would either lose the level axis or
pretend a token index is a pixel.

Every payload is immutable by convention -- nodes return new objects rather than
editing in place, because ComfyUI hands the same object to every downstream node
and caches it for re-execution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class LevelStack:
    """A full coarse-to-fine trajectory: `latents[l]` is the canvas at level l."""

    latents: np.ndarray  # [L, H, W, C] float32
    class_id: int
    seed: int
    history: tuple[str, ...] = ()
    # Level whose canvas was edited without the finer levels being re-sampled
    # yet. TF Resume From Level clears it; TF Decode warns when it is set,
    # because levels above it still show the pre-edit trajectory.
    dirty_level: int | None = None
    # The pipeline that produced this trajectory, so downstream nodes do not
    # each need their own wire back to TF Load Pipeline -- six of them in one
    # example workflow, crossing the whole graph. A trajectory always comes from
    # a pipeline, except when TF Load Levels restores one from disk, which is
    # why every consumer still accepts an explicit pipeline as an override.
    pipeline: object | None = None

    def __post_init__(self):
        arr = np.asarray(self.latents, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"TF_LEVELS expects [L,H,W,C], got {arr.shape}")
        object.__setattr__(self, "latents", arr)

    @property
    def num_levels(self) -> int:
        return int(self.latents.shape[0])

    @property
    def grid(self) -> tuple[int, int]:
        return int(self.latents.shape[1]), int(self.latents.shape[2])

    def level(self, index: int) -> np.ndarray:
        """Canvas at `index`, clamped into range -- level widgets are plain ints."""
        return self.latents[self.clamp(index)]

    def clamp(self, index: int) -> int:
        return int(np.clip(int(index), 0, self.num_levels - 1))

    def with_level(self, index: int, canvas: np.ndarray, note: str) -> LevelStack:
        index = self.clamp(index)
        latents = self.latents.copy()
        latents[index] = np.asarray(canvas, dtype=np.float32)
        return replace(
            self,
            latents=latents,
            history=self.history + (note,),
            dirty_level=index,
        )

    def describe(self) -> str:
        lines = [
            f"levels: {self.num_levels}   grid: {self.grid[0]}x{self.grid[1]}"
            f"   channels: {int(self.latents.shape[3])}",
            f"class_id: {self.class_id}   seed: {self.seed}",
        ]
        if self.dirty_level is not None:
            lines.append(
                f"EDITED at level {self.dirty_level}, not yet resumed -- levels "
                f"{self.dirty_level + 1}..{self.num_levels - 1} are stale."
            )
        lines.append("history:")
        lines.extend(f"  {i + 1}. {h}" for i, h in enumerate(self.history))
        return "\n".join(lines)


@dataclass(frozen=True)
class RegionMap:
    """Connected cosine-similarity clusters of one level's tokens."""

    ids: np.ndarray  # [H, W] int32, 0..num_regions-1
    level: int
    threshold: float

    def __post_init__(self):
        object.__setattr__(self, "ids", np.asarray(self.ids, dtype=np.int32))

    @property
    def num_regions(self) -> int:
        return int(self.ids.max()) + 1 if self.ids.size else 0

    def region_of(self, row: int, col: int) -> int:
        return int(self.ids[int(row), int(col)])

    def mask_for(self, region_ids) -> np.ndarray:
        wanted = np.asarray(list(region_ids), dtype=np.int32)
        return np.isin(self.ids, wanted)


@dataclass(frozen=True)
class TokenSelection:
    """A boolean selection over the token grid."""

    mask: np.ndarray  # [H, W] bool
    note: str = ""
    # Which level's regions this was snapped to, when it was snapped to any.
    # Levels share a token grid, so a selection built against level 2's regions
    # applies cleanly to level 1 as far as shapes go -- and means something
    # entirely different, because the regions are not the same. Recording it is
    # what lets the edit nodes catch that.
    level: int | None = None

    def __post_init__(self):
        arr = np.asarray(self.mask, dtype=bool)
        if arr.ndim != 2:
            raise ValueError(f"TF_TOKENS expects [H,W], got {arr.shape}")
        object.__setattr__(self, "mask", arr)

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    @property
    def coords(self) -> list[tuple[int, int]]:
        return [(int(r), int(c)) for r, c in np.argwhere(self.mask)]

    def require_nonempty(self, what: str) -> None:
        if self.count == 0:
            raise ValueError(f"{what} is empty -- select at least one token.")

    def check_grid(self, grid: tuple[int, int], what: str) -> None:
        if self.mask.shape != grid:
            raise ValueError(
                f"{what} is {self.mask.shape[0]}x{self.mask.shape[1]} but the latent "
                f"token grid is {grid[0]}x{grid[1]}."
            )

    def check_level(self, level: int, what: str) -> None:
        """Refuse a selection that was snapped to a different level's regions.

        The shapes always match -- every level uses the same token grid -- so
        nothing else catches this, and the result looks plausible while being
        the wrong region entirely.
        """
        if self.level is not None and int(self.level) != int(level):
            raise ValueError(
                f"{what} was built from level {self.level}'s regions but is being applied "
                f"at level {level}. Point TF Region Map at level {level}, or wire its "
                f"'level' output into this node so the two cannot disagree."
            )
