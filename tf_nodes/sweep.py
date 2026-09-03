"""Planning a sweep: which arms to run, and what varies between them.

Pure numpy and stdlib, no ComfyUI and no JAX, so the arm list -- the part that
is easy to get subtly wrong, and expensive to get wrong on a GPU -- is testable
on its own.

One variable per arm, everything else pinned: the axis supplies its own values
and the three fixed widgets supply the rest, so no arm can differ from another
in two ways at once. That is the repo's own experiment rule, enforced here
rather than left to whoever wires the graph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SEED = "seed"
LEVEL = "level (l*)"
STRENGTH = "strength"
AXES = [SEED, LEVEL, STRENGTH]

# A ceiling on what a single queued prompt can cost. Each arm is two resumes
# and a decode, so a fat-fingered "0-1000" would hold the GPU for an hour with
# no way to stop it short of killing the server.
ARM_CEILING = 64

_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


@dataclass(frozen=True)
class Arm:
    """One run of the edit. `value` is whatever the axis varies."""

    value: int | float
    level: int
    seed: int
    strength: float

    def label(self, axis: str) -> str:
        return f"{axis.split(' ')[0]} {self.value:g}"

    def describe(self) -> str:
        return f"level {self.level}, seed {self.seed}, strength {self.strength:.2f}"


def parse_values(text: str, axis: str) -> list[int | float]:
    """`"1,2,3"` or `"0-3"` into a list, deduplicated, order preserved.

    `a-b` is an inclusive range, matching the `row,col0:col1` shorthand in
    TF Tokens From Coords in spirit: the common case is consecutive, and typing
    `0,1,2,3` for it is the sort of friction that stops people sweeping at all.
    """
    pieces = [p for p in re.split(r"[,\s]+", (text or "").strip()) if p]
    if not pieces:
        raise ValueError(
            f"'values' is empty, so there is nothing to sweep. For axis {axis!r} try "
            f"{_example(axis)!r}."
        )
    out: list[int | float] = []
    for piece in pieces:
        found = _RANGE.match(piece)
        if found:
            lo, hi = int(found.group(1)), int(found.group(2))
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(_one(piece, axis))
    # Duplicates are a typo, not a request to pay for the same arm twice. The
    # report lists what actually ran, so the drop is visible rather than silent.
    seen, unique = set(), []
    for value in out:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _one(piece: str, axis: str) -> int | float:
    try:
        return float(piece) if axis == STRENGTH else int(piece)
    except ValueError:
        raise ValueError(
            f"Cannot read {piece!r} as a value for axis {axis!r}. Use numbers separated by "
            f"commas or spaces, or 'a-b' for a range -- e.g. {_example(axis)!r}."
        ) from None


def _example(axis: str) -> str:
    return {SEED: "1,2,3,4", LEVEL: "0-3", STRENGTH: "0.25,0.5,0.75,1.0"}[axis]


def _mismatch(problem: str, axis: str) -> str:
    """Say the likely cause, not just the symptom.

    `axis` and `values` are two widgets that have to agree, and changing the
    dropdown leaves the text field behind -- so the first failure most people
    meet is a seed list being read as strengths. "Strength 2.0 is outside 0..1"
    is true and tells them nothing about which of the two to fix.
    """
    return (
        f"{problem}. 'values' is being read as {axis} because that is what 'axis' is set to -- "
        f"if you changed the axis, 'values' needs changing too. For {axis!r} try "
        f"{_example(axis)!r}."
    )


def plan(
    axis: str,
    values: str,
    *,
    level: int,
    seed: int,
    strength: float,
    num_levels: int,
    limit: int,
) -> list[Arm]:
    """The arms to run: one per value, with the other two settings pinned."""
    if axis not in AXES:
        raise ValueError(f"Unknown sweep axis {axis!r}; expected one of {AXES}.")
    parsed = parse_values(values, axis)
    cap = max(1, min(int(limit), ARM_CEILING))
    if len(parsed) > cap:
        raise ValueError(
            f"{len(parsed)} arms requested but 'arm_limit' is {cap}. Each arm costs two "
            "re-samples and a decode, so this is a guard against a mistyped range rather "
            "than a hard limit -- raise 'arm_limit' (advanced) if you meant it."
        )

    # The pinned level is clamped, like every other level widget in the
    # extension. The *swept* level is not -- see below.
    pinned_level = min(max(int(level), 0), int(num_levels) - 1)
    arms = []
    for value in parsed:
        if axis == SEED:
            if value < 0:
                raise ValueError(_mismatch(f"Seed {value} is negative", axis))
            arms.append(Arm(value, pinned_level, int(value), float(strength)))
        elif axis == LEVEL:
            # Not clamped, unlike the level widgets elsewhere: clamping a sweep
            # axis turns "0-9" into four real arms and six silent duplicates of
            # level 3, and the table would not say so.
            if not 0 <= value < num_levels:
                raise ValueError(_mismatch(
                    f"Level {value} is outside this trajectory's 0..{num_levels - 1}", axis))
            arms.append(Arm(value, int(value), int(seed), float(strength)))
        else:
            if not 0.0 <= value <= 1.0:
                raise ValueError(_mismatch(f"Strength {value} is outside 0..1", axis))
            arms.append(Arm(value, pinned_level, int(seed), float(value)))
    return arms
