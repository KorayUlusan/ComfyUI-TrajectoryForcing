#!/usr/bin/env python3
"""Generate the four PCA token mosaics for many ImageNet classes, as a tile bank.

Companion to `class_grid.py`, which banks the *decoded* levels. This banks the
*latent* view of the same trajectories.

Motivation. The teaser's PCA beats (SHOTS.md group D) all draw on a single
trajectory -- class 213, seed 592, from the 2026-09-04 gpu_smoke run -- because
that is the only PCA output that exists. A wall of 66 tiles built from 4 images
is 62 repeats, and it reads as one. Feedback on the shot previews was exactly
that: "show multiple latents not just one".

This run is asset generation, not an experiment, and its criteria are written
accordingly: they establish that the bank is REPRODUCIBLE and USABLE, not that
the method works. The scientific claims about the trajectory belong to
`class_grid.py`, which is where the coarse-to-fine criterion lives. Stating that
plainly here is deliberate -- a criterion invented to look like a hypothesis
would be the "criterion satisfied by the absence of a result" failure this
repo's README already records twice.

Why this is cheap. A PCA tile is a projection of the token grid onto three
axes; it skips the RAE decoder entirely, which is the expensive half. Job
451491 measured 0.04s generate + 0.05s decode per class, so a decode-free
1000-class sweep is bounded by the generate term alone.

Exit criteria, fixed before the run, each printed PASS/FAIL:

  1. control-decode -- class 213 / seed 592 decodes bit-for-bit to the
                       2026-09-04 gpu_smoke `01-original-*.png` (caption strip
                       cropped). This is the same control class_grid.py uses.
                       It costs one decode and it is what makes every tile in
                       the bank comparable to every tile in the decode bank: if
                       sampling has drifted, the two banks are of DIFFERENT
                       trajectories and pairing them in a shot is a lie.
                       If this fails the run is VOID.

  2. control-pca    -- the same trajectory's PCA tiles reproduce gpu_smoke's
                       `02-pca-*.png` bit-for-bit. Reproducing it requires
                       repeating that run's call exactly, INCLUDING its
                       `palette_from` (class 207, seed 593): a jointly-fitted
                       palette puts both trajectories on shared axes, so it
                       gives different colours than the per-trajectory fit the
                       bank itself uses. That is why this check re-runs the
                       reference call rather than comparing a bank tile.
                       If this fails the run is VOID.

  3. distinct       -- within a trajectory the 4 mosaics are distinct images,
                       and across classes the final-level mosaics are distinct.
                       A bank of near-duplicates does not fix the problem that
                       motivated the run.

  4. non-degenerate -- no level's token grid is constant, and no PCA tile is a
                       single flat colour. GUARD, not decoration: a collapsed
                       level is a legitimate result that must be RECORDED, and
                       a flat tile would sail through criterion 3 (four flat
                       colours are four distinct images) while being useless on
                       screen. Degenerate classes are listed and excluded from
                       the curated pick list rather than crashing the run.

  5. timing         -- per-class generate and PCA seconds, extrapolated. Read
                       the smoke arm's number before sizing anything larger.

Curation, reported but NOT a pass/fail claim: per level, the mean absolute
difference between 4-neighbour tokens in the latent, normalised by that level's
own scale. The expectation is that it rises with level -- a coarse level holds
few large regions, a fine level many small ones. It is reported rather than
asserted because the same number is maximised by noise, so on its own it cannot
distinguish "resolved detail" from "no structure at all"; it is useful for
ranking tiles, not for claiming the method worked.

Usage:
    ./slurm/submit.sh slurm/pca_bank.sbatch                  # 4-class smoke
    CLASSES=1000 ./slurm/submit.sh slurm/pca_bank.sbatch     # the full bank
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EXT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT_ROOT))

OUT = EXT_ROOT / "outputs" / "pca_bank"
REF = EXT_ROOT / "outputs" / "gpu_smoke"

# Pinned explicitly. One variable per arm: the bank varies CLASS, so the seed is
# fixed. These three must stay equal to gpu_smoke.py's, because criteria 1 and 2
# compare against that run's output.
SEED = 592
CONTROL_CLASS = 213   # Irish setter
PALETTE_CLASS = 207   # golden retriever -- gpu_smoke's `palette_from`
PALETTE_SEED = SEED + 1
CAPTION_ROWS_RGB = 22   # 01-original-*.png is 278x256; strip the baked caption
CAPTION_ROWS_PCA = 32   # 02-pca-*.png is 544x512

# Same spread as class_grid.py's smoke arm, so the two banks' smoke runs cover
# the same classes and can be eyeballed side by side.
SMOKE_CLASSES = [213, 207, 285, 980]

SIZE = 512  # 16x16 tokens at 32px each -- whole-pixel blocks, so `pixelated`
            # upscaling in the browser stays exact at any frame size.

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)
    return bool(ok)


def roughness(z: np.ndarray) -> float:
    """Mean 4-neighbour absolute difference in the token grid, scale-normalised.

    Normalising by the level's own std is what makes levels comparable: without
    it this measures how large the latents are, which is not the question.
    """
    a = np.asarray(z, dtype=np.float64)
    s = a.std()
    if s <= 0:
        return 0.0
    dv = np.abs(np.diff(a, axis=0)).mean()
    dh = np.abs(np.diff(a, axis=1)).mean()
    return float((dv + dh) / 2 / s)


def image_batch_to_uint8(tensor) -> list[np.ndarray]:
    return [(np.asarray(f) * 255).round().astype(np.uint8) for f in tensor.numpy()]


def main() -> int:
    from tf_nodes.locate import register_model_folder
    from tf_nodes.tf_import import configure_jax_env

    configure_jax_env()
    register_model_folder()

    from tf_nodes.nodes_pipeline import TFDecode, TFGenerate, TFLatentPreview, TFLoadPipeline
    from PIL import Image

    n_classes = int(os.environ.get("CLASSES", "0"))
    classes = SMOKE_CLASSES if n_classes <= 0 else (
        SMOKE_CLASSES + [c for c in range(1000) if c not in SMOKE_CLASSES]
    )[:max(n_classes, len(SMOKE_CLASSES))]
    print(f"seed={SEED}  classes={len(classes)}: {classes[:8]}"
          + (" ..." if len(classes) > 8 else ""), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipeline, info = TFLoadPipeline.execute(
        checkpoint="auto (download TF_L_edit)", config="edit_env_config.yml", warmup=True)
    load_s = time.perf_counter() - t0
    print(f"{info}\npipeline loaded in {load_s:.1f}s (paid once for the whole sweep)",
          flush=True)

    # --- 1 & 2. the control, before the sweep -------------------------------
    # Read the control before the arms: if sampling has drifted there is no
    # point spending the sweep, and no tile from it could be cited anyway.
    ctrl, = TFGenerate.execute(pipeline=pipeline, class_id=CONTROL_CLASS, seed=SEED)

    ref_rgb = [REF / f"01-original-{i}.png" for i in range(4)]
    ok, detail = False, "reference frames not found"
    if all(p.exists() for p in ref_rgb):
        images, _ = TFDecode.execute(
            pipeline=pipeline, levels=ctrl, which="all levels", level_override=-1,
            label_levels=False, sheet_layout="separate frames")
        got = image_batch_to_uint8(images)
        same = []
        for i, p in enumerate(ref_rgb):
            ref = np.asarray(Image.open(p).convert("RGB"))
            ref = ref[: ref.shape[0] - CAPTION_ROWS_RGB]
            same.append(ref.shape == got[i].shape and np.array_equal(ref, got[i]))
        ok = all(same)
        detail = (f"levels matching 2026-09-04 gpu_smoke: {sum(same)}/4"
                  + ("" if ok else "  -> RUN IS VOID"))
    check("control-decode", ok, detail)

    ref_pca = [REF / f"02-pca-{i}.png" for i in range(4)]
    ok, detail = False, "reference frames not found"
    if all(p.exists() for p in ref_pca):
        # Repeat gpu_smoke's call exactly, joint palette included -- see the
        # docstring. This is NOT how the bank itself is coloured.
        other, = TFGenerate.execute(
            pipeline=pipeline, class_id=PALETTE_CLASS, seed=PALETTE_SEED)
        previews, = TFLatentPreview.execute(
            pipeline=pipeline, levels=ctrl, which="all levels", level_override=-1,
            size=SIZE, label_levels=True, palette_from=other,
            sheet_layout="separate frames")
        got = image_batch_to_uint8(previews)
        same = []
        for i, p in enumerate(ref_pca):
            ref = np.asarray(Image.open(p).convert("RGB"))
            same.append(ref.shape == got[i].shape and np.array_equal(ref, got[i]))
        ok = all(same)
        detail = (f"mosaics matching 2026-09-04 gpu_smoke: {sum(same)}/4"
                  + ("" if ok else "  -> RUN IS VOID"))
    check("control-pca", ok, detail)

    # Abort before the sweep if either control is void. This is what lets the
    # sweep be submitted directly instead of as a separate job chained behind a
    # smoke arm: the two controls above exercise generate, decode and PCA --
    # every code path the loop uses -- and stopping here is the `set -e` the
    # repo's "smoke at the top of the sweep script" rule asks for. A queue wait
    # costs more than the minute of GPU this saves, so it is not worth two jobs.
    if any(not ok for _, ok, _ in _results):
        print("\nCONTROL FAILED -- run is void, sweep not attempted. "
              "No tile from this job can be cited.", flush=True)
        return 1

    # --- the sweep ----------------------------------------------------------
    gen_times: list[float] = []
    pca_times: list[float] = []
    finals: dict[int, bytes] = {}
    per_class_distinct = True
    degenerate: list[int] = []
    rough: dict[int, list[float]] = {}

    for ci in classes:
        t0 = time.perf_counter()
        levels, = TFGenerate.execute(pipeline=pipeline, class_id=ci, seed=SEED)
        gen_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        # palette=None -> fit per trajectory, jointly over that trajectory's own
        # four levels. Both halves of that matter. Per TRAJECTORY is what makes
        # a wall of many classes read as many different things rather than one
        # colour scheme. Jointly over its own LEVELS is what makes a
        # level-to-level cross-fade read as structure resolving rather than as a
        # colour flash, because the axes do not move under the transition.
        previews, = TFLatentPreview.execute(
            pipeline=pipeline, levels=levels, which="all levels", level_override=-1,
            size=SIZE, label_levels=False, sheet_layout="separate frames")
        pca_times.append(time.perf_counter() - t0)

        frames = image_batch_to_uint8(previews)
        for li, f in enumerate(frames):
            Image.fromarray(f).save(OUT / f"{ci:04d}-s{SEED}-L{li}.png")

        z = np.asarray(levels.latents, dtype=np.float32)
        rough[ci] = [roughness(z[li]) for li in range(z.shape[0])]

        # Criterion 4, evaluated per class so a collapse is recorded rather than
        # crashing the sweep 900 classes in.
        flat_latent = any(z[li].std() <= 0 for li in range(z.shape[0]))
        flat_tile = any(len(np.unique(f.reshape(-1, 3), axis=0)) <= 1 for f in frames)
        if flat_latent or flat_tile:
            degenerate.append(ci)

        seen = {f.tobytes() for f in frames}
        per_class_distinct &= len(seen) == 4
        finals[ci] = frames[-1].tobytes()

        if len(gen_times) <= 8 or len(gen_times) % 100 == 0:
            print(f"  class {ci:4d}: gen {gen_times[-1]:5.2f}s  pca {pca_times[-1]:5.2f}s  "
                  "roughness " + " ".join(f"{v:.3f}" for v in rough[ci]), flush=True)

    # --- 3. distinct --------------------------------------------------------
    check("distinct", per_class_distinct and len(set(finals.values())) == len(classes),
          f"{len(set(finals.values()))}/{len(classes)} classes give a distinct final "
          f"mosaic; all 4 levels distinct within a class: {per_class_distinct}")

    # --- 4. non-degenerate --------------------------------------------------
    check("non-degenerate", not degenerate,
          f"{len(degenerate)} of {len(classes)} classes collapsed"
          + (f": {degenerate[:20]}" if degenerate else ""))

    # --- curation signal, not a claim ---------------------------------------
    r = np.array([rough[c] for c in classes])          # [classes, levels]
    rising = int((r[:, -1] > r[:, 0]).sum())
    print("\ncuration (reported, not a pass/fail claim -- noise maximises this):")
    print("  mean 4-neighbour token difference by level: "
          + " ".join(f"L{i} {v:.3f}" for i, v in enumerate(r.mean(0))))
    print(f"  L3 > L0 for {rising}/{len(classes)} classes", flush=True)

    # Rank for the shot code: a tile is a good wall tile when its finest level
    # has structure to show. Deliberately NOT ranked by L3/L0 -- class_grid.py
    # learned that ratio selects for a blank L0, which makes an ugly grid.
    order = sorted((c for c in classes if c not in degenerate),
                   key=lambda c: rough[c][-1], reverse=True)
    (OUT / "curated.json").write_text(json.dumps({
        "seed": SEED,
        "size": SIZE,
        "degenerate": degenerate,
        "by_final_roughness": order,
    }, indent=1))

    # --- 5. timing ----------------------------------------------------------
    per = float(np.mean(gen_times) + np.mean(pca_times))
    print(f"\nper class: gen {np.mean(gen_times):.2f}s + pca {np.mean(pca_times):.2f}s "
          f"= {per:.2f}s   (n={len(classes)}, load {load_s:.0f}s paid once)")
    for n in (100, 1000):
        print(f"  extrapolated {n:4d} classes: {(load_s + n * per) / 60:6.1f} min")
    check("timing", per > 0, f"{per:.2f}s per class")

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} criteria passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    print(f"tiles -> {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
