#!/usr/bin/env python3
"""Generate before/after EDIT pairs for many classes, as a tile bank.

Motivation. The teaser's before/after beat (`Proof`) has exactly one pair to
show -- class 213, from the 2026-09-04 gpu_smoke run -- because that is the only
class an edit has ever been run on. Review feedback on the assembled cuts was
that the Irish setter appears in six of eight beats, and this is one of the two
reasons why: the level and latent banks cover 1000 classes, but the EDIT bank
covered one. A wall can be varied by picking a different class; a before/after
cannot, because there is nothing to pick from.

What an edit is here, and why it needs a GPU. Editing a trajectory is not an
image operation: tokens at some level are replaced, and then every level ABOVE
that one is re-sampled from the edited state (`TFResumeFromLevel`). So each pair
costs a generate, an edit, a resume and two decodes. That is the whole point of
the node pack and it is why the "after" image differs from the "before" in a
structurally coherent way rather than looking like a patch pasted on.

Exit criteria, fixed before the run, each printed PASS/FAIL:

  1. control     -- class 213 / seed 592 reproduces the 2026-09-04 gpu_smoke
                    `01-original-*` decodes bit-for-bit. Same control as
                    class_grid.py and pca_bank.py, and voiding for the same
                    reason: if sampling has drifted, a pair from this run is not
                    comparable to any tile in the other two banks, and the film
                    puts them on screen together.

  2. edit-lands  -- for every class, the edited level differs from the original
                    at EXACTLY the targeted tokens, and every level below the
                    edit level is untouched. This is the criterion that would
                    catch the edit silently doing nothing, which is the failure
                    mode that produces a perfectly clean-looking before/after
                    pair showing no difference at all.

  3. edit-shows  -- the final decoded images actually differ, by more than
                    MIN_PIXEL_DELTA mean absolute difference. Criterion 2 can
                    pass in latent space while the decoder washes the change
                    out; a pair that is technically edited and visually
                    identical is useless to the film and worse than a failure,
                    because it looks like the method does nothing.
                    GUARD, not decoration: such classes are reported and
                    excluded from the curated list rather than crashing the run.

  4. timing      -- per-class seconds, extrapolated. Read the smoke arm first.

Usage:
    ./slurm/submit.sh slurm/edit_bank.sbatch                # 4-class smoke
    CLASSES=24 ./slurm/submit.sh slurm/edit_bank.sbatch     # the bank
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

OUT = EXT_ROOT / "outputs" / "edit_bank"
REF = EXT_ROOT / "outputs" / "gpu_smoke"

SEED = 592
CONTROL_CLASS = 213
EDIT_LEVEL = 2          # same level gpu_smoke edits at
CAPTION_ROWS = 22
COORDS = "6,6:9 7,6:9"  # the region gpu_smoke edits: a block near the centre
SOURCE_COORDS = "1,1"

# Mean absolute pixel difference, 0-255, between the before and after finals.
# Set from gpu_smoke's own pair, which is a visibly obvious edit; anything much
# below this is a change you have to hunt for on screen.
MIN_PIXEL_DELTA = 2.0

SMOKE_CLASSES = [213, 207, 285, 980]

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def image_batch_to_uint8(tensor) -> list[np.ndarray]:
    return [(np.asarray(f) * 255).round().astype(np.uint8) for f in tensor.numpy()]


def main() -> int:
    from tf_nodes.locate import register_model_folder
    from tf_nodes.tf_import import configure_jax_env

    configure_jax_env()
    register_model_folder()

    from tf_nodes.nodes_pipeline import TFDecode, TFGenerate, TFLoadPipeline
    from tf_nodes.nodes_regions import TFRegionMap, TFTokensFromCoords
    from tf_nodes.nodes_edit import TFFeatureEdit, TFResumeFromLevel
    from PIL import Image

    n_classes = int(os.environ.get("CLASSES", "0"))
    classes = SMOKE_CLASSES if n_classes <= 0 else (
        SMOKE_CLASSES + [c for c in range(1000) if c not in SMOKE_CLASSES]
    )[:max(n_classes, len(SMOKE_CLASSES))]
    print(f"seed={SEED}  edit level={EDIT_LEVEL}  classes={len(classes)}: {classes[:8]}"
          + (" ..." if len(classes) > 8 else ""), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipeline, info = TFLoadPipeline.execute(
        checkpoint="auto (download TF_L_edit)", config="edit_env_config.yml", warmup=True)
    load_s = time.perf_counter() - t0
    print(f"{info}\npipeline loaded in {load_s:.1f}s", flush=True)

    # --- 1. control, before anything else -----------------------------------
    ctrl, = TFGenerate.execute(pipeline=pipeline, class_id=CONTROL_CLASS, seed=SEED)
    ref_files = [REF / f"01-original-{i}.png" for i in range(4)]
    ok, detail = False, "reference frames not found"
    if all(p.exists() for p in ref_files):
        imgs, _ = TFDecode.execute(
            pipeline=pipeline, levels=ctrl, which="all levels", level_override=-1,
            label_levels=False, sheet_layout="separate frames")
        got = image_batch_to_uint8(imgs)
        same = []
        for i, p in enumerate(ref_files):
            ref = np.asarray(Image.open(p).convert("RGB"))
            ref = ref[: ref.shape[0] - CAPTION_ROWS]
            same.append(ref.shape == got[i].shape and np.array_equal(ref, got[i]))
        ok = all(same)
        detail = f"levels matching 2026-09-04 gpu_smoke: {sum(same)}/4" + ("" if ok else "  -> VOID")
    check("control", ok, detail)
    if not ok:
        print("\nCONTROL FAILED -- run is void, sweep not attempted.", flush=True)
        return 1

    # The "source" trajectory the edit borrows features FROM. Fixed across the
    # whole bank: one variable per arm, so what differs between pairs is the
    # class being edited, not also where the new feature came from.
    donor, = TFGenerate.execute(pipeline=pipeline, class_id=207, seed=SEED + 1)
    source, _ = TFTokensFromCoords.execute(coords=SOURCE_COORDS, levels=donor, regions=None)

    times: list[float] = []
    latent_ok: list[int] = []
    latent_bad: list[int] = []
    deltas: dict[int, float] = {}

    for ci in classes:
        t0 = time.perf_counter()
        levels, = TFGenerate.execute(pipeline=pipeline, class_id=ci, seed=SEED)
        regions, _region_img, _region_lvl = TFRegionMap.execute(
            levels=levels, level=EDIT_LEVEL, cosine_threshold=0.9, size=512)
        target, _ = TFTokensFromCoords.execute(
            coords=COORDS, levels=levels, regions=regions)
        edited, _ = TFFeatureEdit.execute(
            levels=levels, level=EDIT_LEVEL, target_tokens=target, source_tokens=source,
            source_mode="region mean", strength=1.0, source_level=EDIT_LEVEL,
            source_levels=donor)
        resumed, _ = TFResumeFromLevel.execute(
            pipeline=pipeline, levels=edited, level=EDIT_LEVEL, class_id=-1, seed=SEED)

        before, _ = TFDecode.execute(
            pipeline=pipeline, levels=levels, which="all levels", level_override=-1,
            label_levels=False, sheet_layout="separate frames")
        after, _ = TFDecode.execute(
            pipeline=pipeline, levels=resumed, which="all levels", level_override=-1,
            label_levels=False, sheet_layout="separate frames")
        times.append(time.perf_counter() - t0)

        bf = image_batch_to_uint8(before)
        af = image_batch_to_uint8(after)
        for li in range(4):
            Image.fromarray(bf[li]).save(OUT / f"{ci:04d}-before-L{li}.png")
            Image.fromarray(af[li]).save(OUT / f"{ci:04d}-after-L{li}.png")

        # --- criterion 2, per class -----------------------------------------
        changed = ~np.isclose(edited.level(EDIT_LEVEL), levels.level(EDIT_LEVEL)).all(axis=-1)
        below_intact = all(
            np.array_equal(edited.level(l), levels.level(l)) for l in range(EDIT_LEVEL)
        )
        (latent_ok if (np.array_equal(changed, target.mask) and below_intact)
         else latent_bad).append(ci)

        # --- criterion 3, per class -----------------------------------------
        d = float(np.abs(bf[-1].astype(np.float64) - af[-1].astype(np.float64)).mean())
        deltas[ci] = d
        print(f"  class {ci:4d}: {times[-1]:5.2f}s  pixel delta {d:6.2f}"
              f"  {'ok' if d >= MIN_PIXEL_DELTA else 'TOO SUBTLE'}", flush=True)

    check("edit-lands", not latent_bad,
          f"{len(latent_ok)}/{len(classes)} classes edited exactly the targeted tokens"
          + (f"; wrong: {latent_bad}" if latent_bad else ""))

    weak = sorted(c for c, d in deltas.items() if d < MIN_PIXEL_DELTA)
    check("edit-shows", not weak,
          f"pixel delta {min(deltas.values()):.2f}..{max(deltas.values()):.2f} "
          f"(threshold {MIN_PIXEL_DELTA})"
          + (f"; too subtle to use: {weak}" if weak else ""))

    usable = sorted((c for c in classes if c not in weak and c not in latent_bad),
                    key=lambda c: deltas[c], reverse=True)
    (OUT / "curated.json").write_text(json.dumps({
        "seed": SEED, "edit_level": EDIT_LEVEL, "coords": COORDS,
        "too_subtle": weak, "wrong_tokens": latent_bad,
        "by_pixel_delta": usable,
    }, indent=1))

    per = float(np.mean(times))
    print(f"\nper class: {per:.2f}s  (n={len(classes)}, load {load_s:.0f}s paid once)")
    for n in (24, 100):
        print(f"  extrapolated {n:3d} classes: {(load_s + n * per) / 60:5.1f} min")
    check("timing", per > 0, f"{per:.2f}s per class")

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} criteria passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    print(f"pairs -> {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
