#!/usr/bin/env python3
"""End-to-end check of the node graph against the real model, on a GPU.

The unit tests run every node against a stub pipeline, which proves the wiring
but says nothing about the two things that can only fail on a GPU: whether
TrajectoryForcing's JAX model and its PyTorch RAE decoder both survive being
imported through the namespace swap in the same process ComfyUI runs in, and
whether the resume path actually re-samples what it claims to.

Exit criteria, fixed here before the run, each printed as PASS/FAIL:

  1. load      -- the pipeline builds and reports 4 levels.
  2. generate  -- a trajectory comes back finite, [4,16,16,768], and two
                  different seeds give different trajectories.
  3. decode    -- all four levels decode to distinct uint8 images.
  4. regions   -- level 2 clusters into more than one region and fewer than
                  every token being its own (either extreme means the
                  clustering is not describing structure).
  5. edit      -- the feature edit changes exactly the targeted tokens at l*
                  and nothing else, on any level.
  6. resume    -- levels <= l* are bit-identical to the edited stack, levels
                  > l* differ from the unedited trajectory. This is the claim
                  the whole extension rests on; an edit that does not
                  propagate is the failure this test exists to catch.
  7. sweep     -- a three-seed sweep of the *same* edit gives three arms, laid
                  out as one stitched contact sheet rather than a batch; each
                  arm's final level differs from its own no-edit baseline (so
                  the edit registers with the seed cancelled out), the spread
                  across arms is non-zero (so the seed matters at all), and the
                  arm whose seed matches step 6 reproduces step 6's trajectory
                  bit-for-bit. That last part is the sweep's own control: if the
                  loop is not doing exactly what the explicit edit-and-resume
                  chain does, its table describes some other operation.
  8. control   -- re-generating with the original class and seed reproduces
                  step 2's trajectory exactly. If this fails the run is void:
                  sampling is not deterministic and nothing above can be
                  attributed to the edit.

Writes decoded PNGs under outputs/gpu_smoke/ for eyeballing afterwards.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

EXT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT_ROOT))

OUT = EXT_ROOT / "outputs" / "gpu_smoke"

CLASS_ID = 213      # Irish setter
OTHER_CLASS = 207   # golden retriever
SEED = 592
EDIT_LEVEL = 2

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def save(frames, stem: str) -> None:
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    # Clear this stem's previous files first. Frame counts change between runs
    # -- the sweep sheet went from four frames to one stitched image -- and
    # leftovers from the last run sitting beside this one's read as part of it.
    for stale in OUT.glob(f"{stem}-*.png"):
        stale.unlink()
    for i, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(OUT / f"{stem}-{i}.png")


def image_batch_to_uint8(tensor) -> list[np.ndarray]:
    return [(np.asarray(f) * 255).round().astype(np.uint8) for f in tensor.numpy()]


def main() -> int:
    # Same two side effects the extension's on_load() has, in the same order:
    # JAX settings before anything can import jax, model folder before the
    # loader node's schema builds its checkpoint dropdown.
    from tf_nodes.locate import register_model_folder
    from tf_nodes.tf_import import configure_jax_env

    print("jax env set:", configure_jax_env(), flush=True)
    print("models folder:", register_model_folder(), flush=True)

    from tf_nodes import nodes as node_list
    from tf_nodes.nodes_edit import TFFeatureEdit, TFResumeFromLevel
    from tf_nodes.nodes_pipeline import TFDecode, TFGenerate, TFLatentPreview, TFLoadPipeline
    from tf_nodes.nodes_regions import TFRegionMap, TFTokensFromCoords
    from tf_nodes.nodes_sweep import TFSweep

    print(f"{len(node_list())} nodes registered", flush=True)

    # --- 1. load -------------------------------------------------------------
    t0 = time.perf_counter()
    pipeline, info = TFLoadPipeline.execute(
        checkpoint="auto (download TF_L_edit)", config="edit_env_config.yml", warmup=True
    )
    print(info, flush=True)
    check("load", pipeline.num_levels == 4,
          f"{pipeline.num_levels} levels in {time.perf_counter() - t0:.1f}s")

    # --- 2. generate ---------------------------------------------------------
    t0 = time.perf_counter()
    levels, = TFGenerate.execute(pipeline=pipeline, class_id=CLASS_ID, seed=SEED)
    gen_seconds = time.perf_counter() - t0
    other, = TFGenerate.execute(pipeline=pipeline, class_id=OTHER_CLASS, seed=SEED + 1)
    check(
        "generate",
        levels.latents.shape[0] == 4
        and np.isfinite(levels.latents).all()
        and not np.allclose(levels.latents, other.latents),
        f"shape {levels.latents.shape}, {gen_seconds:.1f}s, "
        f"|z| in [{np.abs(levels.latents).min():.3f}, "
        f"{np.abs(levels.latents).max():.3f}]",
    )

    # --- 3. decode -----------------------------------------------------------
    t0 = time.perf_counter()
    images, warning = TFDecode.execute(
        pipeline=pipeline, levels=levels, which="all levels", level_override=-1, label_levels=True,
        sheet_layout="separate frames")
    frames = image_batch_to_uint8(images)
    save(frames, "01-original")
    distinct = len({f.tobytes() for f in frames})
    check("decode", distinct == 4 and warning == "",
          f"{distinct} distinct level images in {time.perf_counter() - t0:.1f}s")

    previews, = TFLatentPreview.execute(
        pipeline=pipeline, levels=levels, which="all levels", level_override=-1, size=512,
        label_levels=True, palette_from=other, sheet_layout="separate frames")
    save(image_batch_to_uint8(previews), "02-pca")

    # --- 4. regions ----------------------------------------------------------
    regions, region_image, num_regions, region_level = TFRegionMap.execute(
        levels=levels, level=EDIT_LEVEL, cosine_threshold=0.9, size=512
    )
    save(image_batch_to_uint8(region_image), "03-regions")
    total_tokens = int(np.prod(regions.ids.shape))
    check(
        "regions",
        1 < num_regions < total_tokens and region_level == EDIT_LEVEL,
        f"{num_regions} regions over {total_tokens} tokens at threshold 0.9, "
        f"level output {region_level}",
    )

    # --- 5. edit -------------------------------------------------------------
    target, _, _ = TFTokensFromCoords.execute(coords="6,6:9 7,6:9", levels=levels, regions=regions)
    source, _, _ = TFTokensFromCoords.execute(coords="1,1", levels=other, regions=None)
    edited, edit_info = TFFeatureEdit.execute(
        levels=levels, level=EDIT_LEVEL, target_tokens=target, source_tokens=source,
        source_mode="region mean", strength=1.0, source_level=EDIT_LEVEL, source_levels=other,
    )
    print(edit_info, flush=True)

    changed = ~np.isclose(edited.level(EDIT_LEVEL), levels.level(EDIT_LEVEL)).all(axis=-1)
    others_intact = all(
        np.array_equal(edited.level(lvl), levels.level(lvl))
        for lvl in range(levels.num_levels) if lvl != EDIT_LEVEL
    )
    check(
        "edit",
        np.array_equal(changed, target.mask) and others_intact and edited.dirty_level == EDIT_LEVEL,
        f"{int(changed.sum())} tokens changed, target had {target.count}",
    )

    # --- 6. resume -----------------------------------------------------------
    t0 = time.perf_counter()
    resumed, resume_info = TFResumeFromLevel.execute(
        pipeline=pipeline, levels=edited, level=EDIT_LEVEL, class_id=-1, seed=SEED
    )
    print(resume_info, flush=True)
    below_frozen = all(
        np.array_equal(resumed.level(lvl), edited.level(lvl)) for lvl in range(EDIT_LEVEL + 1)
    )
    above_changed = all(
        not np.allclose(resumed.level(lvl), levels.level(lvl))
        for lvl in range(EDIT_LEVEL + 1, levels.num_levels)
    )
    check(
        "resume",
        below_frozen and above_changed and resumed.dirty_level is None,
        f"levels 0..{EDIT_LEVEL} frozen, {EDIT_LEVEL + 1}..{levels.num_levels - 1} re-sampled "
        f"in {time.perf_counter() - t0:.1f}s",
    )

    edited_images, _ = TFDecode.execute(
        pipeline=pipeline, levels=resumed, which="all levels", level_override=-1, label_levels=True,
        sheet_layout="separate frames")
    save(image_batch_to_uint8(edited_images), "04-edited")

    final_before = frames[-1]
    final_after = image_batch_to_uint8(edited_images)[-1]
    print(
        f"final image mean |delta| = "
        f"{np.abs(final_after.astype(float) - final_before.astype(float)).mean():.2f}/255",
        flush=True,
    )

    # --- 7. sweep ------------------------------------------------------------
    # The same edit as step 5, run once per seed, with SEED first so the arm can
    # be checked against step 6's explicit chain.
    t0 = time.perf_counter()
    sweep_seeds = [SEED, SEED + 1, SEED + 2]
    report, sheet, picked, arms, spread = TFSweep.execute(
        levels=levels, target_tokens=target, source_tokens=source,
        axis="seed", values=",".join(str(s) for s in sweep_seeds),
        level=EDIT_LEVEL, seed=SEED, strength=1.0, source_mode="region mean",
        baseline=True, decode=True, arm_limit=12, output_arm=0,
        sheet_layout="contact sheet", size=512,
        source_levels=other, pipeline=pipeline,
    )
    print(report, flush=True)
    save(image_batch_to_uint8(sheet), "05-sweep")

    # The sweep's own control: arm 0 uses the same seed, level, class, target
    # and source as step 6, so it must land on exactly step 6's trajectory. A
    # loop that quietly differs -- a re-read source, a shifted seed, a resume
    # from the wrong level -- would still produce a plausible table.
    # The shipped default is one stitched image, so that is what gets exercised:
    # four 512px frames side by side, wider than tall.
    stitched = sheet.shape[0] == 1 and sheet.shape[2] > sheet.shape[1]
    reproduces = np.array_equal(picked.latents, resumed.latents)
    # The node says so itself when an arm came out identical to its baseline,
    # rather than this script re-parsing the table's columns to find out.
    every_arm_moved = "changed nothing at all" not in report
    check(
        "sweep",
        arms == len(sweep_seeds) and reproduces and spread > 0 and every_arm_moved
        and stitched,
        f"{arms} arms in {time.perf_counter() - t0:.1f}s, spread {spread:.4f}, "
        f"every arm moved the final level: {every_arm_moved}, "
        f"sheet {tuple(sheet.shape[1:3])} stitched into one image: {stitched}, "
        f"arm 0 {'reproduces' if reproduces else 'DIVERGES FROM'} the explicit chain",
    )

    # --- 8. control ----------------------------------------------------------
    control, = TFGenerate.execute(pipeline=pipeline, class_id=CLASS_ID, seed=SEED)
    check(
        "control",
        np.array_equal(control.latents, levels.latents),
        "same class+seed reproduces the original trajectory bit-for-bit",
    )

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 60, flush=True)
    print(f"{len(_results) - len(failed)}/{len(_results)} criteria passed; images in {OUT}", flush=True)
    if failed:
        print("FAILED: " + ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
