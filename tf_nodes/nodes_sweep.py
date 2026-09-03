"""Running one edit many times, varying exactly one thing.

The reason to prefer a graph over the editing env's Gradio app is that the same
edit can be run across a seed list, or across l*, and tabulated. Doing that by
hand means duplicating the edit-and-resume chain once per arm, re-typing the
coordinates each time, and then comparing four previews by eye -- at which point
the tool is a slower demo rather than a faster experiment.

This node is the whole chain (feature edit -> resume -> measure) run once per
value, with everything else pinned. Two things make it an experiment tool rather
than a batch button:

* **Every arm is measured against its own baseline** -- the same trajectory
  resumed from the same level with the same seed and *no* edit. Without that,
  "level 3 changed a lot" cannot be told apart from "a different resume seed
  changed a lot", and a seed sweep measures mostly the seed.
* **The spread across arms is reported.** Whether the axis moves the outcome at
  all is the first thing a table wants to say, and it is the one number no
  amount of staring at a contact sheet gives you.

TF Feature Edit and TF Resume From Level keep their full surface; this trades
some of it (a fixed source level, no shape edit) for the loop.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from comfy_api.latest import io

from . import measure, render, sweep, tokens
from .nodes_edit import SOURCE_MODES
from .sockets import (
    CATEGORY_EDIT,
    TFLevelsSocket,
    TFTokensSocket,
    level_input,
    node_preview,
    pipeline_input,
    resolve_pipeline,
    seed_input,
)


def _progress(total: int):
    """ComfyUI's progress bar when there is one, else nothing.

    A sweep is the one node here that runs long enough for a still queue to look
    like a hang. Imported lazily and guarded because `comfy.utils` is the
    server's, not the API's, and the unit tests run without it.
    """
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(total)
    except Exception:  # pragma: no cover - depends on how ComfyUI was started
        return None


class TFSweep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFSweep",
            search_aliases=["sweep", "batch", "seeds", "grid search", "ablation", "arms"],
            display_name="TF Sweep Edit",
            category=CATEGORY_EDIT,
            has_intermediate_output=True,
            description=(
                "Run one feature edit once per value of a single axis -- a seed list, the levels, "
                "or a strength ramp -- and tabulate the result.\n\n"
                "Each arm is compared against the same trajectory resumed with the same seed and "
                "no edit, so the number is the edit's effect rather than the seed's. The report "
                "also gives the spread across arms: how much the axis moves the outcome at all.\n\n"
                "Costs two re-samples and one decode per arm, so four arms is a few seconds and "
                "forty is a coffee. Everything not on the axis is pinned to the widget below it."
            ),
            inputs=[
                TFLevelsSocket.Input(
                    "levels",
                    tooltip="The unedited trajectory. Every arm starts from this one, so the "
                            "arms differ only in the swept value.",
                ),
                TFTokensSocket.Input("target_tokens", tooltip="Tokens that receive the new feature."),
                TFTokensSocket.Input("source_tokens", tooltip="Tokens the feature is taken from."),
                io.Combo.Input(
                    "axis", options=sweep.AXES,
                    tooltip="The one thing that varies between arms. Everything else is pinned "
                            "to its widget below.",
                ),
                io.String.Input(
                    "values", default="1,2,3,4", multiline=False,
                    placeholder="1,2,3,4      or      0-3",
                    tooltip="Values for the axis: numbers separated by commas or spaces, with "
                            "'a-b' for an inclusive range. Duplicates are dropped.",
                ),
                level_input(tooltip="l*, the level being edited. Pinned for every arm unless "
                                    "'level (l*)' is the axis, in which case 'values' supplies it."),
                seed_input(),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Pinned for every arm unless 'strength' is the axis. 1.0 replaces "
                            "outright, which is the paper's edit.",
                ),
                io.Combo.Input(
                    "source_mode", options=SOURCE_MODES, advanced=True,
                    tooltip="As TF Feature Edit. The source feature is read from the arm's own "
                            "level, so a level sweep re-reads it at each one.",
                ),
                io.Boolean.Input(
                    "baseline", default=True, advanced=True,
                    tooltip="Resume each arm a second time without the edit and measure against "
                            "that. Off, arms are measured against the input trajectory instead, "
                            "which folds the resume seed into every number -- cheaper, and only "
                            "honest for a strength sweep.",
                ),
                io.Boolean.Input(
                    "decode", default=True, advanced=True,
                    tooltip="Decode each arm's final level to RGB for the contact sheet. Off, the "
                            "sheet uses PCA tiles instead, which costs nothing; the table is "
                            "unaffected either way.",
                ),
                io.Int.Input(
                    "arm_limit", default=12, min=1, max=sweep.ARM_CEILING, advanced=True,
                    tooltip="Refuse to start if 'values' expands past this. A guard against a "
                            "mistyped range holding the GPU, not a real ceiling.",
                ),
                io.Int.Input(
                    "output_arm", default=0, min=0, max=sweep.ARM_CEILING - 1, advanced=True,
                    tooltip="Which arm leaves on the 'levels' output, counting from 0, for "
                            "feeding TF Save Levels or TF Compare Levels.",
                ),
                io.Int.Input("size", default=384, min=128, max=1024, step=64, advanced=True),
                TFLevelsSocket.Input(
                    "source_levels", optional=True,
                    tooltip="Take the source feature from a different trajectory. Unwired means "
                            "the same one.",
                ),
                pipeline_input(),
            ],
            outputs=[
                io.String.Output("report"),
                io.Image.Output(
                    "sheet",
                    tooltip="Contact sheet: the no-edit baseline first, then one frame per arm.",
                ),
                TFLevelsSocket.Output(
                    "levels", tooltip="The arm named by 'output_arm', already resumed.",
                ),
                io.Int.Output("arms"),
                io.Float.Output(
                    "spread",
                    tooltip="Mean pairwise cosine distance between arms at the final level. Near "
                            "zero means the axis did not change the outcome.",
                ),
            ],
        )

    @classmethod
    def execute(cls, levels, target_tokens, source_tokens, axis, values, level, seed,
                strength, source_mode, baseline, decode, arm_limit, output_arm, size,
                source_levels=None, pipeline=None) -> io.NodeOutput:
        pipe = resolve_pipeline(pipeline, levels, "TF Sweep Edit")
        arms = sweep.plan(
            axis, values, level=level, seed=seed, strength=strength,
            num_levels=levels.num_levels, limit=arm_limit,
        )

        source_stack = source_levels if source_levels is not None else levels
        target_tokens.check_grid(levels.grid, "target_tokens")
        target_tokens.require_nonempty("target_tokens")
        source_tokens.check_grid(source_stack.grid, "source_tokens")
        source_tokens.require_nonempty("source_tokens")
        # A selection snapped to one level's regions describes that level only.
        # Sweeping l* violates that by construction -- holding the token *set*
        # fixed is what "the same edit at every level" has to mean -- so the
        # check is replaced by a line in the report rather than dropped.
        caveat = ""
        if axis == sweep.LEVEL:
            caveat = _level_caveat(target_tokens, source_tokens)
        else:
            target_tokens.check_level(levels.clamp(level), "target_tokens")
            source_tokens.check_level(source_stack.clamp(level), "source_tokens")

        bar = _progress(len(arms) * (2 if baseline else 1))
        baselines: dict[tuple[int, int], np.ndarray] = {}
        results, rows, finals = [], [], []
        for arm in arms:
            feature = tokens.source_feature(
                source_stack.level(arm.level), source_tokens, source_mode)
            canvas = tokens.write_feature(
                levels.level(arm.level), target_tokens, feature, arm.strength)
            edited = levels.with_level(arm.level, canvas, f"sweep arm: {arm.describe()}")
            after = pipe.resume(edited.latents, arm.level, levels.class_id, arm.seed)
            if bar:
                bar.update(1)

            key = (arm.level, arm.seed)
            if baseline:
                if key not in baselines:
                    baselines[key] = pipe.resume(
                        levels.latents, arm.level, levels.class_id, arm.seed)
                    if bar:
                        bar.update(1)
                reference = baselines[key]
            else:
                reference = levels.latents

            distance = measure.per_token_distance(after[-1], reference[-1])
            rows.append((
                arm,
                int((distance > measure.CHANGED).sum()),
                float(distance.mean()),
                float(distance.max()),
            ))
            finals.append(after[-1])
            results.append(after)

        spread = measure.mean_pairwise_distance(finals)
        picked = min(int(output_arm), len(arms) - 1)
        report = _report(
            axis, arms, rows, spread, levels, target_tokens, source_tokens,
            source_levels, source_mode, picked, int(output_arm), baseline, caveat,
        )
        sheet = _sheet(
            pipe, levels, arms, results, baselines.get((arms[0].level, arms[0].seed)),
            axis, decode, size,
        )
        out = replace(
            levels,
            latents=np.asarray(results[picked], dtype=np.float32),
            seed=arms[picked].seed,
            history=levels.history + (
                f"sweep arm {picked} of {len(arms)} ({axis} = {arms[picked].value:g}): "
                f"{arms[picked].describe()}",
            ),
            dirty_level=None,
            pipeline=pipe,
        )
        return io.NodeOutput(
            report, sheet, out, len(arms), round(spread, 6),
            ui=node_preview(image=sheet, text=report),
        )


def _level_caveat(target, source) -> str:
    """What a level sweep does to a region-snapped selection, stated once."""
    bound = sorted({s.level for s in (target, source) if s.level is not None})
    if not bound:
        return ""
    return (
        f"NOTE: the selection was snapped to level {bound[0]}'s regions. Every arm applies the "
        "same token set, but those tokens are not a whole region at the other levels -- which "
        "is what holding the edit fixed while l* varies has to mean. Use typed coordinates if "
        "you want the token set to be the thing you chose."
    )


def _report(axis, arms, rows, spread, levels, target, source, source_levels,
            source_mode, picked, asked, baseline, caveat) -> str:
    """The table, and enough of the setup to read it a month later."""
    origin = "same trajectory" if source_levels is None else f"class {source_levels.class_id}"
    pinned = {
        sweep.SEED: f"level {levels.clamp(arms[0].level)}, strength {arms[0].strength:.2f}",
        sweep.LEVEL: f"seed {arms[0].seed}, strength {arms[0].strength:.2f}",
        sweep.STRENGTH: f"level {levels.clamp(arms[0].level)}, seed {arms[0].seed}",
    }[axis]
    lines = [
        f"sweep {axis} = {', '.join(f'{a.value:g}' for a in arms)}   ({len(arms)} arms)",
        f"pinned: {pinned}, class {levels.class_id}",
        f"edit:   {target.count} target tokens <- {source_mode} of {source.count} "
        f"source tokens ({origin})",
        (
            "measured against: the same trajectory resumed identically without the edit, one "
            "baseline per arm"
            if baseline else
            "measured against: the input trajectory (baseline off -- these numbers include the "
            "resume seed's own effect, not just the edit's)"
        ),
        "",
        f"{axis:<12} tokens changed   mean dist   max dist",
    ]
    for arm, changed, mean, peak in rows:
        total = levels.grid[0] * levels.grid[1]
        lines.append(
            f"{arm.value:<12g} {changed:>4} / {total:<8} {mean:>9.4f}  {peak:>9.4f}"
        )

    dead = [f"{a.value:g}" for (a, changed, _, _) in rows if changed == 0]
    lines += ["", f"spread across arms: {spread:.4f} mean pairwise cosine distance at level "
                  f"{levels.num_levels - 1}"]
    if len(arms) < 2:
        lines[-1] = "spread across arms: n/a with a single arm"
    elif spread <= measure.CHANGED:
        lines.append(
            f"  -> below the {measure.CHANGED} 'changed' threshold: every arm landed in "
            f"effectively the same place, so {axis} is not what decides this edit's outcome."
        )
    else:
        lines.append(f"  -> {axis} does change the outcome.")
    if dead:
        lines.append(
            f"  -> arm(s) {', '.join(dead)} changed nothing at all measured against the "
            "baseline. That is a result, not a failure -- but check the selection if you "
            "expected otherwise."
        )
    # `describe()` already names the swept field, so the axis is not repeated
    # in front of it -- "arm 0 (seed = 592, level 2, seed 592, ...)" was what
    # the first real run printed.
    lines.append(
        f"'levels' output: arm {picked} ({arms[picked].describe()})"
        + (f"   [output_arm was {asked}, clamped to the last arm]" if asked != picked else "")
    )
    if caveat:
        lines += ["", caveat]
    return "\n".join(lines)


def _sheet(pipe, levels, arms, results, baseline_latents, axis, decode, size):
    """Baseline first, then one frame per arm, each captioned with its value."""
    def frame(latents) -> np.ndarray:
        if decode:
            return pipe.decode(latents, final_only=True)[0]
        return pipe.pca_tiles(latents)[-1]

    frames = []
    if baseline_latents is not None:
        frames.append((frame(baseline_latents), "no edit (baseline)"))
    frames += [(frame(latents), f"{axis} = {arm.value:g}")
               for arm, latents in zip(arms, results, strict=True)]
    return render.to_image([
        render.caption(render.fit_to_grid(picture, levels.grid, size), text)
        for picture, text in frames
    ])
