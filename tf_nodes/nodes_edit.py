"""The edits themselves, and the resume that propagates them.

Both edits reduce to the same primitive from the paper's Sec. 4.4 --
`z_i <- f_src` for the selected tokens i -- and differ only in where `f_src`
comes from:

* Feature edit takes it from a source selection, on this canvas or on a second
  trajectory entirely, giving the target region the source's semantic identity.
* Shape edit takes it from the region that is absorbing the tokens, which moves
  a boundary without inventing any new feature content.

Neither node samples anything. They produce the edited canvas z~ at level l*;
TF Resume From Level is what re-generates the finer levels from it, and it does
not care which of the two produced its input.
"""
from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from . import tokens
from .sockets import (
    AUTO,
    CATEGORY_EDIT,
    TFLevelsSocket,
    TFRegionsSocket,
    TFTokensSocket,
    auto_label,
    auto_level_input,
    level_input,
    node_preview,
    pipeline_input,
    resolve_pipeline,
    seed_input,
)

SOURCE_MODES = ["region mean", "token cycle"]


class TFFeatureEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFFeatureEdit",
            search_aliases=["edit", "feature", "swap", "transfer", "replace", "token exchange"],
            has_intermediate_output=True,
            display_name="TF Feature Edit",
            category=CATEGORY_EDIT,
            description=(
                "Replace the selected tokens' features with one sourced from elsewhere. Because "
                "semantically similar content sits near each other in DINOv2 space, this transfers "
                "the source region's identity onto the target region.\n\n"
                "Leave 'source_levels' unwired to source from the same trajectory."
            ),
            inputs=[
                TFLevelsSocket.Input("levels", tooltip="The trajectory being edited."),
                level_input(tooltip="l*, the level whose canvas is edited. Coarser edits cascade "
                                    "through more levels; finer ones stay local."),
                TFTokensSocket.Input("target_tokens", tooltip="Tokens that receive the new feature."),
                TFTokensSocket.Input("source_tokens", tooltip="Tokens the feature is taken from."),
                io.Combo.Input(
                    "source_mode", options=SOURCE_MODES, advanced=True,
                    tooltip="'region mean' is the paper's f_src: one averaged vector fills the whole "
                            "target. 'token cycle' copies token-for-token, keeping the source's "
                            "internal variation.",
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Interpolate towards the source feature instead of replacing outright. "
                            "1.0 is the paper's edit.",
                ),
                TFLevelsSocket.Input(
                    "source_levels", optional=True,
                    tooltip="Take the source feature from a different trajectory -- the cross-image "
                            "token exchange. Unwired means same trajectory.",
                ),
                auto_level_input(
                    "source_level", "reads the source from the same level as the edit",
                    "Levels share a token grid, so a different one is legal and occasionally "
                    "what you want -- coarse content written into a fine canvas.",
                ),
            ],
            outputs=[TFLevelsSocket.Output("levels"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, levels, level, target_tokens, source_tokens, source_mode, strength,
                source_level, source_levels=None) -> io.NodeOutput:
        index = levels.clamp(level)
        target_tokens.check_grid(levels.grid, "target_tokens")
        # Every level shares a token grid, so a selection snapped to another
        # level's regions fits perfectly and means something else entirely.
        # Shape Edit has always caught this; without the same check here the
        # edit lands on the wrong region and nothing says so.
        target_tokens.check_level(index, "target_tokens")
        target_tokens.require_nonempty("target_tokens")

        source_stack = source_levels if source_levels is not None else levels
        source_tokens.check_grid(source_stack.grid, "source_tokens")
        src_index = source_stack.clamp(index if int(source_level) < 0 else source_level)
        source_tokens.check_level(src_index, "source_tokens")
        source_tokens.require_nonempty("source_tokens")

        feature = tokens.source_feature(source_stack.level(src_index), source_tokens, source_mode)
        canvas = tokens.write_feature(levels.level(index), target_tokens, feature, strength)

        origin = "same trajectory" if source_levels is None else f"class {source_stack.class_id}"
        from_level = f"level {src_index}" + (" (auto: same as the edit)"
                                             if int(source_level) < 0 else "")
        note = (
            f"feature edit: {target_tokens.count} tokens at level {index} <- "
            f"{source_mode} of {source_tokens.count} tokens at {from_level} ({origin}), "
            f"strength {strength:.2f}"
        )
        return io.NodeOutput(levels.with_level(index, canvas, note), note,
                             ui=node_preview(text=note))


class TFShapeEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFShapeEdit",
            search_aliases=["shape", "boundary", "reassign", "grow", "shrink", "edit"],
            has_intermediate_output=True,
            display_name="TF Shape Edit",
            category=CATEGORY_EDIT,
            description=(
                "Move boundary tokens from one region into a neighbouring one: the tokens take on "
                "the receiving region's mean feature, so the region's spatial extent changes while "
                "its feature content does not.\n\n"
                "Paint the tokens to hand over into 'boundary_tokens', and one token inside the "
                "region that is absorbing them into 'receiving_tokens'."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                level_input(),
                TFRegionsSocket.Input("regions", tooltip="Region map for this level, from TF Region Map."),
                TFTokensSocket.Input(
                    "boundary_tokens",
                    tooltip="Tokens being reassigned. These are the ones that change.",
                ),
                TFTokensSocket.Input(
                    "receiving_tokens",
                    tooltip="Any token inside the region absorbing them; the whole region it names "
                            "supplies the mean feature.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               advanced=True),
            ],
            outputs=[TFLevelsSocket.Output("levels"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, levels, level, regions, boundary_tokens, receiving_tokens, strength) -> io.NodeOutput:
        index = levels.clamp(level)
        boundary_tokens.check_grid(levels.grid, "boundary_tokens")
        boundary_tokens.check_level(index, "boundary_tokens")
        boundary_tokens.require_nonempty("boundary_tokens")
        receiving_tokens.check_grid(levels.grid, "receiving_tokens")
        receiving_tokens.check_level(index, "receiving_tokens")
        receiving_tokens.require_nonempty("receiving_tokens")
        if regions.ids.shape != levels.grid:
            raise ValueError(
                f"Region map is {regions.ids.shape} but the token grid is {levels.grid}."
            )
        if regions.level != index:
            raise ValueError(
                f"Region map was built on level {regions.level} but this edit is at level {index}. "
                "Point TF Region Map at the same level, or the boundaries do not describe this canvas."
            )

        canvas = levels.level(index)
        # The receiving region is named by whatever tokens the user pointed at,
        # but f_src is the mean of the *whole* region, not of those tokens: a
        # shape edit must not also change what the region looks like.
        receiving_ids = sorted({regions.region_of(r, c) for r, c in receiving_tokens.coords})
        donor_ids = sorted({regions.region_of(r, c) for r, c in boundary_tokens.coords})
        if set(donor_ids) == set(receiving_ids):
            raise ValueError(
                f"boundary_tokens and receiving_tokens are both inside region(s) {receiving_ids}; "
                "a shape edit needs two different regions."
            )
        receiving_mask = regions.mask_for(receiving_ids)
        feature = canvas[receiving_mask].mean(axis=0, keepdims=True)
        edited = tokens.write_feature(canvas, boundary_tokens, feature, strength)

        note = (
            f"shape edit: {boundary_tokens.count} tokens at level {index} moved from region(s) "
            f"{donor_ids} into region(s) {receiving_ids} "
            f"({int(receiving_mask.sum())} tokens), strength {strength:.2f}"
        )
        return io.NodeOutput(levels.with_level(index, edited, note), note,
                             ui=node_preview(text=note))


class TFResumeFromLevel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFResumeFromLevel",
            search_aliases=["resume", "regenerate", "propagate", "re-sample", "apply"],
            has_intermediate_output=True,
            display_name="TF Resume From Level",
            category=CATEGORY_EDIT,
            description=(
                "Re-sample every level above l*, conditioned on the (edited) canvas sitting there. "
                "Levels below l* are left exactly as they were -- sampling is Markov in the level "
                "index, so an edit only ever propagates upward.\n\n"
                "Left alone it follows the upstream edit, so l* cannot disagree with where the "
                "edit actually landed."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                seed_input(),
                auto_level_input(
                    "level", "resumes from whichever level the upstream edit wrote to",
                    "Set a level to resume from somewhere else instead. Levels below it are "
                    "left untouched either way.",
                ),
                io.Int.Input(
                    "class_id", display_name=auto_label("class_id"),
                    default=AUTO, min=AUTO, max=999, advanced=True,
                    tooltip="Class the re-sampled levels are conditioned on. -1 (auto) keeps "
                            "the trajectory's own class, which is what an edit normally wants.",
                ),
                pipeline_input(),
            ],
            outputs=[TFLevelsSocket.Output("levels"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, levels, seed, level, class_id, pipeline=None) -> io.NodeOutput:
        pipeline = resolve_pipeline(pipeline, levels, "TF Resume From Level")
        start = level
        if int(level) < 0:
            if levels.dirty_level is None:
                raise ValueError(
                    "This trajectory carries no edit to follow, so there is no level to resume "
                    "from. Wire an edit node in, or set 'level' (advanced) to resume from a "
                    "level of your choosing."
                )
            start = levels.dirty_level
        start = levels.clamp(start)
        effective_class = levels.class_id if int(class_id) < 0 else int(class_id)

        latents = pipeline.resume(levels.latents, start, effective_class, seed)
        # Spelling out what -1 turned into: the label says a widget is on auto,
        # this says what auto decided, which is the half a sentinel hides.
        why_level = ("auto: the level the edit wrote to" if int(level) < 0
                     else "set on the node")
        why_class = ("auto: the trajectory's own" if int(class_id) < 0
                     else "set on the node")
        note = (
            f"resume from level {start} ({why_level}); "
            f"class {effective_class} ({why_class}); seed {int(seed)}\n"
            + (f"levels {start + 1}..{levels.num_levels - 1} re-sampled"
               if start < levels.num_levels - 1
               else "nothing above it to re-sample")
        )
        # Resuming refreshes everything above `start`, so it settles a pending
        # edit only if the edit was at or below it. Resuming from *above* one --
        # possible with follow_edit off -- leaves those levels exactly as stale
        # as they were, and must not clear the marker that says so.
        still_dirty = (
            None if levels.dirty_level is None or start <= levels.dirty_level else levels.dirty_level
        )
        from dataclasses import replace

        out = replace(
            levels,
            latents=np.asarray(latents, dtype=np.float32),
            class_id=effective_class,
            seed=int(seed),
            history=levels.history + (note,),
            dirty_level=still_dirty,
            pipeline=pipeline,
        )
        return io.NodeOutput(out, note, ui=node_preview(text=note))
