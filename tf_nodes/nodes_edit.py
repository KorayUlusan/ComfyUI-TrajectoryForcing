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
    CATEGORY,
    TFLevelsSocket,
    TFPipelineSocket,
    TFRegionsSocket,
    TFTokensSocket,
    level_input,
    seed_input,
)

SOURCE_MODES = ["region mean", "token cycle"]


class TFFeatureEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFFeatureEdit",
            display_name="TF Feature Edit",
            category=CATEGORY,
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
                    "source_mode", options=SOURCE_MODES,
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
                level_input("source_level", tooltip="Level to read the source feature from. Levels "
                                                    "share a token grid, so this need not match l*."),
            ],
            outputs=[TFLevelsSocket.Output("levels"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, levels, level, target_tokens, source_tokens, source_mode, strength,
                source_level, source_levels=None) -> io.NodeOutput:
        index = levels.clamp(level)
        target_tokens.check_grid(levels.grid, "target_tokens")
        target_tokens.require_nonempty("target_tokens")

        source_stack = source_levels if source_levels is not None else levels
        source_tokens.check_grid(source_stack.grid, "source_tokens")
        source_tokens.require_nonempty("source_tokens")
        src_index = source_stack.clamp(source_level)

        feature = tokens.source_feature(source_stack.level(src_index), source_tokens, source_mode)
        canvas = tokens.write_feature(levels.level(index), target_tokens, feature, strength)

        origin = "same trajectory" if source_levels is None else f"class {source_stack.class_id}"
        note = (
            f"feature edit: {target_tokens.count} tokens at level {index} <- "
            f"{source_mode} of {source_tokens.count} tokens at level {src_index} ({origin}), "
            f"strength {strength:.2f}"
        )
        return io.NodeOutput(levels.with_level(index, canvas, note), note)


class TFShapeEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFShapeEdit",
            display_name="TF Shape Edit",
            category=CATEGORY,
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
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05),
            ],
            outputs=[TFLevelsSocket.Output("levels"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, levels, level, regions, boundary_tokens, receiving_tokens, strength) -> io.NodeOutput:
        index = levels.clamp(level)
        boundary_tokens.check_grid(levels.grid, "boundary_tokens")
        boundary_tokens.require_nonempty("boundary_tokens")
        receiving_tokens.check_grid(levels.grid, "receiving_tokens")
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
        return io.NodeOutput(levels.with_level(index, edited, note), note)


class TFResumeFromLevel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFResumeFromLevel",
            display_name="TF Resume From Level",
            category=CATEGORY,
            description=(
                "Re-sample every level above l*, conditioned on the (edited) canvas sitting there. "
                "Levels below l* are left exactly as they were -- sampling is Markov in the level "
                "index, so an edit only ever propagates upward."
            ),
            inputs=[
                TFPipelineSocket.Input("pipeline"),
                TFLevelsSocket.Input("levels"),
                level_input(tooltip="l*. Defaults to the level the upstream edit touched when "
                                    "'follow_edit' is on."),
                io.Boolean.Input(
                    "follow_edit", default=True,
                    tooltip="Resume from whichever level the upstream edit node wrote to, ignoring "
                            "the level widget. Off means the widget decides.",
                ),
                io.Int.Input(
                    "class_id", default=-1, min=-1, max=999,
                    tooltip="Class the re-sampled levels are conditioned on. -1 keeps the "
                            "trajectory's own class, which is what an edit normally wants.",
                ),
                seed_input(),
            ],
            outputs=[TFLevelsSocket.Output("levels"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, pipeline, levels, level, follow_edit, class_id, seed) -> io.NodeOutput:
        start = level
        if follow_edit:
            if levels.dirty_level is None:
                raise ValueError(
                    "'follow_edit' is on but this trajectory carries no edit to follow. Turn it off "
                    "to resume from the level widget instead."
                )
            start = levels.dirty_level
        start = levels.clamp(start)
        effective_class = levels.class_id if int(class_id) < 0 else int(class_id)

        latents = pipeline.resume(levels.latents, start, effective_class, seed)
        note = (
            f"resume from level {start} (class {effective_class}, seed {int(seed)}): "
            f"levels {start + 1}..{levels.num_levels - 1} re-sampled"
            if start < levels.num_levels - 1
            else f"resume from level {start}: nothing above it to re-sample"
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
        )
        return io.NodeOutput(out, note)
