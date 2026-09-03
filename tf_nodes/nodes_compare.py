"""Did the edit do anything, and where?

The question every edit prompts, and until this node the only answer was two
previews and your eyes. Eyes are bad at "did level 1 move a little or not at
all", and a writeup needs a number rather than an impression.

Everything here is measured in the latent space the edits act on, per level and
per token, so the answer localises: a feature edit at l* should show zero change
below l*, exactly the edited tokens at l*, and diffuse change above it. A
result that does not look like that is the interesting kind of wrong.
"""
from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from . import render
from .measure import CHANGED, heat, per_token_distance
from .sockets import (
    CATEGORY_EDIT,
    TFLevelsSocket,
    node_preview,
    pipeline_input,
    resolve_pipeline,
    sheet_layout_input,
)


class TFCompareLevels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFCompareLevels",
            search_aliases=["compare", "diff", "difference", "changed", "measure"],
            display_name="TF Compare Levels",
            category=CATEGORY_EDIT,
            has_intermediate_output=True,
            description=(
                "Measure what changed between two trajectories, per level and per token.\n\n"
                "Wire the original into 'before' and the edited-and-resumed result into "
                "'after'. The report gives the number of tokens that moved at each level; "
                "the heatmap shows where. A feature edit at l* should leave every level "
                "below it untouched."
            ),
            inputs=[
                TFLevelsSocket.Input("before", tooltip="Usually the trajectory straight from TF Generate."),
                TFLevelsSocket.Input("after", tooltip="Usually the output of TF Resume From Level."),
                io.Int.Input("size", default=512, min=128, max=2048, step=64, advanced=True),
                io.Boolean.Input(
                    "decode_difference", default=False,
                    tooltip="Also decode both final levels and show the pixel difference. "
                            "Costs two ViT-XL passes; the latent heatmap is free.",
                ),
                sheet_layout_input("the per-level heatmaps"),
                pipeline_input(tooltip="Only needed when 'decode_difference' is on."),
            ],
            outputs=[
                io.String.Output("report"),
                io.Image.Output("heatmap", tooltip="One tile per level; brighter is more changed."),
                io.Int.Output("changed_tokens", tooltip="Total across all levels."),
                io.Float.Output("max_distance", tooltip="Largest per-token cosine distance."),
            ],
        )

    @classmethod
    def execute(cls, before, after, size, decode_difference, sheet_layout,
                pipeline=None) -> io.NodeOutput:
        if before.latents.shape != after.latents.shape:
            raise ValueError(
                f"Cannot compare {before.latents.shape} with {after.latents.shape} -- these are "
                "not two versions of the same trajectory."
            )

        lines = [
            f"before: class {before.class_id}, seed {before.seed}",
            f"after:  class {after.class_id}, seed {after.seed}",
            "",
            "level              tokens changed   mean dist   max dist",
        ]
        # Every frame goes through fit_to_grid at the same target, because
        # to_image stacks them into one batch and numpy will not stack frames of
        # different sizes -- which a decoded difference is, being 256px.
        tile_px = max(128, int(size) // 2)
        tiles, changed_per_level, peak = [], [], 0.0
        for level in range(before.num_levels):
            distance = per_token_distance(before.level(level), after.level(level))
            changed = int((distance > CHANGED).sum())
            changed_per_level.append(changed)
            peak = max(peak, float(distance.max()))
            name = render.level_caption(level, before.num_levels)
            lines.append(
                f"{name:<18} {changed:>4} / {distance.size:<8} "
                f"{distance.mean():>9.4f}  {distance.max():>9.4f}"
            )
            tile = render.fit_to_grid(heat(distance), distance.shape, tile_px)
            tiles.append(render.caption(render.draw_ticks(tile, distance.shape), name))

        total = sum(changed_per_level)
        if total == 0:
            lines += ["", "Nothing changed. The two trajectories are identical."]
        else:
            first = next(i for i, n in enumerate(changed_per_level) if n)
            lines += ["", f"First level that differs: {first}."]
            if first:
                lines[-1] += (f" Levels 0..{first - 1} are untouched, which is what an edit "
                              f"at level {first} should do.")

        images = tiles
        if decode_difference:
            resolved = resolve_pipeline(pipeline, after, "TF Compare Levels")
            a = resolved.decode(before.latents, final_only=True)[0].astype(np.int16)
            b = resolved.decode(after.latents, final_only=True)[0].astype(np.int16)
            delta = np.abs(b - a)
            lines += ["", f"decoded final image: mean |delta| = {delta.mean():.2f}/255"]
            images = images + [render.caption(
                render.fit_to_grid(delta.astype(np.uint8), before.grid, tile_px),
                "decoded |difference|",
            )]

        report = "\n".join(lines)
        heatmap = render.lay_out(images, sheet_layout)
        return io.NodeOutput(
            report, heatmap, total, round(peak, 6),
            ui=node_preview(image=heatmap, text=report),
        )
