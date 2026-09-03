"""Choosing what to edit: the level canvas, the region map, and token selections.

TF's editing operates on regions of a 16x16 token grid, which is not a resolution
any ComfyUI mask tool knows about. Rather than ship a bespoke LiteGraph canvas,
TF Level Canvas renders the level as a paintable image with the token grid drawn
on it, core Painter supplies the brush, and TF Tokens From Mask converts what was
painted back down to tokens -- optionally snapping to whole cosine regions, which
is the one-click "grab the whole cluster" of the editing env's region select.
"""
from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from . import render, tokens
from .sockets import (
    CATEGORY,
    TFLevelsSocket,
    TFPipelineSocket,
    TFRegionsSocket,
    TFTokensSocket,
    level_input,
)


class TFLevelCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLevelCanvas",
            display_name="TF Level Canvas",
            category=CATEGORY,
            description=(
                "Render one level as a paintable canvas: PCA latent or decoded RGB, with the token "
                "grid and optionally the region boundaries drawn on. Feed it to core Painter and "
                "the mask that comes back lines up with the token grid."
            ),
            inputs=[
                TFPipelineSocket.Input("pipeline"),
                TFLevelsSocket.Input("levels"),
                level_input(),
                io.Combo.Input(
                    "view", options=["latent PCA", "decoded RGB"],
                    tooltip="PCA shows the token structure the edit acts on; decoded RGB shows what "
                            "that structure looks like as an image.",
                ),
                io.Boolean.Input("draw_grid", default=True),
                io.Int.Input(
                    "size", default=512, min=128, max=2048, step=64,
                    tooltip="Rounded so each token is a whole number of pixels.",
                ),
                TFRegionsSocket.Input(
                    "regions", optional=True,
                    tooltip="Draw region boundaries from a TF Region Map, so a brush stroke can "
                            "follow the cluster it is meant to select.",
                ),
                TFTokensSocket.Input(
                    "highlight", optional=True,
                    tooltip="Tint an existing selection, to check what a mask resolved to.",
                ),
            ],
            outputs=[io.Image.Output("canvas"), io.Int.Output("level")],
        )

    @classmethod
    def execute(cls, pipeline, levels, level, view, draw_grid, size,
                regions=None, highlight=None) -> io.NodeOutput:
        index = levels.clamp(level)
        if view == "decoded RGB":
            single = levels.latents[index:index + 1]
            base = pipeline.decode(single, final_only=True)[0]
        else:
            base = pipeline.pca_tiles(levels.latents)[index]
        canvas = render.fit_to_grid(base, levels.grid, size)
        # Painted in increasing order of "what the user is looking for": region
        # boundaries are context, the grid is a ruler, the current selection is
        # the answer -- so the selection goes on last and nothing overdraws it.
        if regions is not None:
            canvas = render.draw_region_boundaries(canvas, regions)
        if draw_grid:
            canvas = render.draw_grid(canvas, levels.grid)
        if highlight is not None:
            highlight.check_grid(levels.grid, "highlight")
            canvas = render.draw_selection(canvas, highlight)
        return io.NodeOutput(render.to_image(canvas), index)


class TFRegionMap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFRegionMap",
            display_name="TF Region Map",
            category=CATEGORY,
            description=(
                "Cluster a level's tokens into connected regions by cosine similarity. These are "
                "the R in the paper's edits: a feature edit replaces one region's feature, a shape "
                "edit moves tokens from one region to a neighbour."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                level_input(),
                io.Float.Input(
                    "cosine_threshold", default=0.9, min=0.5, max=0.99, step=0.01,
                    tooltip="Higher splits into more, smaller regions. "
                            "0.9 matches the editing env's default.",
                ),
                io.Int.Input("size", default=512, min=128, max=2048, step=64),
            ],
            outputs=[
                TFRegionsSocket.Output("regions"),
                io.Image.Output("map"),
                io.Int.Output("num_regions"),
            ],
        )

    @classmethod
    def execute(cls, levels, level, cosine_threshold, size) -> io.NodeOutput:
        index = levels.clamp(level)
        regions = tokens.build_region_map(levels.level(index), index, cosine_threshold)
        picture = render.draw_grid(render.render_regions(regions, size), regions.ids.shape)
        return io.NodeOutput(regions, render.to_image(picture), regions.num_regions)


class TFTokensFromMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensFromMask",
            display_name="TF Tokens From Mask",
            category=CATEGORY,
            description=(
                "Reduce a painted mask to a token selection. A token counts as selected once "
                "enough of its footprint is painted, so a stroke that clips a corner does not "
                "silently overwrite that token's whole feature vector."
            ),
            inputs=[
                io.Mask.Input("mask"),
                TFLevelsSocket.Input("levels", tooltip="Only its token-grid size is read."),
                io.Float.Input(
                    "coverage", default=0.35, min=0.01, max=1.0, step=0.01,
                    tooltip="Fraction of a token's pixels that must be painted for it to count.",
                ),
                TFRegionsSocket.Input(
                    "regions", optional=True,
                    tooltip="Snap the selection to whole regions from this map instead of taking "
                            "the painted tokens literally.",
                ),
                io.Float.Input(
                    "region_overlap", default=0.3, min=0.01, max=1.0, step=0.01,
                    tooltip="With 'regions' wired: a region joins the selection once this fraction "
                            "of its tokens is painted. Ignored otherwise.",
                ),
            ],
            outputs=[
                TFTokensSocket.Output("tokens"),
                io.Int.Output("count"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, mask, levels, coverage, regions=None, region_overlap=0.3) -> io.NodeOutput:
        selection = tokens.mask_to_tokens(render.from_mask(mask), levels.grid, coverage)
        if regions is not None:
            selection = tokens.snap_to_regions(selection, regions, region_overlap)
        return io.NodeOutput(selection, selection.count, _describe(selection))


class TFTokensFromCoords(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensFromCoords",
            display_name="TF Tokens From Coords",
            category=CATEGORY,
            description=(
                "Type token coordinates directly: 'row,col' pairs, with 'row,col0:col1' for a run. "
                "Reproducible in a way a brush stroke is not, which is what a written-up experiment needs."
            ),
            inputs=[
                io.String.Input(
                    "coords", default="", multiline=True,
                    placeholder="7,7  7,8  8,7  8,8      or      7,6:9",
                ),
                TFLevelsSocket.Input("levels", tooltip="Only its token-grid size is read."),
                TFRegionsSocket.Input(
                    "regions", optional=True,
                    tooltip="Expand each typed token to the whole region containing it -- the "
                            "editing env's cluster-select, as one click per region.",
                ),
            ],
            outputs=[
                TFTokensSocket.Output("tokens"),
                io.Int.Output("count"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, coords, levels, regions=None) -> io.NodeOutput:
        selection = tokens.parse_coords(coords, levels.grid)
        if regions is not None:
            # Any overlap at all expands the region: a typed coordinate is a
            # deliberate pick of one token, not a rough stroke to be thresholded.
            selection = tokens.snap_to_regions(selection, regions, 0.0)
        return io.NodeOutput(selection, selection.count, _describe(selection))


class TFTokensCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensCombine",
            display_name="TF Tokens Combine",
            category=CATEGORY,
            description="Set operations on token selections -- build a region up from several strokes.",
            inputs=[
                TFTokensSocket.Input("a"),
                io.Combo.Input(
                    "operation",
                    options=["union", "intersection", "difference", "symmetric difference", "invert"],
                    tooltip="'invert' uses only input a.",
                ),
                TFTokensSocket.Input("b", optional=True),
            ],
            outputs=[TFTokensSocket.Output("tokens"), io.Int.Output("count")],
        )

    @classmethod
    def execute(cls, a, operation, b=None) -> io.NodeOutput:
        out = tokens.combine(a, b, operation)
        return io.NodeOutput(out, out.count)


class TFTokensPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensPreview",
            display_name="TF Tokens Preview",
            category=CATEGORY,
            description="Draw a token selection on its own, without needing the pipeline loaded.",
            inputs=[
                TFTokensSocket.Input("tokens"),
                io.Int.Input("size", default=512, min=128, max=2048, step=64),
            ],
            outputs=[io.Image.Output("image"), io.String.Output("coords")],
        )

    @classmethod
    def execute(cls, tokens, size) -> io.NodeOutput:  # shadows the module; nothing here needs it
        selection = tokens
        grid = selection.mask.shape
        blank = np.full((grid[0], grid[1], 3), 24, dtype=np.uint8)
        canvas = render.fit_to_grid(blank, grid, size)
        canvas = render.draw_grid(render.draw_selection(canvas, selection), grid)
        coords = " ".join(f"{r},{c}" for r, c in selection.coords)
        return io.NodeOutput(render.to_image(canvas), coords)


def _describe(selection) -> str:
    if selection.count == 0:
        return f"0 tokens selected ({selection.note})"
    rows, cols = np.argwhere(selection.mask).T
    return (
        f"{selection.count} tokens selected ({selection.note})\n"
        f"rows {rows.min()}..{rows.max()}, cols {cols.min()}..{cols.max()}"
    )
