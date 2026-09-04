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
from comfy_execution.graph_utils import ExecutionBlocker

from . import render
from . import tokens as token_ops
from .locate import vue_nodes_enabled
from .sockets import (
    CATEGORY_SELECT,
    TFLevelsSocket,
    TFRegionsSocket,
    TFTokensSocket,
    level_input,
    node_preview,
    pipeline_input,
    resolve_pipeline,
)

# Coordinates are typed by hand against these pictures, so they carry row/column
# labels unless something downstream is going to consume them as a plain image.
DEFAULT_GRID = (16, 16)


class TFLevelCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLevelCanvas",
            search_aliases=["canvas", "paint", "grid", "brush", "painter", "mask"],
            display_name="TF Level Canvas",
            category=CATEGORY_SELECT,
            # Publishing this node's image as a UI preview is what makes the
            # Painter downstream show something to paint over. Painter resolves
            # its backdrop with `nodeOutputStore.getNodeImageUrls(inputNode)` --
            # the *stored preview* of whatever feeds its image slot, not the
            # tensor on the wire. Without a preview the Painter is a blank
            # square and you are painting blind.
            has_intermediate_output=True,
            description=(
                "Render one level as a paintable canvas: PCA latent or decoded RGB, with the token "
                "grid and optionally the region boundaries drawn on. Feed it to core Painter and "
                "the mask that comes back lines up with the token grid.\n\n"
                "The Painter node needs ComfyUI's Node 2.0 rendering; this node says so if it is "
                "switched off."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                level_input(),
                io.Combo.Input(
                    "view", options=["latent PCA", "decoded RGB"],
                    tooltip="PCA shows the token structure the edit acts on; decoded RGB shows what "
                            "that structure looks like as an image.",
                ),
                io.Boolean.Input("draw_grid", default=True, advanced=True),
                io.Boolean.Input(
                    "label_coords", default=True, advanced=True,
                    tooltip="Number the rows and columns, so a coordinate for TF Tokens From "
                            "Coords can be read off instead of counted.",
                ),
                io.Int.Input(
                    "size", default=512, min=128, max=2048, step=64, advanced=True,
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
                pipeline_input(),
            ],
            outputs=[io.Image.Output("canvas"), io.Int.Output("level")],
        )

    @classmethod
    def execute(cls, levels, level, view, draw_grid, label_coords, size,
                regions=None, highlight=None, pipeline=None) -> io.NodeOutput:
        pipeline = resolve_pipeline(pipeline, levels, "TF Level Canvas")
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
        if label_coords:
            canvas = render.draw_ticks(canvas, levels.grid)
        image = render.to_image(canvas)
        notice = _node2_notice()
        # The image is what the Painter downstream uses as its backdrop, so it
        # is published even alongside the notice.
        return io.NodeOutput(image, index, ui=node_preview(image=image, text=notice))


class TFRegionMap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFRegionMap",
            search_aliases=["region", "cluster", "segment", "cosine", "parts"],
            has_intermediate_output=True,
            display_name="TF Region Map",
            category=CATEGORY_SELECT,
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
                io.Int.Input("size", default=512, min=128, max=2048, step=64, advanced=True),
            ],
            # No `num_regions` socket: the count is on the RegionMap the
            # 'regions' output already carries, and it is in this node's body.
            # 'level' stays because it drives a widget -- the edit node's own
            # 'level' -- which is the test a scalar output has to pass.
            outputs=[
                TFRegionsSocket.Output("regions"),
                io.Image.Output("map"),
                io.Int.Output(
                    "level",
                    tooltip="The level these regions describe. Wire it into the edit node's "
                            "'level' so the two cannot drift apart.",
                ),
            ],
        )

    @classmethod
    def execute(cls, levels, level, cosine_threshold, size) -> io.NodeOutput:
        index = levels.clamp(level)
        regions = token_ops.build_region_map(levels.level(index), index, cosine_threshold)
        picture = render.draw_grid(render.render_regions(regions, size), regions.ids.shape)
        picture = render.draw_ticks(picture, regions.ids.shape)
        image = render.to_image(picture)
        return io.NodeOutput(
            regions, image, index,
            ui=node_preview(image=image,
                            text=f"{regions.num_regions} regions at level {index} "
                                 f"(threshold {cosine_threshold:.2f})"),
        )


class TFTokensFromMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensFromMask",
            search_aliases=["mask", "paint", "brush", "tokens", "select"],
            display_name="TF Tokens From Mask",
            category=CATEGORY_SELECT,
            # Keeps this node's own text visible on a re-run and across a page
            # refresh. Without it the "nothing painted yet" notice shows once
            # and is gone the next time, when the node is served from cache.
            has_intermediate_output=True,
            description=(
                "Reduce a painted mask to a token selection. A token counts as selected once "
                "enough of its footprint is painted, so a stroke that clips a corner does not "
                "silently overwrite that token's whole feature vector.\n\n"
                "With nothing painted yet it stops the graph here rather than failing downstream: "
                "the first run of a painting workflow exists to produce the canvas."
            ),
            inputs=[
                io.Mask.Input("mask"),
                TFLevelsSocket.Input(
                    "levels", optional=True,
                    tooltip="Only its token-grid size is read. Unwired assumes 16x16.",
                ),
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
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, mask, coverage, levels=None, regions=None, region_overlap=0.3) -> io.NodeOutput:
        grid = levels.grid if levels is not None else (
            regions.ids.shape if regions is not None else DEFAULT_GRID)
        painted = render.from_mask(mask)
        selection = token_ops.mask_to_tokens(painted, grid, coverage)
        if regions is not None:
            selection = token_ops.snap_to_regions(selection, regions, region_overlap)
        if selection.count == 0:
            # Painting workflows are two-pass by construction -- the first run
            # renders the canvas you paint on, so at that point there is nothing
            # to reduce. Stopping the graph here is right; how to stop it took
            # two wrong turns worth recording.
            #
            # Not `block_execution=reason`: a blocker carrying a message is
            # reported to the browser as "Node threw an error during execution",
            # and only on the *first* run -- the second finds this node cached,
            # never calls execute, and blocks in silence. Alarming and
            # inconsistent, which is worse than either alone.
            #
            # Not a bare `return ExecutionBlocker(None)` either: EXECUTE_NORMALIZED
            # turns that into NodeOutput(block_execution=None), i.e. no block at
            # all. One blocker per declared output is what actually stops the
            # graph, quietly and the same way on every run, and it leaves `ui`
            # free to say why in this node's own body.
            stop = ExecutionBlocker(None)
            reason = _nothing_painted(painted, coverage, regions, region_overlap)
            return io.NodeOutput(stop, stop, ui=node_preview(text=reason))
        info = _describe(selection)
        return io.NodeOutput(selection, info, ui=node_preview(text=info))


class TFTokensFromCoords(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensFromCoords",
            search_aliases=["coords", "tokens", "select", "row", "col", "typed"],
            has_intermediate_output=True,
            display_name="TF Tokens From Coords",
            category=CATEGORY_SELECT,
            description=(
                "Pick tokens on a 16x16 grid: click cells, or drag to paint several. Typing works "
                "too -- 'row,col' pairs, with 'row,col0:col1' for a run -- and the two stay in "
                "step, because the clickable grid writes into the text field rather than replacing "
                "it.\n\n"
                "With a region map wired in, one click takes the whole region -- the paper's "
                "R_tgt, a semantic part rather than an arbitrary set of tokens -- and the grid "
                "draws the boundaries. It learns them from the first run, so before that a click "
                "picks one token and the node expands it.\n\n"
                "Alt-click (option on a Mac) writes just that one coordinate. It does not "
                "narrow the selection -- with a map wired the node snaps to whole regions "
                "either way -- but '7,7' is a better line in a writeup than a nine-run "
                "coordinate list. Unwire 'regions' for genuinely sub-region tokens.\n\n"
                "Either way the selection ends up as text you can paste into a writeup, which is "
                "what a brush stroke cannot give you and what a written-up experiment needs."
            ),
            inputs=[
                io.String.Input(
                    "coords", default="", multiline=True,
                    placeholder="7,7  7,8  8,7  8,8      or      7,6:9",
                    tooltip="Click the grid above, or type here -- they are the same value. "
                            "'7,6:9' means row 7, columns 6 through 9.",
                ),
                TFLevelsSocket.Input(
                    "levels", optional=True,
                    tooltip="Only its token-grid size is read. Unwired assumes 16x16, which "
                            "is what every released checkpoint uses.",
                ),
                TFRegionsSocket.Input(
                    "regions", optional=True,
                    tooltip="Expand each typed token to the whole region containing it -- the "
                            "editing env's cluster-select, as one click per region.",
                ),
            ],
            outputs=[
                TFTokensSocket.Output("tokens"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, coords, levels=None, regions=None) -> io.NodeOutput:
        grid = levels.grid if levels is not None else (
            regions.ids.shape if regions is not None else DEFAULT_GRID)
        selection = token_ops.parse_coords(coords, grid)
        if regions is not None:
            # Any overlap at all expands the region: a typed coordinate is a
            # deliberate pick of one token, not a rough stroke to be thresholded.
            selection = token_ops.snap_to_regions(selection, regions, 0.0)
        if selection.count == 0:
            # The same quiet stop as TF Tokens From Mask, for the same reason: an
            # empty selection is not an error here, it is "nothing picked yet".
            # Left to travel downstream it surfaces two nodes later as a raw
            # traceback out of TF Feature Edit, which reads as a crash rather
            # than as something the user has not done. One ExecutionBlocker per
            # declared output and never a message-carrying one -- see
            # TFTokensFromMask for why that route is the wrong one.
            stop = ExecutionBlocker(None)
            return io.NodeOutput(stop, stop,
                                 ui=node_preview(text=_nothing_selected()))
        info = _describe(selection)
        # Hand the region map back to this node's own clickable grid, so what it
        # highlights is what the node actually selected. Without this the widget
        # shows the one token you clicked while the snap above quietly expands it
        # to the whole region -- often forty tokens -- and the count beside the
        # grid is wrong every time a map is wired in, which is the default in
        # workflows 02 and 05. Read back in web/tf_token_grid.js via onExecuted;
        # a 16x16 grid of small ints is about a kilobyte of JSON.
        return io.NodeOutput(
            selection, info,
            ui=_coords_ui(info, regions),
        )


class TFTokensCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensCombine",
            search_aliases=["combine", "union", "intersect", "subtract", "invert", "boolean"],
            has_intermediate_output=True,
            display_name="TF Tokens Combine",
            category=CATEGORY_SELECT,
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
            # 'info' rather than the old 'count' INT, so this node has the same
            # shape as the two that also produce a selection -- and so the
            # number lands somewhere, which a bare INT never did.
            outputs=[TFTokensSocket.Output("tokens"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, a, operation, b=None) -> io.NodeOutput:
        out = token_ops.combine(a, b, operation)
        info = f"{out.count} tokens after {operation}"
        return io.NodeOutput(out, info, ui=node_preview(text=info))


class TFTokensPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFTokensPreview",
            search_aliases=["preview", "tokens", "selection", "show", "check"],
            has_intermediate_output=True,
            display_name="TF Tokens Preview",
            category=CATEGORY_SELECT,
            description="Draw a token selection on its own, without needing the pipeline loaded.",
            inputs=[
                TFTokensSocket.Input("tokens"),
                io.Int.Input("size", default=512, min=128, max=2048, step=64, advanced=True),
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
        canvas = render.draw_ticks(canvas, grid)
        # Same notation someone would have typed, runs and all, so it pastes
        # straight back into TF Tokens From Coords.
        coords = token_ops.format_coords(selection.mask)
        image = render.to_image(canvas)
        return io.NodeOutput(
            image, coords,
            ui=node_preview(image=image, text=f"{selection.count} tokens: {coords or '(none)'}"),
        )


def _nothing_selected() -> str:
    return (
        "Nothing selected. Click the grid above to pick tokens, or type coordinates -- "
        "'7,7' is one token on the 16x16 grid, '7,6:9' a run along row 7.\n\n"
        "The graph stops here on purpose. An empty selection reaching TF Feature Edit "
        "raises there instead, two nodes further on, where it reads as a crash rather "
        "than as something you have not done yet."
    )


def _coords_ui(info: str, regions) -> dict:
    """This node's own body, plus the region map its grid widget needs."""
    payload = dict(node_preview(text=info) or {})
    if regions is not None:
        payload["tf_regions"] = [{
            "ids": regions.ids.tolist(),
            "level": int(regions.level),
            "num_regions": int(regions.num_regions),
        }]
    return payload


def _node2_notice() -> str:
    """Warn only the users who actually need it, and tell them exactly where to click.

    The Painter widget exists solely in ComfyUI's Vue node rendering; under the
    classic renderer it shows the text "Node 2.0 only" and cannot be painted on.
    Whether it is on is readable from ComfyUI's own settings file, so this stays
    silent for the majority who already have it rather than nagging everyone.
    """
    if vue_nodes_enabled() is not False:
        return ""
    return (
        "The Painter node needs Node 2.0, which is currently off -- it will show "
        "\"Node 2.0 only\" instead of a brush.\n\n"
        "Turn it on:  Settings (gear, bottom left)  ->  search \"Node 2.0\"  ->  "
        "enable it, then reload the page.\n\n"
        "This canvas is still correct; only the painting step is blocked. Workflow 02 "
        "types coordinates instead and needs none of this."
    )


def _nothing_painted(painted: np.ndarray, coverage: float, regions, region_overlap: float) -> str:
    """Why the selection came out empty, and what to do about it.

    The three causes need different fixes, and guessing wrong wastes a run:
    nothing painted at all, painted too thinly for the coverage threshold, or
    painted across regions none of which cleared the overlap threshold.
    """
    if not painted.any():
        return (
            "Nothing painted yet. Paint over the region you want on the Painter node, then run "
            "again. (The first run of this workflow exists to produce the canvas to paint on.)"
        )
    if regions is not None:
        return (
            f"Painted, but no region reached the {region_overlap:.2f} overlap threshold. Paint "
            "more of one region -- TF Region Map's preview shows where the boundaries are -- or "
            "lower 'region_overlap'."
        )
    return (
        f"Painted, but no token reached {coverage:.2f} coverage. Use a bigger brush, or lower "
        "'coverage'."
    )


def _describe(selection) -> str:
    if selection.count == 0:
        return f"0 tokens selected ({selection.note})"
    rows, cols = np.argwhere(selection.mask).T
    return (
        f"{selection.count} tokens selected ({selection.note})\n"
        f"rows {rows.min()}..{rows.max()}, cols {cols.min()}..{cols.max()}"
    )
