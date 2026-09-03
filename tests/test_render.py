"""Drawing: sizes, ranges, and the token-grid alignment the whole edit UI leans on.

The one that matters is `fit_to_grid`: the mask a user paints on the canvas is
mapped back to tokens by dividing the image into a grid, so if the canvas is not
an exact multiple of the token count, a stroke on the token they can see selects
a different token.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tf_nodes import render
from tf_nodes import tokens as token_ops
from tf_nodes.data import RegionMap, TokenSelection


class TestConversion:
    def test_single_frame_becomes_a_batch_of_one(self):
        image = render.to_image(np.zeros((8, 8, 3), dtype=np.uint8))
        assert image.shape == (1, 8, 8, 3)
        assert image.dtype == torch.float32

    def test_values_are_scaled_to_zero_one(self):
        image = render.to_image(np.full((2, 2, 3), 255, dtype=np.uint8))
        assert float(image.max()) == 1.0

    def test_a_list_becomes_a_batch(self):
        assert render.to_image([np.zeros((4, 4, 3), np.uint8)] * 3).shape == (3, 4, 4, 3)

    def test_mask_batch_takes_the_first_frame(self):
        mask = torch.zeros((2, 6, 6))
        mask[0] = 1.0
        assert render.from_mask(mask).mean() == 1.0

    def test_mask_rejects_the_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\[H,W\]"):
            render.from_mask(torch.zeros((2, 2, 6, 6)))


class TestFitToGrid:
    @pytest.mark.parametrize("target", [128, 256, 512, 300])
    def test_output_is_always_a_whole_number_of_tokens(self, target):
        out = render.fit_to_grid(np.zeros((16, 16, 3), np.uint8), (16, 16), target)
        assert out.shape[0] % 16 == 0 and out.shape[1] % 16 == 0

    def test_upscaling_a_tile_keeps_hard_token_edges(self):
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        tile[0, 0] = 255
        out = render.fit_to_grid(tile, (4, 4), 128)
        cell = out.shape[0] // 4
        assert np.all(out[:cell, :cell] == 255), "nearest, not interpolated"
        assert np.all(out[:cell, cell:] == 0)

    def test_a_decoded_image_is_downscaled_to_the_same_canvas(self):
        decoded = np.zeros((256, 256, 3), dtype=np.uint8)
        assert render.fit_to_grid(decoded, (16, 16), 512).shape == (512, 512, 3)

    def test_a_painted_stroke_round_trips_to_the_token_it_covers(self):
        # the alignment guarantee: paint over token (3,5) on the canvas, get (3,5)
        canvas = render.fit_to_grid(np.zeros((16, 16, 3), np.uint8), (16, 16), 512)
        cell = canvas.shape[0] // 16
        mask = np.zeros(canvas.shape[:2], dtype=np.float32)
        mask[3 * cell:4 * cell, 5 * cell:6 * cell] = 1.0
        assert token_ops.mask_to_tokens(mask, (16, 16), 0.9).coords == [(3, 5)]


class TestOverlays:
    def test_grid_lines_do_not_change_the_size(self):
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        assert render.draw_grid(base, (8, 8)).shape == base.shape

    def test_grid_lines_land_on_token_boundaries(self):
        out = render.draw_grid(np.zeros((64, 64, 3), np.uint8), (8, 8))
        assert np.array_equal(out[8], np.tile(render.GRID_LINE, (64, 1)))
        assert out[4, 1:7].sum() == 0, "nothing drawn inside a token"

    def test_selection_marks_only_the_selected_token(self):
        selection = TokenSelection(mask=np.zeros((8, 8), dtype=bool))
        selection.mask[2, 3] = True
        out = render.draw_selection(np.zeros((64, 64, 3), np.uint8), selection)
        assert out[16:24, 24:32].sum() > 0
        assert out[0:8, 0:8].sum() == 0

    def test_selection_does_not_modify_its_input(self):
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        selection = TokenSelection(mask=np.ones((8, 8), dtype=bool))
        render.draw_selection(base, selection)
        assert base.sum() == 0

    def test_boundaries_are_drawn_between_differing_regions_only(self):
        ids = np.zeros((8, 8), dtype=np.int32)
        ids[:, 4:] = 1
        out = render.draw_region_boundaries(np.zeros((64, 64, 3), np.uint8), RegionMap(ids, 0, 0.9))
        assert out[:, 30:34].sum() > 0, "the vertical seam at column 4"
        assert out[:, 0:8].sum() == 0, "no seam inside a region"

    def test_region_colours_are_distinct_and_deterministic(self):
        colours = render.region_colours(24)
        assert len({tuple(c) for c in colours}) == 24
        np.testing.assert_array_equal(colours, render.region_colours(24))

    def test_region_render_is_grid_aligned(self):
        regions = RegionMap(np.arange(64, dtype=np.int32).reshape(8, 8), 0, 0.9)
        assert render.render_regions(regions, 256).shape == (256, 256, 3)


class TestCaption:
    def test_a_strip_is_added_below(self):
        out = render.caption(np.zeros((64, 64, 3), np.uint8), "Level 0")
        assert out.shape[1] == 64
        assert out.shape[0] > 64
        assert out[:64].sum() == 0, "the image itself is untouched"

    @pytest.mark.parametrize(
        "index,levels,expected",
        [(0, 4, "Level 0 (object/bg)"), (1, 4, "Level 1 (parts)"),
         (2, 4, "Level 2 (subparts)"), (3, 4, "Level 3 (fine)"), (5, 9, "Level 5")],
    )
    def test_level_captions(self, index, levels, expected):
        assert render.level_caption(index, levels) == expected


class TestContactSheet:
    """One image rather than a batch, because a sweep's arms only mean
    anything next to each other."""

    def frames(self, n, h=20, w=30):
        return [np.full((h, w, 3), 10 * (i + 1), dtype=np.uint8) for i in range(n)]

    def test_a_short_sweep_is_one_row(self):
        sheet = render.contact_sheet(self.frames(4))
        assert sheet.shape[0] == 20
        assert sheet.shape[1] == 4 * 30 + 3 * render.SHEET_GAP

    def test_a_long_sweep_wraps_into_a_near_square_grid(self):
        # Twelve frames in a row is 4600px wide at the real frame size, an
        # aspect ratio no screen and no page wants.
        sheet = render.contact_sheet(self.frames(12))
        assert sheet.shape[0] == 3 * 20 + 2 * render.SHEET_GAP   # 4 cols x 3 rows
        assert sheet.shape[1] == 4 * 30 + 3 * render.SHEET_GAP

    def test_the_boundary_is_where_a_row_stops_being_readable(self):
        assert render.contact_sheet(self.frames(render.SHEET_MAX_ROW)).shape[0] == 20
        assert render.contact_sheet(self.frames(render.SHEET_MAX_ROW + 1)).shape[0] > 20

    def test_frames_keep_their_reading_order(self):
        sheet = render.contact_sheet(self.frames(3))
        for i in range(3):
            assert sheet[0, i * (30 + render.SHEET_GAP), 0] == 10 * (i + 1)

    def test_a_part_row_is_filled_not_left_as_garbage(self):
        # Seven frames is a 3x3 grid with two cells empty; uninitialised memory
        # there would read as noise beside the last arm.
        sheet = render.contact_sheet(self.frames(7))
        assert np.array_equal(sheet[-1, -1], render.SHEET_BG)

    def test_one_frame_is_returned_unchanged(self):
        only = self.frames(1)[0]
        assert np.array_equal(render.contact_sheet([only]), only)

    def test_ragged_frames_are_refused_with_their_sizes(self):
        with pytest.raises(ValueError, match="same size"):
            render.contact_sheet(self.frames(2) + [np.zeros((5, 5, 3), dtype=np.uint8)])

    def test_no_frames_at_all_is_refused(self):
        with pytest.raises(ValueError, match="at least one frame"):
            render.contact_sheet([])
