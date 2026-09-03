"""The payload types, especially the copy-on-write discipline.

ComfyUI hands the same output object to every downstream node and keeps it in
its execution cache, so a node that edits a payload in place corrupts its own
siblings and every future re-run of the graph. That is the class of bug these
tests exist for.
"""
from __future__ import annotations

import numpy as np
import pytest

from tf_nodes.data import LevelStack, RegionMap, TokenSelection


def stack(levels: int = 4, grid: int = 8, channels: int = 3) -> LevelStack:
    latents = np.arange(levels * grid * grid * channels, dtype=np.float32)
    return LevelStack(latents=latents.reshape(levels, grid, grid, channels), class_id=213, seed=592)


class TestLevelStack:
    def test_reports_its_shape(self):
        s = stack()
        assert (s.num_levels, s.grid) == (4, (8, 8))

    def test_rejects_the_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\[L,H,W,C\]"):
            LevelStack(latents=np.zeros((4, 8, 8), dtype=np.float32), class_id=0, seed=0)

    def test_casts_to_float32(self):
        s = LevelStack(latents=np.zeros((1, 2, 2, 2), dtype=np.float64), class_id=0, seed=0)
        assert s.latents.dtype == np.float32

    @pytest.mark.parametrize("asked,expected", [(-5, 0), (0, 0), (3, 3), (99, 3)])
    def test_level_index_is_clamped(self, asked, expected):
        assert stack().clamp(asked) == expected

    def test_with_level_does_not_touch_the_original(self):
        s = stack()
        before = s.latents.copy()
        edited = s.with_level(1, np.zeros((8, 8, 3), dtype=np.float32), "zeroed")
        np.testing.assert_array_equal(s.latents, before)
        assert edited.latents[1].sum() == 0
        np.testing.assert_array_equal(edited.latents[0], before[0])

    def test_with_level_records_the_edit(self):
        edited = stack().with_level(2, np.zeros((8, 8, 3), dtype=np.float32), "zeroed")
        assert edited.dirty_level == 2
        assert edited.history[-1] == "zeroed"

    def test_with_level_clamps_too(self):
        assert stack().with_level(99, np.zeros((8, 8, 3), dtype=np.float32), "x").dirty_level == 3

    def test_describe_flags_a_stale_stack(self):
        assert "not yet resumed" in stack().with_level(1, np.zeros((8, 8, 3), np.float32), "e").describe()
        assert "not yet resumed" not in stack().describe()

    def test_level_returns_a_view_of_the_right_slice(self):
        s = stack()
        np.testing.assert_array_equal(s.level(2), s.latents[2])


class TestTokenSelection:
    def test_counts_and_coords_agree(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 2] = mask[3, 0] = True
        selection = TokenSelection(mask=mask)
        assert selection.count == 2
        assert sorted(selection.coords) == [(1, 2), (3, 0)]

    def test_rejects_the_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\[H,W\]"):
            TokenSelection(mask=np.zeros((4, 4, 4), dtype=bool))

    def test_require_nonempty_names_what_is_empty(self):
        with pytest.raises(ValueError, match="source_tokens is empty"):
            TokenSelection(mask=np.zeros((2, 2), dtype=bool)).require_nonempty("source_tokens")

    def test_check_grid_reports_both_shapes(self):
        selection = TokenSelection(mask=np.ones((8, 8), dtype=bool))
        selection.check_grid((8, 8), "tokens")
        with pytest.raises(ValueError, match="8x8 but the latent token grid is 16x16"):
            selection.check_grid((16, 16), "tokens")


class TestRegionMap:
    def test_counts_regions_from_the_highest_id(self):
        assert RegionMap(ids=np.array([[0, 2], [2, 1]]), level=0, threshold=0.9).num_regions == 3

    def test_mask_for_accepts_several_ids(self):
        regions = RegionMap(ids=np.array([[0, 1], [2, 2]]), level=0, threshold=0.9)
        assert regions.mask_for([0, 2]).sum() == 3
        assert regions.mask_for([]).sum() == 0
