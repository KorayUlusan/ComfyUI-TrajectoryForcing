"""The editing math, with no GPU and no ComfyUI in sight."""
from __future__ import annotations

import numpy as np
import pytest

from tf_nodes import tokens
from tf_nodes.data import RegionMap, TokenSelection


def two_region_canvas(size: int = 8, channels: int = 6) -> np.ndarray:
    """Left half one direction in feature space, right half an orthogonal one."""
    canvas = np.zeros((size, size, channels), dtype=np.float32)
    canvas[:, : size // 2, 0] = 1.0
    canvas[:, size // 2 :, 1] = 1.0
    return canvas


class TestRegionIds:
    def test_splits_a_two_region_canvas(self):
        ids = tokens.region_ids(two_region_canvas(), threshold=0.9)
        assert ids.shape == (8, 8)
        assert len(np.unique(ids)) == 2
        assert len(np.unique(ids[:, :4])) == 1
        assert np.all(ids[:, :4] != ids[0, 4])

    def test_uniform_canvas_is_one_region(self):
        canvas = np.ones((6, 6, 4), dtype=np.float32)
        assert len(np.unique(tokens.region_ids(canvas, 0.9))) == 1

    def test_threshold_of_one_shatters_random_tokens(self):
        rng = np.random.default_rng(0)
        canvas = rng.normal(size=(5, 5, 8)).astype(np.float32)
        assert len(np.unique(tokens.region_ids(canvas, 0.999))) == 25

    def test_zero_tokens_do_not_produce_nans(self):
        # A collapsed level is a result, not a crash: cosine on a zero vector is
        # 0/0, and the guard has to be in the normalisation, not downstream.
        canvas = np.zeros((4, 4, 3), dtype=np.float32)
        ids = tokens.region_ids(canvas, 0.5)
        assert np.isfinite(ids).all()

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\[H,W,C\]"):
            tokens.region_ids(np.zeros((4, 4), dtype=np.float32), 0.9)


class TestMaskToTokens:
    def test_full_mask_selects_everything(self):
        mask = np.ones((64, 64), dtype=np.float32)
        assert tokens.mask_to_tokens(mask, (8, 8), 0.5).count == 64

    def test_one_painted_cell(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:24, 32:40] = 1.0  # token (2, 4) at 8 px per token
        selection = tokens.mask_to_tokens(mask, (8, 8), 0.5)
        assert selection.coords == [(2, 4)]

    def test_coverage_threshold_rejects_a_clipped_corner(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:18, 32:34] = 1.0  # 4 of 64 pixels of token (2,4)
        assert tokens.mask_to_tokens(mask, (8, 8), 0.5).count == 0
        assert tokens.mask_to_tokens(mask, (8, 8), 0.05).count == 1

    def test_non_multiple_mask_size_still_works(self):
        mask = np.ones((100, 100), dtype=np.float32)
        assert tokens.mask_to_tokens(mask, (16, 16), 0.5).count == 256

    def test_rejects_mask_smaller_than_grid(self):
        with pytest.raises(ValueError, match="smaller than"):
            tokens.mask_to_tokens(np.ones((4, 4), dtype=np.float32), (16, 16), 0.5)


class TestParseCoords:
    def test_pairs(self):
        selection = tokens.parse_coords("3,4 7,2", (16, 16))
        assert sorted(selection.coords) == [(3, 4), (7, 2)]

    def test_inclusive_run(self):
        selection = tokens.parse_coords("1,10:12", (16, 16))
        assert sorted(selection.coords) == [(1, 10), (1, 11), (1, 12)]

    def test_tolerates_punctuation(self):
        assert tokens.parse_coords("(3,4), (5,6)", (16, 16)).count == 2

    def test_empty_is_an_empty_selection(self):
        assert tokens.parse_coords("   ", (16, 16)).count == 0

    def test_out_of_range_is_an_error_not_a_silent_drop(self):
        with pytest.raises(ValueError, match="outside"):
            tokens.parse_coords("20,1", (16, 16))
        with pytest.raises(ValueError, match="outside"):
            tokens.parse_coords("1,14:20", (16, 16))

    def test_garbage_that_is_not_empty_is_an_error(self):
        with pytest.raises(ValueError, match="No coordinates"):
            tokens.parse_coords("top left", (16, 16))


class TestSnapToRegions:
    def setup_method(self):
        self.regions = tokens.build_region_map(two_region_canvas(), level=1, threshold=0.9)

    def test_one_token_grabs_its_whole_region(self):
        seed = tokens.parse_coords("0,0", (8, 8))
        snapped = tokens.snap_to_regions(seed, self.regions, 0.0)
        assert snapped.count == 32
        assert all(c < 4 for _, c in snapped.coords)

    def test_overlap_threshold_drops_a_barely_touched_region(self):
        seed = tokens.parse_coords("0,0 0,4", (8, 8))
        # region 1 has 1/32 of its tokens picked, which does not clear 0.5
        snapped = tokens.snap_to_regions(seed, self.regions, 0.5)
        assert snapped.count == 0

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="does not match"):
            tokens.snap_to_regions(tokens.parse_coords("0,0", (16, 16)), self.regions, 0.0)


class TestCombine:
    def setup_method(self):
        self.a = tokens.parse_coords("0,0 0,1", (4, 4))
        self.b = tokens.parse_coords("0,1 0,2", (4, 4))

    @pytest.mark.parametrize(
        "op,expected",
        [
            ("union", {(0, 0), (0, 1), (0, 2)}),
            ("intersection", {(0, 1)}),
            ("difference", {(0, 0)}),
            ("symmetric difference", {(0, 0), (0, 2)}),
        ],
    )
    def test_set_ops(self, op, expected):
        assert set(tokens.combine(self.a, self.b, op).coords) == expected

    def test_invert_needs_no_second_input(self):
        assert tokens.combine(self.a, None, "invert").count == 14

    def test_binary_op_without_b_is_an_error(self):
        with pytest.raises(ValueError, match="needs a second"):
            tokens.combine(self.a, None, "union")


class TestEditPrimitive:
    def test_region_mean_collapses_the_source_to_one_vector(self):
        canvas = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
        source = tokens.parse_coords("0,0 0,1", (4, 4))
        feature = tokens.source_feature(canvas, source, "region mean")
        assert feature.shape == (1, 2)
        np.testing.assert_allclose(feature[0], (canvas[0, 0] + canvas[0, 1]) / 2)

    def test_token_cycle_keeps_every_source_vector(self):
        canvas = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
        feature = tokens.source_feature(canvas, tokens.parse_coords("0,0 0,1", (4, 4)), "token cycle")
        assert feature.shape == (2, 2)

    def test_write_replaces_only_the_selected_tokens(self):
        canvas = np.zeros((4, 4, 3), dtype=np.float32)
        target = tokens.parse_coords("1,1 2,2", (4, 4))
        out = tokens.write_feature(canvas, target, np.ones((1, 3), dtype=np.float32))
        assert out[1, 1].tolist() == [1, 1, 1]
        assert out[2, 2].tolist() == [1, 1, 1]
        assert out.sum() == 6
        assert canvas.sum() == 0, "the input canvas must not be edited in place"

    def test_one_source_vector_broadcasts_over_many_targets(self):
        canvas = np.zeros((4, 4, 3), dtype=np.float32)
        out = tokens.write_feature(
            canvas, tokens.parse_coords("0,0:3", (4, 4)), np.full((1, 3), 5.0, dtype=np.float32)
        )
        assert np.all(out[0] == 5.0)

    def test_several_source_vectors_cycle(self):
        canvas = np.zeros((1, 4, 2), dtype=np.float32)
        feature = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        out = tokens.write_feature(canvas, tokens.parse_coords("0,0:3", (1, 4)), feature)
        assert out[0, :, 0].tolist() == [1.0, 2.0, 1.0, 2.0]

    def test_strength_interpolates(self):
        canvas = np.zeros((2, 2, 2), dtype=np.float32)
        out = tokens.write_feature(
            canvas, tokens.parse_coords("0,0", (2, 2)), np.ones((1, 2), dtype=np.float32), strength=0.25
        )
        np.testing.assert_allclose(out[0, 0], [0.25, 0.25])

    def test_empty_target_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            tokens.write_feature(
                np.zeros((2, 2, 2), dtype=np.float32),
                TokenSelection(mask=np.zeros((2, 2), dtype=bool)),
                np.ones((1, 2), dtype=np.float32),
            )

    def test_empty_source_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            tokens.source_feature(
                np.zeros((2, 2, 2), dtype=np.float32),
                TokenSelection(mask=np.zeros((2, 2), dtype=bool)),
                "region mean",
            )


class TestRegionMapPayload:
    def test_mask_for_selects_named_regions(self):
        regions = RegionMap(ids=np.array([[0, 1], [1, 2]], dtype=np.int32), level=0, threshold=0.9)
        assert regions.num_regions == 3
        assert regions.mask_for([1]).sum() == 2
        assert regions.region_of(1, 1) == 2
