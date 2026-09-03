"""The sweep planner and the measure, without ComfyUI or a GPU.

The arm list is the part of a sweep that is both easy to get subtly wrong and
expensive to get wrong: a mistake here is not a crash, it is a table whose rows
differ in two ways at once, which reads as a finding.
"""
from __future__ import annotations

import numpy as np
import pytest

from tf_nodes import measure, sweep


class TestParsingValues:
    def test_a_comma_list(self):
        assert sweep.parse_values("1,2,3", sweep.SEED) == [1, 2, 3]

    def test_whitespace_is_a_separator_too(self):
        assert sweep.parse_values("1 2  3", sweep.SEED) == [1, 2, 3]
        assert sweep.parse_values("1, 2 ,3", sweep.SEED) == [1, 2, 3]

    def test_an_inclusive_range(self):
        assert sweep.parse_values("0-3", sweep.LEVEL) == [0, 1, 2, 3]

    def test_a_descending_range(self):
        assert sweep.parse_values("3-0", sweep.LEVEL) == [3, 2, 1, 0]

    def test_ranges_and_singles_mix(self):
        assert sweep.parse_values("0-2, 7", sweep.SEED) == [0, 1, 2, 7]

    def test_strength_takes_floats(self):
        assert sweep.parse_values("0.25,0.5,1", sweep.STRENGTH) == [0.25, 0.5, 1.0]

    def test_seeds_stay_integers(self):
        # A float seed would be silently truncated by the sampler; refuse it here.
        with pytest.raises(ValueError, match="0.5"):
            sweep.parse_values("0.5", sweep.SEED)

    def test_duplicates_are_dropped_in_order(self):
        # Paying twice for the same arm is a typo, never a request.
        assert sweep.parse_values("2,1,2,1", sweep.SEED) == [2, 1]

    def test_empty_says_what_to_type(self):
        with pytest.raises(ValueError) as caught:
            sweep.parse_values("   ", sweep.STRENGTH)
        assert "0.25,0.5,0.75,1.0" in str(caught.value)

    def test_nonsense_names_the_offender_and_the_fix(self):
        with pytest.raises(ValueError) as caught:
            sweep.parse_values("1,two,3", sweep.SEED)
        assert "'two'" in str(caught.value)
        assert "1,2,3,4" in str(caught.value)


class TestPlanningArms:
    KW = dict(level=2, seed=7, strength=1.0, num_levels=4, limit=12)

    def test_a_seed_sweep_pins_everything_else(self):
        arms = sweep.plan(sweep.SEED, "1,2,3", **self.KW)
        assert [a.seed for a in arms] == [1, 2, 3]
        assert {a.level for a in arms} == {2}
        assert {a.strength for a in arms} == {1.0}

    def test_a_level_sweep_pins_everything_else(self):
        arms = sweep.plan(sweep.LEVEL, "0-3", **self.KW)
        assert [a.level for a in arms] == [0, 1, 2, 3]
        assert {a.seed for a in arms} == {7}
        assert {a.strength for a in arms} == {1.0}

    def test_a_strength_sweep_pins_everything_else(self):
        arms = sweep.plan(sweep.STRENGTH, "0.5,1.0", **self.KW)
        assert [a.strength for a in arms] == [0.5, 1.0]
        assert {a.seed for a in arms} == {7}
        assert {a.level for a in arms} == {2}

    def test_exactly_one_thing_varies_per_arm(self):
        # The repo's own experiment rule, enforced by the node rather than by
        # whoever wires the graph: two arms may differ in one field, never two.
        for axis, values in ((sweep.SEED, "1,2"), (sweep.LEVEL, "0,1"),
                             (sweep.STRENGTH, "0.5,1.0")):
            a, b = sweep.plan(axis, values, **self.KW)
            differing = [f for f in ("level", "seed", "strength")
                         if getattr(a, f) != getattr(b, f)]
            assert len(differing) == 1, f"{axis}: {differing}"

    def test_a_level_outside_the_trajectory_is_refused_not_clamped(self):
        # Clamping would turn "0-9" into four real arms and six silent copies of
        # the last one, and the table would not say so.
        with pytest.raises(ValueError, match="outside this trajectory's 0..3"):
            sweep.plan(sweep.LEVEL, "0-9", **self.KW)

    def test_a_strength_outside_zero_to_one_is_refused(self):
        with pytest.raises(ValueError, match="outside 0..1"):
            sweep.plan(sweep.STRENGTH, "0.5,2.0", **self.KW)

    def test_too_many_arms_is_refused_before_any_gpu_work(self):
        with pytest.raises(ValueError) as caught:
            sweep.plan(sweep.SEED, "0-99", **{**self.KW, "limit": 4})
        assert "arm_limit" in str(caught.value)

    def test_the_ceiling_holds_even_if_the_limit_is_raised(self):
        with pytest.raises(ValueError):
            sweep.plan(sweep.SEED, f"0-{sweep.ARM_CEILING + 10}",
                       **{**self.KW, "limit": 10_000})

    def test_an_unknown_axis_is_refused(self):
        with pytest.raises(ValueError, match="Unknown sweep axis"):
            sweep.plan("temperature", "1,2", **self.KW)

    @pytest.mark.parametrize(
        ("axis", "values", "example"),
        [(sweep.STRENGTH, "1,2,3,4", "0.25,0.5,0.75,1.0"),
         (sweep.LEVEL, "592,593", "0-3")],
    )
    def test_changing_the_axis_and_not_the_values_says_which_to_fix(
            self, axis, values, example):
        # `axis` and `values` are two widgets that have to agree, and the
        # default `values` only fits the default `axis` -- so the first failure
        # most people meet is a seed list read as strengths. "Strength 2.0 is
        # outside 0..1" is true and says nothing about which widget is wrong.
        with pytest.raises(ValueError) as caught:
            sweep.plan(axis, values, **self.KW)
        message = str(caught.value)
        assert "'axis' is set to" in message
        assert example in message


class TestTheMeasure:
    def test_identical_canvases_are_zero(self):
        rng = np.random.default_rng(0)
        canvas = rng.normal(size=(4, 4, 6)).astype(np.float32)
        assert measure.per_token_distance(canvas, canvas).max() == pytest.approx(0.0, abs=1e-6)

    def test_an_opposed_token_is_the_maximum(self):
        canvas = np.ones((1, 1, 3), dtype=np.float32)
        assert measure.per_token_distance(canvas, -canvas)[0, 0] == pytest.approx(2.0)

    def test_a_zero_vector_on_one_side_counts_as_changed(self):
        # Cosine is undefined against a zero vector; a NaN here would poison
        # every mean downstream and report as a plausible number.
        canvas = np.ones((1, 1, 3), dtype=np.float32)
        distance = measure.per_token_distance(canvas, np.zeros_like(canvas))
        assert distance[0, 0] == 1.0
        assert np.isfinite(distance).all()

    def test_two_zero_vectors_count_as_identical(self):
        zero = np.zeros((1, 1, 3), dtype=np.float32)
        assert measure.per_token_distance(zero, zero)[0, 0] == 0.0

    def test_spread_is_zero_when_every_arm_landed_in_the_same_place(self):
        canvas = np.random.default_rng(1).normal(size=(4, 4, 6)).astype(np.float32)
        assert measure.mean_pairwise_distance([canvas] * 3) == pytest.approx(0.0, abs=1e-6)

    def test_spread_is_zero_for_a_single_arm(self):
        canvas = np.ones((2, 2, 3), dtype=np.float32)
        assert measure.mean_pairwise_distance([canvas]) == 0.0

    def test_spread_grows_when_the_arms_disagree(self):
        rng = np.random.default_rng(2)
        same = [np.ones((4, 4, 6), dtype=np.float32)] * 3
        different = [rng.normal(size=(4, 4, 6)).astype(np.float32) for _ in range(3)]
        assert measure.mean_pairwise_distance(different) > measure.mean_pairwise_distance(same)
