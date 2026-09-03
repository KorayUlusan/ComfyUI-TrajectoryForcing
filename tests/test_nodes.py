"""Every node's schema and execute path, against a stub pipeline.

This is the part that catches the mistakes a pure-numpy test cannot: a schema
input whose id does not match the `execute` parameter it feeds, an output tuple
of the wrong length, an optional input that is not actually optional. All of
those are invisible until someone queues the node in a browser, and none of them
needs a GPU to find.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import requires_comfy

pytestmark = requires_comfy

LEVELS, GRID, CHANNELS = 4, 8, 6


class StubPipeline:
    """Stands in for TFPipeline: same surface, deterministic, no JAX."""

    def __init__(self):
        self.calls = []

    @property
    def num_levels(self):
        return LEVELS

    def describe(self):
        return "stub pipeline"

    def generate(self, class_id, seed):
        self.calls.append(("generate", class_id, seed))
        rng = np.random.default_rng(seed * 1000 + class_id)
        return rng.normal(size=(LEVELS, GRID, GRID, CHANNELS)).astype(np.float32)

    def resume(self, latents, start_level, class_id, seed):
        self.calls.append(("resume", start_level, class_id, seed))
        out = np.array(latents, dtype=np.float32, copy=True)
        # Mimic the real thing: levels above start_level are re-sampled, levels
        # below are untouched, and each re-sampled level is conditioned on the
        # one under it. The conditioning is not decoration -- without it an edit
        # at l* leaves the final level bit-identical, so anything measuring the
        # edit's effect at the top (TF Compare Levels, TF Sweep Edit) would be
        # reading the seed and nothing else, and the test would pass regardless.
        rng = np.random.default_rng(seed)
        for level in range(start_level + 1, LEVELS):
            noise = rng.normal(size=(GRID, GRID, CHANNELS)).astype(np.float32)
            out[level] = noise + out[level - 1]
        return out

    def decode(self, latents, final_only):
        n = 1 if final_only else int(np.shape(latents)[0])
        return [np.full((32, 32, 3), 10 * (i + 1), dtype=np.uint8) for i in range(n)]

    def pca_tiles(self, latents, palette=None):
        self.calls.append(("pca_tiles", palette is not None))
        return [np.full((GRID, GRID, 3), 20 * (i + 1), dtype=np.uint8) for i in range(len(latents))]

    def fit_palette(self, stacks):
        self.calls.append(("fit_palette", len(stacks)))
        return "palette"


@pytest.fixture
def pipeline():
    return StubPipeline()


@pytest.fixture
def levels(pipeline):
    from tf_nodes.data import LevelStack

    return LevelStack(latents=pipeline.generate(213, 1), class_id=213, seed=1, history=("stub",))


@pytest.fixture
def two_region_levels():
    """A stack whose level 2 splits cleanly into a left and a right region."""
    from tf_nodes.data import LevelStack

    latents = np.zeros((LEVELS, GRID, GRID, CHANNELS), dtype=np.float32)
    latents[:, :, : GRID // 2, 0] = 1.0
    latents[:, :, GRID // 2 :, 1] = 1.0
    return LevelStack(latents=latents, class_id=7, seed=3)


@pytest.fixture(scope="session")
def node_classes(extension):
    """Register the model folder, then hand back every node class."""
    import asyncio

    asyncio.run(extension.TrajectoryForcingExtension().on_load())
    from tf_nodes import nodes

    return nodes()


def ui_text(out) -> str:
    """What the node shows in its own body.

    `ui` is a plain dict rather than a _UIOutput because a node that produces
    both a picture and a number should show both, and each preview class owns
    only its own key.
    """
    payload = out.ui.as_dict() if hasattr(out.ui, "as_dict") else (out.ui or {})
    return " ".join(payload.get("text", ()))


def ui_images(out) -> list:
    payload = out.ui.as_dict() if hasattr(out.ui, "as_dict") else (out.ui or {})
    return list(payload.get("images", ()))


def node(node_classes, node_id):
    for cls in node_classes:
        if cls.define_schema().node_id == node_id:
            return cls
    raise AssertionError(f"no node with id {node_id}")


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
class TestSchemas:
    def test_every_node_has_a_valid_schema(self, node_classes):
        for cls in node_classes:
            cls.GET_SCHEMA()

    def test_node_ids_and_display_names_are_unique(self, node_classes):
        ids = [c.define_schema().node_id for c in node_classes]
        names = [c.define_schema().display_name for c in node_classes]
        assert len(set(ids)) == len(ids)
        assert len(set(names)) == len(names)

    def test_every_node_id_is_namespaced(self, node_classes):
        # ComfyUI has one flat node-id space shared with every other extension.
        for cls in node_classes:
            assert cls.define_schema().node_id.startswith("TF")

    def test_execute_accepts_exactly_the_schema_inputs(self, node_classes):
        for cls in node_classes:
            schema = cls.define_schema()
            params = inspect.signature(cls.execute.__func__).parameters
            accepted = set(params) - {"cls"}
            declared = {i.id for i in schema.inputs}
            assert declared <= accepted, (
                f"{schema.node_id}: execute is missing {declared - accepted}")
            assert accepted <= declared, (
                f"{schema.node_id}: execute has extra {accepted - declared}")
            for input_ in schema.inputs:
                if input_.optional:
                    assert params[input_.id].default is not inspect.Parameter.empty, (
                        f"{schema.node_id}: optional input {input_.id!r} has no default"
                    )

    def test_every_node_documents_itself(self, node_classes):
        for cls in node_classes:
            schema = cls.define_schema()
            assert schema.description, f"{schema.node_id} has no description"
            assert schema.category.startswith("TrajectoryForcing")


# ---------------------------------------------------------------------------
# generate / decode / preview
# ---------------------------------------------------------------------------
class TestGenerateAndDecode:
    def test_generate_records_class_and_seed(self, node_classes, pipeline):
        out = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=42, seed=7)
        stack = out[0]
        assert stack.latents.shape == (LEVELS, GRID, GRID, CHANNELS)
        assert (stack.class_id, stack.seed) == (42, 7)
        assert stack.dirty_level is None
        assert "class=42" in stack.history[0]

    def test_imagenet_class_round_trips(self, node_classes):
        # Taken from the node's own option list, not written out: ComfyUI
        # validates a combo value against that list and rejects the whole prompt
        # if it is one word off, which is how a hand-written workflow broke.
        cls = node(node_classes, "TFImageNetClass")
        options = cls.define_schema().inputs[0].options
        assert len(options) == 1000
        chosen = next(o for o in options if o.startswith("213 - "))
        class_id, name = cls.execute(class_name=chosen)
        assert class_id == 213
        assert chosen == f"213 - {name}"

    @pytest.mark.parametrize(
        "which,expected", [("all levels", LEVELS), ("final level only", 1), ("level 1", 1)]
    )
    def test_decode_frame_counts(self, node_classes, pipeline, levels, which, expected):
        images, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which=which, label_levels=False, level_override=-1,
            sheet_layout="separate frames")
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] == expected
        assert images.dtype == torch.float32
        assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0

    def test_decode_captions_add_a_strip(self, node_classes, pipeline, levels):
        plain, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="all levels", label_levels=False, level_override=-1,
            sheet_layout="separate frames")
        labelled, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="all levels", label_levels=True, level_override=-1,
            sheet_layout="separate frames")
        assert labelled.shape[1] > plain.shape[1]
        assert labelled.shape[2] == plain.shape[2]

    def test_decode_warns_when_the_stack_is_stale(self, node_classes, pipeline, levels):
        edited = levels.with_level(1, levels.level(1), "edit")
        _, warning = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=edited, which="all levels", label_levels=False, level_override=-1,
            sheet_layout="separate frames")
        assert "stale" in warning

    def test_decode_of_a_level_below_the_edit_does_not_warn(self, node_classes, pipeline, levels):
        edited = levels.with_level(2, levels.level(2), "edit")
        _, warning = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=edited, which="level 1", label_levels=False, level_override=-1,
            sheet_layout="separate frames")
        assert warning == ""

    def test_latent_preview_fits_the_requested_size(self, node_classes, pipeline, levels):
        images, = node(node_classes, "TFLatentPreview").execute(
            pipeline=pipeline, levels=levels, which="all levels", size=256,
            label_levels=False, level_override=-1, palette_from=None, sheet_layout="separate frames")
        assert images.shape == (LEVELS, 256, 256, 3)

    def test_palette_from_fits_jointly(self, node_classes, pipeline, levels):
        node(node_classes, "TFLatentPreview").execute(
            pipeline=pipeline, levels=levels, which="all levels", size=128,
            label_levels=False, level_override=-1, palette_from=levels, sheet_layout="separate frames")
        assert ("fit_palette", 2) in pipeline.calls
        assert ("pca_tiles", True) in pipeline.calls

    def test_levels_info_reports_the_history(self, node_classes, levels):
        info, n, class_id, class_name, seed = node(node_classes, "TFLevelsInfo").execute(levels=levels)
        assert (n, class_id, seed) == (LEVELS, 213, 1)
        assert "stub" in info


# ---------------------------------------------------------------------------
# regions and selections
# ---------------------------------------------------------------------------
class TestRegionNodes:
    def test_region_map_finds_two_regions(self, node_classes, two_region_levels):
        regions, image, count, level_out = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=2, cosine_threshold=0.9, size=256
        )
        assert count == 2
        assert regions.level == 2
        assert image.shape == (1, 256, 256, 3)

    def test_tokens_from_mask_and_back_to_coords(self, node_classes, levels):
        mask = torch.zeros((1, 64, 64))
        mask[0, 0:8, 0:8] = 1.0  # token (0,0) on an 8x8 grid at 8 px per token
        tokens, count, info = node(node_classes, "TFTokensFromMask").execute(
            mask=mask, levels=levels, coverage=0.5, regions=None, region_overlap=0.3
        )
        assert count == 1
        assert tokens.coords == [(0, 0)]
        assert "1 tokens" in info

    def test_tokens_from_mask_snapped_to_a_region(self, node_classes, two_region_levels):
        regions, _, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=2, cosine_threshold=0.9, size=128
        )
        mask = torch.zeros((1, 64, 64))
        mask[0, 0:8, 0:8] = 1.0
        tokens, count, _ = node(node_classes, "TFTokensFromMask").execute(
            mask=mask, levels=two_region_levels, coverage=0.5, regions=regions, region_overlap=0.0
        )
        assert count == GRID * GRID // 2
        assert all(c < GRID // 2 for _, c in tokens.coords)

    @staticmethod
    def _blocked_reason(out, schema):
        """The stop is one silent blocker per output, plus the reason as node text.

        A message-carrying `block_execution` would instead be reported to the
        browser as "Node threw an error during execution", and only on the first
        run -- the second finds the node cached and blocks in silence. Both
        halves of that are what these tests pin down.
        """
        from comfy_execution.graph_utils import ExecutionBlocker

        assert out.block_execution is None, "a message here becomes an error dialog"
        assert out.result is not None and len(out.result) == len(schema.outputs), (
            "one blocker per declared output; fewer leaves the node's outputs "
            "empty and ComfyUI's cache bookkeeping raises IndexError"
        )
        assert all(isinstance(r, ExecutionBlocker) for r in out.result)
        assert all(r.message is None for r in out.result), "a message blocks loudly"
        return ui_text(out)

    def test_an_unpainted_mask_stops_quietly_and_says_why(self, node_classes, levels):
        cls = node(node_classes, "TFTokensFromMask")
        out = cls.execute(
            mask=torch.zeros((1, 64, 64)), levels=levels, coverage=0.5,
            regions=None, region_overlap=0.3,
        )
        assert "Nothing painted yet" in self._blocked_reason(out, cls.define_schema())

    def test_the_notice_survives_a_cached_rerun(self, node_classes):
        # Without has_intermediate_output the text shows on the first run and is
        # gone on the second, when the node is served from cache.
        assert node(node_classes, "TFTokensFromMask").define_schema().has_intermediate_output

    def test_a_stroke_too_thin_to_count_says_so(self, node_classes, levels):
        cls = node(node_classes, "TFTokensFromMask")
        mask = torch.zeros((1, 64, 64))
        mask[0, 0:2, 0:2] = 1.0  # 4 of a token's 64 pixels
        out = cls.execute(
            mask=mask, levels=levels, coverage=0.5, regions=None, region_overlap=0.3,
        )
        assert "no token reached 0.50 coverage" in self._blocked_reason(out, cls.define_schema())

    def test_a_stroke_missing_every_region_says_so(self, node_classes, two_region_levels):
        cls = node(node_classes, "TFTokensFromMask")
        regions, _, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=2, cosine_threshold=0.9, size=128
        )
        mask = torch.zeros((1, 64, 64))
        mask[0, 0:8, 0:8] = 1.0  # one token of a 32-token region
        out = cls.execute(
            mask=mask, levels=two_region_levels, coverage=0.5,
            regions=regions, region_overlap=0.9,
        )
        reason = self._blocked_reason(out, cls.define_schema())
        assert "no region reached the 0.90 overlap threshold" in reason

    def test_a_real_selection_reports_itself_in_the_node(self, node_classes, levels):
        mask = torch.zeros((1, 64, 64))
        mask[0, 0:8, 0:8] = 1.0
        out = node(node_classes, "TFTokensFromMask").execute(
            mask=mask, levels=levels, coverage=0.5, regions=None, region_overlap=0.3,
        )
        assert out.block_execution is None
        assert out[1] == 1
        assert "1 tokens selected" in ui_text(out)

    def test_tokens_from_coords(self, node_classes, levels):
        tokens, count, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="1,2 3,4:6", levels=levels, regions=None
        )
        assert count == 4
        assert (1, 2) in tokens.coords

    def test_tokens_combine(self, node_classes, levels):
        coords_node = node(node_classes, "TFTokensFromCoords")
        a, _, _ = coords_node.execute(coords="0,0 0,1", levels=levels, regions=None)
        b, _, _ = coords_node.execute(coords="0,1", levels=levels, regions=None)
        _, count = node(node_classes, "TFTokensCombine").execute(a=a, operation="difference", b=b)
        assert count == 1

    def test_tokens_preview_lists_its_coords(self, node_classes, levels):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="2,3", levels=levels, regions=None
        )
        image, coords = node(node_classes, "TFTokensPreview").execute(tokens=selection, size=128)
        assert coords == "2,3"
        assert image.shape == (1, 128, 128, 3)

    def test_level_canvas_both_views(self, node_classes, pipeline, levels):
        for view in ("latent PCA", "decoded RGB"):
            image, level = node(node_classes, "TFLevelCanvas").execute(
                pipeline=pipeline, levels=levels, level=2, view=view,
                draw_grid=True, label_coords=True, size=256, regions=None, highlight=None,
            )
            assert image.shape == (1, 256, 256, 3)
            assert level == 2

    def test_level_canvas_clamps_an_out_of_range_level(self, node_classes, pipeline, levels):
        _, level = node(node_classes, "TFLevelCanvas").execute(
            pipeline=pipeline, levels=levels, level=15, view="latent PCA",
            draw_grid=False, label_coords=False, size=128, regions=None, highlight=None,
        )
        assert level == LEVELS - 1

    def test_level_canvas_rejects_a_mismatched_highlight(self, node_classes, pipeline, levels):
        from tf_nodes.data import TokenSelection

        with pytest.raises(ValueError, match="token grid is"):
            node(node_classes, "TFLevelCanvas").execute(
                pipeline=pipeline, levels=levels, level=0, view="latent PCA",
                draw_grid=False, label_coords=False, size=128, regions=None,
                highlight=TokenSelection(mask=np.ones((16, 16), dtype=bool)),
            )


# ---------------------------------------------------------------------------
# the edits
# ---------------------------------------------------------------------------
class TestEditNodes:
    def _tokens(self, node_classes, levels, coords):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords=coords, levels=levels, regions=None
        )
        return selection

    def test_feature_edit_touches_only_the_target_tokens(self, node_classes, levels):
        target = self._tokens(node_classes, levels, "0,0 0,1")
        source = self._tokens(node_classes, levels, "5,5")
        out, info = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=2, target_tokens=target, source_tokens=source,
            source_mode="region mean", strength=1.0, source_level=2, source_levels=None,
        )
        before, after = levels.level(2), out.level(2)
        np.testing.assert_allclose(after[0, 0], before[5, 5])
        np.testing.assert_allclose(after[0, 1], before[5, 5])
        np.testing.assert_allclose(after[3:], before[3:])
        assert out.dirty_level == 2
        assert "feature edit" in info

    def test_feature_edit_leaves_other_levels_alone(self, node_classes, levels):
        out, _ = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=1,
            target_tokens=self._tokens(node_classes, levels, "0,0"),
            source_tokens=self._tokens(node_classes, levels, "5,5"),
            source_mode="region mean", strength=1.0, source_level=1, source_levels=None,
        )
        for other in (0, 2, 3):
            np.testing.assert_allclose(out.level(other), levels.level(other))

    def test_feature_edit_does_not_mutate_its_input(self, node_classes, levels):
        original = levels.latents.copy()
        node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=2,
            target_tokens=self._tokens(node_classes, levels, "0,0"),
            source_tokens=self._tokens(node_classes, levels, "5,5"),
            source_mode="region mean", strength=1.0, source_level=2, source_levels=None,
        )
        np.testing.assert_array_equal(levels.latents, original)

    def test_feature_edit_across_two_trajectories(self, node_classes, pipeline, levels):
        from tf_nodes.data import LevelStack

        other = LevelStack(latents=pipeline.generate(9, 2), class_id=9, seed=2)
        out, info = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=2,
            target_tokens=self._tokens(node_classes, levels, "0,0"),
            source_tokens=self._tokens(node_classes, levels, "1,1"),
            source_mode="region mean", strength=1.0, source_level=3, source_levels=other,
        )
        np.testing.assert_allclose(out.level(2)[0, 0], other.level(3)[1, 1])
        assert "class 9" in info

    def test_feature_edit_strength_interpolates(self, node_classes, levels):
        out, _ = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=2,
            target_tokens=self._tokens(node_classes, levels, "0,0"),
            source_tokens=self._tokens(node_classes, levels, "5,5"),
            source_mode="region mean", strength=0.5, source_level=2, source_levels=None,
        )
        expected = 0.5 * levels.level(2)[0, 0] + 0.5 * levels.level(2)[5, 5]
        np.testing.assert_allclose(out.level(2)[0, 0], expected, rtol=1e-6)

    def test_feature_edit_rejects_an_empty_target(self, node_classes, levels):
        with pytest.raises(ValueError, match="empty"):
            node(node_classes, "TFFeatureEdit").execute(
                levels=levels, level=2,
                target_tokens=self._tokens(node_classes, levels, ""),
                source_tokens=self._tokens(node_classes, levels, "5,5"),
                source_mode="region mean", strength=1.0, source_level=2, source_levels=None,
            )

    def test_shape_edit_moves_a_boundary_without_changing_features(self, node_classes, two_region_levels):
        regions, _, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=2, cosine_threshold=0.9, size=128
        )
        boundary = self._tokens(node_classes, two_region_levels, "0,4")   # leftmost right-region token
        receiving = self._tokens(node_classes, two_region_levels, "0,0")  # inside the left region
        out, info = node(node_classes, "TFShapeEdit").execute(
            levels=two_region_levels, level=2, regions=regions,
            boundary_tokens=boundary, receiving_tokens=receiving, strength=1.0,
        )
        # the handed-over token now carries the left region's feature exactly
        np.testing.assert_allclose(out.level(2)[0, 4], two_region_levels.level(2)[0, 0])
        assert out.dirty_level == 2
        assert "shape edit" in info

    def test_shape_edit_rejects_a_region_map_from_another_level(self, node_classes, two_region_levels):
        regions, _, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=1, cosine_threshold=0.9, size=128
        )
        with pytest.raises(ValueError, match="built on level 1"):
            node(node_classes, "TFShapeEdit").execute(
                levels=two_region_levels, level=2, regions=regions,
                boundary_tokens=self._tokens(node_classes, two_region_levels, "0,4"),
                receiving_tokens=self._tokens(node_classes, two_region_levels, "0,0"),
                strength=1.0,
            )

    def test_shape_edit_rejects_one_region_talking_to_itself(self, node_classes, two_region_levels):
        regions, _, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=2, cosine_threshold=0.9, size=128
        )
        with pytest.raises(ValueError, match="two different regions"):
            node(node_classes, "TFShapeEdit").execute(
                levels=two_region_levels, level=2, regions=regions,
                boundary_tokens=self._tokens(node_classes, two_region_levels, "0,1"),
                receiving_tokens=self._tokens(node_classes, two_region_levels, "0,0"),
                strength=1.0,
            )


class TestResume:
    def _edited(self, node_classes, levels, level=1):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=levels, regions=None
        )
        out, _ = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=level, target_tokens=selection, source_tokens=selection,
            source_mode="region mean", strength=1.0, source_level=level, source_levels=None,
        )
        return out

    def test_resume_follows_the_edit(self, node_classes, pipeline, levels):
        edited = self._edited(node_classes, levels, level=1)
        out, info = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=-1, class_id=-1, seed=5
        )
        assert ("resume", 1, 213, 5) in pipeline.calls, "the edit's level and the stack's class win"
        assert out.dirty_level is None
        assert "resume from level 1" in info

    def test_resume_keeps_levels_below_l_star(self, node_classes, pipeline, levels):
        edited = self._edited(node_classes, levels, level=2)
        out, _ = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=-1, class_id=-1, seed=5
        )
        for below in (0, 1, 2):
            np.testing.assert_allclose(out.level(below), edited.level(below))
        assert not np.allclose(out.level(3), edited.level(3))

    def test_resume_can_override_the_class(self, node_classes, pipeline, levels):
        node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=levels, level=1, class_id=77, seed=2
        )
        assert ("resume", 1, 77, 2) in pipeline.calls

    def test_resuming_with_nothing_to_follow_is_an_error(self, node_classes, pipeline, levels):
        with pytest.raises(ValueError, match="no edit to follow"):
            node(node_classes, "TFResumeFromLevel").execute(
                pipeline=pipeline, levels=levels, level=-1, class_id=-1, seed=1
            )

    def test_resuming_above_a_pending_edit_keeps_it_marked_stale(self, node_classes, pipeline, levels):
        # Resuming refreshes only the levels above the start, so an edit further
        # down is still unpropagated -- clearing the marker would hide that.
        edited = self._edited(node_classes, levels, level=1)
        out, _ = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=2, class_id=-1, seed=5
        )
        assert out.dirty_level == 1

    def test_resuming_below_a_pending_edit_settles_it(self, node_classes, pipeline, levels):
        edited = self._edited(node_classes, levels, level=2)
        out, _ = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=1, class_id=-1, seed=5
        )
        assert out.dirty_level is None

    def test_resuming_at_the_last_level_says_so(self, node_classes, pipeline, levels):
        _, info = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=levels, level=LEVELS - 1, class_id=-1, seed=1
        )
        assert "nothing above it" in info


class TestSaveLoad:
    def test_round_trip_preserves_everything(self, node_classes, levels, tmp_path, monkeypatch):
        import tf_nodes.nodes_io as nodes_io

        monkeypatch.setattr(nodes_io, "_output_dir", lambda: tmp_path)
        edited = levels.with_level(1, levels.level(1), "an edit")
        path, = node(node_classes, "TFSaveLevels").execute(levels=edited, name="round trip!", overwrite=True)
        assert path.endswith("roundtrip.npz")

        loaded, info = node(node_classes, "TFLoadLevels").execute(file="roundtrip.npz", path_override="")
        np.testing.assert_array_equal(loaded.latents, edited.latents)
        assert (loaded.class_id, loaded.seed) == (edited.class_id, edited.seed)
        assert loaded.dirty_level == 1
        assert "an edit" in info

    def test_no_overwrite_appends_a_suffix(self, node_classes, levels, tmp_path, monkeypatch):
        import tf_nodes.nodes_io as nodes_io

        monkeypatch.setattr(nodes_io, "_output_dir", lambda: tmp_path)
        first, = node(node_classes, "TFSaveLevels").execute(levels=levels, name="t", overwrite=False)
        second, = node(node_classes, "TFSaveLevels").execute(levels=levels, name="t", overwrite=False)
        assert first != second
        assert second.endswith("t-001.npz")

    def test_missing_file_is_a_clear_error(self, node_classes, tmp_path, monkeypatch):
        import tf_nodes.nodes_io as nodes_io

        monkeypatch.setattr(nodes_io, "_output_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="No saved trajectory"):
            node(node_classes, "TFLoadLevels").execute(file="nope.npz", path_override="")


class TestPipelineTravelsWithTheTrajectory:
    """TF_LEVELS carries the pipeline that made it, so consumers need no wire.

    Six TF_PIPELINE wires crossed one example workflow before this; the socket
    survives only as an override and for trajectories restored from disk.
    """

    def test_generate_attaches_the_pipeline(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(
            pipeline=pipeline, class_id=1, seed=1)
        assert stack.pipeline is pipeline

    def test_decode_finds_it_without_a_wire(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(
            pipeline=pipeline, class_id=1, seed=1)
        images, _ = node(node_classes, "TFDecode").execute(
            levels=stack, which="all levels", label_levels=False, level_override=-1,
            sheet_layout="separate frames")
        assert images.shape[0] == LEVELS

    def test_resume_passes_it_on(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(
            pipeline=pipeline, class_id=1, seed=1)
        out, _ = node(node_classes, "TFResumeFromLevel").execute(
            levels=stack, level=1, class_id=-1, seed=2)
        assert out.pipeline is pipeline

    def test_an_explicit_pipeline_overrides_the_carried_one(self, node_classes, pipeline):
        other = StubPipeline()
        stack, = node(node_classes, "TFGenerate").execute(
            pipeline=pipeline, class_id=1, seed=1)
        node(node_classes, "TFDecode").execute(
            levels=stack, which="final level only", label_levels=False,
            level_override=-1, pipeline=other, sheet_layout="separate frames")
        assert any(c[0] == "generate" for c in pipeline.calls)
        assert not other.calls, "decode uses no recorded call, but the override must be the one used"

    def test_a_trajectory_from_disk_says_what_to_wire(self, node_classes, levels):
        # `levels` is built directly, as TF Load Levels builds one: no pipeline.
        assert levels.pipeline is None
        with pytest.raises(ValueError, match="Wire TF Load Pipeline"):
            node(node_classes, "TFDecode").execute(
                levels=levels, which="all levels", label_levels=False, level_override=-1,
                sheet_layout="separate frames")


class TestLevelAgreement:
    """A selection snapped to one level's regions must not be applied at another.

    Every level shares a token grid, so the shapes match and the result looks
    plausible while being the wrong region entirely. TF Shape Edit always caught
    this; TF Feature Edit silently accepted it.
    """

    def _regions_at(self, node_classes, stack, level):
        regions, _, _, level_out = node(node_classes, "TFRegionMap").execute(
            levels=stack, level=level, cosine_threshold=0.9, size=64)
        assert level_out == level, "TF Region Map reports the level it clustered"
        return regions

    def test_region_map_level_output_can_drive_the_edit(self, node_classes, two_region_levels):
        regions = self._regions_at(node_classes, two_region_levels, 2)
        assert regions.level == 2

    def test_feature_edit_rejects_a_selection_from_another_level(
            self, node_classes, two_region_levels):
        regions = self._regions_at(node_classes, two_region_levels, 2)
        target, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=two_region_levels, regions=regions)
        source, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="7,7", levels=two_region_levels)
        with pytest.raises(ValueError, match="level 2's regions but is being applied at level 1"):
            node(node_classes, "TFFeatureEdit").execute(
                levels=two_region_levels, level=1, target_tokens=target, source_tokens=source,
                source_mode="region mean", strength=1.0, source_level=1, source_levels=None)

    def test_a_matching_level_is_fine(self, node_classes, two_region_levels):
        regions = self._regions_at(node_classes, two_region_levels, 2)
        target, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=two_region_levels, regions=regions)
        source, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="7,7", levels=two_region_levels)
        out, _ = node(node_classes, "TFFeatureEdit").execute(
            levels=two_region_levels, level=2, target_tokens=target, source_tokens=source,
            source_mode="region mean", strength=1.0, source_level=2, source_levels=None)
        assert out.dirty_level == 2

    def test_an_unsnapped_selection_is_accepted_anywhere(self, node_classes, two_region_levels):
        # Typed coordinates with no region map name no level, so they carry none.
        plain, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=two_region_levels)
        assert plain.level is None
        node(node_classes, "TFFeatureEdit").execute(
            levels=two_region_levels, level=0, target_tokens=plain, source_tokens=plain,
            source_mode="region mean", strength=1.0, source_level=0, source_levels=None)

    def test_combining_selections_from_two_levels_is_refused(
            self, node_classes, two_region_levels):
        a_tokens, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=two_region_levels,
            regions=self._regions_at(node_classes, two_region_levels, 1))
        b_tokens, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=two_region_levels,
            regions=self._regions_at(node_classes, two_region_levels, 2))
        with pytest.raises(ValueError, match="level 1's regions with one from level 2"):
            node(node_classes, "TFTokensCombine").execute(
                a=a_tokens, operation="union", b=b_tokens)


class TestCoordinateErgonomics:
    def test_coords_need_no_levels_wire(self, node_classes):
        tokens, count, _ = node(node_classes, "TFTokensFromCoords").execute(coords="15,15")
        assert count == 1 and tokens.mask.shape == (16, 16), "defaults to the released 16x16 grid"

    def test_coords_still_honour_a_wired_grid(self, node_classes, levels):
        with pytest.raises(ValueError, match="outside"):
            node(node_classes, "TFTokensFromCoords").execute(coords="15,15", levels=levels)

    def test_the_canvas_can_number_its_axes(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=1, seed=1)
        kw = dict(levels=stack, level=2, view="latent PCA", draw_grid=True,
                  size=256, regions=None, highlight=None)
        plain, _ = node(node_classes, "TFLevelCanvas").execute(label_coords=False, **kw)
        ticked, _ = node(node_classes, "TFLevelCanvas").execute(label_coords=True, **kw)
        assert plain.shape == ticked.shape
        assert not torch.equal(plain, ticked), "labels should actually be drawn"

    def test_levels_info_names_the_class(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=213, seed=1)
        _, _, class_id, class_name, _ = node(node_classes, "TFLevelsInfo").execute(levels=stack)
        assert class_id == 213
        assert class_name == "Irish setter"


class TestCompareLevels:
    """The node that answers "did the edit do anything, and where"."""

    def _edit_and_resume(self, node_classes, pipeline, level=2):
        before, = node(node_classes, "TFGenerate").execute(
            pipeline=pipeline, class_id=213, seed=1)
        target, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="1,1", levels=before)
        # A different source token: writing a token's own value onto itself is
        # an identity edit, and then only the re-sampled levels above it move.
        source, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="5,5", levels=before)
        edited, _ = node(node_classes, "TFFeatureEdit").execute(
            levels=before, level=level, target_tokens=target, source_tokens=source,
            source_mode="region mean", strength=1.0, source_level=level, source_levels=None)
        after, _ = node(node_classes, "TFResumeFromLevel").execute(
            levels=edited, level=-1, class_id=-1, seed=1)
        return before, after

    def test_identical_trajectories_report_no_change(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=1, seed=1)
        report, images, changed, peak = node(node_classes, "TFCompareLevels").execute(
            before=stack, after=stack, size=256, decode_difference=False, sheet_layout="separate frames")
        assert changed == 0
        assert peak == 0.0
        assert "Nothing changed" in report
        assert images.shape[0] == LEVELS, "one heatmap tile per level"

    def test_it_localises_the_edit_to_the_right_levels(self, node_classes, pipeline):
        before, after = self._edit_and_resume(node_classes, pipeline, level=2)
        report, _, changed, peak = node(node_classes, "TFCompareLevels").execute(
            before=before, after=after, size=256, decode_difference=False, sheet_layout="separate frames")
        assert changed > 0 and peak > 0
        # the stub re-samples only above the edited level, so 0..1 must be clean
        assert "First level that differs: 2." in report
        assert "Levels 0..1 are untouched" in report

    def test_the_report_has_a_row_per_level(self, node_classes, pipeline):
        before, after = self._edit_and_resume(node_classes, pipeline)
        report, _, _, _ = node(node_classes, "TFCompareLevels").execute(
            before=before, after=after, size=256, decode_difference=False, sheet_layout="separate frames")
        for level in range(LEVELS):
            assert f"Level {level}" in report

    def test_decoding_the_difference_adds_a_frame(self, node_classes, pipeline):
        before, after = self._edit_and_resume(node_classes, pipeline)
        plain, _, _, _ = node(node_classes, "TFCompareLevels").execute(
            before=before, after=after, size=256, decode_difference=False, sheet_layout="separate frames")
        report, images, _, _ = node(node_classes, "TFCompareLevels").execute(
            before=before, after=after, size=256, decode_difference=True, sheet_layout="separate frames")
        assert images.shape[0] == LEVELS + 1
        assert "mean |delta|" in report and "mean |delta|" not in plain

    def test_frames_are_all_the_same_size(self, node_classes, pipeline):
        # to_image stacks them into one batch; numpy will not stack ragged frames,
        # and a decoded difference is a different size from a heatmap tile.
        before, after = self._edit_and_resume(node_classes, pipeline)
        _, images, _, _ = node(node_classes, "TFCompareLevels").execute(
            before=before, after=after, size=640, decode_difference=True, sheet_layout="separate frames")
        assert images.shape[0] == LEVELS + 1

    def test_mismatched_trajectories_are_refused(self, node_classes, pipeline):
        from tf_nodes.data import LevelStack

        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=1, seed=1)
        odd = LevelStack(latents=np.zeros((2, 4, 4, 3), np.float32), class_id=1, seed=1)
        with pytest.raises(ValueError, match="not two versions of the same trajectory"):
            node(node_classes, "TFCompareLevels").execute(
                before=stack, after=odd, size=256, decode_difference=False, sheet_layout="separate frames")

    def test_a_zero_token_does_not_produce_a_nan(self, node_classes, pipeline):
        # A collapsed token has no direction, so cosine is 0/0. That is a result
        # to report, not a NaN to average into every statistic.
        from tf_nodes.data import LevelStack

        a = np.ones((LEVELS, GRID, GRID, CHANNELS), np.float32)
        b = a.copy()
        b[2, 0, 0] = 0.0
        before = LevelStack(latents=a, class_id=1, seed=1, pipeline=pipeline)
        after = LevelStack(latents=b, class_id=1, seed=1, pipeline=pipeline)
        report, _, changed, peak = node(node_classes, "TFCompareLevels").execute(
            before=before, after=after, size=256, decode_difference=False, sheet_layout="separate frames")
        assert changed == 1
        assert np.isfinite(peak) and peak == 1.0
        assert "nan" not in report.lower()


class TestSweepEdit:
    """One edit run many times, varying exactly one thing.

    The node the extension exists for: the same edit across seeds or across
    l*, tabulated. Doing it by hand means duplicating the chain per arm.
    """

    DEFAULTS = dict(
        axis="seed", values="1,2,3", level=2, seed=7, strength=1.0,
        source_mode="region mean", baseline=True, decode=True,
        arm_limit=12, output_arm=0, sheet_layout="contact sheet", size=128,
        source_levels=None,
    )

    @pytest.fixture
    def levels(self, pipeline):
        """Overrides the module fixture: a stack that carries its pipeline.

        The module-level one deliberately has none, to stand in for a
        trajectory restored from disk. A sweep needs the real thing, because
        it re-samples once per arm rather than reading the latents it was given.
        """
        from tf_nodes.data import LevelStack

        return LevelStack(
            latents=pipeline.generate(213, 1), class_id=213, seed=1,
            history=("stub",), pipeline=pipeline,
        )

    def _run(self, node_classes, levels, **overrides):
        target, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="1,1 1,2", levels=levels)
        source, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="5,5", levels=levels)
        kwargs = {**self.DEFAULTS, "target_tokens": target, "source_tokens": source}
        kwargs.update(overrides)
        return node(node_classes, "TFSweep").execute(levels=levels, **kwargs)

    # ----- the loop itself -----
    def test_one_arm_per_value(self, node_classes, levels):
        report, sheet, _, arms, _ = self._run(
            node_classes, levels, values="1,2,3", sheet_layout="separate frames")
        assert arms == 3
        assert sheet.shape[0] == 4, "the no-edit baseline, then one frame per arm"
        for seed in (1, 2, 3):
            assert f"\n{seed:<12g}" in f"\n{report}" or f"\n{seed} " in report

    def test_each_arm_is_edited_and_resumed_at_its_own_value(self, node_classes, pipeline, levels):
        self._run(node_classes, levels, values="4,5")
        for seed in (4, 5):
            assert ("resume", 2, 213, seed) in pipeline.calls

    def test_a_level_sweep_resumes_from_each_level(self, node_classes, pipeline, levels):
        self._run(node_classes, levels, axis="level (l*)", values="0-2")
        for level in (0, 1, 2):
            assert ("resume", level, 213, 7) in pipeline.calls

    def test_the_baseline_is_resumed_once_per_distinct_arm_not_once_per_arm(
            self, node_classes, pipeline, levels):
        # A strength sweep pins the level and the seed, so every arm shares one
        # baseline. Re-resuming it per arm would double the sweep's cost for
        # nothing -- and on real hardware that is the whole run.
        self._run(node_classes, levels, axis="strength", values="0.25,0.5,0.75,1.0")
        resumes = [c for c in pipeline.calls if c[0] == "resume"]
        assert len(resumes) == 5, f"4 arms + 1 shared baseline, got {resumes}"

    def test_baseline_off_skips_the_extra_resumes(self, node_classes, pipeline, levels):
        self._run(node_classes, levels, values="1,2", baseline=False)
        assert len([c for c in pipeline.calls if c[0] == "resume"]) == 2

    # ----- the numbers -----
    def test_every_arm_is_measured_against_its_own_baseline(self, node_classes, levels):
        # Not against the input trajectory: with a seed axis that folds the
        # seed's own effect into every row, and the table then says the edit
        # did something when only the re-sample did.
        report, _, _, _, _ = self._run(node_classes, levels, values="1,2,3")
        assert "resumed identically without the edit" in report
        off, _, _, _, _ = self._run(node_classes, levels, values="1,2,3", baseline=False)
        assert "baseline off" in off

    def test_the_table_has_a_row_per_arm_with_a_change_count(self, node_classes, levels):
        report, _, _, _, _ = self._run(node_classes, levels, values="1,2,3")
        body = report.splitlines()
        rows = [ln for ln in body if "/" in ln and ln[0].isdigit()]
        assert len(rows) == 3, body
        total = levels.grid[0] * levels.grid[1]
        assert all(f"/ {total}" in row for row in rows)

    def test_the_edit_registers_against_the_baseline(self, node_classes, levels):
        # The stub resume conditions each level on the one below, so an edit at
        # l* must still be visible at the top. If this ever reads zero the
        # measurement is wired to the wrong reference.
        report, _, _, _, spread = self._run(node_classes, levels, values="1,2,3")
        assert "changed nothing at all" not in report
        assert spread > 0

    def test_the_spread_says_whether_the_axis_mattered(self, node_classes, levels):
        report, _, _, _, spread = self._run(node_classes, levels, values="1,2,3")
        assert "spread across arms" in report
        assert spread > 0 and "does change the outcome" in report

    def test_a_single_arm_reports_no_spread_rather_than_zero(self, node_classes, levels):
        # Zero spread over one arm is not "the axis did nothing", it is "there
        # was nothing to compare" -- and the two read identically as a number.
        report, _, _, arms, _ = self._run(node_classes, levels, values="1")
        assert arms == 1
        assert "n/a with a single arm" in report

    # ----- what leaves the node -----
    def test_the_picked_arm_leaves_on_the_levels_output(self, node_classes, levels):
        _, _, first, _, _ = self._run(node_classes, levels, values="4,5,6", output_arm=0)
        _, _, third, _, _ = self._run(node_classes, levels, values="4,5,6", output_arm=2)
        assert (first.seed, third.seed) == (4, 6)
        assert not np.allclose(first.latents[-1], third.latents[-1])

    def test_the_output_arm_carries_no_pending_edit(self, node_classes, levels):
        # It has been resumed, so nothing above l* is stale; leaving the marker
        # set would make TF Decode warn about a trajectory that is fine.
        _, _, out, _, _ = self._run(node_classes, levels, values="1,2")
        assert out.dirty_level is None
        assert "sweep arm 0 of 2" in out.history[-1]

    def test_an_out_of_range_output_arm_clamps_and_says_so(self, node_classes, levels):
        report, _, out, _, _ = self._run(node_classes, levels, values="1,2", output_arm=9)
        assert out.seed == 2
        assert "clamped to the last arm" in report

    def test_the_trajectory_keeps_its_pipeline_so_downstream_nodes_work(
            self, node_classes, levels):
        _, _, out, _, _ = self._run(node_classes, levels, values="1,2")
        report, _, _, _ = node(node_classes, "TFCompareLevels").execute(
            before=levels, after=out, size=128, decode_difference=False, sheet_layout="separate frames")
        assert "tokens changed" in report

    # ----- the contact sheet -----
    def test_the_sheet_frames_are_all_stackable(self, node_classes, levels):
        # to_image stacks them into one batch and numpy will not stack ragged
        # frames; a decoded arm and a PCA tile are different sizes. The same
        # equality is what lets them be stitched into one sheet at all.
        kw = dict(sheet_layout="separate frames")
        _, decoded, _, _, _ = self._run(node_classes, levels, values="1,2", decode=True, **kw)
        _, latent, _, _, _ = self._run(node_classes, levels, values="1,2", decode=False, **kw)
        assert decoded.shape[0] == latent.shape[0] == 3

    def test_the_arms_are_stitched_side_by_side_by_default(self, node_classes, levels):
        # Comparison is the whole reason the node exists, and a batch puts the
        # arms in separate pictures -- five frames sharing a 320px node body
        # compare nothing, and SaveImage writes five unrelated files.
        _, sheet, _, _, _ = self._run(node_classes, levels, values="1,2")
        _, frames, _, _, _ = self._run(node_classes, levels, values="1,2",
                                       sheet_layout="separate frames")
        assert sheet.shape[0] == 1, "one image, not a batch"
        assert frames.shape[0] == 3
        # baseline + two arms in a row, so three frames wide and one tall
        assert sheet.shape[2] > sheet.shape[1]
        assert sheet.shape[2] >= 3 * frames.shape[2]

    def test_a_long_sweep_wraps_instead_of_becoming_a_strip(self, node_classes, levels):
        # Twelve arms in one row is 4600px wide at the default size, an aspect
        # ratio nothing displays usefully.
        _, wide, _, _, _ = self._run(node_classes, levels, values="1-3")
        _, grid, _, _, _ = self._run(node_classes, levels, values="1-9")
        assert wide.shape[1] < grid.shape[1], "the long one gained rows"
        assert grid.shape[2] < 9 * 128, "and stopped growing sideways"

    def test_it_shows_its_own_table_and_sheet(self, node_classes, levels):
        out = self._run(node_classes, levels, values="1,2")
        assert "spread across arms" in ui_text(out)
        # One stitched sheet in the node's own body: the arms are only worth
        # anything side by side, and that is where they land.
        assert len(ui_images(out)) == 1

    # ----- guards -----
    def test_a_mistyped_range_is_refused_before_any_gpu_work(
            self, node_classes, pipeline, levels):
        with pytest.raises(ValueError, match="arm_limit"):
            self._run(node_classes, levels, values="0-99")
        assert not [c for c in pipeline.calls if c[0] == "resume"]

    def test_a_level_outside_the_trajectory_is_refused(self, node_classes, levels):
        with pytest.raises(ValueError, match="outside this trajectory"):
            self._run(node_classes, levels, axis="level (l*)", values="0-9")

    def test_a_cross_level_selection_is_still_caught_on_a_seed_sweep(
            self, node_classes, two_region_levels, pipeline):
        from dataclasses import replace

        levels = replace(two_region_levels, pipeline=pipeline)
        regions, _, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=levels, level=1, cosine_threshold=0.9, size=128)
        snapped, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=levels, regions=regions)
        plain, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="5,5", levels=levels)
        with pytest.raises(ValueError, match="level 1's regions"):
            node(node_classes, "TFSweep").execute(
                levels=levels, target_tokens=snapped, source_tokens=plain,
                **{**self.DEFAULTS, "values": "1,2"})

    def test_a_level_sweep_states_the_caveat_instead_of_refusing(
            self, node_classes, two_region_levels, pipeline):
        # Sweeping l* breaks a region-snapped selection's level binding by
        # construction: holding the token *set* fixed is what "the same edit at
        # every level" has to mean. Refusing would make the axis unusable, so
        # the report says what was actually held fixed.
        from dataclasses import replace

        levels = replace(two_region_levels, pipeline=pipeline)
        regions, _, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=levels, level=1, cosine_threshold=0.9, size=128)
        snapped, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=levels, regions=regions)
        plain, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="5,5", levels=levels)
        report, _, _, _, _ = node(node_classes, "TFSweep").execute(
            levels=levels, target_tokens=snapped, source_tokens=plain,
            **{**self.DEFAULTS, "axis": "level (l*)", "values": "0-2"})
        assert "snapped to level 1's regions" in report

    def test_an_empty_selection_is_refused(self, node_classes, levels):
        empty, _, _ = node(node_classes, "TFTokensFromCoords").execute(coords="", levels=levels)
        source, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="5,5", levels=levels)
        with pytest.raises(ValueError, match="target_tokens is empty"):
            node(node_classes, "TFSweep").execute(
                levels=levels, target_tokens=empty, source_tokens=source, **self.DEFAULTS)

    def test_a_trajectory_with_no_pipeline_says_where_to_wire_one(self, node_classes):
        from tf_nodes.data import LevelStack

        orphan = LevelStack(
            latents=np.zeros((LEVELS, GRID, GRID, CHANNELS), np.float32), class_id=1, seed=1)
        with pytest.raises(ValueError, match="TF Load Pipeline"):
            self._run(node_classes, orphan, values="1,2")

    def test_the_report_records_the_pinned_settings(self, node_classes, levels):
        # A table read a month later has to say what was held fixed, or the
        # numbers in it cannot be reproduced.
        report, _, _, _, _ = self._run(node_classes, levels, values="1,2", level=1, strength=0.5)
        assert "pinned: level 1, strength 0.50, class 213" in report
        assert "2 target tokens" in report and "1 source tokens" in report


class TestSaveReport:
    """The numbers have to be able to leave the graph.

    TF Sweep Edit and TF Compare Levels produce exactly what a results table
    wants, and before this node it existed only as text in a node body.
    """

    @pytest.fixture
    def output_dir(self, tmp_path, monkeypatch):
        import tf_nodes.nodes_io as nodes_io

        monkeypatch.setattr(nodes_io, "_output_dir", lambda: tmp_path)
        return tmp_path

    def test_it_writes_the_report_verbatim(self, node_classes, output_dir):
        table = "seed   changed\n592    27 / 256"
        path, = node(node_classes, "TFSaveReport").execute(
            text=table, name="sweep", levels=None, append=True)
        written = (output_dir / "sweep.md").read_text()
        assert table in written
        assert path.endswith("sweep.md")

    def test_the_table_is_fenced_so_markdown_cannot_reflow_it(self, node_classes, output_dir):
        # These are aligned columns; unfenced, markdown runs them into one
        # paragraph, which destroys the only thing the file is for.
        node(node_classes, "TFSaveReport").execute(
            text="a   b\n1   2", name="r", levels=None, append=True)
        assert (output_dir / "r.md").read_text().count("```") == 2

    def test_provenance_is_written_when_a_trajectory_is_wired(
            self, node_classes, output_dir, levels):
        node(node_classes, "TFSaveReport").execute(
            text="table", name="r", levels=levels, append=True)
        written = (output_dir / "r.md").read_text()
        assert "class: 213" in written and "Irish setter" in written
        assert "seed: 1" in written
        assert "stub" in written, "the edit history is what makes a number traceable"

    def test_appending_accumulates_runs_in_one_file(self, node_classes, output_dir):
        for value in ("first", "second"):
            node(node_classes, "TFSaveReport").execute(
                text=value, name="r", levels=None, append=True)
        written = (output_dir / "r.md").read_text()
        assert "first" in written and "second" in written
        assert written.count("```") == 4
        assert len(list(output_dir.glob("*.md"))) == 1

    def test_not_appending_numbers_the_files_instead(self, node_classes, output_dir):
        for _ in range(3):
            node(node_classes, "TFSaveReport").execute(
                text="x", name="r", levels=None, append=False)
        assert sorted(p.name for p in output_dir.glob("*.md")) == [
            "r-001.md", "r-002.md", "r.md"]

    def test_an_empty_report_says_what_to_wire(self, node_classes, output_dir):
        with pytest.raises(ValueError, match="Wire TF Sweep Edit"):
            node(node_classes, "TFSaveReport").execute(
                text="   ", name="r", levels=None, append=True)

    def test_the_name_cannot_escape_the_output_directory(self, node_classes, output_dir):
        path, = node(node_classes, "TFSaveReport").execute(
            text="x", name="../../etc/passwd", levels=None, append=True)
        assert Path(path).parent == output_dir

    def test_it_is_an_output_node_so_it_runs_with_nothing_downstream(self, node_classes):
        # Its whole purpose is the side effect; without this ComfyUI prunes it.
        assert node(node_classes, "TFSaveReport").define_schema().is_output_node


class TestOneWidgetInsteadOfTwo:
    """`which` + `level` and `follow_edit` + `level` were both a mode plus a
    number the mode silently ignored. Each is now a single control that always
    does something, with the automatic case folded into the value itself."""

    def test_the_dropdown_names_the_level_directly(self, node_classes, pipeline, levels):
        images, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="level 1",
            label_levels=False, level_override=-1, sheet_layout="separate frames")
        assert images.shape[0] == 1

    def test_the_dropdown_covers_every_shipped_level(self, node_classes):
        from tf_nodes.sockets import SHIPPED_LEVELS

        options = next(i for i in node(node_classes, "TFDecode").define_schema().inputs
                       if i.id == "which").options
        assert options[:2] == ["all levels", "final level only"]
        assert options[2:] == [f"level {i}" for i in range(SHIPPED_LEVELS)]

    def test_the_override_wins_and_so_is_never_dead(self, node_classes, pipeline, levels):
        # It exists only for a model deeper than any released one, so it must
        # take effect whatever the dropdown says -- otherwise it is the same
        # ignorable widget under a new name.
        images, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="all levels",
            label_levels=False, level_override=3, sheet_layout="separate frames")
        assert images.shape[0] == 1

    def test_resume_follows_the_edit_when_left_alone(self, node_classes, pipeline, levels):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=levels)
        edited, _ = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=1, target_tokens=selection, source_tokens=selection,
            source_mode="region mean", strength=1.0, source_level=-1, source_levels=None)
        _, info = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=-1, class_id=-1, seed=5)
        assert "resume from level 1" in info

    def test_a_set_level_overrides_the_edit(self, node_classes, pipeline, levels):
        _, info = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=levels, level=2, class_id=-1, seed=5)
        assert "resume from level 2" in info

    def test_source_level_defaults_to_the_edit_level(self, node_classes, levels):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=levels)
        _, info = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=3, target_tokens=selection, source_tokens=selection,
            source_mode="region mean", strength=1.0, source_level=-1, source_levels=None)
        assert "at level 3" in info and "level 2" not in info


class TestAdvancedWidgets:
    """Rarely-touched settings are hidden behind ComfyUI's advanced toggle
    rather than removed: the surface shrinks, the capability does not."""

    def test_roughly_half_the_widgets_are_out_of_the_way(self, node_classes):
        visible = advanced = 0
        for cls in node_classes:
            for i in cls.define_schema().inputs:
                if not hasattr(i, "default"):
                    continue
                if i.advanced:
                    advanced += 1
                else:
                    visible += 1
        assert advanced >= visible * 0.6, f"{visible} visible vs {advanced} advanced"

    def test_the_knobs_you_actually_turn_stay_visible(self, node_classes):
        must_be_visible = {
            "TFGenerate": {"class_id", "seed"},
            "TFFeatureEdit": {"level", "strength"},
            "TFRegionMap": {"level", "cosine_threshold"},
            "TFTokensFromCoords": {"coords"},
            "TFTokensFromMask": {"coverage", "region_overlap"},
            "TFDecode": {"which"},
        }
        for cls in node_classes:
            schema = cls.define_schema()
            wanted = must_be_visible.get(schema.node_id)
            if not wanted:
                continue
            shown = {i.id for i in schema.inputs if hasattr(i, "default") and not i.advanced}
            assert wanted <= shown, f"{schema.node_id} hides {wanted - shown}"


class TestEveryNodeShowsItsOwnResult:
    """Stock ComfyUI has no node that displays a STRING.

    Verified against every registered core class: an `info` output is
    unreachable unless the node that computed it shows it itself. Before this,
    forty-two text and number outputs across the four example workflows went
    nowhere, including TF Levels Info's entire reason to exist.
    """

    TEXT_PRODUCERS = {
        "TFLoadPipeline", "TFImageNetClass", "TFLevelsInfo", "TFDecode",
        "TFLatentPreview", "TFRegionMap", "TFTokensFromMask", "TFTokensFromCoords",
        "TFTokensCombine", "TFTokensPreview", "TFFeatureEdit", "TFShapeEdit",
        "TFResumeFromLevel", "TFCompareLevels", "TFSweep", "TFSaveLevels",
        "TFSaveReport", "TFLoadLevels",
    }

    def test_they_all_declare_a_string_output_or_show_text(self, node_classes):
        for cls in node_classes:
            schema = cls.define_schema()
            if schema.node_id not in self.TEXT_PRODUCERS:
                continue
            assert schema.has_intermediate_output, (
                f"{schema.node_id} shows text, so its preview must survive a cached re-run")

    def test_the_edit_summary_is_visible(self, node_classes, levels, pipeline):
        target, _, _ = node(node_classes, "TFTokensFromCoords").execute(coords="1,1", levels=levels)
        source, _, _ = node(node_classes, "TFTokensFromCoords").execute(coords="5,5", levels=levels)
        out = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=2, target_tokens=target, source_tokens=source,
            source_mode="region mean", strength=1.0, source_level=2, source_levels=None)
        assert "feature edit" in ui_text(out)
        assert ui_text(out) == out[1], "what it shows and what it outputs must agree"

    def test_levels_info_is_not_an_inert_box(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=213, seed=1)
        out = node(node_classes, "TFLevelsInfo").execute(levels=stack)
        shown = ui_text(out)
        assert "Irish setter" in shown and "seed" in shown

    def test_region_map_shows_its_count_and_its_picture(self, node_classes, two_region_levels):
        out = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=2, cosine_threshold=0.9, size=128)
        assert "2 regions at level 2" in ui_text(out)
        assert len(ui_images(out)) == 1, "the map is the point of the node"

    def test_compare_shows_both_the_table_and_the_heatmap(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=1, seed=1)
        out = node(node_classes, "TFCompareLevels").execute(
            before=stack, after=stack, size=256, decode_difference=False, sheet_layout="separate frames")
        assert "tokens changed" in ui_text(out)
        assert len(ui_images(out)) == LEVELS

    def test_the_canvas_publishes_its_image_for_the_painter(self, node_classes, pipeline):
        # The Painter takes its backdrop from the upstream node's stored preview.
        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=1, seed=1)
        out = node(node_classes, "TFLevelCanvas").execute(
            levels=stack, level=2, view="latent PCA", draw_grid=True,
            label_coords=True, size=256, regions=None, highlight=None)
        assert len(ui_images(out)) == 1

    def test_tokens_preview_shows_the_selection_it_drew(self, node_classes, levels):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="2,3", levels=levels)
        out = node(node_classes, "TFTokensPreview").execute(tokens=selection, size=128)
        assert len(ui_images(out)) == 1
        assert "2,3" in ui_text(out)


class TestDiscoverability:
    def test_every_node_has_search_aliases(self, node_classes):
        # ComfyUI's node search matches the display name and these. Without them
        # "paint", "mask" or "diff" find nothing, and the name has to be known
        # before it can be looked up.
        for cls in node_classes:
            schema = cls.define_schema()
            assert schema.search_aliases, f"{schema.node_id} has no search aliases"

    def test_aliases_are_lowercase_and_distinct(self, node_classes):
        for cls in node_classes:
            aliases = cls.define_schema().search_aliases
            assert aliases == [a.lower() for a in aliases]
            assert len(set(aliases)) == len(aliases)

    def test_nodes_are_grouped_into_subcategories(self, node_classes):
        # Eighteen nodes in one flat menu is a scroll.
        categories = {c.define_schema().category for c in node_classes}
        assert len(categories) >= 4
        assert all(c.startswith("TrajectoryForcing") for c in categories)

    def test_the_class_picker_defaults_to_the_same_class_as_generate(self, node_classes):
        # Two nodes that are normally wired together should not disagree about
        # what a default run produces.
        picker = node(node_classes, "TFImageNetClass").define_schema().inputs[0].default
        generate = next(i for i in node(node_classes, "TFGenerate").define_schema().inputs
                        if i.id == "class_id").default
        assert picker.startswith(f"{generate} - ")


class TestTheAutoConvention:
    """Every widget that can decide for itself uses -1, says so on its label,
    and reports what it decided. A sentinel is only obvious to someone who
    already knows the convention."""

    def _auto_widgets(self, node_classes):
        for cls in node_classes:
            schema = cls.define_schema()
            for i in schema.inputs:
                if getattr(i, "default", None) == -1:
                    yield schema.node_id, i

    def test_there_are_some(self, node_classes):
        assert list(self._auto_widgets(node_classes)), "the convention needs users to matter"

    def test_the_label_states_it_without_hovering(self, node_classes):
        from tf_nodes.sockets import AUTO_SUFFIX

        for node_id, widget in self._auto_widgets(node_classes):
            assert widget.display_name, f"{node_id}.{widget.id} shows only its bare id"
            assert widget.display_name.endswith(AUTO_SUFFIX), (
                f"{node_id}.{widget.id} label is {widget.display_name!r}")

    def test_the_tooltip_says_what_auto_does(self, node_classes):
        for node_id, widget in self._auto_widgets(node_classes):
            assert "auto" in (widget.tooltip or "").lower(), f"{node_id}.{widget.id}"

    def test_minus_one_is_the_only_sentinel_value(self, node_classes):
        # Three meanings would be three conventions to learn; one is enough.
        from tf_nodes.sockets import AUTO

        for node_id, widget in self._auto_widgets(node_classes):
            assert widget.min == AUTO, f"{node_id}.{widget.id} allows values below the sentinel"

    def test_resume_reports_what_auto_chose(self, node_classes, pipeline, levels):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=levels)
        edited, _ = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=1, target_tokens=selection, source_tokens=selection,
            source_mode="region mean", strength=1.0, source_level=-1, source_levels=None)
        _, info = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=-1, class_id=-1, seed=5)
        assert "auto: the level the edit wrote to" in info
        assert "auto: the trajectory's own" in info

    def test_a_value_set_by_hand_is_reported_as_such(self, node_classes, pipeline, levels):
        _, info = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=levels, level=2, class_id=7, seed=5)
        assert "set on the node" in info
        assert "auto" not in info

    def test_feature_edit_says_when_the_source_level_was_auto(self, node_classes, levels):
        selection, _, _ = node(node_classes, "TFTokensFromCoords").execute(
            coords="0,0", levels=levels)
        _, auto = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=2, target_tokens=selection, source_tokens=selection,
            source_mode="region mean", strength=1.0, source_level=-1, source_levels=None)
        _, explicit = node(node_classes, "TFFeatureEdit").execute(
            levels=levels, level=2, target_tokens=selection, source_tokens=selection,
            source_mode="region mean", strength=1.0, source_level=2, source_levels=None)
        assert "auto: same as the edit" in auto
        assert "auto" not in explicit


class TestTheSmokeScriptsCallNodesCorrectly:
    """`scripts/gpu_smoke.py` calls node `execute` methods by hand.

    Nothing else checks those call sites, so a renamed or added input leaves the
    script raising TypeError -- and only once it reaches a GPU, minutes into a
    queued job. That has now happened twice in this repo: once when `which` +
    `level` were merged, and once when the sweep gained `sheet_layout`. This
    reads the call sites out of the source and checks them against the live
    schemas, in the five seconds the rest of the suite takes.
    """

    def _calls(self, source: str):
        """Every `SomeNode.execute(...)` in the file, as (node class name, kwargs)."""
        import ast

        for node_ in ast.walk(ast.parse(source)):
            if not isinstance(node_, ast.Call):
                continue
            func = node_.func
            if (isinstance(func, ast.Attribute) and func.attr == "execute"
                    and isinstance(func.value, ast.Name)):
                yield func.value.id, {kw.arg for kw in node_.keywords if kw.arg}

    def test_every_execute_call_matches_its_schema(self, node_classes):
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "scripts" / "gpu_smoke.py"
        by_name = {c.__name__: c for c in node_classes}
        checked = 0
        for name, passed in self._calls(script.read_text()):
            cls = by_name.get(name)
            if cls is None:
                continue
            schema = cls.define_schema()
            declared = {i.id for i in schema.inputs}
            required = {i.id for i in schema.inputs if not i.optional}
            assert passed <= declared, (
                f"gpu_smoke.py passes {sorted(passed - declared)} to {name}, which has no such input")
            assert required <= passed, (
                f"gpu_smoke.py is missing {sorted(required - passed)} for {name}")
            checked += 1
        assert checked >= 6, f"only matched {checked} call sites; the parser has drifted"


class TestSeveralFramesArriveAsOneImage:
    """ComfyUI pages a multi-image output one frame at a time.

    Its renderer draws a small "1/4" button and shows `imageIndex` only:

        if (!(l > 1)) return;
        let C = (t.imageIndex ?? 0) + 1;
        if (drawButton(f - 40, p + n - 40, 30, `${C}/${l}`)) { ... }

    So a node returning four levels as a batch shows *level 0* and a control
    most people never spot -- reported as "I run workflow 1 and get level 0, it
    should give me all levels". Anything meant to be seen together is therefore
    stitched by default, and the batch stays behind `sheet_layout` because it is
    the only shape SaveImage writes as one file per frame.
    """

    def test_decoding_all_levels_gives_one_image_not_a_batch(self, node_classes, pipeline, levels):
        images, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="all levels",
            label_levels=True, level_override=-1, sheet_layout="contact sheet")
        assert images.shape[0] == 1, "four levels must not arrive as four pages"
        assert images.shape[2] > images.shape[1], "laid out as a row"

    def test_the_batch_is_still_available_for_saving(self, node_classes, pipeline, levels):
        images, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="all levels",
            label_levels=True, level_override=-1, sheet_layout="separate frames")
        assert images.shape[0] == LEVELS

    def test_a_single_level_is_unaffected_either_way(self, node_classes, pipeline, levels):
        for layout in ("contact sheet", "separate frames"):
            images, _ = node(node_classes, "TFDecode").execute(
                pipeline=pipeline, levels=levels, which="level 1",
                label_levels=False, level_override=-1, sheet_layout=layout)
            assert images.shape[0] == 1

    def test_the_latent_preview_stitches_too(self, node_classes, pipeline, levels):
        images, = node(node_classes, "TFLatentPreview").execute(
            pipeline=pipeline, levels=levels, which="all levels", size=128,
            label_levels=True, level_override=-1, sheet_layout="contact sheet")
        assert images.shape[0] == 1

    def test_compare_stitches_its_per_level_heatmaps(self, node_classes, pipeline):
        stack, = node(node_classes, "TFGenerate").execute(pipeline=pipeline, class_id=1, seed=1)
        _, heatmap, _, _ = node(node_classes, "TFCompareLevels").execute(
            before=stack, after=stack, size=256, decode_difference=False,
            sheet_layout="contact sheet")
        assert heatmap.shape[0] == 1

    def test_every_multi_frame_node_offers_the_choice(self, node_classes):
        # If a node can emit several frames meant to be compared, it must not be
        # able to hand them over as a batch by accident.
        for node_id in ("TFDecode", "TFLatentPreview", "TFCompareLevels", "TFSweep"):
            schema = node(node_classes, node_id).define_schema()
            widget = next((i for i in schema.inputs if i.id == "sheet_layout"), None)
            assert widget is not None, f"{node_id} has no sheet_layout"
            assert widget.options[0] == "contact sheet", f"{node_id} defaults to a batch"
            assert widget.advanced, f"{node_id} shows sheet_layout by default"
