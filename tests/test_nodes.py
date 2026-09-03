"""Every node's schema and execute path, against a stub pipeline.

This is the part that catches the mistakes a pure-numpy test cannot: a schema
input whose id does not match the `execute` parameter it feeds, an output tuple
of the wrong length, an optional input that is not actually optional. All of
those are invisible until someone queues the node in a browser, and none of them
needs a GPU to find.
"""
from __future__ import annotations

import inspect

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
        # mimic the real thing: levels above start_level are re-sampled, below untouched
        rng = np.random.default_rng(seed)
        for level in range(start_level + 1, LEVELS):
            out[level] = rng.normal(size=(GRID, GRID, CHANNELS)).astype(np.float32)
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
            assert schema.category == "TrajectoryForcing"


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
        "which,expected", [("all levels", LEVELS), ("final level only", 1), ("single level", 1)]
    )
    def test_decode_frame_counts(self, node_classes, pipeline, levels, which, expected):
        images, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which=which, level=1, label_levels=False
        )
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] == expected
        assert images.dtype == torch.float32
        assert 0.0 <= float(images.min()) and float(images.max()) <= 1.0

    def test_decode_captions_add_a_strip(self, node_classes, pipeline, levels):
        plain, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="all levels", level=0, label_levels=False
        )
        labelled, _ = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=levels, which="all levels", level=0, label_levels=True
        )
        assert labelled.shape[1] > plain.shape[1]
        assert labelled.shape[2] == plain.shape[2]

    def test_decode_warns_when_the_stack_is_stale(self, node_classes, pipeline, levels):
        edited = levels.with_level(1, levels.level(1), "edit")
        _, warning = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=edited, which="all levels", level=0, label_levels=False
        )
        assert "stale" in warning

    def test_decode_of_a_level_below_the_edit_does_not_warn(self, node_classes, pipeline, levels):
        edited = levels.with_level(2, levels.level(2), "edit")
        _, warning = node(node_classes, "TFDecode").execute(
            pipeline=pipeline, levels=edited, which="single level", level=1, label_levels=False
        )
        assert warning == ""

    def test_latent_preview_fits_the_requested_size(self, node_classes, pipeline, levels):
        images, = node(node_classes, "TFLatentPreview").execute(
            pipeline=pipeline, levels=levels, which="all levels", level=0, size=256,
            label_levels=False, palette_from=None,
        )
        assert images.shape == (LEVELS, 256, 256, 3)

    def test_palette_from_fits_jointly(self, node_classes, pipeline, levels):
        node(node_classes, "TFLatentPreview").execute(
            pipeline=pipeline, levels=levels, which="all levels", level=0, size=128,
            label_levels=False, palette_from=levels,
        )
        assert ("fit_palette", 2) in pipeline.calls
        assert ("pca_tiles", True) in pipeline.calls

    def test_levels_info_reports_the_history(self, node_classes, levels):
        info, n, class_id, seed = node(node_classes, "TFLevelsInfo").execute(levels=levels)
        assert (n, class_id, seed) == (LEVELS, 213, 1)
        assert "stub" in info


# ---------------------------------------------------------------------------
# regions and selections
# ---------------------------------------------------------------------------
class TestRegionNodes:
    def test_region_map_finds_two_regions(self, node_classes, two_region_levels):
        regions, image, count = node(node_classes, "TFRegionMap").execute(
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
        regions, _, _ = node(node_classes, "TFRegionMap").execute(
            levels=two_region_levels, level=2, cosine_threshold=0.9, size=128
        )
        mask = torch.zeros((1, 64, 64))
        mask[0, 0:8, 0:8] = 1.0
        tokens, count, _ = node(node_classes, "TFTokensFromMask").execute(
            mask=mask, levels=two_region_levels, coverage=0.5, regions=regions, region_overlap=0.0
        )
        assert count == GRID * GRID // 2
        assert all(c < GRID // 2 for _, c in tokens.coords)

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
                draw_grid=True, size=256, regions=None, highlight=None,
            )
            assert image.shape == (1, 256, 256, 3)
            assert level == 2

    def test_level_canvas_clamps_an_out_of_range_level(self, node_classes, pipeline, levels):
        _, level = node(node_classes, "TFLevelCanvas").execute(
            pipeline=pipeline, levels=levels, level=15, view="latent PCA",
            draw_grid=False, size=128, regions=None, highlight=None,
        )
        assert level == LEVELS - 1

    def test_level_canvas_rejects_a_mismatched_highlight(self, node_classes, pipeline, levels):
        from tf_nodes.data import TokenSelection

        with pytest.raises(ValueError, match="token grid is"):
            node(node_classes, "TFLevelCanvas").execute(
                pipeline=pipeline, levels=levels, level=0, view="latent PCA",
                draw_grid=False, size=128, regions=None,
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
        regions, _, _ = node(node_classes, "TFRegionMap").execute(
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
        regions, _, _ = node(node_classes, "TFRegionMap").execute(
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
        regions, _, _ = node(node_classes, "TFRegionMap").execute(
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
            pipeline=pipeline, levels=edited, level=0, follow_edit=True, class_id=-1, seed=5
        )
        assert ("resume", 1, 213, 5) in pipeline.calls, "the edit's level and the stack's class win"
        assert out.dirty_level is None
        assert "resume from level 1" in info

    def test_resume_keeps_levels_below_l_star(self, node_classes, pipeline, levels):
        edited = self._edited(node_classes, levels, level=2)
        out, _ = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=0, follow_edit=True, class_id=-1, seed=5
        )
        for below in (0, 1, 2):
            np.testing.assert_allclose(out.level(below), edited.level(below))
        assert not np.allclose(out.level(3), edited.level(3))

    def test_resume_can_override_the_class(self, node_classes, pipeline, levels):
        node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=levels, level=1, follow_edit=False, class_id=77, seed=2
        )
        assert ("resume", 1, 77, 2) in pipeline.calls

    def test_follow_edit_without_an_edit_is_an_error(self, node_classes, pipeline, levels):
        with pytest.raises(ValueError, match="no edit to follow"):
            node(node_classes, "TFResumeFromLevel").execute(
                pipeline=pipeline, levels=levels, level=1, follow_edit=True, class_id=-1, seed=1
            )

    def test_resuming_above_a_pending_edit_keeps_it_marked_stale(self, node_classes, pipeline, levels):
        # Resuming refreshes only the levels above the start, so an edit further
        # down is still unpropagated -- clearing the marker would hide that.
        edited = self._edited(node_classes, levels, level=1)
        out, _ = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=2, follow_edit=False, class_id=-1, seed=5
        )
        assert out.dirty_level == 1

    def test_resuming_below_a_pending_edit_settles_it(self, node_classes, pipeline, levels):
        edited = self._edited(node_classes, levels, level=2)
        out, _ = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=edited, level=1, follow_edit=False, class_id=-1, seed=5
        )
        assert out.dirty_level is None

    def test_resuming_at_the_last_level_says_so(self, node_classes, pipeline, levels):
        _, info = node(node_classes, "TFResumeFromLevel").execute(
            pipeline=pipeline, levels=levels, level=LEVELS - 1, follow_edit=False, class_id=-1, seed=1
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
