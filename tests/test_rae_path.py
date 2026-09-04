"""The decoder path, and the marker that makes auto-download work.

TrajectoryForcing downloads the RAE decoder itself when the file is missing. It
works out *what* to fetch by splitting the configured path on the literal string
``checkpoints/rae/`` and sending the tail to HuggingFace as the filename
(`utils/rae_decoder.py`, `_RAE_LOCAL_MARKER` and `_maybe_hf_download`).

`_reroot_rae` rewrites that same path so the weights can live somewhere else --
`$TF_RAE_ROOT`, or ComfyUI's `models/` when the checkout has none. The two only
work together as long as the rewritten path still contains the marker, and
nothing enforces that. Rewrite it to a tidier layout and the decoder stops
auto-downloading: not with an error at edit time, but at first decode, in
somebody else's code, on a fresh install where there is no local copy to fall
back on. The developer who makes the change has weights on disk already and
sees nothing.

Verified on a GPU: a run with an empty models directory and its own empty
HF_HOME fetched the 2 GB checkpoint and the 1.6 GB decoder unaided and passed
all eight smoke criteria. These tests are what keep that true.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tf_nodes import pipeline
from tf_nodes.pipeline import _RAE_MARKER, _reroot_rae

# The path TrajectoryForcing's own config ships, which is what the marker has to
# survive. Kept verbatim rather than built from the marker, so a test cannot
# agree with a broken constant by construction.
TF_CONFIG_PATH = "checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
TAIL = "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"


class Section(dict):
    """Enough of an ml_collections ConfigDict: dict `.get`, attribute assignment."""

    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


class Config:
    def __init__(self, path: str):
        self.rae_decoder = Section(pretrained_decoder_path=path)


@pytest.fixture
def rooted(monkeypatch):
    """Point `rae_root()` wherever a test likes."""

    def _set(root: str):
        monkeypatch.setattr(pipeline, "rae_root", lambda: root)

    return _set


class TestTheRewrittenPathKeepsTheDownloadMarker:
    """The invariant: whatever `_reroot_rae` returns, TF can still split it.

    Every assertion here is really the same one. It is written several ways
    because the plausible mistakes differ -- dropping the marker, keeping it but
    mangling the tail, or special-casing a root that happens to end in a slash.
    """

    @pytest.mark.parametrize(
        "root",
        [
            "/home/someone/TrajectoryForcing/checkpoints/rae",
            "/models/trajectory_forcing/checkpoints/rae",
            "/models/trajectory_forcing/checkpoints/rae/",   # trailing slash
            "/a b/with spaces/checkpoints/rae",
            "relative/checkpoints/rae",
        ],
    )
    def test_the_marker_survives_every_root(self, rooted, root):
        rooted(root)
        config = Config(TF_CONFIG_PATH)
        out = _reroot_rae(config)
        assert out is not None
        assert _RAE_MARKER in out.replace("\\", "/"), (
            f"rerooting to {root!r} produced {out!r}, which TrajectoryForcing cannot "
            "split -- auto-download of the decoder is silently dead."
        )

    def test_the_tail_after_the_marker_is_preserved_exactly(self, rooted):
        """That tail is sent to HuggingFace as the filename, so it is not cosmetic."""
        rooted("/models/trajectory_forcing/checkpoints/rae")
        config = Config(TF_CONFIG_PATH)
        out = _reroot_rae(config).replace("\\", "/")
        assert out.split(_RAE_MARKER, 1)[1] == TAIL

    def test_the_config_is_actually_updated(self, rooted):
        rooted("/models/trajectory_forcing/checkpoints/rae")
        config = Config(TF_CONFIG_PATH)
        out = _reroot_rae(config)
        assert config.rae_decoder.pretrained_decoder_path == out

    def test_a_root_without_the_marker_would_be_caught(self, rooted):
        """The failure this whole file exists to prevent, made visible.

        `rae_root()` is built from RAE_SUBDIR, so it ends in checkpoints/rae
        today. If someone flattens that layout, this is what it looks like: a
        perfectly reasonable path that quietly cannot auto-download.
        """
        rooted("/models/trajectory_forcing/rae_weights")
        out = _reroot_rae(Config(TF_CONFIG_PATH)).replace("\\", "/")
        assert _RAE_MARKER not in out

    def test_a_path_that_never_had_the_marker_is_left_alone(self, rooted):
        """Someone pointing the config at an absolute path of their own.

        Rewriting it would move their weights out from under them, so the
        function declines and says so by returning None.
        """
        rooted("/models/trajectory_forcing/checkpoints/rae")
        config = Config("/somewhere/else/decoder.pt")
        assert _reroot_rae(config) is None
        assert config.rae_decoder.pretrained_decoder_path == "/somewhere/else/decoder.pt"

    def test_windows_separators_still_match(self, rooted):
        """`_reroot_rae` normalises backslashes before looking for the marker."""
        rooted("/models/trajectory_forcing/checkpoints/rae")
        config = Config(TF_CONFIG_PATH.replace("/", "\\"))
        out = _reroot_rae(config)
        assert out is not None
        assert out.replace("\\", "/").split(_RAE_MARKER, 1)[1] == TAIL


def _tf_rae_decoder() -> Path | None:
    """TrajectoryForcing's decoder module, if a checkout is already present.

    Auto-fetch is suppressed: a test suite that clones 2 GB of upstream on a
    cold CI runner is a worse problem than the one this checks for.
    """
    os.environ.setdefault("TF_NO_AUTO_FETCH", "1")
    try:
        from tf_nodes import locate

        path = locate.tf_repo() / "utils" / "rae_decoder.py"
    except Exception:
        return None
    return path if path.is_file() else None


class TestUpstreamStillSplitsOnTheSameString:
    """Our marker is a copy of TrajectoryForcing's. Copies drift.

    `TF_REPO_COMMIT` is a pin, so this cannot break on its own -- but it can
    break the moment the pin is bumped, and the symptom would appear far from
    the cause. Bumping the pin is exactly when someone should be told.
    """

    def test_tf_defines_the_marker_we_reroot_against(self):
        source = _tf_rae_decoder()
        if source is None:
            pytest.skip("no TrajectoryForcing checkout; nothing to compare against")
        text = source.read_text(encoding="utf8", errors="replace")
        assert f'"{_RAE_MARKER}"' in text or f"'{_RAE_MARKER}'" in text, (
            f"TrajectoryForcing's utils/rae_decoder.py no longer contains the literal "
            f"{_RAE_MARKER!r}. If upstream changed how it locates downloadable weights, "
            "tf_nodes/pipeline.py:_RAE_MARKER has to change with it or the decoder will "
            "stop auto-downloading on fresh installs."
        )

    def test_tf_still_splits_a_config_path_the_way_we_assume(self):
        source = _tf_rae_decoder()
        if source is None:
            pytest.skip("no TrajectoryForcing checkout; nothing to compare against")
        text = source.read_text(encoding="utf8", errors="replace")
        assert ".split(" in text and "hf_hub_download" in text, (
            "utils/rae_decoder.py no longer splits a path and downloads the tail. "
            "The re-rooting in pipeline.py assumes it does."
        )
