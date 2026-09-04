"""Finding -- or fetching -- the TrajectoryForcing checkout.

`locate.py` imports no ComfyUI, so this runs anywhere. The fetch path is stubbed
rather than exercised: a test that actually clones upstream would be slow, need
network, and break when GitHub is down, none of which says anything about this
code.
"""
from __future__ import annotations

import pytest

from tf_nodes import locate


@pytest.fixture(autouse=True)
def _clear_cache():
    """`tf_repo()` memoises into a module global; each test needs a clean one."""
    locate._TF_REPO = None
    yield
    locate._TF_REPO = None


def make_checkout(path):
    """The two files `_looks_like_tf_repo` keys on."""
    (path / "editing_env").mkdir(parents=True, exist_ok=True)
    (path / "editing_env" / "tf_pipeline.py").write_text("")
    (path / "pmf.py").write_text("")
    return path


class TestAnExistingCheckoutWins:
    """Fetching is the last resort, not the first move.

    The RAE decoder is 2 GB and TrajectoryForcing's own download script already
    puts it inside the checkout, so quietly making a second copy is the failure
    to avoid -- worse than the missing-checkout error it replaces.
    """

    def test_tf_repo_env_var_short_circuits_the_fetch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TF_REPO", str(make_checkout(tmp_path / "tf")))
        monkeypatch.setattr(locate, "fetch_tf_repo", _never_called)
        assert locate.tf_repo() == tmp_path / "tf"

    def test_a_sibling_checkout_short_circuits_the_fetch(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TF_REPO", raising=False)
        monkeypatch.setattr(locate, "EXT_ROOT", tmp_path / "ext")
        make_checkout(tmp_path / "TrajectoryForcing")
        monkeypatch.setattr(locate, "fetch_tf_repo", _never_called)
        assert locate.tf_repo() == tmp_path / "TrajectoryForcing"

    def test_opting_out_reports_where_it_looked(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TF_REPO", raising=False)
        monkeypatch.setattr(locate, "EXT_ROOT", tmp_path / "ext")
        monkeypatch.setenv("TF_NO_AUTO_FETCH", "1")
        monkeypatch.setattr(locate, "fetch_tf_repo", _never_called)
        with pytest.raises(FileNotFoundError, match="TF_NO_AUTO_FETCH"):
            locate.tf_repo()


class TestTheFetchIsPinned:
    """An unpinned fetch is the dependency this is meant to avoid.

    The extension calls into TF's own API, so upstream moving is a change here.
    `clone --depth 1` would take whatever main points at on the day someone
    installs, which is precisely the silent version drift the pin exists for.
    """

    def test_the_pin_is_a_full_sha(self):
        assert len(locate.TF_REPO_COMMIT) == 40
        assert all(c in "0123456789abcdef" for c in locate.TF_REPO_COMMIT), (
            "a branch or tag name here would make the fetch unpinned")

    def test_it_fetches_the_commit_rather_than_a_shallow_clone(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(locate, "_git", lambda *a, **k: calls.append(a))
        # _git is stubbed, so nothing lands on disk; satisfy the post-check.
        monkeypatch.setattr(locate, "_looks_like_tf_repo",
                            lambda p: bool(calls) and p == tmp_path / "tf")
        locate.fetch_tf_repo(dest=tmp_path / "tf")
        flat = [" ".join(c) for c in calls]
        assert any(locate.TF_REPO_COMMIT in f and "fetch" in f for f in flat), flat
        assert not any("clone" in f for f in flat), (
            f"a clone cannot name a commit, so it would be unpinned: {flat}")

    def test_a_failed_fetch_leaves_nothing_behind(self, tmp_path, monkeypatch):
        # A half-clone would send the next attempt down the "exists but is not a
        # checkout" branch, which needs a human to delete a directory.
        def boom(*a, **k):
            raise RuntimeError("network")

        monkeypatch.setattr(locate, "_git", boom)
        with pytest.raises(RuntimeError):
            locate.fetch_tf_repo(dest=tmp_path / "tf")
        assert not (tmp_path / "tf").exists()

    def test_it_refuses_to_fetch_over_something_it_did_not_make(self, tmp_path):
        occupied = tmp_path / "tf"
        occupied.mkdir()
        (occupied / "important.txt").write_text("not ours")
        with pytest.raises(FileNotFoundError, match="does not look like"):
            locate.fetch_tf_repo(dest=occupied)
        assert (occupied / "important.txt").exists()


def _never_called(*args, **kwargs):
    raise AssertionError("fetched when an existing checkout should have been used")


class TestTheEnvironmentIsCheckedBeforeTF:
    """The most likely first failure for a ComfyUI Manager install.

    `requirements.txt` is deliberately empty -- listing jax and a CUDA-matched
    torch would have Manager rewrite torch underneath every other node in the
    install. So the nodes register and the missing half is noticed here, and the
    error has to name the command that fixes it rather than surfacing as a
    ModuleNotFoundError from inside TrajectoryForcing's own code.

    Every test here stubs RUNTIME_DEPS with stdlib names. An earlier version
    called `check_runtime_deps()` against the real list and passed locally, then
    failed in CI -- which installs no JAX, precisely because that is the
    environment the check exists for. A test of the missing-dependency path must
    not itself depend on the dependency being present.
    """

    def test_it_passes_when_everything_is_present(self, monkeypatch):
        import tf_nodes.pipeline as pipeline

        monkeypatch.setattr(pipeline, "RUNTIME_DEPS", ("json", "colorsys"))
        pipeline.check_runtime_deps()

    def test_it_names_only_what_is_missing(self, monkeypatch):
        import tf_nodes.pipeline as pipeline

        monkeypatch.setattr(pipeline, "RUNTIME_DEPS", ("json", "definitely_not_installed_xyz"))
        with pytest.raises(ImportError) as caught:
            pipeline.check_runtime_deps()
        message = str(caught.value)
        assert "bash env/setup.sh" in message, "the error must carry the command that fixes it"
        listed = message.split("needs", 1)[1].split(", which", 1)[0]
        assert "definitely_not_installed_xyz" in listed
        assert "json" not in listed, f"named a package that is installed: {listed!r}"

    def test_it_does_not_import_to_find_out(self, monkeypatch):
        # Importing jax initialises its GPU backend, which configure_jax_env has
        # to run before and which cannot be undone in a live process. find_spec
        # answers the question without that. Checked with a stdlib module that is
        # present but not already imported, so it holds with or without jax.
        import sys

        import tf_nodes.pipeline as pipeline

        monkeypatch.delitem(sys.modules, "colorsys", raising=False)
        monkeypatch.setattr(pipeline, "RUNTIME_DEPS", ("colorsys",))
        pipeline.check_runtime_deps()
        assert "colorsys" not in sys.modules, "check_runtime_deps imported instead of probing"

    def test_the_real_list_covers_the_jax_stack(self):
        # The stubs above say nothing about what is actually checked.
        from tf_nodes.pipeline import RUNTIME_DEPS

        assert "jax" in RUNTIME_DEPS and "flax" in RUNTIME_DEPS, RUNTIME_DEPS


class TestRequirementsCannotBreakSomeoneElsesInstall:
    """A line in the top-level requirements.txt is installed into other people's
    environments by ComfyUI Manager. The pins belong in env/requirements.txt."""

    def test_the_top_level_file_installs_nothing(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        lines = [ln.strip() for ln in (root / "requirements.txt").read_text().splitlines()]
        installable = [ln for ln in lines if ln and not ln.startswith("#")]
        assert not installable, (
            f"{installable} would be pip-installed into whatever venv ComfyUI runs in, "
            "rewriting torch and the CUDA libraries under every other custom node. "
            "Put dependencies in env/requirements.txt instead.")

    def test_the_real_list_still_exists(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        text = (root / "env" / "requirements.txt").read_text()
        assert "jax[cuda12]" in text and "torch==" in text
