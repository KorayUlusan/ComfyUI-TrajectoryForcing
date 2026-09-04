"""Failures have to surface where the person who can fix them is looking.

The defect these cover was one shape seen three ways: a problem detected in one
place and needed in another. `on_load` fetching the checkout meant no network at
startup took all 21 nodes with it, leaving `IMPORT FAILED` in the Manager and
the real reason in a console nobody reads. The dependency check ran only when a
graph was executed. `install.py`'s decline printed into collapsed installer
output.

So the rule these tests hold: **nothing on the startup path raises**, and every
problem detected there is still available later, when the user does something
that depends on it.
"""
from __future__ import annotations

import pytest

from tf_nodes import health


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """A marker file per test, and no memory of the last one."""
    monkeypatch.setattr(health, "SETUP_MARKER", tmp_path / "SETUP-REQUIRED.txt")
    monkeypatch.setattr(health, "LAST_REPORT", None)
    yield
    monkeypatch.setattr(health, "LAST_REPORT", None)


def working_checkout(monkeypatch, tmp_path):
    """A checkout that is simply there.

    Stubbed rather than resolved for real: `health.collect()` is allowed to
    fetch, because `on_load` is supposed to, and a unit suite that clones from
    the network on a cold runner is slow, flaky, and -- as this file found out
    -- leaves a checkout behind that later tests then trip over.
    """
    from tf_nodes import locate

    monkeypatch.setattr(locate, "tf_repo", lambda *a, **k: tmp_path / "TrajectoryForcing")


def break_checkout(monkeypatch, exc=None):
    """Make locating TrajectoryForcing fail the way a cold start can."""
    from tf_nodes import locate

    def boom(*_a, **_k):
        raise exc or FileNotFoundError("Could not find the TrajectoryForcing checkout")

    monkeypatch.setattr(locate, "tf_repo", boom)


class TestNothingOnTheStartupPathRaises:
    """The whole point. A report is a report, never an exception.

    `collect()` is what `on_load` calls, and `on_load` raising is what used to
    delete the entire node pack from the menu.
    """

    def test_a_missing_checkout_is_reported_not_raised(self, monkeypatch):
        break_checkout(monkeypatch)
        report = health.collect()
        assert not report.ok
        assert any("model code is not available" in p.title for p in report.problems)

    def test_the_original_error_text_survives_into_the_report(self, monkeypatch):
        break_checkout(monkeypatch, FileNotFoundError("Tried: /a /b. Set TF_REPO."))
        problem = next(p for p in health.collect().problems if "model code" in p.title)
        assert "Set TF_REPO." in problem.detail, "the useful part of the message was dropped"

    @pytest.mark.parametrize(
        "exc",
        [
            FileNotFoundError("no checkout"),
            PermissionError("read-only filesystem"),
            OSError("network unreachable"),
            RuntimeError("git exploded"),
        ],
    )
    def test_any_failure_shape_is_survivable(self, monkeypatch, exc):
        """A `git` subprocess can fail in more ways than one, and none of them
        should be the difference between having nodes and not having nodes."""
        break_checkout(monkeypatch, exc)
        assert health.collect() is not None

    def test_report_at_startup_never_raises_and_remembers(self, monkeypatch):
        break_checkout(monkeypatch)
        report = health.report_at_startup()
        assert health.LAST_REPORT is report

    def test_a_healthy_install_reports_nothing(self, monkeypatch, tmp_path):
        working_checkout(monkeypatch, tmp_path)
        monkeypatch.setattr(health, "missing_runtime_deps", lambda: [])
        report = health.collect()
        assert report.ok, [p.title for p in report.problems]
        assert report.repo is not None


class TestTheProblemIsStillThereWhenTheUserRunsSomething:
    """Detected at startup, needed at execute time. Different moments."""

    def test_a_missing_checkout_blocks_a_node(self, monkeypatch):
        break_checkout(monkeypatch)
        health.report_at_startup()
        assert health.blocking_problem() is not None

    def test_missing_dependencies_block_a_node(self, monkeypatch, tmp_path):
        # The checkout is stubbed present so the dependency problem is the only
        # one, and `blocking_problem` is therefore being asked about the thing
        # this test names rather than about a missing checkout.
        working_checkout(monkeypatch, tmp_path)
        monkeypatch.setattr(health, "missing_runtime_deps", lambda: ["jax", "flax"])
        health.report_at_startup()
        problem = health.blocking_problem()
        assert problem is not None
        assert "env/setup.sh" in problem.fix

    def test_a_healthy_install_blocks_nothing(self, monkeypatch, tmp_path):
        working_checkout(monkeypatch, tmp_path)
        monkeypatch.setattr(health, "missing_runtime_deps", lambda: [])
        health.report_at_startup()
        assert health.blocking_problem() is None

    def test_the_pipeline_guard_raises_the_startup_diagnosis(self, monkeypatch):
        """Not the downstream symptom, several frames inside TrajectoryForcing."""
        from tf_nodes import pipeline

        break_checkout(monkeypatch, FileNotFoundError("Set TF_REPO to the directory"))
        health.report_at_startup()
        with pytest.raises(RuntimeError) as excinfo:
            pipeline.check_startup_problems()
        assert "Set TF_REPO to the directory" in str(excinfo.value)

    def test_the_guard_is_silent_on_a_healthy_install(self, monkeypatch, tmp_path):
        from tf_nodes import pipeline

        working_checkout(monkeypatch, tmp_path)
        monkeypatch.setattr(health, "missing_runtime_deps", lambda: [])
        health.report_at_startup()
        pipeline.check_startup_problems()

    def test_the_guard_is_silent_before_any_startup_report(self):
        """Imported without on_load ever running -- tests, scripts, tooling."""
        from tf_nodes import pipeline

        pipeline.check_startup_problems()


class TestTheInstallerNoteOutlivesTheInstaller:
    """install.py runs in another process, under collapsed output, once."""

    def test_a_note_is_read_back_as_a_problem(self, monkeypatch, tmp_path):
        working_checkout(monkeypatch, tmp_path)
        health.write_setup_marker("torch 2.14.0+cpu is a CPU build.")
        problem = next(
            (p for p in health.collect().problems if "installer left a note" in p.title), None
        )
        assert problem is not None
        assert "CPU build" in problem.detail

    def test_clearing_it_removes_the_problem(self, monkeypatch, tmp_path):
        working_checkout(monkeypatch, tmp_path)
        health.write_setup_marker("something")
        health.clear_setup_marker()
        assert not any("installer left a note" in p.title for p in health.collect().problems)

    def test_clearing_a_note_that_was_never_written_is_fine(self):
        health.clear_setup_marker()
        health.clear_setup_marker()

    def test_an_unwritable_location_is_not_fatal(self, monkeypatch, tmp_path):
        """install.py runs under someone else's interpreter in someone else's
        directory. Failing to leave a note must not fail their install."""
        monkeypatch.setattr(health, "SETUP_MARKER", tmp_path / "nope" / "deep" / "marker.txt")
        assert health.write_setup_marker("text") is None

    def test_a_note_alone_does_not_block_a_node(self, monkeypatch, tmp_path):
        """The user may have fixed it by hand and not deleted the file. Nagging
        in the log is fair; refusing to run is not."""
        working_checkout(monkeypatch, tmp_path)
        monkeypatch.setattr(health, "missing_runtime_deps", lambda: [])
        health.write_setup_marker("stale advice")
        health.report_at_startup()
        assert health.blocking_problem() is None


class TestTheBannerIsWorthReading:
    def test_it_names_the_problem_and_the_fix(self, monkeypatch):
        break_checkout(monkeypatch)
        text = health.format_report(health.collect())
        assert "model code is not available" in text
        assert "TF_REPO" in text

    def test_it_says_the_nodes_still_loaded(self, monkeypatch):
        """Otherwise the banner reads like a crash, and someone reinstalls
        looking for a broken download."""
        break_checkout(monkeypatch)
        assert "registered either way" in health.format_report(health.collect())


class TestASecondCopyIsNoticed:
    """Two installs register the same node names, and ComfyUI picks silently.

    The symptom is that edits appear to do nothing, or a fix keeps not taking.
    Nothing about that points at the cause, so it is worth saying outright.
    """

    def _fake_tree(self, tmp_path, *names, ext="ComfyUI-TrajectoryForcing"):
        custom_nodes = tmp_path / "custom_nodes"
        for name in (ext, *names):
            (custom_nodes / name / "tf_nodes").mkdir(parents=True)
            (custom_nodes / name / "tf_nodes" / "nodes.py").write_text("")
        return custom_nodes / ext

    def test_one_copy_is_not_a_problem(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health, "EXT_ROOT", self._fake_tree(tmp_path))
        assert health._duplicate_install_problem() is None

    def test_a_second_copy_is_reported(self, tmp_path, monkeypatch):
        root = self._fake_tree(tmp_path, "comfyui-trajectoryforcing")
        monkeypatch.setattr(health, "EXT_ROOT", root)
        problem = health._duplicate_install_problem()
        assert problem is not None
        assert "comfyui-trajectoryforcing" in problem.detail

    def test_the_registry_and_git_names_differ_which_is_the_point(self, tmp_path, monkeypatch):
        """A name-based check would miss exactly the case worth catching: the
        registry unpacks lowercase, a clone is usually CamelCase."""
        root = self._fake_tree(tmp_path, "comfyui-trajectoryforcing")
        monkeypatch.setattr(health, "EXT_ROOT", root)
        assert root.name != "comfyui-trajectoryforcing"
        assert health._duplicate_install_problem() is not None

    def test_a_disabled_copy_is_ignored(self, tmp_path, monkeypatch):
        """ComfyUI-Manager disables by renaming; a disabled pack registers
        nothing, so warning about it would be noise."""
        root = self._fake_tree(tmp_path, "comfyui-trajectoryforcing.disabled")
        monkeypatch.setattr(health, "EXT_ROOT", root)
        assert health._duplicate_install_problem() is None

    def test_an_unrelated_custom_node_is_ignored(self, tmp_path, monkeypatch):
        root = self._fake_tree(tmp_path)
        (root.parent / "some-other-pack").mkdir()
        (root.parent / "some-other-pack" / "__init__.py").write_text("")
        monkeypatch.setattr(health, "EXT_ROOT", root)
        assert health._duplicate_install_problem() is None

    def test_an_unreadable_parent_is_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health, "EXT_ROOT", tmp_path / "gone" / "ext")
        assert health._duplicate_install_problem() is None

    def test_a_duplicate_does_not_block_a_node(self, tmp_path, monkeypatch):
        """It is worth saying, but the nodes do work -- one of them, anyway.
        Refusing to run would be a worse outcome than the ambiguity."""
        root = self._fake_tree(tmp_path, "comfyui-trajectoryforcing")
        monkeypatch.setattr(health, "EXT_ROOT", root)
        working_checkout(monkeypatch, tmp_path)
        monkeypatch.setattr(health, "missing_runtime_deps", lambda: [])
        health.report_at_startup()
        assert health.blocking_problem() is None
