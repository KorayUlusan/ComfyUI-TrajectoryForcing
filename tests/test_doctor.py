"""The install report, which has to work when nothing else does.

A diagnostic that only runs on a healthy install is worse than none: it reports
"healthy" and nothing else, and the person reading it concludes their problem is
somewhere more exotic than it is. So the property under test is not that the
checks are right, it is that a failing check becomes a line in the report rather
than a traceback.

The one probe with a side effect, `--devices`, is not exercised here. Importing
jax initialises its GPU backend for the whole process, which would leak into
every test that ran afterwards -- the same reason the doctor keeps it behind a
flag.
"""
from __future__ import annotations

import pytest

from tf_nodes import doctor


class TestABrokenProbeIsAFindingNotACrash:
    """The whole contract. This tool runs when things are already wrong."""

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("torch is a smoking hole"),
            KeyError("trajectory_forcing"),
            OSError("filesystem went away"),
            ImportError("no module named anything"),
        ],
    )
    def test_an_exploding_check_lands_in_the_report(self, exc):
        f = doctor.Findings()

        def boom():
            raise exc

        f.check("thing", boom, "the fix")
        status, label, value = f.rows[0]
        assert status == doctor.BAD
        assert label == "thing"
        assert "could not be determined" in value
        assert "the fix" in f.fixes

    def test_a_full_run_never_raises(self):
        """Whatever this machine looks like, `run()` returns a report."""
        assert doctor.run().rows

    def test_every_check_produces_exactly_one_row(self):
        f = doctor.run()
        labels = [label for _, label, _ in f.rows]
        assert len(labels) == len(set(labels)), f"duplicate rows: {labels}"

    def test_the_key_error_that_this_tool_hit_itself_is_handled(self):
        """`model_roots()` raises KeyError until `register_model_folder()` runs,
        which is exactly the state the doctor is most useful in. It reported
        `could not be determined: KeyError` on its first run."""
        roots = doctor._model_roots()
        assert roots and all(hasattr(r, "is_dir") for r in roots)


class TestItDoesNotDisturbWhatItMeasures:
    def test_probing_the_jax_stack_does_not_import_jax(self):
        """Importing jax initialises its GPU backend, which configure_jax_env
        has to precede and which cannot be undone in a running process."""
        import sys

        had_jax = "jax" in sys.modules
        doctor._jax_stack()
        assert had_jax or "jax" not in sys.modules

    def test_looking_for_the_checkout_does_not_clone(self, monkeypatch):
        """A diagnostic that silently downloads is doing something unasked, at
        the worst moment. Asked via `allow_fetch=False`, not the env var."""
        called = []
        from tf_nodes import locate

        monkeypatch.setattr(locate, "fetch_tf_repo", lambda *a, **k: called.append(1))
        monkeypatch.delenv("TF_NO_AUTO_FETCH", raising=False)
        try:
            doctor._checkout()
        except Exception:  # noqa: BLE001 - a missing checkout is fine, a clone is not
            pass
        assert not called

    def test_a_full_run_never_clones(self, monkeypatch, tmp_path):
        """CI caught this one, and the local suite could not.

        `_checkout` passed allow_fetch=False, which looked like enough. But
        `_weights` reaches `rae_root()`, which called `tf_repo()` with fetching
        allowed -- so on a runner with no checkout the doctor cloned upstream
        while answering "where would the decoder live". Here every candidate is
        an empty temp directory and git is an error, so the machine running the
        test cannot hide it the way a dev box does.
        """
        from tf_nodes import locate

        monkeypatch.setattr(locate, "EXT_ROOT", tmp_path / "ext")
        monkeypatch.setattr(locate, "TF_REPO_FETCH_DIR", tmp_path / "ext" / "TrajectoryForcing")
        monkeypatch.setattr(locate, "_TF_REPO", None)
        monkeypatch.delenv("TF_REPO", raising=False)
        monkeypatch.delenv("TF_RAE_ROOT", raising=False)

        def boom(*a, **k):
            raise AssertionError("the doctor tried to clone TrajectoryForcing")

        monkeypatch.setattr(locate, "_git", boom)
        monkeypatch.setattr(locate, "fetch_tf_repo", boom)

        doctor.run()  # must not raise, and must not reach either boom

    def test_it_does_not_change_the_environment(self, monkeypatch):
        """The regression that turned CI red.

        `_checkout` used to `os.environ.setdefault("TF_NO_AUTO_FETCH", "1")`,
        which is process-wide. Every test that ran afterwards then found
        fetching disabled, and on a runner -- where there is no checkout until
        something fetches one -- eighteen of them failed a long way from the
        cause. It passed here only because a sibling checkout exists locally and
        no fetch was ever needed.

        Nothing in this module may write to os.environ. A probe that changes what
        it is probing is not a probe.
        """
        import os

        before = dict(os.environ)
        try:
            doctor.run()
        except Exception:  # noqa: BLE001 - a broken install is the normal case here
            pass
        assert dict(os.environ) == before, (
            "doctor mutated the environment: "
            f"{set(os.environ.items()) ^ set(before.items())}"
        )


class TestTheReportIsActionable:
    def test_identical_advice_from_several_checks_appears_once(self):
        """torch and the JAX stack fail together on a fresh Manager install, and
        both are fixed by the same command. Three copies of it is noise."""
        f = doctor.Findings()
        f.add(doctor.BAD, "torch", "cpu build", doctor.SETUP_FIX)
        f.add(doctor.BAD, "jax stack", "missing", doctor.SETUP_FIX)
        text = doctor.format_findings(f)
        assert text.count(doctor.SETUP_FIX) == 1

    def test_a_passing_check_contributes_no_advice(self):
        f = doctor.Findings()
        f.add(doctor.OK, "torch", "2.8.0+cu128", doctor.SETUP_FIX)
        assert "What to do" not in doctor.format_findings(f)
        assert "looks complete" in doctor.format_findings(f)

    def test_a_warning_still_contributes_advice(self):
        """A note is not a failure, but it is still something to act on."""
        f = doctor.Findings()
        f.add(doctor.WARN, "weights", "empty", "they download on first run")
        assert "they download on first run" in doctor.format_findings(f)

    def test_only_a_hard_failure_sets_the_exit_code(self):
        f = doctor.Findings()
        f.add(doctor.WARN, "weights", "empty")
        assert not f.blocking
        f.add(doctor.BAD, "torch", "not installed")
        assert f.blocking

    def test_every_row_renders(self):
        text = doctor.format_findings(doctor.run())
        for _, label, _ in doctor.run().rows:
            assert label in text


def fake_torch_without_a_gpu(monkeypatch):
    """A torch that imports and sees no device.

    A bare class is not enough: `_gpu` probes with `find_spec` first, which
    raises ValueError on an object with no `__spec__`.
    """
    import importlib.util
    import sys
    import types

    module = types.ModuleType("torch")
    module.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
    module.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", module)


class TestExpectedStatesAreNotFailures:
    """A doctor that cries FAIL on a correct install gets ignored.

    Both of these were reported as failures on a machine that had just been
    installed exactly as the README says. The person reading it had to reason
    their way past two FAILs to conclude nothing was wrong, which is the
    opposite of what this tool is for.
    """

    def test_an_unfetched_checkout_is_a_note_not_a_failure(self, monkeypatch, tmp_path):
        """ComfyUI fetches it at startup, so before the first start there is
        legitimately none."""
        from tf_nodes import locate

        monkeypatch.setattr(locate, "EXT_ROOT", tmp_path / "ext")
        monkeypatch.setattr(locate, "TF_REPO_FETCH_DIR", tmp_path / "ext" / "TrajectoryForcing")
        monkeypatch.setattr(locate, "_TF_REPO", None)
        monkeypatch.delenv("TF_REPO", raising=False)

        status, detail = doctor._checkout()
        assert status == doctor.WARN, f"got {status}: {detail}"
        assert "startup" in detail

    def test_no_gpu_on_a_slurm_login_node_is_a_note(self, monkeypatch):
        """Preparing an install on a login node and running it in a job is the
        documented route, so a missing GPU there is expected."""
        import shutil as _shutil

        def which(name):
            return "/usr/bin/sbatch" if name == "sbatch" else _shutil.which(name)

        monkeypatch.setattr(doctor.shutil, "which", which)

        fake_torch_without_a_gpu(monkeypatch)
        status, detail = doctor._gpu()
        assert status == doctor.WARN
        assert "login node" in detail

    def test_no_gpu_without_slurm_is_still_a_failure(self, monkeypatch):
        """On an ordinary machine a missing GPU means the nodes cannot run."""
        monkeypatch.setattr(doctor.shutil, "which", lambda n: None)

        fake_torch_without_a_gpu(monkeypatch)
        status, _ = doctor._gpu()
        assert status == doctor.BAD
