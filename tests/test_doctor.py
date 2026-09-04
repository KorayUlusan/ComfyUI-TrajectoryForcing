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
        the worst moment. `_checkout` sets TF_NO_AUTO_FETCH before looking."""
        called = []
        from tf_nodes import locate

        monkeypatch.setattr(locate, "fetch_tf_repo", lambda *a, **k: called.append(1))
        monkeypatch.delenv("TF_NO_AUTO_FETCH", raising=False)
        try:
            doctor._checkout()
        except Exception:  # noqa: BLE001 - a missing checkout is fine, a clone is not
            pass
        assert not called


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
