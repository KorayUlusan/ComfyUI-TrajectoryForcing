"""Say what is wrong, where the person who can fix it is actually looking.

Three failures used to surface a long way from that person:

* `on_load` fetched the TrajectoryForcing checkout, unguarded. No network, no
  `git`, or a half-finished clone, and the exception propagated out of the
  entry point -- so all 21 nodes vanished and ComfyUI-Manager said only
  `IMPORT FAILED`. The genuinely useful message went to the server console,
  which someone driving a browser never reads.
* the dependency check lived in `TFPipeline.__init__`, so an incomplete
  environment looked fine until a graph was wired and run.
* `install.py`'s decline printed during the Manager install, where output is
  collapsed by default and scrolls away.

The fix is the same in all three cases and is not a better message -- the
messages were already good. It is that a problem has to be *recorded* when it is
detected and *repeated* when the user next does something, because those are
different moments. So: nothing here raises, everything is collected, and the
result is logged at startup and available to nodes at run time.

Deliberately import-light. It runs before anything else and must not be the
reason a load fails.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parent.parent

#: Written by `install.py` when it declines to touch the environment, read back
#: here at load. A file rather than a log line because the two events are in
#: different processes, minutes or days apart.
SETUP_MARKER = EXT_ROOT / "SETUP-REQUIRED.txt"

log = logging.getLogger("TrajectoryForcing")


@dataclass
class Problem:
    """One thing wrong, and the single next thing to do about it."""

    title: str
    detail: str
    fix: str


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    #: Where TrajectoryForcing was found, when it was.
    repo: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


#: Filled in by `on_load` so nodes can repeat the startup diagnosis at run time
#: instead of failing with something further downstream.
LAST_REPORT: Report | None = None


def _importable(name: str) -> bool:
    """Present, without importing it.

    Importing jax initialises its GPU backend, which `configure_jax_env` has to
    happen before and which cannot be undone inside a running process.
    """
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):  # broken or half-installed
        return False


def missing_runtime_deps() -> list[str]:
    """Which of the JAX-stack modules are absent.

    Reads `pipeline.RUNTIME_DEPS` through the module rather than importing the
    names, so a test that monkeypatches that list is honoured here too.
    """
    from . import pipeline

    return [name for name in pipeline.RUNTIME_DEPS if not _importable(name)]


def _setup_marker_problem() -> Problem | None:
    try:
        text = SETUP_MARKER.read_text(encoding="utf8", errors="replace").strip()
    except OSError:
        return None
    return Problem(
        title="The installer left a note",
        detail=text,
        fix=f"When it no longer applies, delete {SETUP_MARKER.name}.",
    )


def _dependency_problem() -> Problem | None:
    missing = missing_runtime_deps()
    if not missing:
        return None
    return Problem(
        title=f"The model stack is not installed ({', '.join(missing)} missing)",
        detail=(
            "The nodes will load and appear in the menu, but TF Load Pipeline will "
            "fail. This extension runs a JAX model inside the ComfyUI process, and "
            "the two dependency sets only coexist on CUDA 12 at torch 2.8 or above."
        ),
        fix="bash env/setup.sh, then start ComfyUI from the venv it builds.",
    )


def _checkout_problem(report: Report) -> Problem | None:
    """Locate TrajectoryForcing, recording rather than raising.

    This is the one that used to take the whole extension down with it, and the
    one most likely to fail for a reason outside the user's control -- a fetch
    needs the network to be up at exactly the moment ComfyUI starts.
    """
    try:
        from .locate import tf_repo

        report.repo = tf_repo()
        return None
    except Exception as exc:  # noqa: BLE001 - any failure here is a report, not a crash
        return Problem(
            title="TrajectoryForcing's model code is not available",
            detail=str(exc),
            fix=(
                "Check the network, or clone it yourself and set TF_REPO to the "
                "directory holding pmf.py and editing_env/."
            ),
        )


def _duplicate_install_problem() -> Problem | None:
    """A second copy of these nodes in the same custom_nodes/.

    Easy to end up with: clone the repo to work on it, then install the registry
    package to try something, and now two directories register the same node
    names. ComfyUI does not say which won, so edits appear to do nothing, or a
    bug fixed in one tree keeps happening. Cheap to detect, unreasonably
    annoying to diagnose.

    Keyed on a file the package always has rather than on the directory name,
    since the registry unpacks to `comfyui-trajectoryforcing` and a git clone is
    usually `ComfyUI-TrajectoryForcing` -- the same name check that would look
    obvious here is the one that misses the case worth catching.
    """
    marker = Path("tf_nodes") / "nodes.py"
    try:
        siblings = sorted(EXT_ROOT.parent.iterdir())
    except OSError:
        return None

    twins = []
    for entry in siblings:
        try:
            if entry.name.endswith(".disabled") or entry.resolve() == EXT_ROOT:
                continue
            if (entry / marker).is_file():
                twins.append(entry)
        except OSError:
            continue
    if not twins:
        return None
    return Problem(
        title="These nodes are installed twice",
        detail=(
            f"This copy: {EXT_ROOT}\nAlso found: "
            + "\n            ".join(str(t) for t in twins)
            + "\nBoth register the same node names, and which one ComfyUI uses is "
            "not defined."
        ),
        fix="Remove or rename one of them, then restart ComfyUI.",
    )


def collect() -> Report:
    """Everything wrong with this install. Never raises."""
    report = Report()
    for problem in (
        _checkout_problem(report),
        _dependency_problem(),
        _duplicate_install_problem(),
        _setup_marker_problem(),
    ):
        if problem is not None:
            report.problems.append(problem)
    return report


def format_report(report: Report) -> str:
    width = 78
    lines = ["", "=" * width, "  Trajectory Forcing: the nodes loaded, but something needs attention.", ""]
    for n, problem in enumerate(report.problems, 1):
        lines.append(f"  {n}. {problem.title}")
        for line in problem.detail.splitlines():
            lines.append(f"     {line}")
        lines.append(f"     -> {problem.fix}")
        lines.append("")
    lines.append("  The nodes are registered either way; this is what will fail when you run one.")
    lines.append("=" * width)
    lines.append("")
    return "\n".join(lines)


def report_at_startup() -> Report:
    """Collect, log, remember. Called once from the extension entry point."""
    global LAST_REPORT

    report = collect()
    LAST_REPORT = report
    if report.ok:
        log.info("TrajectoryForcing: repo=%s, model stack present", report.repo)
    else:
        # warning, not error: nothing is broken yet, and an error-level banner on
        # every start would train people to ignore it.
        log.warning(format_report(report))
    return report


def blocking_problem() -> Problem | None:
    """The startup problem that will stop a node from running, if any.

    Nodes call this to fail with the diagnosis made at startup rather than with
    whatever error the missing piece eventually causes. Returns the checkout or
    dependency problem; the installer's note alone is not blocking, since the
    user may have resolved it by hand.
    """
    if LAST_REPORT is None:
        return None
    for problem in LAST_REPORT.problems:
        if problem.title.startswith(("TrajectoryForcing's model code", "The model stack")):
            return problem
    return None


def write_setup_marker(text: str) -> Path | None:
    """Record an installer decision where the next ComfyUI start will find it."""
    try:
        SETUP_MARKER.write_text(text.rstrip() + "\n", encoding="utf8")
    except OSError:
        return None
    return SETUP_MARKER


def clear_setup_marker() -> None:
    """Drop a stale note, so a fixed install stops nagging."""
    try:
        os.unlink(SETUP_MARKER)
    except OSError:
        pass
