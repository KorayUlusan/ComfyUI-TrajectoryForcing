"""One place that says what this install has, and what to do about what it hasn't.

    python -m tf_nodes.doctor

Every other diagnostic here answers one question at the moment it goes wrong.
This answers all of them at once, on demand, which is what someone needs when
the thing they are looking at is a symptom two steps downstream -- and what
should be pasted into a bug report instead of a screenshot of a traceback.

Two rules, both learned the hard way in this repo:

* **Never import jax to find out whether jax is there.** Importing it
  initialises the GPU backend, which `configure_jax_env` has to precede and
  which cannot be undone in a running process. A doctor that breaks the patient
  is not a doctor. Presence is probed with `find_spec`; devices are only listed
  when the caller opts in with --devices.
* **Never let a check take the report down.** Each one is wrapped, because the
  whole point is to run when things are broken. A doctor that only works on a
  healthy install reports "healthy" and nothing else.

Exit code is 0 when nothing is blocking and 1 otherwise, so it can gate a
script.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parent.parent

OK = "ok"
WARN = "warn"
BAD = "problem"

_MARK = {OK: "  ok  ", WARN: " note ", BAD: " FAIL "}

#: One string, so the same advice from several failing checks collapses to one
#: line instead of three near-identical ones.
SETUP_FIX = "bash env/setup.sh -- builds a venv with a matching torch and the JAX stack."


class Findings:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.fixes: list[str] = []

    def add(self, status: str, label: str, value: str, fix: str | None = None) -> None:
        self.rows.append((status, label, value))
        if fix and status != OK:
            self.fixes.append(fix)

    def check(self, label: str, fn, fix: str | None = None) -> None:
        """Run one probe. A probe that explodes is itself a finding."""
        try:
            status, value = fn()
        except Exception as exc:  # noqa: BLE001 - this tool exists for broken installs
            self.add(BAD, label, f"could not be determined: {type(exc).__name__}: {exc}", fix)
            return
        self.add(status, label, value, fix)

    @property
    def blocking(self) -> bool:
        return any(status == BAD for status, _, _ in self.rows)


def _python() -> tuple[str, str]:
    v = sys.version_info
    status = OK if (v.major, v.minor) == (3, 11) else WARN
    return status, f"{platform.python_version()} at {sys.executable}"


def _torch() -> tuple[str, str]:
    if find_spec("torch") is None:
        return BAD, "not installed"
    import torch

    cuda = torch.version.cuda
    if cuda is None:
        return BAD, f"{torch.__version__} (CPU build; these nodes need a CUDA build)"
    major = cuda.split(".")[0]
    if major != "12":
        return BAD, f"{torch.__version__}, CUDA {cuda} (jax 0.4.36 needs CUDA 12)"
    return OK, f"{torch.__version__}, CUDA {cuda}"


def _gpu() -> tuple[str, str]:
    if find_spec("torch") is None:
        return WARN, "unknown (no torch to ask)"
    import torch

    if not torch.cuda.is_available():
        # Expected, and fine, on a cluster login node: the install is being
        # prepared here and run somewhere else. Say so, or this reads as a
        # broken install to someone doing exactly the right thing.
        where = " (normal on a Slurm login node -- run this again inside the job)"
        return BAD, "torch cannot see a GPU" + (where if shutil.which("sbatch") else "")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    status = OK if total >= 7.5 else WARN
    return status, f"{name}, {total:.1f} GiB"


def _jax_stack() -> tuple[str, str]:
    from . import pipeline

    missing = [n for n in pipeline.RUNTIME_DEPS if find_spec(n) is None]
    if missing:
        return BAD, f"missing: {', '.join(missing)}"
    return OK, "jax, flax, orbax, ml_collections all present"


def _jax_devices() -> tuple[str, str]:
    """Only ever called behind --devices; this is the one probe with side effects."""
    import jax

    devices = jax.devices()
    kinds = {d.platform for d in devices}
    if "gpu" not in kinds and "cuda" not in kinds:
        return BAD, f"jax {jax.__version__} sees no GPU: {devices}"
    return OK, f"jax {jax.__version__} on {devices[0]}"


def _checkout() -> tuple[str, str]:
    from .locate import TF_REPO_COMMIT, tf_repo

    # Suppressed: a diagnostic that silently clones several hundred MB is doing
    # something the user did not ask for, at the worst possible moment.
    os.environ.setdefault("TF_NO_AUTO_FETCH", "1")
    path = tf_repo()
    head = "unknown"
    git = shutil.which("git")
    if git:
        import subprocess

        out = subprocess.run(
            (git, "rev-parse", "HEAD"), cwd=path, capture_output=True, text=True
        )
        head = out.stdout.strip() or "unknown"
    if head not in ("unknown", TF_REPO_COMMIT):
        return WARN, f"{path}\n       at {head[:12]}, pinned to {TF_REPO_COMMIT[:12]}"
    return OK, f"{path} @ {head[:12]}"


def _model_roots() -> list[Path]:
    """Where checkpoints live, without needing the extension to have loaded.

    `locate.model_roots()` asks `folder_paths`, which raises KeyError until
    `register_model_folder()` has run -- and this tool is most useful precisely
    when ComfyUI has not got that far. Fall back to the path that registration
    would have produced, rather than registering (and creating directories) as a
    side effect of asking a question.
    """
    from .locate import MODELS_FOLDER, model_roots

    try:
        return model_roots()
    except KeyError:
        import folder_paths

        return [Path(folder_paths.models_dir) / MODELS_FOLDER]


def _weights() -> tuple[str, str]:
    from .locate import rae_root

    lines, missing = [], False
    for root in _model_roots():
        entries = (
            [e.name for e in sorted(root.iterdir()) if e.name not in (".cache", "checkpoints")]
            if root.is_dir()
            else []
        )
        lines.append(f"{root}: {', '.join(entries) if entries else 'empty'}")
        missing = missing or not entries
    decoder = Path(rae_root()) / "decoders"
    lines.append(f"RAE decoder: {decoder}{'' if decoder.is_dir() else '  (absent)'}")
    # Absent weights are not a problem: they download on first use. Saying so is
    # the point, since 3.5 GB of silence otherwise looks like a hang.
    return (WARN if missing or not decoder.is_dir() else OK), "\n       ".join(lines)


def _disk() -> tuple[str, str]:
    free = shutil.disk_usage(EXT_ROOT).free / 1024**3
    status = OK if free >= 15 else (WARN if free >= 5 else BAD)
    return status, f"{free:.1f} GiB free at {EXT_ROOT}"


def _installer_note() -> tuple[str, str]:
    from .health import SETUP_MARKER

    if not SETUP_MARKER.exists():
        return OK, "none"
    return WARN, SETUP_MARKER.read_text(encoding="utf8", errors="replace").strip()


def _duplicates() -> tuple[str, str]:
    from .health import _duplicate_install_problem

    problem = _duplicate_install_problem()
    if problem is None:
        return OK, "one copy"
    return WARN, problem.detail.replace("\n", "\n       ")


def run(devices: bool = False) -> Findings:
    f = Findings()
    f.check("python", _python, "These nodes are built and tested on 3.11.")
    f.check("torch", _torch, SETUP_FIX)
    f.check("gpu", _gpu, "Check nvidia-smi, and that ComfyUI runs from the right venv.")
    f.check("jax stack", _jax_stack, SETUP_FIX)
    if devices:
        f.check("jax devices", _jax_devices, SETUP_FIX)
    f.check(
        "TrajectoryForcing",
        _checkout,
        "Set TF_REPO to a checkout holding pmf.py and editing_env/, or let it fetch one.",
    )
    f.check("weights", _weights, "Absent weights download on first run; allow ~3.5 GB and time.")
    f.check("disk", _disk, "First run downloads ~3.5 GB of weights.")
    f.check("copies installed", _duplicates, "Remove or rename one, then restart ComfyUI.")
    f.check("installer note", _installer_note, "Delete SETUP-REQUIRED.txt once it no longer applies.")
    return f


def format_findings(f: Findings) -> str:
    out = ["", "Trajectory Forcing -- install report", "=" * 72]
    for status, label, value in f.rows:
        out.append(f"[{_MARK[status]}] {label:18} {value}")
    out.append("=" * 72)
    if f.fixes:
        out.append("")
        out.append("What to do:")
        for fix in dict.fromkeys(f.fixes):  # de-duplicated, order kept
            out.append(f"  - {fix}")
    else:
        out.append("Nothing to do; this install looks complete.")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tf_nodes.doctor",
        description="Report what this Trajectory Forcing install has and what it needs.",
    )
    parser.add_argument(
        "--devices",
        action="store_true",
        help="also ask JAX what it can see. Initialises the GPU backend, so do "
        "not use it inside a running ComfyUI.",
    )
    args = parser.parse_args(argv)
    findings = run(devices=args.devices)
    print(format_findings(findings))
    return 1 if findings.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
