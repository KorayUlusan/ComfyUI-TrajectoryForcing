"""Where the TrajectoryForcing checkout and the TF weights live.

Deliberately free of any ComfyUI import at module scope: the pure-numpy half of
this extension (and its tests) has to run outside a ComfyUI process, and only
the two model-folder helpers at the bottom need `folder_paths`.
"""
from __future__ import annotations

import os
from pathlib import Path

# .../ComfyUI-TrajectoryForcing/tf_nodes/locate.py -> .../ComfyUI-TrajectoryForcing
# resolve() follows the custom_nodes symlink back to the git checkout, which is
# what makes the sibling lookup in tf_repo() work when ComfyUI loads us.
EXT_ROOT = Path(__file__).resolve().parent.parent

# ComfyUI models/<MODELS_FOLDER>/ holds the flow checkpoints this extension loads.
MODELS_FOLDER = "trajectory_forcing"

# Subdirectory of MODELS_FOLDER reserved for RAE decoder weights. The name is not
# ours to choose: utils/rae_decoder.py only auto-downloads a missing decoder when
# the path contains the literal marker "checkpoints/rae/", from which it derives
# the repo-relative filename.
RAE_SUBDIR = Path("checkpoints") / "rae"

_TF_REPO: Path | None = None

TF_REPO_URL = "https://github.com/mervekocabas/TrajectoryForcing.git"

# The upstream commit this extension is developed and tested against. Pinned
# rather than tracking main for the same reason the model code is imported
# rather than vendored: the extension calls into TF's own API, so a change up
# there is a change here, and it should arrive when someone bumps this line and
# re-runs the smoke tests -- not silently, on a stranger's first install.
#
# To update: bump the sha, run `./slurm/submit.sh slurm/gpu_smoke.sbatch`, and
# keep the job id with the result.
TF_REPO_COMMIT = "2fab8a6acd08efa2532b0312ee03fb68f8ef8e7e"

# Where an automatic fetch puts it. Inside the extension, so a Manager install is
# self-contained and uninstalling takes the checkout with it.
TF_REPO_FETCH_DIR = EXT_ROOT / "TrajectoryForcing"


def _candidates() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    env = os.environ.get("TF_REPO", "").strip()
    if env:
        out.append(("$TF_REPO", Path(env).expanduser()))
    out.append(("sibling of the extension", EXT_ROOT.parent / "TrajectoryForcing"))
    out.append(("inside the extension", TF_REPO_FETCH_DIR))
    return out


def _looks_like_tf_repo(path: Path) -> bool:
    return (path / "editing_env" / "tf_pipeline.py").is_file() and (path / "pmf.py").is_file()


def _git(*args: str, cwd: Path | None = None) -> None:
    import subprocess

    subprocess.run(("git", *args), cwd=cwd, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def fetch_tf_repo(dest: Path = TF_REPO_FETCH_DIR, progress=None) -> Path:
    """Clone TrajectoryForcing at the pinned commit.

    Most people arrive through ComfyUI Manager, which installs this extension and
    nothing else -- so "now clone a second repository by hand" is a wall in front
    of the first run, and one the Manager gives no way to satisfy. The model code
    still is not vendored; it is fetched, pinned, and left as its own checkout.

    `git init` + a fetch of the one commit rather than `clone --depth 1`, because
    a shallow clone gets whatever main points at today, which is exactly the
    unpinned dependency this is avoiding. If the server refuses a fetch by sha
    (some mirrors do), that is reported rather than silently falling back to an
    unpinned tip.
    """
    if _looks_like_tf_repo(dest):
        return dest
    if dest.exists() and any(dest.iterdir()):
        raise FileNotFoundError(
            f"{dest} exists but does not look like a TrajectoryForcing checkout "
            f"(no pmf.py / editing_env/). Remove it, or set TF_REPO to a good one."
        )
    import shutil

    if shutil.which("git") is None:
        raise FileNotFoundError(
            "TrajectoryForcing is not present and `git` is not installed to fetch it. "
            f"Clone {TF_REPO_URL} yourself and point TF_REPO at it."
        )
    # This is the one slow thing that can happen with no node on screen to show
    # it -- TF Load Pipeline's own bar does not start until the checkout is
    # found. ComfyUI prints these to its console and, since the launchers keep
    # rolling logs, to a file afterwards; without them a first run looks hung,
    # which is exactly how the slow load was reported before it got a bar.
    import logging

    note = f"fetching TrajectoryForcing @ {TF_REPO_COMMIT[:8]} into {dest} (one time)"
    logging.getLogger(__name__).info(note)
    if progress is not None:
        progress(note)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        _git("init", "-q", str(dest))
        _git("remote", "add", "origin", TF_REPO_URL, cwd=dest)
        _git("fetch", "-q", "--depth", "1", "origin", TF_REPO_COMMIT, cwd=dest)
        _git("checkout", "-q", "FETCH_HEAD", cwd=dest)
    except Exception:
        # Leaving a half-clone behind would make the next attempt take the
        # "exists but is not a checkout" branch above and need manual cleanup.
        shutil.rmtree(dest, ignore_errors=True)
        raise
    if not _looks_like_tf_repo(dest):
        shutil.rmtree(dest, ignore_errors=True)
        raise FileNotFoundError(
            f"Fetched {TF_REPO_URL} into {dest} but it has no pmf.py / editing_env/."
        )
    return dest


def tf_repo(progress=None) -> Path:
    """The TrajectoryForcing checkout this extension runs against.

    Everything (model code, configs, the RAE decoder sources) is imported from
    there rather than vendored, so the extension always tracks the pinned
    upstream instead of a stale copy of its math. An existing checkout is always
    preferred over fetching one -- `$TF_REPO` first, then a sibling of this
    extension -- so nobody ends up with two copies of a 2 GB decoder.
    """
    global _TF_REPO
    if _TF_REPO is not None:
        return _TF_REPO
    tried = []
    for label, path in _candidates():
        if _looks_like_tf_repo(path):
            _TF_REPO = path
            return _TF_REPO
        tried.append(f"  {label}: {path}")
    if os.environ.get("TF_NO_AUTO_FETCH", "").strip():
        raise FileNotFoundError(
            "Could not find the TrajectoryForcing checkout, and TF_NO_AUTO_FETCH is set. "
            "Tried:\n" + "\n".join(tried)
            + "\nSet TF_REPO to the directory containing pmf.py and editing_env/."
        )
    _TF_REPO = fetch_tf_repo(progress=progress)
    return _TF_REPO


def tf_configs() -> list[str]:
    """YAML config filenames shipped by TrajectoryForcing, edit-env one first."""
    names = sorted(p.name for p in (tf_repo() / "configs").glob("*.yml"))
    preferred = "edit_env_config.yml"
    if preferred in names:
        names.remove(preferred)
        names.insert(0, preferred)
    return names


def tf_config_path(name: str) -> Path:
    path = tf_repo() / "configs" / name
    if not path.is_file():
        raise FileNotFoundError(f"No such TrajectoryForcing config: {path}")
    return path


def imagenet_classes() -> dict[int, str]:
    """ImageNet-1k id -> class name, read from the editing env's JSON."""
    import json

    with open(tf_repo() / "editing_env" / "imagenet_classes.json", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


# ---------------------------------------------------------------------------
# ComfyUI models/ folder (registered by the extension's comfy_entrypoint)
# ---------------------------------------------------------------------------
AUTO_CHECKPOINT = "auto (download TF_L_edit)"
HF_CKPT_REPO = "mervekocabas/TrajectoryForcing"
HF_CKPT_FILE = "TF_L_edit"


def register_model_folder() -> Path:
    """Register models/trajectory_forcing/ with ComfyUI and return its path.

    Registered with an empty extension set, which `folder_paths` reads as "list
    everything" -- TF's checkpoints are extensionless files (`TF_L_edit`), so an
    extension filter would hide every one of them.
    """
    import folder_paths

    root = Path(folder_paths.models_dir) / MODELS_FOLDER
    # Created eagerly: an empty folder is where a user drops a checkpoint, and a
    # missing one just looks like the extension did not load.
    root.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(MODELS_FOLDER, str(root), is_default=True)
    return root


def model_roots() -> list[Path]:
    import folder_paths

    return [Path(p) for p in folder_paths.get_folder_paths(MODELS_FOLDER)]


def list_checkpoints() -> list[str]:
    """Flow checkpoints available in models/trajectory_forcing/.

    Both files and directories are listed: a released TF checkpoint is a single
    extensionless file, but a checkpoint exported by `third_party/fd_loss` is an
    orbax directory. `checkpoints/` is skipped -- that subtree is RAE weights.
    """
    names: list[str] = []
    for root in model_roots():
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if entry.name.startswith(".") or entry.name == "checkpoints":
                continue
            names.append(entry.name)
    return [AUTO_CHECKPOINT] + names


def resolve_checkpoint(name: str) -> str:
    """Turn a dropdown entry into an absolute checkpoint path, downloading if asked."""
    if name == AUTO_CHECKPOINT:
        return download_default_checkpoint()
    for root in model_roots():
        candidate = root / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Checkpoint {name!r} not found under {[str(p) for p in model_roots()]}. "
        f"Pick {AUTO_CHECKPOINT!r} to fetch the released one."
    )


def download_default_checkpoint() -> str:
    """Fetch TF_L_edit into models/trajectory_forcing/ so it shows up in the dropdown."""
    from huggingface_hub import hf_hub_download

    root = model_roots()[0]
    root.mkdir(parents=True, exist_ok=True)
    target = root / HF_CKPT_FILE
    if target.exists():
        return str(target)
    return hf_hub_download(
        repo_id=os.environ.get("TF_CKPT_REPO", HF_CKPT_REPO),
        filename=os.environ.get("TF_CKPT_FILE", HF_CKPT_FILE),
        repo_type="model",
        local_dir=str(root),
        token=os.environ.get("HF_TOKEN"),
    )


# ComfyUI's "Node 2.0" (Vue) node rendering. The Painter widget only exists
# there; under classic LiteGraph rendering it degrades to the text "Node 2.0
# only" and cannot be painted on at all.
VUE_NODES_SETTING = "Comfy.VueNodes.Enabled"


def vue_nodes_enabled() -> bool | None:
    """Whether the browser user has Node 2.0 on, or None if it cannot be told.

    Read from ComfyUI's own per-user settings file rather than guessed, so the
    canvas node can say something specific instead of a blanket warning that
    would nag the majority who already have it on. Returns None when there is no
    settings file yet (a fresh install), where nagging would be wrong.
    """
    import json

    import folder_paths

    users = Path(folder_paths.get_user_directory())
    found = None
    for settings in sorted(users.glob("*/comfy.settings.json")):
        try:
            value = json.loads(settings.read_text()).get(VUE_NODES_SETTING)
        except (OSError, ValueError):
            continue
        if value is not None:
            found = bool(value)
            # Any user with it off is the one who needs telling.
            if not found:
                return False
    return found


def rae_root() -> str:
    """Directory the RAE decoder weights are read from (and downloaded into).

    Prefers a copy already sitting in the TrajectoryForcing checkout -- the
    editing env and `scripts/download_models.sh` both put it there, and a second
    2 GB copy under ComfyUI's models/ buys nothing. Falls back to the ComfyUI
    convention when the checkout has none, so a fresh clone downloads to the
    place a ComfyUI user expects to find weights.
    """
    override = os.environ.get("TF_RAE_ROOT", "").strip()
    if override:
        return str(Path(override).expanduser())
    in_repo = tf_repo() / RAE_SUBDIR
    if (in_repo / "decoders").is_dir():
        return str(in_repo)
    return str(model_roots()[0] / RAE_SUBDIR)
