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


def _candidates() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    env = os.environ.get("TF_REPO", "").strip()
    if env:
        out.append(("$TF_REPO", Path(env).expanduser()))
    out.append(("sibling of the extension", EXT_ROOT.parent / "TrajectoryForcing"))
    out.append(("inside the extension", EXT_ROOT / "TrajectoryForcing"))
    return out


def _looks_like_tf_repo(path: Path) -> bool:
    return (path / "editing_env" / "tf_pipeline.py").is_file() and (path / "pmf.py").is_file()


def tf_repo() -> Path:
    """The TrajectoryForcing checkout this extension runs against.

    Everything (model code, configs, the RAE decoder sources) is imported from
    there rather than vendored, so the extension always tracks the pinned
    submodule instead of a stale copy of its math.
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
    raise FileNotFoundError(
        "Could not find the TrajectoryForcing checkout. Tried:\n"
        + "\n".join(tried)
        + "\nSet TF_REPO to the directory containing pmf.py and editing_env/."
    )


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
