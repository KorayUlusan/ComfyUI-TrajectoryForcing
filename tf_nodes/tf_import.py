"""Importing TrajectoryForcing inside ComfyUI without the two shadowing each other.

TrajectoryForcing is a flat repo: `pmf.py`, `models/`, `utils/`, `configs/` and
`third_party/` all sit at its root, and its own code imports them by those bare
names (`from utils.rae_decoder import get_decoder`). Putting that root on
`sys.path` inside ComfyUI is not enough, and is actively harmful:

* ComfyUI ships its own top-level `utils` package (a real package, with
  `__init__.py`, imported during startup). Once it is in `sys.modules`, TF's
  `import utils.rae_decoder` resolves against ComfyUI's `utils.__path__` and
  fails -- `sys.path` is never consulted for an already-imported package.
* The reverse holds too: leaving TF's `utils` cached would break ComfyUI's own
  `utils.json_util` on the next request.
* `models` and `main` collide the same way.

So the two namespaces are swapped rather than merged. `tf_scope()` moves the
shadowed names out of `sys.modules` into a side table on entry and puts them
back on exit, keeping TF's modules warm for the next entry. Every call into
TrajectoryForcing has to happen inside it -- not just the import -- because
`utils/rae_decoder.py` imports `utils.logging_util` and `third_party.rae_decoder`
lazily, from inside functions that first run at decode time.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from .locate import tf_repo

# Top-level names owned by the TrajectoryForcing tree. `utils`, `models` and
# `main` are the ones that actually collide with ComfyUI today; the rest are
# listed so a future collision (with ComfyUI or another custom node) cannot make
# the two trees silently share a module.
_TF_TOP_LEVEL = frozenset({
    "configs", "main", "models", "pmf", "prepare_ref", "scripts", "third_party",
    "tf_pipeline", "train", "utils",
})

_TF_MODULES: dict[str, object] = {}
_LOCK = threading.RLock()
_DEPTH = 0


def _swap_in(namespace: dict[str, object]) -> dict[str, object]:
    """Replace every shadowed top-level module with `namespace`, returning the evictees."""
    evicted = {}
    for name in [n for n in sys.modules if n.split(".", 1)[0] in _TF_TOP_LEVEL]:
        evicted[name] = sys.modules.pop(name)
    sys.modules.update(namespace)
    return evicted


def _tf_paths() -> list[str]:
    root = tf_repo()
    # editing_env/ is on the path for `import tf_pipeline`; the repo root is what
    # tf_pipeline's own imports (pmf, utils, configs, third_party) resolve against.
    return [str(root), str(root / "editing_env")]


def _namespace_dirs(root: Path, prefix: str):
    """Every directory under `root` that Python would treat as a namespace package."""
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "__")):
            continue
        name = f"{prefix}.{child.name}" if prefix else child.name
        if not (child / "__init__.py").exists():
            yield name, child
        yield from _namespace_dirs(child, name)


def _bind_namespace_packages() -> None:
    """Point the shadowed package names straight at TrajectoryForcing's directories.

    Two problems, one fix.

    First, putting TF's root first on `sys.path` is not enough. None of TF's
    package directories has an `__init__.py`, so each is only a *namespace
    portion*, and the import system keeps scanning the rest of the path after
    finding one -- a regular package further along (ComfyUI's
    `utils/__init__.py`) then wins outright, however early TF's directory sits.

    Second, and less obvious: CPython gives every namespace package
    ``__file__ = None`` rather than no ``__file__`` at all. Any library that
    walks `sys.modules` guarded by ``hasattr(m, "__file__")`` -- `inspect.getmodule`
    does, and pydantic calls it while building a model class, and wandb builds
    pydantic models while TF imports it -- passes that guard and then dies in
    `inspect.getfile` with "is a built-in module". That crash is what took down
    TF Load Pipeline inside a running ComfyUI server while the same import
    succeeded in a bare script: it depends on whether some other pydantic model
    happened to warm `inspect`'s cache first. Deleting the null `__file__` makes
    these modules look like what they are -- modules with no file -- so the walk
    skips them.

    Every namespace directory in TF's tree is bound up front, not just the top
    level, so no nested one (`utils/jax_fid`, `third_party/fd_loss/...`) can be
    created lazily with the null `__file__` back in place.

    Plain modules at TF's root (`pmf.py`, `tf_pipeline.py`) need none of this:
    a module file at an earlier path entry wins immediately.
    """
    root = tf_repo()
    for name, directory in _namespace_dirs(root, ""):
        if name.split(".", 1)[0] not in _TF_TOP_LEVEL or name in sys.modules:
            continue
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(directory)]
        module = importlib.util.module_from_spec(spec)
        if getattr(module, "__file__", "") is None:
            del module.__file__
        sys.modules[name] = module


@contextmanager
def tf_scope():
    """Run a block with TrajectoryForcing's top-level modules visible instead of ComfyUI's.

    Reentrant, and serialised: ComfyUI executes one node at a time on its worker
    thread, but the swap is process-global while held, so nothing else may
    import `utils` from underneath it.
    """
    global _DEPTH
    with _LOCK:
        outermost = _DEPTH == 0
        added: list[str] = []
        comfy_modules: dict[str, object] = {}
        if outermost:
            comfy_modules = _swap_in(_TF_MODULES)
            for entry in reversed(_tf_paths()):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
                    added.append(entry)
            _bind_namespace_packages()
        _DEPTH += 1
        try:
            yield
        finally:
            _DEPTH -= 1
            if outermost:
                _TF_MODULES.clear()
                _TF_MODULES.update(_swap_in(comfy_modules))
                for entry in added:
                    sys.path.remove(entry)


def configure_jax_env() -> list[str]:
    """Set the JAX knobs that let it share one GPU with ComfyUI's torch models.

    Carried over from `editing_env/run.sh`, which already runs a JAX DiT and a
    PyTorch RAE decoder in one process. Applied at extension import so it lands
    before the first `import jax` -- the loader node's, since nothing else in
    this extension touches JAX. `run_comfyui.sh` sets the same values earlier
    still; anything already exported wins, so the two never fight.

    Returns the names of the variables this call actually set.
    """
    if "jax" in sys.modules:
        import logging

        logging.getLogger(__name__).warning(
            "TrajectoryForcing: jax was already imported before this extension loaded; "
            "XLA memory settings may not apply. Launch ComfyUI via run_comfyui.sh."
        )

    defaults = {
        # Grow VRAM on demand instead of claiming the whole card up front, so
        # comfy.model_management still has room for the torch models it juggles.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # Persist XLA compiles, so a ComfyUI restart does not re-pay JIT time.
        "JAX_COMPILATION_CACHE_DIR": str(tf_repo() / ".jax_cache"),
    }
    # A hard ceiling on JAX's share is only meaningful together with
    # preallocation -- MEM_FRACTION sizes the up-front block and does nothing
    # when PREALLOCATE is false. Setting TF_XLA_MEM_FRACTION flips both.
    fraction = os.environ.get("TF_XLA_MEM_FRACTION", "").strip()
    if fraction:
        defaults["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        defaults["XLA_PYTHON_CLIENT_MEM_FRACTION"] = fraction

    applied = []
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    os.makedirs(os.environ["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)
    return applied
