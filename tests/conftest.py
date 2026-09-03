"""Put the extension and ComfyUI on the import path.

`tf_nodes` is an ordinary package, so tests import it directly once the
extension root is on `sys.path`. Only the top-level `__init__.py` needs the
file-location dance, because ComfyUI loads it under a directory name that is not
a legal module name.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

EXT_ROOT = Path(__file__).resolve().parent.parent


def _comfy_root() -> Path | None:
    env = os.environ.get("COMFYUI_ROOT", "").strip()
    candidates = [Path(env)] if env else []
    candidates += [EXT_ROOT.parent.parent / "ComfyUI", Path.home() / "ComfyUI"]
    for path in candidates:
        if (path / "comfy_api" / "latest" / "__init__.py").is_file():
            return path
    return None


COMFY_ROOT = _comfy_root()

if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))
if COMFY_ROOT is not None and str(COMFY_ROOT) not in sys.path:
    sys.path.append(str(COMFY_ROOT))

requires_comfy = pytest.mark.skipif(COMFY_ROOT is None, reason="no ComfyUI checkout found")


@pytest.fixture(scope="session")
def extension():
    """The top-level package, loaded the way ComfyUI loads a custom node directory."""
    if COMFY_ROOT is None:
        pytest.skip("no ComfyUI checkout found")
    name = "comfyui_trajectoryforcing"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, EXT_ROOT / "__init__.py", submodule_search_locations=[str(EXT_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
