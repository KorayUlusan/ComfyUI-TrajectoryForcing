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


# ---------------------------------------------------------------------------
# Guards against the class of bug that turned CI red twice
# ---------------------------------------------------------------------------
# Both failures were test code mutating process-global state, and both passed
# locally because a sibling TrajectoryForcing checkout exists on a dev machine
# and makes the mutation invisible. A green local run was not evidence, so these
# make the mutation itself the failure, on the machine where it happens.
#
#   1. `os.environ.setdefault("TF_NO_AUTO_FETCH", "1")` in a probe. Process-wide,
#      so every test that ran afterwards found fetching disabled -- and on a
#      runner there is no checkout until something fetches one.
#   2. `health.collect()` resolving with fetching allowed, which cloned upstream
#      into the extension directory mid-run, so later candidate lookups found a
#      checkout that had not been there at collection time.

#: Environment this extension reads. A test may set these through monkeypatch,
#: which restores them; assigning to os.environ directly does not.
_OWNED_ENV = (
    "TF_REPO", "TF_NO_AUTO_FETCH", "TF_RAE_ROOT", "TF_CKPT_REPO", "TF_CKPT_FILE",
    "TF_NO_AUTO_DEPS", "TF_XLA_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "JAX_COMPILATION_CACHE_DIR",
)


@pytest.fixture(autouse=True)
def _no_global_leaks(request):
    """Fail the test that leaks, not the twenty that trip over it afterwards.

    Ordinary `monkeypatch.setenv` is unaffected: pytest restores it before this
    check runs. What this catches is a bare assignment from library or test
    code, which is invisible until some later test depends on the variable.
    """
    before_env = {name: os.environ.get(name) for name in _OWNED_ENV}
    fetch_dir = EXT_ROOT / "TrajectoryForcing"
    fetch_existed = fetch_dir.exists()

    yield

    after_env = {name: os.environ.get(name) for name in _OWNED_ENV}
    changed = {k: (before_env[k], after_env[k]) for k in _OWNED_ENV if before_env[k] != after_env[k]}
    assert not changed, (
        f"this test changed the environment and left it changed: {changed}. "
        "Use monkeypatch.setenv/delenv, which restores; a bare assignment leaks "
        "into every test that runs afterwards."
    )
    # Fetching is legitimate for the tests that genuinely need upstream -- on a
    # runner that is how the suite gets a checkout at all -- so they are marked
    # and exempt. Anything else doing it is the bug: it makes the "inside the
    # extension" candidate exist for every test that follows, which is what
    # stopped test_locate's opt-out case from raising.
    allowed = request.node.get_closest_marker("needs_tf_checkout") is not None
    assert allowed or fetch_existed or not fetch_dir.exists(), (
        f"this test fetched a TrajectoryForcing checkout into {fetch_dir}. Stub the "
        "resolver, or mark the test needs_tf_checkout if it genuinely needs upstream."
    )


def pytest_addoption(parser):
    parser.addoption(
        "--no-checkout",
        action="store_true",
        help="Simulate a fresh runner: no TrajectoryForcing checkout anywhere, and "
             "cloning one is an error. Tests marked needs_tf_checkout are skipped.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_tf_checkout: requires a real TrajectoryForcing checkout, so it is "
        "skipped under --no-checkout.",
    )
    if not config.getoption("--no-checkout"):
        return

    import tempfile

    from tf_nodes import locate

    # Point both filesystem candidates at an empty directory rather than
    # replacing `_candidates`, so the tests that exercise the resolver by
    # monkeypatching these same names still work -- monkeypatch restores to
    # whatever is set here.
    empty = Path(tempfile.mkdtemp(prefix="tf-no-checkout-"))
    locate.EXT_ROOT = empty / "ext"
    locate.TF_REPO_FETCH_DIR = empty / "ext" / "TrajectoryForcing"
    locate._TF_REPO = None
    os.environ.pop("TF_REPO", None)

    # Blocked at the git boundary, not at `fetch_tf_repo`: the fetch tests stub
    # `_git` themselves, so they keep working while a real clone cannot happen.
    def _no_git(*args, **kwargs):
        raise AssertionError(
            "this test tried to clone TrajectoryForcing. Under --no-checkout that is "
            "the failure being looked for: on a runner it would leave a checkout "
            "behind for every later test. Stub the resolver, or mark the test "
            "needs_tf_checkout if it genuinely needs upstream."
        )

    locate._git = _no_git


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--no-checkout"):
        return
    skip = pytest.mark.skip(reason="needs a real TrajectoryForcing checkout (--no-checkout)")
    for item in items:
        if "needs_tf_checkout" in item.keywords:
            item.add_marker(skip)
