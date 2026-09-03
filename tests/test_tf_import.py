"""The namespace swap, which is the load-bearing trick of this extension.

ComfyUI and TrajectoryForcing both own a top-level `utils` package. If the swap
leaks in either direction the failure is remote from its cause -- ComfyUI's
`utils.json_util` disappearing halfway through a session, or TF's decoder
failing to find `utils.logging_util` only at decode time -- so it is worth
proving both directions here.
"""
from __future__ import annotations

import sys

import pytest
from conftest import requires_comfy

pytestmark = requires_comfy


@pytest.fixture(autouse=True)
def comfy_utils_loaded():
    """Put ComfyUI's `utils` package in sys.modules, as its startup does."""
    import utils.json_util  # noqa: F401

    yield
    assert "utils" in sys.modules


def test_comfy_owns_utils_outside_the_scope():
    import utils

    assert "ComfyUI" in utils.__file__


def test_tf_owns_utils_inside_the_scope():
    from tf_nodes.tf_import import tf_scope

    with tf_scope():
        import utils.vis_util

        assert "TrajectoryForcing" in utils.vis_util.__file__


def test_comfy_utils_still_works_afterwards():
    from tf_nodes.tf_import import tf_scope

    with tf_scope():
        import utils.vis_util  # noqa: F401

    import utils.json_util

    assert "ComfyUI" in utils.json_util.__file__
    # and a submodule not yet imported still resolves against ComfyUI
    import utils.mime_types

    assert "ComfyUI" in utils.mime_types.__file__


def test_tf_modules_stay_warm_between_scopes():
    from tf_nodes.tf_import import tf_scope

    with tf_scope():
        import utils.vis_util

        first = utils.vis_util
    with tf_scope():
        import utils.vis_util

        assert utils.vis_util is first, "re-entering the scope must not re-import TF's tree"


def test_tf_pipeline_imports_without_touching_the_gpu():
    """tf_pipeline defers every JAX import into Pipeline.__init__ -- rely on that."""
    from tf_nodes.tf_import import tf_scope

    with tf_scope():
        import tf_pipeline

        assert hasattr(tf_pipeline, "build_pipeline")
    assert "jax" not in sys.modules or sys.modules["jax"] is not None


def test_the_scope_is_reentrant():
    from tf_nodes.tf_import import tf_scope

    with tf_scope():
        with tf_scope():
            import utils.vis_util  # noqa: F401
        # the inner exit must not have restored ComfyUI's utils underneath us
        import utils.vis_util

        assert "TrajectoryForcing" in utils.vis_util.__file__
    import utils

    assert "ComfyUI" in utils.__file__


def test_a_swapped_module_does_not_break_a_sys_modules_walk():
    """The regression that took down TF Load Pipeline inside a running server.

    `inspect.getmodule` walks `sys.modules` and remembers each module's
    `__file__` in `inspect._filesbymodname`. A module whose `__file__` is None
    is skipped -- but only while the remembered value is *also* None. Swap a
    name that was already recorded with a real path (ComfyUI's `utils`) for one
    with a null `__file__` (a namespace package, which is every directory in
    TrajectoryForcing's tree) and the walk stops skipping it and dies in
    `inspect.getfile` with "is a built-in module".

    pydantic calls `inspect.getmodule` while building a model class, and wandb
    builds pydantic models while TrajectoryForcing imports it, so this crashed
    the pipeline load itself. It reproduced only in a long-lived process where
    something had already recorded ComfyUI's `utils` -- which is why a bare
    script kept passing.
    """
    import inspect

    # Prime the cache with ComfyUI's utils, the way a running server does.
    # Both caches are cleared first: they are process-global, so an earlier test
    # that entered the scope would otherwise decide this one's starting state.
    import utils.json_util  # noqa: F401

    from tf_nodes.tf_import import tf_scope

    inspect.modulesbyfile.clear()
    inspect._filesbymodname.clear()
    inspect.getmodule(inspect.currentframe())
    assert inspect._filesbymodname.get("utils"), "the walk should have recorded ComfyUI's utils"

    with tf_scope():
        import utils.jax_fid  # noqa: F401  -- a nested namespace package
        import utils.vis_util  # noqa: F401

        # getmodule only walks sys.modules when it has not already resolved the
        # caller's file. In production that walk is forced by wandb's modules
        # being seen for the first time; here, by dropping the resolved cache.
        inspect.modulesbyfile.clear()
        inspect.getmodule(inspect.currentframe())  # raised TypeError before the fix


def test_nested_namespace_packages_are_bound_too():
    """utils/jax_fid and third_party/fd_loss/* are namespace packages as well;
    if they were left to be created lazily the null __file__ would come back."""
    from tf_nodes.tf_import import tf_scope

    with tf_scope():
        assert "utils.jax_fid" in sys.modules
        assert not hasattr(sys.modules["utils.jax_fid"], "__file__")


def test_sys_path_is_left_as_it_was():
    from tf_nodes.tf_import import tf_scope

    before = list(sys.path)
    with tf_scope():
        pass
    assert sys.path == before


def test_configure_jax_env_is_idempotent(monkeypatch):
    from tf_nodes.tf_import import configure_jax_env

    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("TF_XLA_MEM_FRACTION", raising=False)

    first = configure_jax_env()
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" in first
    import os

    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert configure_jax_env() == [], "a second call must not overwrite what is already exported"


def test_mem_fraction_switches_preallocation_on(monkeypatch):
    from tf_nodes.tf_import import configure_jax_env

    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    monkeypatch.setenv("TF_XLA_MEM_FRACTION", "0.4")

    configure_jax_env()
    import os

    # a ceiling is meaningless without preallocation, so asking for one flips it
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.4"
