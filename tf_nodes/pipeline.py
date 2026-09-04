"""Adapter over TrajectoryForcing's `editing_env/tf_pipeline.py`.

Only this module imports JAX, and every entry point holds `tf_scope()` for the
whole call -- the RAE decoder imports `utils.logging_util` and
`third_party.rae_decoder` lazily at decode time, so a scope that covered only
the initial import would break on the first TF Decode.

Nothing here re-derives TF's sampling math. `resume` in particular goes through
the public `Pipeline.edit` with a no-op token copy (one token onto itself),
because `edit` is exactly "install this canvas at level l*, then re-sample every
finer level" -- which is the resume operation with a token exchange bolted on
the front. This extension does its edits in numpy (see `tokens.py`) and hands
the finished canvas here, so the bolted-on part has to be neutralised rather
than reimplemented.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .locate import rae_root, resolve_checkpoint, tf_config_path
from .tf_import import tf_scope

log = logging.getLogger(__name__)

_CACHE: dict[tuple[str, str], TFPipeline] = {}
_CACHE_LOCK = threading.Lock()

# Marker `utils/rae_decoder.py` splits paths on to work out what to fetch from
# HuggingFace. Re-rooting has to keep it, or auto-download stops working.
_RAE_MARKER = "checkpoints/rae/"


def _reroot_rae(config) -> str | None:
    """Point the decoder weights at `rae_root()` while keeping the config's choice of decoder.

    The config names them relative to the TrajectoryForcing checkout
    (`checkpoints/rae/decoders/dinov2/...`). TrajectoryForcing resolves those
    against its own repo root, not the cwd, so they already work under ComfyUI --
    but they hard-code where the weights live, and the extension lets that be
    redirected (TF_RAE_ROOT, or ComfyUI's models/ when the checkout has none).
    """
    current = str(config.rae_decoder.get("pretrained_decoder_path", "")).replace("\\", "/")
    if _RAE_MARKER not in current:
        return None
    tail = current.split(_RAE_MARKER, 1)[1]
    absolute = str(Path(rae_root()) / tail)
    config.rae_decoder.pretrained_decoder_path = absolute
    return absolute


#: Imported by TrajectoryForcing, not by this extension directly, so a missing
#: one surfaces as a ModuleNotFoundError from inside TF's own code.
RUNTIME_DEPS = ("jax", "flax", "orbax", "ml_collections")


def check_runtime_deps() -> None:
    """Fail with the command that fixes it, rather than from inside TF.

    This is the most likely first failure for anyone who installed through
    ComfyUI Manager. Manager installs the node's `requirements.txt`, which is
    deliberately empty here -- listing jax and a CUDA-matched torch would have
    Manager rewrite torch underneath every other node in that install. So the
    nodes register and this is where the missing half is noticed.

    Checked with `find_spec`, not by importing: importing jax initialises its GPU
    backend, which `configure_jax_env` has to happen before, and which there is
    no undoing inside a running process.
    """
    from importlib.util import find_spec

    missing = []
    for name in RUNTIME_DEPS:
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):  # a broken or half-installed package
            missing.append(name)
    if not missing:
        return
    raise ImportError(
        f"TrajectoryForcing needs {', '.join(missing)}, which {'is' if len(missing) == 1 else 'are'} "
        "not installed in the environment ComfyUI is running in.\n\n"
        "This extension runs a JAX model inside the ComfyUI process, and the two dependency "
        "sets only coexist at one specific torch version -- so it needs its own environment "
        "rather than additions to an existing one. From the extension directory:\n\n"
        "    bash env/setup.sh\n\n"
        "then start ComfyUI from the venv it builds. env/requirements.txt lists what goes in "
        "and why; the top-level requirements.txt is empty on purpose, so that installing this "
        "node through ComfyUI Manager cannot rewrite the torch every other node depends on."
    )


def check_startup_problems() -> None:
    """Fail with the diagnosis made at startup, not with its downstream symptom.

    `on_load` records rather than raises, so a missing checkout no longer takes
    the extension down. The cost of that is a node which would otherwise have
    failed at import now failing here -- and it should fail saying the thing
    startup already worked out, rather than with whatever error the absence
    eventually causes several frames inside TrajectoryForcing.
    """
    from .health import blocking_problem

    problem = blocking_problem()
    if problem is None:
        return
    raise RuntimeError(f"{problem.title}\n\n{problem.detail}\n\n-> {problem.fix}")


#: Substrings that mean "the card is full", from either framework. Matched on
#: the message rather than the exception class so that nothing here has to
#: import jax or torch to recognise one -- and so a jaxlib that renames
#: XlaRuntimeError again does not silently stop being handled.
_OOM_SIGNS = (
    "RESOURCE_EXHAUSTED",
    "out of memory",
    "OutOfMemory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "failed to allocate",
)

_OOM_ADVICE = """TrajectoryForcing needs about 6.6 GiB at peak on its own: 2.5 for the model,
4.6 once the sampler has compiled, 6.6 once the RAE decoder is built. Two
trajectories held for an edit do not add to that. Measured on an H100 by
scripts/measure_resources.py.

So on a card with room for the model, this almost always means something else
in the graph is holding VRAM:

  * Keep this the only model in the workflow. ComfyUI does not know about JAX's
    allocation and will happily load a checkpoint into memory JAX has taken.
  * If you must share, cap JAX and tell ComfyUI to stay clear:
        TF_XLA_MEM_FRACTION=0.5 TF_RESERVE_VRAM=8 ./run_comfyui.sh
    Capping requires preallocation, so this trades flexibility for a hard
    ceiling; it is off by default because it costs memory on a card that does
    not need it.
  * On an 8 GB card, decode one level at a time (TF Decode Levels -> final
    only, or TF Latent Preview -> which) rather than all four.

python -m tf_nodes.doctor prints the card and how much it has."""


@contextmanager
def _vram_advice(what: str):
    """Turn an allocator failure into something a user can act on.

    An XLA OOM is a page of buffer arithmetic and assignment tables, and the one
    thing it does not say is what to do. The original is chained, so it is still
    there for a bug report.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - re-raised either way
        text = f"{type(exc).__name__}: {exc}"
        if not any(sign.lower() in text.lower() for sign in _OOM_SIGNS):
            raise
        raise RuntimeError(f"{what} ran out of GPU memory.\n\n{_OOM_ADVICE}") from exc


class TFPipeline:
    """One loaded TrajectoryForcing model, shared by every node that references it."""

    def __init__(self, config_name: str, checkpoint_name: str):
        check_runtime_deps()
        check_startup_problems()
        self.config_name = config_name
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = resolve_checkpoint(checkpoint_name)
        started = time.perf_counter()
        with tf_scope():
            from tf_pipeline import build_pipeline

            self._pipe = build_pipeline(str(tf_config_path(config_name)), self.checkpoint_path)
            self.decoder_path = _reroot_rae(self._pipe.config)
        self.load_seconds = time.perf_counter() - started
        log.info(
            "TrajectoryForcing: loaded %s with %s in %.1fs",
            checkpoint_name, config_name, self.load_seconds,
        )

        # TrajectoryForcing fetches the RAE decoder itself, lazily, at the first
        # decode -- which happens inside warm_up, behind a progress bar that has
        # no idea it is waiting on 1.6 GB. Work out now whether that is coming,
        # while there is still somewhere to say it.
        self.decoder_will_download = bool(
            self.decoder_path and not Path(self.decoder_path).exists()
        )
        if self.decoder_will_download:
            log.info(
                "TrajectoryForcing: the RAE decoder is not on disk yet, so the first "
                "decode will download about 1.6 GB into %s. First run only.",
                Path(self.decoder_path).parent,
            )

    @property
    def num_levels(self) -> int:
        return int(self._pipe.num_levels)

    @property
    def num_classes(self) -> int:
        return int(self._pipe.num_classes)

    def describe(self) -> str:
        return (
            f"config: {self.config_name}\n"
            f"checkpoint: {self.checkpoint_path}\n"
            f"rae decoder: {self.decoder_path}\n"
            f"levels: {self.num_levels}   classes: {self.num_classes}   "
            f"steps/level: {int(self._pipe.num_steps)}\n"
            f"load time: {self.load_seconds:.1f}s"
        )

    # ----- inference -----
    def generate(self, class_id: int, seed: int) -> np.ndarray:
        with _vram_advice("Generating a trajectory"), tf_scope():
            return np.asarray(self._pipe.generate(int(class_id), seed=int(seed)), dtype=np.float32)

    def resume(self, latents: np.ndarray, start_level: int, class_id: int, seed: int) -> np.ndarray:
        """Re-sample levels above `start_level`, conditioned on the canvas sitting there.

        Routed through `Pipeline.edit` with source == target and a single token
        copied onto itself, which makes the token exchange a no-op and leaves
        just the resume. See the module docstring.
        """
        arr = np.asarray(latents, dtype=np.float32)
        level = int(start_level)
        with _vram_advice("Resuming from an edited level"), tf_scope():
            out = self._pipe.edit(
                arr, arr, level, level, [(0, 0)], [(0, 0)],
                class_id=int(class_id), seed=int(seed),
            )
        return np.asarray(out, dtype=np.float32)

    def decode(self, latents: np.ndarray, final_only: bool) -> list[np.ndarray]:
        # The decoder is the peak: it is what takes the 4.6 GiB after compiling
        # to the 6.6 GiB the README quotes.
        with _vram_advice("Decoding levels to images"), tf_scope():
            if final_only:
                return [np.asarray(self._pipe.decode_last(latents), dtype=np.uint8)]
            return [np.asarray(x, dtype=np.uint8) for x in self._pipe.decode_all(latents)]

    # ----- latent visualisation -----
    def fit_palette(self, stacks: list[np.ndarray]):
        with tf_scope():
            return self._pipe.fit_palette(stacks)

    def pca_tiles(self, latents: np.ndarray, palette=None) -> list[np.ndarray]:
        with tf_scope():
            return [np.asarray(t, dtype=np.uint8) for t in self._pipe.pca_tiles(latents, palette=palette)]

    def warm_up(self, on_step=None) -> None:
        """Pay the JIT compile and the decoder build once, at load.

        Ported from `editing_env/app.py::_warmup`. Without it the first queued
        prompt sits for a minute or two with no progress, which in ComfyUI reads
        as a hung graph rather than as a compile.

        `on_step(label)` is called as each stage finishes, so the node above can
        drive a progress bar: moving the cost here only helps if something says
        it is being paid.
        """
        started = time.perf_counter()
        levels = self.generate(0, 0)
        compiled = time.perf_counter()
        if on_step:
            on_step("sampler compiled")
        if on_step and getattr(self, "decoder_will_download", False):
            # The bar is the only thing the user can see, so the one stage that
            # takes minutes rather than seconds should say why.
            on_step("downloading RAE decoder (~1.6 GB, first run)")
        self.decode(levels, final_only=True)
        if on_step:
            on_step("decoder built")
        log.info(
            "TrajectoryForcing: warmup done -- sampler %.1fs, decoder %.1fs",
            compiled - started, time.perf_counter() - compiled,
        )


def load_pipeline(config_name: str, checkpoint_name: str, warmup: bool, on_step=None) -> TFPipeline:
    """Get the pipeline for this (config, checkpoint), building it at most once.

    ComfyUI caches node outputs by input signature, but that cache is dropped on
    a workflow edit and never survives a "free model memory". A multi-GB flax
    restore plus an XLA compile is far too expensive to redo on either, so the
    handle is kept here for the life of the process.
    """
    key = (config_name, checkpoint_name)
    with _CACHE_LOCK:
        pipe = _CACHE.get(key)
        if pipe is None:
            pipe = TFPipeline(config_name, checkpoint_name)
            _CACHE[key] = pipe
            if on_step:
                on_step("checkpoint restored")
            if warmup:
                pipe.warm_up(on_step)
        elif on_step:
            # Cached: nothing to wait for, so finish the bar rather than leaving
            # it stopped a third of the way across, which reads as stuck.
            for label in ("checkpoint restored", "sampler compiled", "decoder built"):
                on_step(f"{label} (cached)")
        return pipe
