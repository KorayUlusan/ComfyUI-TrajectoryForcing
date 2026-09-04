#!/usr/bin/env python3
"""Finish the install, when the ComfyUI venv can take it.

ComfyUI-Manager runs this file with the ComfyUI python straight after unpacking
the node (`glob/manager_core.py`, `execute_install_script`). It is the only
hook that gets to touch the environment, so it is the only chance to spare
people the separate `bash env/setup.sh` step.

It follows one rule: never change a package that is already installed. Not
upgrade, not downgrade, not remove. Someone whose ComfyUI works today must have
a ComfyUI that works tomorrow, whatever this node decides. Everything below
falls out of that:

  * torch is never installed or touched. It is the one package this extension
    genuinely constrains, and the constraint is narrow (see `SUPPORTED` below),
    so instead of imposing it this script reads the torch that is there and
    steps aside when it does not fit.
  * only *missing* packages are installed. transformers is the cautionary
    case: `env/requirements.txt` pins 5.3.0, and blindly applying that pin in a
    venv that already had 5.16.1 downgraded transformers and tokenizers
    underneath everything else in that install. Measured, not guessed --
    building the two environments is what surfaced it.
  * a wrong-version JAX is reported, not corrected.

Evidence that adding JAX after torch is safe at all: an H100 run comparing a
JAX-first venv (what `env/setup.sh` builds) against a torch-first one (what
this script produces). Both passed the same five checks -- torch still sees the
GPU, its matmul still agrees with a CPU reference, `comfy.quant_ops` still
imports, JAX gets a CUDA device, and both frameworks compute correctly in one
process. `pip freeze` either side of the JAX install showed no `nvidia-*` wheel
moving. Without that result this script would not exist, because `env/setup.sh`
installs JAX *first* on purpose and this necessarily does the reverse.

Nothing here is required. `bash env/setup.sh` builds a dedicated venv and is
still the answer for anyone this script declines to help.

Set TF_NO_AUTO_DEPS=1 to skip it entirely.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.util import find_spec

# The torch this extension is known to work with. Deliberately narrow: 2.8.0 is
# the version the GPU comparison above actually ran, and CUDA 12 is what jax
# 0.4.36 is built for -- two CUDA majors in one process is a fight nobody wins.
#
# The floor is not arbitrary either. TrajectoryForcing pins torch 2.6.0, but
# ComfyUI's comfy-kitchen registers a custom op annotated `list[int]` that
# torch 2.6's `infer_schema` rejects, so `import comfy.quant_ops` fails and
# ComfyUI does not start. Anything below 2.8 is therefore either known broken
# or untested, and this script treats both the same way: it declines.
MIN_TORCH = (2, 8)
CUDA_MAJOR = 12

# TrajectoryForcing's JAX stack, pinned exactly as TF pins it. Kept together
# because these versions were resolved as a set; installing a subset of them
# against a different jax is not something this script tries to do.
JAX_STACK: tuple[tuple[str, str], ...] = (
    ("jax", "jax[cuda12]==0.4.36"),
    ("flax", "flax==0.10.4"),
    ("optax", "optax==0.2.5"),
    ("orbax", "orbax-checkpoint==0.11.0"),
    ("chex", "chex==0.1.87"),
    ("ml_dtypes", "ml_dtypes==0.5.1"),
    ("tensorstore", "tensorstore==0.1.76"),
    ("ml_collections", "ml_collections==1.1.0"),
)

# What TF's inference path needs beyond that. Most of these ship with ComfyUI
# already, which is the point of checking each one by import name rather than
# handing pip the whole list: a package that is present is left exactly as it
# is, pin or no pin.
EXTRAS: tuple[tuple[str, str], ...] = (
    ("transformers", "transformers==5.3.0"),
    ("huggingface_hub", "huggingface_hub"),
    ("yaml", "PyYAML"),
    ("numpy", "numpy>=2.2"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("PIL", "pillow"),
    ("einops", "einops"),
    ("timm", "timm"),
    # Not optional: TF's utils/logging_util.py imports wandb at module scope,
    # and utils/ckpt_util.py -- which Pipeline.__init__ imports -- pulls it in.
    ("wandb", "wandb==0.22.0"),
)


class Plan:
    """What this script decided, and why. Pure data, so tests can assert on it."""

    def __init__(self, ok: bool, reason: str, install: list[str] | None = None):
        self.ok = ok
        self.reason = reason
        self.install = install or []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Plan(ok={self.ok!r}, reason={self.reason!r}, install={self.install!r})"


def parse_torch_version(raw: str) -> tuple[int, ...]:
    """`2.8.0+cu128` -> (2, 8, 0). Local version suffixes are not comparable."""
    core = raw.split("+", 1)[0]
    parts = re.findall(r"\d+", core)
    return tuple(int(p) for p in parts[:3])


def make_plan(torch_version: str | None, cuda_version: str | None, present) -> Plan:
    """Decide what to install. `present(module) -> bool`, injected for tests.

    Split out from `main` so the decisions can be tested on a machine with no
    torch, no GPU and no network -- which is every CI runner this repo has.
    """
    if torch_version is None:
        return Plan(False, "torch is not installed in this environment.")

    if cuda_version is None:
        return Plan(
            False,
            f"torch {torch_version} is a CPU build. These nodes run a JAX model on the GPU.",
        )

    cuda_major = parse_torch_version(cuda_version)
    if not cuda_major or cuda_major[0] != CUDA_MAJOR:
        return Plan(
            False,
            f"torch {torch_version} is built for CUDA {cuda_version}, "
            f"and TrajectoryForcing's jax 0.4.36 needs CUDA {CUDA_MAJOR}.",
        )

    version = parse_torch_version(torch_version)
    if version < MIN_TORCH:
        return Plan(
            False,
            f"torch {torch_version} is below {'.'.join(map(str, MIN_TORCH))}. "
            "ComfyUI's comfy-kitchen registers an op that torch below 2.8 rejects, "
            "so raising it here would break this ComfyUI rather than fix it.",
        )

    # A JAX that is already here belongs to something else, or to a previous run
    # of this script. Either way it is not ours to move.
    jax_present = [mod for mod, _ in JAX_STACK if present(mod)]
    if jax_present and len(jax_present) != len(JAX_STACK):
        return Plan(
            False,
            f"a partial JAX stack is already installed ({', '.join(jax_present)}). "
            "Leaving it alone rather than resolving against it.",
        )

    wanted = [spec for mod, spec in JAX_STACK + EXTRAS if not present(mod)]
    if not wanted:
        return Plan(True, "everything is already installed.", [])
    return Plan(True, f"torch {torch_version} (CUDA {cuda_version}) is compatible.", wanted)


def read_torch() -> tuple[str | None, str | None]:
    try:
        import torch
    except Exception:  # noqa: BLE001 - a torch that will not import is a "no"
        return None, None
    return torch.__version__, torch.version.cuda


def _present(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):  # a broken or half-installed package
        return False


def say(*lines: str) -> None:
    print("\n".join(lines), flush=True)


def main() -> int:
    if os.environ.get("TF_NO_AUTO_DEPS", "").strip():
        say("[TrajectoryForcing] TF_NO_AUTO_DEPS is set; not touching this environment.")
        return 0

    torch_version, cuda_version = read_torch()
    plan = make_plan(torch_version, cuda_version, _present)

    if not plan.ok:
        say(
            "",
            "=" * 72,
            "  Trajectory Forcing: the nodes are installed, the model stack is not.",
            "",
            f"  {plan.reason}",
            "",
            "  Nothing in this environment was changed. These nodes need JAX and a",
            "  CUDA-matched torch in one process, which only coexist at a narrow set",
            "  of versions -- so rather than move the torch every other node here",
            "  depends on, this extension builds its own environment:",
            "",
            "      cd custom_nodes/comfyui-trajectoryforcing",
            "      bash env/setup.sh",
            "",
            "  then start ComfyUI from the venv that builds. env/requirements.txt",
            "  lists what goes in and why.",
            "=" * 72,
            "",
        )
        return 0

    if not plan.install:
        say(f"[TrajectoryForcing] {plan.reason} Nothing to do.")
        return 0

    say(
        "",
        f"[TrajectoryForcing] {plan.reason}",
        f"[TrajectoryForcing] Installing {len(plan.install)} missing package(s). "
        "This pulls CUDA wheels and is a few GB.",
        f"[TrajectoryForcing] Already-installed packages are left untouched: {_skipped_note()}",
        "",
    )
    cmd = [sys.executable, "-m", "pip", "install", *plan.install]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        say(
            "",
            "=" * 72,
            "  Trajectory Forcing: dependency install failed.",
            "",
            "  The nodes are installed but will not run yet. Nothing was removed, so",
            "  this ComfyUI is no worse off. The reliable route is a dedicated venv:",
            "",
            "      cd custom_nodes/comfyui-trajectoryforcing",
            "      bash env/setup.sh",
            "=" * 72,
            "",
        )
        # Deliberately 0: the node itself unpacked fine, and reporting the whole
        # install as failed would send people looking for a broken download.
        return 0

    say("", "[TrajectoryForcing] Done. Restart ComfyUI to load the nodes.", "")
    return 0


def _skipped_note() -> str:
    skipped = [mod for mod, _ in JAX_STACK + EXTRAS if _present(mod)]
    return ", ".join(skipped) if skipped else "(none were)"


if __name__ == "__main__":
    sys.exit(main())
