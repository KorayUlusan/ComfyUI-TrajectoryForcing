"""The Manager install hook, and the promise it has to keep.

`install.py` is the only code in this repo that installs packages into someone
else's environment. The promise is that it never changes a package that is
already there, and these tests are what hold it to that -- the decision logic is
split out into `make_plan` precisely so it can be checked on a runner with no
torch, no GPU and no network.

What is deliberately not tested here: that `pip install` works, and that JAX
functions once installed. The second is not a unit-test question at all -- it
was settled on an H100 by running a JAX-first venv and a torch-first one through
the same five checks, and the day that stops being true no assertion in this
file will notice.
"""
from __future__ import annotations

import install


def presence(*installed: str):
    """A stand-in for `find_spec`, so a test can describe any environment."""
    have = set(installed)
    return lambda module: module in have


NOTHING = presence()
EVERYTHING = presence(*[m for m, _ in install.JAX_STACK + install.EXTRAS])


class TestItDeclinesRatherThanBreakSomething:
    """Every branch here ends with the environment untouched.

    This is the whole reason the script is allowed to exist. A refusal costs a
    user one command; getting it wrong costs them the ComfyUI they already had.
    """

    def test_no_torch_at_all_is_a_decline(self):
        plan = install.make_plan(None, None, NOTHING)
        assert not plan.ok
        assert plan.install == []

    def test_a_cpu_torch_is_a_decline(self):
        plan = install.make_plan("2.8.0", None, NOTHING)
        assert not plan.ok
        assert "CPU build" in plan.reason

    def test_the_wrong_cuda_major_is_a_decline(self):
        """jax 0.4.36 is a CUDA 12 build; two CUDA majors in one process is a fight."""
        plan = install.make_plan("2.9.0+cu130", "13.0", NOTHING)
        assert not plan.ok
        assert "CUDA 13.0" in plan.reason
        assert plan.install == []

    def test_torch_below_the_floor_is_a_decline(self):
        """2.6 is the version that makes `import comfy.quant_ops` fail outright.

        Raising torch here would be the one change guaranteed to break the rest
        of the install, so this branch has to refuse rather than "fix" it.
        """
        plan = install.make_plan("2.6.0+cu124", "12.4", NOTHING)
        assert not plan.ok
        assert "2.8" in plan.reason

    def test_a_partial_jax_stack_is_left_alone(self):
        """Half a JAX stack belongs to something else, and resolving against it
        is exactly the kind of guess this script must not make."""
        plan = install.make_plan("2.8.0+cu128", "12.8", presence("jax", "flax"))
        assert not plan.ok
        assert "partial JAX stack" in plan.reason
        assert plan.install == []

    def test_no_decline_ever_carries_an_install_list(self):
        declines = [
            install.make_plan(None, None, NOTHING),
            install.make_plan("2.8.0", None, NOTHING),
            install.make_plan("2.9.0+cu130", "13.0", NOTHING),
            install.make_plan("2.6.0+cu124", "12.4", NOTHING),
            install.make_plan("2.8.0+cu128", "12.8", presence("jax")),
        ]
        assert all(not p.ok and p.install == [] for p in declines)


class TestItNeverTouchesWhatIsAlreadyThere:
    """The measured failure, encoded.

    Building the two experiment venvs showed `transformers==5.3.0` downgrading a
    venv that already had 5.16.1, taking tokenizers 0.23.2 -> 0.22.2 with it.
    That is a broken install by any reasonable reading, caused by this
    extension. Hence: present means skipped, pin or no pin.
    """

    def test_an_already_installed_package_is_never_in_the_list(self):
        plan = install.make_plan("2.8.0+cu128", "12.8", presence("transformers", "numpy"))
        assert plan.ok
        assert not any(spec.startswith("transformers") for spec in plan.install)
        assert not any(spec.startswith("numpy") for spec in plan.install)

    def test_a_fully_provisioned_venv_installs_nothing(self):
        plan = install.make_plan("2.8.0+cu128", "12.8", EVERYTHING)
        assert plan.ok
        assert plan.install == []

    def test_torch_is_never_installed(self):
        """The one package this extension constrains is the one it must not move."""
        for present in (NOTHING, EVERYTHING, presence("numpy")):
            plan = install.make_plan("2.8.0+cu128", "12.8", present)
            assert not any(
                spec.split("=")[0].split("[")[0] in {"torch", "torchvision", "torchaudio"}
                for spec in plan.install
            ), plan.install

    def test_an_empty_environment_gets_the_whole_stack(self):
        plan = install.make_plan("2.8.0+cu128", "12.8", NOTHING)
        assert plan.ok
        assert len(plan.install) == len(install.JAX_STACK) + len(install.EXTRAS)
        assert "jax[cuda12]==0.4.36" in plan.install


class TestThePinsMatchTheOnesThatWereTested:
    """install.py and env/requirements.txt describe the same environment.

    They are separate files that a future edit can move apart, and the failure
    is silent: the venv setup.sh builds and the venv install.py builds would
    quietly stop being the thing the GPU comparison was run against.
    """

    def test_every_jax_pin_appears_in_env_requirements(self):
        from pathlib import Path

        req = Path(install.__file__).parent / "env" / "requirements.txt"
        # Comment lines carry version numbers in prose (torch 2.6.0, and so on),
        # which a naive scan happily mistakes for pins.
        body = "\n".join(
            line for line in req.read_text().splitlines() if not line.strip().startswith("#")
        )
        for _, spec in install.JAX_STACK:
            assert spec in body, f"{spec} is in install.py but not env/requirements.txt"

    def test_the_torch_floor_is_the_version_env_requirements_pins(self):
        from pathlib import Path

        req = Path(install.__file__).parent / "env" / "requirements.txt"
        body = "\n".join(
            line for line in req.read_text().splitlines() if not line.strip().startswith("#")
        )
        assert "torch==2.8.0" in body
        assert install.MIN_TORCH == (2, 8)


class TestVersionParsing:
    def test_a_local_version_suffix_does_not_confuse_the_comparison(self):
        """`2.8.0+cu128` must not sort below `2.8.0`; the suffix is not a number."""
        assert install.parse_torch_version("2.8.0+cu128") == (2, 8, 0)
        assert install.parse_torch_version("2.8.0+cu128") >= install.MIN_TORCH

    def test_a_two_part_cuda_version_yields_its_major(self):
        assert install.parse_torch_version("12.8")[0] == 12
        assert install.parse_torch_version("13.0")[0] == 13

    def test_a_release_candidate_still_parses(self):
        assert install.parse_torch_version("2.9.0rc1+cu128")[:2] == (2, 9)
