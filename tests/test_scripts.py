"""The launcher scripts, run against stubbed Slurm.

`run_comfyui_slurm.sh` is the thing a user types every session, and it had no
test at all. It has now broken twice in ways only the user could find: once
leaking `ncat` orphans that held the port, and once exiting silently before
printing anything, because an empty log glob made `ls` exit 2 and
`set -o pipefail` carried that out through `set -e`.

Both failures are invisible to `bash -n`. What catches them is actually running
the script with `srun`, `squeue`, `ncat` and friends replaced by stubs: it goes
through every line of setup and stops at the point where it would ask Slurm for
a GPU.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

EXT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = (sorted(EXT_ROOT.glob("*.sh")) + sorted(EXT_ROOT.glob("env/*.sh"))
           + sorted(EXT_ROOT.glob("slurm/*.sh")))

# `squeue` is asked two different questions and has to answer them differently,
# or the script waits on itself: the bridge polls for a running node (%N) and
# must get one, while cleanup polls for the job still existing (%i) and must
# not -- otherwise it sits in its ten-second wind-down loop.
STUBS = {
    "srun": "#!/usr/bin/env bash\necho \"stub srun: $*\"\nexit 0\n",
    "squeue": (
        "#!/usr/bin/env bash\n"
        'case "$*" in *%N*) echo mlcbm999 ;; *) : ;; esac\n'
        "exit 0\n"
    ),
    "scancel": "#!/usr/bin/env bash\nexit 0\n",
    "ncat": "#!/usr/bin/env bash\nexit 0\n",
    "ss": "#!/usr/bin/env bash\nexit 0\n",
    "pgrep": "#!/usr/bin/env bash\nexit 1\n",
    "setsid": '#!/usr/bin/env bash\nexec "$@"\n',
}


@pytest.fixture
def stub_path(tmp_path):
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    for name, body in STUBS.items():
        script = stub_dir / name
        script.write_text(body)
        script.chmod(0o755)
    return stub_dir


def run_launcher(stub_path, log_dir, extra_env=None):
    env = {
        **os.environ,
        "PATH": f"{stub_path}:{os.environ['PATH']}",
        "TF_LOG_DIR": str(log_dir),
        # Point away from the repo's own .env unless a test says otherwise.
        # Without this every launcher test would read whatever the developer
        # happens to have configured -- so the suite would pass here, fail in CI
        # where no .env exists, and neither result would mean anything.
        "TF_ENV_FILE": str(log_dir / "no-such.env"),
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(EXT_ROOT / "run_comfyui_slurm.sh"), "8199"],
        env=env, capture_output=True, text=True, timeout=120,
    )


class TestTheLocalEnvFile:
    """`.env` overrides, exercised rather than assumed.

    This is the shape that has already cost a session here: the silent-exit bug
    was one untested line of setup in a launcher, invisible to `bash -n` because
    it was a runtime status and not a syntax error. Sourcing a file under
    `set -euo pipefail` is the same kind of line, in six scripts now.

    `TF_ENV_FILE` exists so these can point somewhere other than the repo root.
    Writing a `.env` into EXT_ROOT to test it would clobber the developer's own.
    """

    def env_file(self, tmp_path, body: str):
        path = tmp_path / "test.env"
        path.write_text(body)
        return path

    def test_a_value_in_the_file_reaches_the_script(self, stub_path, tmp_path):
        # TF_PARTITION is echoed back by the launcher's own banner.
        env_file = self.env_file(tmp_path, 'TF_PARTITION="${TF_PARTITION:-from-the-env-file}"\n')
        out = run_launcher(stub_path, tmp_path / "logs",
                           {"TF_ENV_FILE": str(env_file)})
        assert "from-the-env-file" in out.stdout, out.stdout[-2000:]

    def test_the_command_line_still_wins_over_the_file(self, stub_path, tmp_path):
        # The scripts source with `set -a`, so a plain `KEY=value` in the file
        # would overwrite an exported one -- silently reversing the precedence
        # everyone expects. .env.example is written `KEY="${KEY:-default}"` for
        # exactly this, and that is only true as long as something checks it.
        env_file = self.env_file(tmp_path, 'TF_PARTITION="${TF_PARTITION:-from-the-env-file}"\n')
        out = run_launcher(stub_path, tmp_path / "logs",
                           {"TF_ENV_FILE": str(env_file), "TF_PARTITION": "from-the-command-line"})
        assert "from-the-command-line" in out.stdout, out.stdout[-2000:]
        assert "from-the-env-file" not in out.stdout

    def test_no_env_file_is_not_an_error(self, stub_path, tmp_path):
        # `set -e` plus a test that fails is how the launcher silently exited 2
        # before. A missing .env is the normal case, so it must not do that.
        out = run_launcher(stub_path, tmp_path / "logs",
                           {"TF_ENV_FILE": str(tmp_path / "absent.env")})
        assert "This asks Slurm for a GPU" in out.stdout, (
            f"launcher stopped before its banner; rc={out.returncode}\n{out.stdout[-2000:]}\n"
            f"{out.stderr[-2000:]}")

    def test_env_example_cannot_overwrite_your_environment(self):
        """Every assignment in the template must be `KEY="${KEY:-default}"`.

        The test above proves the *mechanism* keeps command-line precedence when
        the file is written that way. This is the other half: that the file
        people actually copy is written that way. A plain `WORK=/some/path` here
        would be sourced under `set -a` and silently beat an exported WORK.
        """
        import re

        offenders = []
        for line in (EXT_ROOT / ".env.example").read_text().splitlines():
            stripped = line.lstrip("# ").strip()
            match = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)$', stripped)
            if match and f"${{{match.group(1)}:-" not in match.group(2):
                offenders.append(stripped)
        assert not offenders, (
            f"{offenders} would overwrite an exported value when sourced with `set -a`. "
            'Write them as KEY="${KEY:-default}".')

    @pytest.mark.parametrize(
        "script",
        ["run_comfyui.sh", "run_comfyui_slurm.sh", "env/setup.sh", "slurm/submit.sh",
         "slurm/gpu_smoke.sbatch", "slurm/server_smoke.sbatch", "slurm/measure_resources.sbatch"],
    )
    def test_every_script_reads_it(self, script):
        # Adding a seventh script and forgetting the two lines would leave one
        # entry point ignoring the config file the docs point everyone at.
        text = (EXT_ROOT / script).read_text()
        assert "TF_ENV_FILE" in text, f"{script} does not source the local .env"


class TestShellSyntax:
    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_it_parses(self, script):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0

    def test_the_slurm_launcher_is_named_for_what_it_does(self):
        # `serve.sh` did not say it was asking Slurm for a GPU. The pair now
        # reads as one being the other on a cluster.
        assert (EXT_ROOT / "run_comfyui.sh").is_file()
        assert (EXT_ROOT / "run_comfyui_slurm.sh").is_file()
        assert not (EXT_ROOT / "serve.sh").exists()


class TestTheSlurmLauncher:
    def test_it_gets_as_far_as_asking_slurm_for_a_gpu(self, stub_path, tmp_path):
        # The regression that started this: it exited 0 having printed nothing
        # at all, before the banner, because pruning an empty log directory
        # failed under `set -euo pipefail`.
        result = run_launcher(stub_path, tmp_path / "logs")
        assert result.returncode == 0, result.stderr
        assert "stub srun:" in result.stdout, (
            f"never reached srun.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    def test_it_prints_the_instructions_before_queueing(self, stub_path, tmp_path):
        result = run_launcher(stub_path, tmp_path / "logs")
        for expected in ("ComfyUI + Trajectory Forcing", "WAIT FOR THIS LINE",
                         "http://localhost:8199", "Ctrl-C"):
            assert expected in result.stdout, f"missing {expected!r} from:\n{result.stdout}"

    def test_it_passes_the_port_through_to_srun(self, stub_path, tmp_path):
        result = run_launcher(stub_path, tmp_path / "logs")
        assert "run_comfyui.sh 8199" in result.stdout

    def test_it_says_where_the_logs_are_on_the_way_out(self, stub_path, tmp_path):
        # The point of keeping them: a user reporting "nothing happened" needs
        # to be told where to look without asking.
        result = run_launcher(stub_path, tmp_path / "logs")
        assert "Logs from this session" in result.stdout
        assert "bridge:" in result.stdout and "server:" in result.stdout

    def test_it_creates_the_log_directory(self, stub_path, tmp_path):
        logs = tmp_path / "logs"
        run_launcher(stub_path, logs)
        assert logs.is_dir()

    def test_an_empty_log_directory_is_not_fatal(self, stub_path, tmp_path):
        # The exact shape of the bug: nothing to prune must not fail.
        logs = tmp_path / "logs"
        logs.mkdir()
        assert run_launcher(stub_path, logs).returncode == 0

    def test_it_prunes_old_bridge_logs_but_keeps_the_recent_ones(self, stub_path, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        for i in range(30):
            (logs / f"bridge-2020010{i % 10}-00000{i}-{i}.log").write_text("old")
        (logs / "comfyui-keepme.log").write_text("not a bridge log")

        assert run_launcher(stub_path, logs).returncode == 0
        # 20 kept of the 30, plus the one this run created.
        assert len(list(logs.glob("bridge-*.log"))) == 21
        assert (logs / "comfyui-keepme.log").exists(), "it must only prune its own family"

    def test_it_cancels_by_job_name_never_by_pattern(self):
        # A `pkill -f` pattern that matches this script's own command line takes
        # out the shell running it. This has cost time in this repo before.
        source = (EXT_ROOT / "run_comfyui_slurm.sh").read_text()
        assert "scancel --name=" in source
        assert "pkill" not in source

    def test_the_bridge_is_killed_by_process_group(self):
        # `ncat --keep-open` forks per connection and each fork inherits the
        # listening socket, so killing the leader alone leaves the port bound --
        # which is how 34 orphans once accumulated on port 8188.
        source = (EXT_ROOT / "run_comfyui_slurm.sh").read_text()
        assert "setsid" in source
        assert 'kill -- -"$bridge_pid"' in source


class TestThePinsAgree:
    """`env/setup.sh` and `env/requirements.txt` list the same versions.

    They cannot be merged into one file: torch comes from
    download.pytorch.org/whl/cu128 while everything else comes from PyPI, and the
    JAX stack has to be resolved *before* torch is read or pip pulls a different
    CUDA build underneath it. That needs three staged `pip install` calls, which a
    single requirements file cannot express. So the pins live in two places, and
    two places that must agree is the shape this repo has already paid for four
    times over in stale smoke scripts.
    """

    def pins(self, text: str) -> set[str]:
        """Pins from real lines only.

        Both files explain the torch conflict in prose, and that prose names
        versions -- `torch==2.6.0` (TrajectoryForcing's pin) and
        `comfy-kitchen==0.2.31` (what forced the move). Reading those as
        requirements is how this check first failed on two files that agree.
        """
        import re

        code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        return set(re.findall(r'[a-zA-Z0-9_.-]+(?:\[[a-z0-9]+\])?==[0-9][0-9a-z.]*',
                              "\n".join(code)))

    def test_setup_sh_and_the_requirements_file_pin_the_same_versions(self):
        from_script = self.pins((EXT_ROOT / "env" / "setup.sh").read_text())
        from_file = self.pins((EXT_ROOT / "env" / "requirements.txt").read_text())
        assert from_script == from_file, (
            f"only in setup.sh: {sorted(from_script - from_file)}; "
            f"only in env/requirements.txt: {sorted(from_file - from_script)}")

    def test_there_are_pins_to_compare(self):
        # A parser that silently matches nothing would make the check above pass
        # for the wrong reason -- the failure mode of every stale-expectation bug
        # in this repo so far.
        assert len(self.pins((EXT_ROOT / "env" / "setup.sh").read_text())) >= 10


class TestTheLauncherPicksTheRightComfyUI:
    """`run_comfyui.sh`'s symlink and workspace handling.

    Both cases here are from one report. A user installed into a fresh
    directory, launched it, and got:

        ln: failed to create symbolic link '.../custom_nodes/...': File exists
        srun: error: task 0: Exited with exit code 1

    Two faults at once. The guard was `[[ ! -e "$LINK" ]]`, and `-e` follows the
    link, so it is *false* for a dangling one -- the link left behind by an
    install that had since been deleted. And COMFY_DIR had fallen back to its
    $HOME default because the .env carried only TF_PARTITION and TF_QOS, so the
    launcher of the new install was about to start an unrelated ComfyUI.

    `run_comfyui_slurm.sh` stubs srun, so none of this is reachable through the
    harness above: it runs on the compute node. These drive the script directly.
    """

    def workspace(self, tmp_path, name="comfyui-trajectoryforcing"):
        """A ComfyUI with this extension inside its custom_nodes/."""
        comfy = tmp_path / "ComfyUI"
        (comfy / "custom_nodes").mkdir(parents=True)
        (comfy / "main.py").write_text("")
        ext = comfy / "custom_nodes" / name
        ext.mkdir()
        shutil.copy(EXT_ROOT / "run_comfyui.sh", ext / "run_comfyui.sh")

        venv = tmp_path / "venv" / "bin"
        venv.mkdir(parents=True)
        python = venv / "python"
        # Satisfies the CUDA probe, the device-name print, and the final exec.
        python.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "-c" ]]; then echo stub; exit 0; fi\n'
            'echo "STUB SERVER"\n'
        )
        python.chmod(0o755)
        return comfy, ext, venv.parent

    def launch(self, ext, comfy, venv, tmp_path):
        return subprocess.run(
            ["bash", str(ext / "run_comfyui.sh"), "8199"],
            env={
                **os.environ,
                "COMFY_DIR": str(comfy),
                "COMFY_VENV": str(venv),
                "TF_ENV_FILE": str(tmp_path / "no-such.env"),
                "TF_LOG_DIR": str(tmp_path / "logs"),
            },
            capture_output=True, text=True, timeout=120,
        )

    def test_a_dangling_link_is_replaced_rather_than_fatal(self, tmp_path):
        """The reported crash. `-e` is false for a dangling link, so the old
        guard fell through to `ln` and `set -e` killed the launch."""
        comfy, ext, venv = self.workspace(tmp_path)
        stale = comfy / "custom_nodes" / "other-copy"
        stale.symlink_to(tmp_path / "deleted-install")
        assert not stale.exists() and stale.is_symlink(), "not a dangling link"

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        shutil.copy(EXT_ROOT / "run_comfyui.sh", elsewhere / "run_comfyui.sh")
        (comfy / "custom_nodes" / "elsewhere").symlink_to(tmp_path / "gone")

        out = self.launch(elsewhere, comfy, venv, tmp_path)
        assert out.returncode == 0, out.stderr
        assert (comfy / "custom_nodes" / "elsewhere").resolve() == elsewhere.resolve()

    def test_a_link_to_a_different_copy_stops_with_both_paths(self, tmp_path):
        """Two copies register the same node names and which wins is undefined,
        so this is worth refusing rather than guessing."""
        comfy, _, venv = self.workspace(tmp_path)
        mine = tmp_path / "mine"
        mine.mkdir()
        shutil.copy(EXT_ROOT / "run_comfyui.sh", mine / "run_comfyui.sh")
        theirs = tmp_path / "theirs"
        theirs.mkdir()
        (comfy / "custom_nodes" / "mine").symlink_to(theirs)

        out = self.launch(mine, comfy, venv, tmp_path)
        assert out.returncode != 0
        assert "another copy" in out.stderr
        assert str(theirs) in out.stderr

    def test_it_uses_the_comfyui_it_actually_lives_in(self, tmp_path):
        """The silent half of the report: COMFY_DIR pointing somewhere else."""
        comfy, ext, venv = self.workspace(tmp_path)
        unrelated = tmp_path / "unrelated-ComfyUI"
        (unrelated / "custom_nodes").mkdir(parents=True)
        (unrelated / "main.py").write_text("")

        out = self.launch(ext, unrelated, venv, tmp_path)
        assert out.returncode == 0, out.stderr
        assert str(comfy) in out.stdout
        assert "not the configured" in out.stdout

    def test_an_extension_already_in_place_needs_no_link(self, tmp_path):
        comfy, ext, venv = self.workspace(tmp_path)
        out = self.launch(ext, comfy, venv, tmp_path)
        assert out.returncode == 0, out.stderr
        assert "linked" not in out.stdout
