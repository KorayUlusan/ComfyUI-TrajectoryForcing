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
import subprocess
from pathlib import Path

import pytest

EXT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(EXT_ROOT.glob("*.sh")) + sorted(EXT_ROOT.glob("env/*.sh"))

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
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(EXT_ROOT / "run_comfyui_slurm.sh"), "8199"],
        env=env, capture_output=True, text=True, timeout=120,
    )


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
