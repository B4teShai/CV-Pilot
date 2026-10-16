"""The documented restart recipe calls ``find_resume_point`` and ``recover_run`` before
``cvpilot.init``. Their events must be JSON like every other event, not structlog's default
console format."""

import json
import os
import subprocess
import sys
from pathlib import Path

_CRASHED_JOB = """
import os
import cvpilot
from cvpilot.ckpt import CheckpointManager

run = cvpilot.init("rp", task="classification")
CheckpointManager(run, every_n_steps=1).maybe_save(1, model={"w": [1.0]})
os._exit(0)  # a crash: the run stays 'running'
"""

_RESTART = """
from cvpilot.ckpt import find_resume_point
from cvpilot.runs import recover_run

point = find_resume_point("rp")  # before cvpilot.init, as in the documented recipe
assert point is not None and point.needs_recover
recover_run("rp", point.run_id)
"""


def _run(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "CVPILOT_STORE_DIR": str(tmp_path / "store"),
        "CVPILOT_CHECKPOINT_DIR": str(tmp_path / "ckpt"),
        "GIT_CEILING_DIRECTORIES": str(tmp_path.parent),
    }
    for name in ("CVPILOT_LOG_LEVEL", "CVPILOT_ENV", "CVPILOT_CKPT_KEY"):
        env.pop(name, None)
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )


def test_events_before_init_are_json(tmp_path: Path) -> None:
    assert _run(_CRASHED_JOB, tmp_path).returncode == 0
    restart = _run(_RESTART, tmp_path)
    assert restart.returncode == 0, restart.stderr
    assert restart.stdout == "", "events belong on stderr; stdout is for results"
    lines = [line for line in restart.stderr.splitlines() if line.strip()]
    assert lines, "the restart recipe should log resume_point_selected and run_finished"
    events = [json.loads(line) for line in lines]  # a console-format line fails here
    assert {"resume_point_selected", "run_finished"} <= {e["message"] for e in events}
    required = {"@timestamp", "level", "service", "env", "request_id", "logger"}
    assert all(required <= e.keys() for e in events)
