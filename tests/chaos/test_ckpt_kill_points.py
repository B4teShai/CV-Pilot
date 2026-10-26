"""Seeded kill-point suite: 200 real ``SIGKILL``s in each of four groups of places inside a
checkpoint save (11 named places, each hit at least 20 times), plus 200 at a random moment. After
every kill a fresh process must restore exactly the newest complete checkpoint, byte for byte what
that step saved, never a partial file, and must be able to save and restore the next step. See
``_ckpt_kill_points.py`` for the places and what each iteration varies.

Run alone with ``uv run pytest -m chaos tests/chaos/test_ckpt_kill_points.py -s`` to see the
per-place counts.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="needs fork, SIGSTOP and SIGKILL"),
]
HARNESS = Path(__file__).with_name("_ckpt_kill_points.py")
ITERATIONS = 200
SEED = 20261026
TIME_LIMIT_S = 600  # about 40 s per point on a laptop; the bound only stops a hung run


def run_group(argv: list[str], env: dict[str, str], timeout: float) -> tuple[int, str, str]:
    """Run ``argv`` in its own session and process group and kill the whole group afterwards: the
    harness forks writers that stop themselves with SIGSTOP, and if the harness dies or times out
    they must not be left behind stopped."""
    proc = subprocess.Popen(  # noqa: S603
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        start_new_session=True,
    )  # fmt: skip
    try:
        out, err = proc.communicate(timeout=timeout)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # whatever of the group is still alive
        proc.wait()
    return proc.returncode, out, err


def run_point(tmp_path: Path, point: str) -> list[dict[str, Any]]:
    home = tmp_path / "home"
    home.mkdir()
    code, out, err = run_group(
        [sys.executable, str(HARNESS), str(tmp_path / "work"), point, str(SEED), str(ITERATIONS)],
        {"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONHASHSEED": "0"},
        TIME_LIMIT_S,
    )
    assert code == 0, err[-3000:]
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_a_stopped_writer_left_by_a_dying_harness_is_killed_with_its_group(
    tmp_path: Path,
) -> None:
    script = (
        "import os, signal, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.close(1)\n"
        "    os.close(2)\n"  # like the harness's writers, which detach from its pipes
        "    os.kill(os.getpid(), signal.SIGSTOP)\n"  # a writer stopped for ever
        "    os._exit(0)\n"
        "print(pid, flush=True)\n"
        "os._exit(0)\n"  # the harness dies without killing its child
    )
    code, out, _ = run_group([sys.executable, "-c", script], {"PATH": "/usr/bin:/bin"}, 30)
    child = int(out.split()[0])
    assert code == 0
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return  # gone
        time.sleep(0.05)
    os.kill(child, signal.SIGKILL)
    pytest.fail("the stopped child survived the group kill")


def summary(point: str, rows: list[dict[str, Any]]) -> str:
    places = collections.Counter(r["place"] for r in rows)
    restored = collections.Counter(
        "newer" if r.get("restored") == r["acks"] + 1 else "acknowledged" for r in rows if r["ok"]
    )
    keyed = sum(1 for r in rows if r["keyed"])
    in_save = sum(1 for r in rows if r.get("in_save"))
    torn = sum(1 for r in rows if r.get("temp_files"))
    return (
        f"{point}: {len(rows)} kills, {sum(r['ok'] for r in rows)} restored correctly; "
        f"places {dict(sorted(places.items()))}; restored {dict(restored)}; "
        f"with a key {keyed}; temporary files left {torn}; killed with a save in flight {in_save}"
    )


@pytest.mark.parametrize(
    "point",
    [
        "before_temp_write",
        "mid_write",
        "after_write_before_rename",
        "after_rename_before_chain",
        "random_moment",
    ],
)
def test_every_kill_inside_a_save_restores_the_newest_complete_checkpoint(
    tmp_path: Path, point: str
) -> None:
    rows = run_point(tmp_path, point)
    print(summary(point, rows))
    assert len(rows) == ITERATIONS
    assert all(r["killed"] for r in rows), [r for r in rows if not r["killed"]][:2]
    failed = [r for r in rows if not r["ok"]]
    assert failed == [], json.dumps(failed[:3], indent=1)[:4000]
    if point == "random_moment":
        assert sum(1 for r in rows if r["in_save"]) >= 1  # the count is printed, not assumed
    if point == "random_moment":
        assert sum(1 for r in rows if r["in_save"]) >= 1  # the count is printed, not assumed
    places = collections.Counter(r["place"] for r in rows)
    assert all(count >= 20 for count in places.values()), places  # every place is exercised
    for r in rows:
        if r["place"] == "data_bytes":
            assert r["temp_files"] >= 1  # the torn file was on disk, and it was not loaded
        if r["place"] == "manifest_linked":
            # a manifest without its data is refused for a reason that depends on the policy: the
            # store never recorded it (not_anchored) under anchored, its data is missing otherwise
            reason = "not_anchored" if r["anchored"] else "data_missing"
            assert reason in r["skipped"], r
            other = {"not_anchored", "data_missing"} - {reason}
            assert not other & set(r["skipped"]), r
        if point != "random_moment":
            assert r["restored"] == r["expected"]


def test_the_docs_count_the_named_places_the_harness_really_has() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("kill_points_harness", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    named = {
        place
        for point, places in module.POINTS.items()
        if point != "random_moment"
        for place in places
    }
    assert (
        len(named) == 11
    )  # "11 named places" in the changelog, status page, ADR-0012 and testing guide
    assert set(module.POINTS) == {
        "before_temp_write", "mid_write", "after_write_before_rename", "after_rename_before_chain",
        "random_moment",
    }  # fmt: skip
