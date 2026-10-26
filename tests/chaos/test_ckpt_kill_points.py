"""Seeded kill-point suite: 200 real ``SIGKILL``s at each of four places inside a checkpoint save,
plus 200 at a random moment. After every kill a fresh process must restore exactly the newest
complete checkpoint, byte for byte what that step saved, never a partial file, and must be able to
save and restore the next step. See ``_ckpt_kill_points.py`` for the places and what each
iteration varies.

Run alone with ``uv run pytest -m chaos tests/chaos/test_ckpt_kill_points.py -s`` to see the
per-place counts.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
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


def run_point(tmp_path: Path, point: str) -> list[dict[str, Any]]:
    home = tmp_path / "home"
    home.mkdir()
    done = subprocess.run(  # noqa: S603
        [sys.executable, str(HARNESS), str(tmp_path / "work"), point, str(SEED), str(ITERATIONS)],
        capture_output=True, text=True, check=False, timeout=TIME_LIMIT_S,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONHASHSEED": "0"},
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    return [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]


def summary(point: str, rows: list[dict[str, Any]]) -> str:
    places = collections.Counter(r["place"] for r in rows)
    restored = collections.Counter(
        "newer" if r.get("restored") == r["acks"] + 1 else "acknowledged" for r in rows if r["ok"]
    )
    keyed = sum(1 for r in rows if r["keyed"])
    torn = sum(1 for r in rows if r.get("temp_files"))
    return (
        f"{point}: {len(rows)} kills, {sum(r['ok'] for r in rows)} restored correctly; "
        f"places {dict(sorted(places.items()))}; restored {dict(restored)}; "
        f"with a key {keyed}; temporary files left {torn}"
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
    places = collections.Counter(r["place"] for r in rows)
    assert all(count >= 20 for count in places.values()), places  # every place is exercised
    for r in rows:
        if r["place"] == "data_bytes":
            assert r["temp_files"] >= 1  # the torn file was on disk, and it was not loaded
        if r["place"] == "manifest_linked":
            assert "data_missing" in r["skipped"]  # a manifest without its data: skipped
        if point != "random_moment":
            assert r["restored"] == r["expected"]
