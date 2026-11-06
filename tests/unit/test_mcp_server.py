"""The MCP server (``cvpilot mcp``): tool surface, read/write split, path roots, gate verdicts.

Every call goes through a real in-memory client session, so tool errors are seen the way an agent
sees them: a result with ``isError``, not an exception.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from conftest import CliResult  # noqa: E402
from mcp.client import Client  # noqa: E402
from support import make_run, open_project, write_eval_files  # noqa: E402

from cvpilot.mcp_server import create_server  # noqa: E402

Invoke = Callable[..., CliResult]
READ_TOOLS = {
    "runs_list", "run_show", "baselines_list", "baseline_show", "compare", "gate",
    "ckpt_verify", "alerts_check",
}  # fmt: skip


def tools(server: Any) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(server) as client:
            return {t.name: t for t in (await client.list_tools()).tools}

    return asyncio.run(go())


def call(server: Any, name: str, **args: Any) -> tuple[bool, Any]:
    """(is_error, structured result or error text)."""

    async def go() -> tuple[bool, Any]:
        async with Client(server) as client:
            result = await client.call_tool(name, args)
            if result.is_error:
                return True, result.content[0].text
            return False, result.structured_content

    return asyncio.run(go())


def evaluate_into(invoke: Invoke, store_root: Path, folder: Path, **kw: Any) -> str:
    store = open_project(store_root)
    try:
        run = make_run(store)
    finally:
        store.close()
    preds, gts, classes = write_eval_files(folder, **kw)
    res = invoke(
        "eval", run, "--project", "proj", "--preds", str(preds), "--gts", str(gts),
        "--classes", str(classes), "--slices", "camera=camera", "--allow-closed",
        "--resamples", "0",
    )  # fmt: skip
    assert res.code == 0, res.err
    return run


@pytest.fixture
def server(store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("CVPILOT_PROJECT", "proj")
    return create_server(roots=(tmp_path,))


@pytest.fixture
def baseline_run(invoke: Invoke, store_root: Path, tmp_path: Path) -> str:
    run = evaluate_into(invoke, store_root, tmp_path / "base")
    assert invoke("baseline", "set", "prod", run, "--project", "proj").code == 0
    return run


# ---------------------------------------------------------------- surface
def test_read_only_by_default_and_every_tool_is_annotated(server: Any) -> None:
    found = tools(server)
    assert set(found) == READ_TOOLS
    for tool in found.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.title and tool.description


def test_baseline_set_needs_the_flag(
    store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_MCP_ALLOW_WRITE", "1")
    monkeypatch.setenv("CVPILOT_MCP_ROOTS", str(tmp_path))
    found = tools(create_server())
    assert set(found) == READ_TOOLS | {"baseline_set"}
    assert found["baseline_set"].annotations.read_only_hint is False
    assert found["baseline_set"].annotations.destructive_hint is False


def test_a_project_is_required(store_root: Path, tmp_path: Path) -> None:
    error, text = call(create_server(roots=(tmp_path,)), "runs_list")
    assert error and "no project" in text
    error, text = call(create_server(roots=(tmp_path,)), "runs_list", project="../x")
    assert error and "project must match" in text


# ---------------------------------------------------------------- runs and baselines
def test_runs_list_and_run_show(
    server: Any, invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    run = evaluate_into(invoke, store_root, tmp_path / "e")
    error, out = call(server, "runs_list")
    assert not error and out["total"] == 1 and out["runs"][0]["run_id"] == run
    error, out = call(server, "run_show", run_id=run.lower())
    assert not error
    assert out["meta"]["run_id"] == run and out["repro"]["dataset_hash"].startswith("merkle:")
    [snap] = out["snapshots"]
    assert snap["n_images"] > 0 and "mAP50_95" in snap["metrics"]
    error, text = call(server, "run_show", run_id="0" * 26)
    assert error


def test_baselines_list_and_show(server: Any, baseline_run: str) -> None:
    error, out = call(server, "baselines_list")
    assert not error and [b["tag"] for b in out["baselines"]] == ["prod"]
    error, out = call(server, "baseline_show", tag="prod")
    assert not error
    assert out["baseline"]["run_id"] == baseline_run and "mAP50_95" in out["metrics"]


def test_baseline_set_pins_a_tag(
    store_root: Path, tmp_path: Path, invoke: Invoke, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_PROJECT", "proj")
    run = evaluate_into(invoke, store_root, tmp_path / "e")
    writer = create_server(roots=(tmp_path,), allow_write=True)
    error, out = call(writer, "baseline_set", tag="candidate", run_id=run)
    assert not error and out["baseline"]["tag"] == "candidate"
    error, text = call(writer, "baseline_set", tag="candidate", run_id=run)
    assert error  # already pinned: needs replace=True


# ---------------------------------------------------------------- compare and gate
def test_compare_is_bounded_and_has_no_verdict(
    server: Any, invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    error, out = call(server, "compare", run_id=bad, baseline="tag:prod", top_k=1)
    assert not error
    assert out["candidate"]["run_id"] == bad and out["baseline"]["run_id"] == baseline_run
    assert {h["name"] for h in out["headline"]} >= {"mAP50_95"}
    assert len(out["worst_classes"]) <= 1 and out["class_rows"] >= 1
    assert "passed" not in out
    json.dumps(out)


def test_gate_fails_a_regression_and_passes_an_equal_run(
    server: Any, invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    error, out = call(
        server, "gate", run_id=bad, baseline="tag:prod", max_drop=["mAP50_95=0.5"],
        min_images=20,
    )  # fmt: skip
    assert not error and out["passed"] is False
    assert any(v["subject"] == "mAP50_95" for v in out["gate"]["violations"])
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    error, out = call(
        server, "gate", run_id=same, baseline="tag:prod", max_drop=["mAP50_95=0.5"],
        slice_max_drop=["camera=*:1.5"], min_images=20,
    )  # fmt: skip
    assert not error and out["passed"] is True


def test_gate_refuses_bad_thresholds_and_half_a_recompute(
    server: Any, baseline_run: str, tmp_path: Path
) -> None:
    error, text = call(server, "gate", run_id=baseline_run, baseline="tag:prod", max_drop=["x"])
    assert error and "NAME=POINTS" in text
    error, text = call(
        server, "gate", run_id=baseline_run, baseline="tag:prod",
        max_drop=["mAP50_95=0.5", "mAP50_95=1"],
    )  # fmt: skip
    assert error and "twice" in text
    error, text = call(
        server, "gate", run_id=baseline_run, baseline="tag:prod", max_drop=["mAP50_95=0.5"],
        gt=str(tmp_path / "gt.parquet"),
    )  # fmt: skip
    assert error and "together" in text


# ---------------------------------------------------------------- child-process tools and paths
def test_paths_outside_the_roots_are_refused(server: Any, tmp_path: Path) -> None:
    error, text = call(server, "ckpt_verify", directory=str(tmp_path.parent))
    assert error and "outside the allowed roots" in text
    (tmp_path / "link").symlink_to(tmp_path.parent)
    error, text = call(server, "ckpt_verify", directory="link")
    assert error and "outside the allowed roots" in text
    error, text = call(server, "alerts_check", config="missing.json")
    assert error and "does not exist" in text


def test_ckpt_verify_reports_an_empty_directory_as_not_ok(server: Any, tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir(mode=0o755)
    (tmp_path / "shared").chmod(0o755)
    error, text = call(server, "ckpt_verify", directory="shared")
    assert error and "chmod 700" in text  # the command line's refusal (exit 2) is a tool error
    (tmp_path / "ckpt").mkdir(mode=0o700)
    error, out = call(server, "ckpt_verify", directory="ckpt")
    assert not error and out["ok"] is False and out["total"] == 0
    assert any("no checkpoints" in n for n in out["notes"])


def test_alerts_check_validates_and_replays_without_sending(
    server: Any, invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    config = tmp_path / "alerts.json"
    config.write_text(json.dumps({"rate_limit_s": 60, "rules": [{"rule": "run_failed"}]}))
    run = evaluate_into(invoke, store_root, tmp_path / "e")
    error, out = call(server, "alerts_check", config="alerts.json", run=run)
    assert not error and out["ok"] is True
    assert out["lines"][0].startswith("config: ok (1 rules")
    assert any(line.startswith("replay: run") for line in out["lines"])
    config.write_text('{"rules": [{"rule": "exec"}]}')
    error, out = call(server, "alerts_check", config="alerts.json")
    assert not error and out["ok"] is False and out["error"]


# ---------------------------------------------------------------- failures and the command line
def test_an_unexpected_failure_reaches_the_client_without_internals(
    server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("/home/someone/private/path")

    monkeypatch.setattr("cvpilot.runs.list_runs", boom)
    error, text = call(server, "runs_list")
    assert error and "internal error (RuntimeError)" in text and "someone" not in text


def test_the_command_needs_the_extra(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "mcp" else real(name, *a)
    )
    res = invoke("mcp")
    assert res.code == 2 and "cvpilot[mcp]" in res.err


def test_the_command_serves_the_server(invoke: Invoke, monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[bool] = []
    monkeypatch.setattr("cvpilot.mcp_server.main", lambda: served.append(True))
    assert invoke("mcp").code == 0 and served == [True]


def test_a_recompute_gate_runs_in_a_child_process_under_the_cli_memory_limit(
    server: Any,
    invoke: Invoke,
    store_root: Path,
    tmp_path: Path,
    baseline_run: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-evaluation loads the ground truth: it must never run inside the long-lived server."""
    import cvpilot.compare.service as service

    def in_process(*a: Any, **kw: Any) -> None:
        if kw.get("recompute") is not None:
            raise AssertionError("recompute ran inside the server process")

    monkeypatch.setattr(service, "run_gate", in_process)  # the child process imports its own copy
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    gts, classes = tmp_path / "same" / "g.parquet", tmp_path / "same" / "c.txt"
    assert gts.is_file() and classes.is_file()
    error, out = call(
        server, "gate", run_id=same, baseline="tag:prod", max_drop=["mAP50_95=0.5"],
        min_images=20, gt=str(gts), classes=str(classes),
    )  # fmt: skip
    assert not error, out
    assert out["passed"] is True and out["comparison"]["candidate"]["run_id"] == same
    error, text = call(
        server, "gate", run_id=same, baseline="tag:missing", max_drop=["mAP50_95=0.5"],
        gt=str(gts), classes=str(classes),
    )  # fmt: skip
    assert error  # the child's refusal (exit 2 or no report) is a tool error
