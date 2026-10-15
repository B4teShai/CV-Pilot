"""Smaller hardening items: exit mapping, impossible limits, support, artifacts, workers, text."""

from __future__ import annotations

import io
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import CliResult
from fake_forwarders import Recording
from support import (
    PARAMS,
    add_snapshot,
    batch,
    build_snapshot,
    make_run,
    meta_repro,
    open_project,
    write_eval_files,
)
from workflow_lint import lint

from cvpilot.compare import service
from cvpilot.compare.engine import compare
from cvpilot.compare.gate import ThresholdError, Thresholds, parse_drop
from cvpilot.core.errors import CompareRefusedError, CvpilotError, StoreError
from cvpilot.runs import forwarding

Invoke = Callable[..., CliResult]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def base(invoke: Invoke, store_root: Path, tmp_path: Path) -> str:
    return seed(invoke, store_root, tmp_path / "base", pin=True)


def seed(invoke: Invoke, store_root: Path, folder: Path, *, pin: bool = False, **kw: Any) -> str:
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
    if pin:
        assert invoke("baseline", "set", "prod", run, "--project", "proj").code == 0
    return run


def gate(invoke: Invoke, run: str, *extra: str) -> CliResult:
    return invoke(
        "gate", run, "--project", "proj", "--baseline", "tag:prod", "--max-drop", "mAP50_95=0.5",
        "--min-images", "20", *extra,
    )  # fmt: skip


# -- 13: the error boundary maps every way out --


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (SystemExit(0), 1),
        (SystemExit(None), 1),
        (SystemExit("bye"), 1),
        (SystemExit(3), 3),
        (KeyboardInterrupt(), 130),
    ],
)
def test_a_system_exit_or_interrupt_inside_the_gate_never_becomes_success(
    invoke: Invoke, base: str, monkeypatch: pytest.MonkeyPatch, raised: BaseException, code: int
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise raised

    monkeypatch.setattr(service, "gate", boom)
    res = gate(invoke, base)
    assert res.code == code
    assert not res.event("gate_passed")
    if code == 1:
        assert res.event("command_failed")


def test_clean_exits_still_succeed(invoke: Invoke) -> None:
    assert invoke("--version").code == 0
    assert invoke("doctor").code == 0
    assert invoke("--help").code == 0


# -- 15: limits that cannot fail --


@pytest.mark.parametrize("text", ["mAP50_95=100", "mAP50_95=100.0", "mAP50_95=100.5"])
def test_a_limit_of_100_points_is_refused_because_it_can_never_fail(text: str) -> None:
    with pytest.raises(ThresholdError, match="never fail|above"):
        parse_drop(text)
    with pytest.raises(ThresholdError):
        Thresholds(max_drop={"mAP50_95": 100.0})


def test_the_largest_accepted_limit_is_below_100() -> None:
    assert parse_drop("mAP50_95=99.999999") == ("mAP50_95", 99.999999)


def test_the_cli_refuses_a_limit_of_100_and_a_zero_floor(invoke: Invoke, base: str) -> None:
    assert (
        invoke(
            "gate",
            base,
            "--project",
            "proj",
            "--baseline",
            "tag:prod",
            "--max-drop",
            "mAP50_95=100",
        ).code
        == 2
    )
    assert (
        invoke(
            "gate",
            base,
            "--project",
            "proj",
            "--baseline",
            "tag:prod",
            "--max-drop",
            "mAP50_95=1",
            "--min-images",
            "0",
        ).code
        == 2
    )


# -- 16: per-class support --


def classes(support: dict[str, float]) -> dict[str, dict[str, float]]:
    return {n: {"AP50_95": 0.5, "support": s} for n, s in support.items()}


def test_snapshots_with_different_per_class_support_are_refused() -> None:
    names = ["cat", "dog", "bird"]
    base = build_snapshot(class_names=names, per_class=classes({"cat": 10, "dog": 10, "bird": 10}))
    cand = build_snapshot(class_names=names, per_class=classes({"cat": 10, "dog": 9, "bird": 11}))
    with pytest.raises(CompareRefusedError) as exc:
        compare(cand, base)
    assert exc.value.reason == "class_support"
    compare(base, base)  # equal support compares normally


def test_a_class_without_a_support_value_on_one_side_is_refused() -> None:
    names = ["cat", "dog", "bird"]
    full = classes({"cat": 10, "dog": 10, "bird": 10})
    partial = {**full, "dog": {"AP50_95": 0.5}}
    with pytest.raises(CompareRefusedError) as exc:
        compare(
            build_snapshot(class_names=names, per_class=partial),
            build_snapshot(class_names=names, per_class=full),
        )
    assert exc.value.reason == "class_support"


# -- 14: artifacts are verified on every gate --


def test_a_tampered_confusion_artifact_fails_the_gate_without_any_report_flag(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    cand = seed(invoke, store_root, tmp_path / "cand")
    store = open_project(store_root)
    try:
        path = store.artifact_path(store.list_snapshots(cand)[0].confusion_sha256)
    finally:
        store.close()
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"x")
    res = gate(invoke, cand)
    assert res.code == 1 and res.event("snapshot_integrity_failed") and not res.event("gate_passed")


# -- 17: the artifact that was hashed is the artifact that is read --


def test_open_artifact_returns_the_verified_bytes_from_memory(tmp_path: Path) -> None:
    store = open_project(tmp_path)
    try:
        run = make_run(store)
        art = store.put_artifact_bytes(run, b"verified bytes", "predictions", {})
        handle = store.open_artifact(art.sha256)
        assert isinstance(handle, io.BytesIO)
        path = store.artifact_path(art.sha256)
        path.chmod(0o644)
        path.write_bytes(b"swapped after verification")  # a later change cannot reach the reader
        assert handle.read() == b"verified bytes"
    finally:
        store.close()


def test_open_artifact_has_a_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cvpilot.store import local

    store = open_project(tmp_path)
    try:
        run = make_run(store)
        art = store.put_artifact_bytes(run, b"0123456789", "predictions", {})
        monkeypatch.setattr(local, "MAX_OPEN_ARTIFACT_BYTES", 5)
        with pytest.raises(StoreError, match="too large"):
            store.open_artifact(art.sha256)
    finally:
        store.close()


# -- 18: a forwarder worker survives anything an adapter throws --


class Exits(Recording):
    name = "exits"

    def log_points(self, points: Any) -> None:
        self.calls.append(("attempt", len(points)))
        raise SystemExit(1)  # an SDK that calls sys.exit


def test_a_base_exception_from_an_adapter_disables_it_logs_once_and_closes_it() -> None:
    meta, repro = meta_repro()
    adapter = Exits()
    fan = forwarding.Fanout([adapter], meta, repro)
    fan.start()
    for step in range(5):
        fan.submit(batch(meta, step))
    stats = fan.close("finished")
    assert stats["exits"].disabled_reason == "base_exception"
    assert [c for c in adapter.calls if c[0] == "attempt"] == [
        ("attempt", 3)
    ]  # never offered again
    assert [c for c in adapter.calls if c[0] == "close"] == [("close", "failed")]  # closed once
    assert not [t for t in threading.enumerate() if t.name.startswith("cvpilot-forward-exits")]


# -- 19: error lines are made printable --


def test_error_lines_never_carry_control_or_bidirectional_characters(
    invoke: Invoke, base: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise CvpilotError("bad name \x1b[31mred‮txt\x07 and ::error::pwn")

    monkeypatch.setattr(service, "gate", boom)
    res = gate(invoke, base)
    assert res.code == 1
    line = next(x for x in res.human.splitlines() if x.startswith("error:"))
    assert "\x1b" not in line and "‮" not in line and "\x07" not in line
    assert "\\u202e" in line and "\\u001b" in line


# -- 20: checkout does not keep credentials --


def test_every_checkout_in_every_workflow_drops_the_credentials() -> None:
    docs = [
        yaml.safe_load(p.read_text())
        for p in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    ]
    assert docs
    for doc in docs:
        assert lint(doc) == []
        for job in doc["jobs"].values():
            for step in job["steps"]:
                if str(step.get("uses", "")).startswith("actions/checkout@"):
                    assert (step.get("with") or {}).get("persist-credentials") is False
    assert PARAMS and add_snapshot  # keep shared helpers imported
    assert sys.version_info >= (3, 10)
