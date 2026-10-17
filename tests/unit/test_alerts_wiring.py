"""Alerts wired through ``init(alerts=...)`` and ``CVPILOT_ALERTS`` (log, log_eval, finish)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import structlog
from fake_channels import (
    BlockingChannel,
    FailingChannel,
    RecordingChannel,
    use_channels,
)
from fake_forwarders import LEAK_MARKER
from support import write_eval_files

from cvpilot import api, runs
from cvpilot.alerts import engine as alert_engine
from cvpilot.alerts.config import parse_alert_config
from cvpilot.compare import baselines
from cvpilot.core.errors import AlertConfigError
from cvpilot.eval.files import load_ground_truth, load_predictions
from cvpilot.forward._common import ForwardConfigError
from cvpilot.logging import clear_request_context
from cvpilot.registry import UnknownPluginError, tables
from cvpilot.runs import forwarding
from cvpilot.settings import SettingsError, load_settings

CONFIG: dict[str, Any] = {
    "channels": ["recording"],
    "rate_limit_s": 1,
    "rules": [
        {"rule": "nan_loss"},
        {"rule": "no_progress", "patience": 3},
        {"rule": "run_failed"},
        {"rule": "run_preempted"},
    ],
}


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO", "CVPILOT_ALERTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


def events(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    err = capsys.readouterr().err
    return [json.loads(x) for x in err.splitlines() if x.startswith("{")]


# -- the setting -----------------------------------------------------------------------------


def test_the_alerts_setting_is_a_path_and_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert load_settings({}).alerts is None
    assert (
        load_settings({"CVPILOT_ALERTS": str(tmp_path / "a.json")}).alerts
        == (tmp_path / "a.json").resolve()
    )


@pytest.mark.parametrize("value", ["", "   ", "a\x00b"])
def test_an_invalid_alerts_setting_names_the_variable_and_not_the_value(value: str) -> None:
    with pytest.raises(SettingsError, match="CVPILOT_ALERTS") as info:
        load_settings({"CVPILOT_ALERTS": value})
    assert "\x00" not in str(info.value)


# -- on log and finish -----------------------------------------------------------------------


def test_a_nan_loss_and_a_failed_run_reach_the_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    run.log({"train/loss": 0.5}, step=0)
    run.log({"train/loss": float("nan")}, step=1)
    run.finish("failed")
    (chan,) = RecordingChannel.instances
    assert [(r, s) for r, s, _ in chan.sent] == [("nan_loss", 1), ("run_failed", None)]
    assert chan.sent[0][2] == f"[proj] Loss train/loss is NaN at step 1 (run {run.run_id})."
    assert chan.calls[0][0] == "open" and chan.calls[-1] == ("close", "failed")


def test_a_preempted_run_fires_and_a_finished_run_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    run.finish("preempted")
    assert [r for r, _, _ in RecordingChannel.instances[0].sent] == ["run_preempted"]
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    run.finish("finished")
    assert RecordingChannel.instances[0].sent == []


def test_an_exception_in_the_with_block_fails_the_run_and_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    with pytest.raises(RuntimeError):
        with api.init("proj", task="detection", alerts=CONFIG):
            raise RuntimeError("boom")
    assert [r for r, _, _ in RecordingChannel.instances[0].sent] == ["run_failed"]


def test_a_plateau_alerts_through_the_run_log_path(monkeypatch: pytest.MonkeyPatch) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    for step, loss in enumerate([1.0, 0.5, 0.6, 0.6, 0.6]):
        run.log({"loss": loss}, step=step)
    run.finish()
    (chan,) = RecordingChannel.instances
    assert [r for r, _, _ in chan.sent] == ["no_progress"] and chan.sent[0][1] == 4


def test_the_file_the_variable_points_to_is_used_and_an_argument_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "alerts.json"
    path.write_text(json.dumps(CONFIG))
    monkeypatch.setenv("CVPILOT_ALERTS", str(path))
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection")
    run.log({"loss": float("nan")}, step=0)
    run.finish()
    assert [r for r, _, _ in RecordingChannel.instances[0].sent] == ["nan_loss"]
    use_channels(monkeypatch, recording=RecordingChannel)
    other = {**CONFIG, "rules": [{"rule": "run_failed"}]}
    run = api.init("proj", task="detection", alerts=other)
    run.log({"loss": float("nan")}, step=0)
    run.finish()
    assert RecordingChannel.instances[0].sent == []  # the argument's rules, not the file's
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=str(path))  # a path as an argument
    run.log({"loss": float("nan")}, step=0)
    run.finish()
    assert len(RecordingChannel.instances[0].sent) == 1


def test_without_channels_alerts_are_only_logged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_channels(monkeypatch)
    run = api.init("proj", task="detection", alerts={"rules": [{"rule": "nan_loss"}]})
    run.log({"loss": float("nan")}, step=0)
    run.finish()
    fired = [e for e in events(capsys) if e["message"] == "alert_fired"]
    assert len(fired) == 1 and fired[0]["rule"] == "nan_loss" and fired[0]["run_id"] == run.run_id
    assert fired[0]["request_id"].startswith(run.run_id)


# -- configuration errors are loud and early --------------------------------------------------


def test_a_bad_configuration_fails_before_any_run_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    with pytest.raises(AlertConfigError):
        api.init("proj", task="detection", alerts={"rules": [{"rule": "nope"}]})
    with pytest.raises(AlertConfigError):
        api.init("proj", task="detection", alerts=tmp_path / "missing.json")
    assert runs.list_runs("proj", store_dir=tmp_path / "store") == [] and api._active is None


def test_an_unknown_channel_fails_before_any_run_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    with pytest.raises(UnknownPluginError, match="pagerduty"):
        api.init("proj", task="detection", alerts={**CONFIG, "channels": ["pagerduty"]})
    assert runs.list_runs("proj", store_dir=tmp_path / "store") == [] and api._active is None


def test_a_channel_without_its_environment_fails_before_any_run_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("CVPILOT_ALERT_SLACK_URL", "SSL_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ForwardConfigError, match="CVPILOT_ALERT_SLACK_URL"):
        api.init("proj", task="detection", alerts={**CONFIG, "channels": ["slack"]})
    assert runs.list_runs("proj", store_dir=tmp_path / "store") == [] and api._active is None


def test_only_rank_zero_alerts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RANK", "1")
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    run.log({"loss": float("nan")}, step=0)
    run.finish("failed")
    assert RecordingChannel.instances == []
    assert any(e["message"] == "alert_skipped_rank" for e in events(capsys))


def test_the_development_setting_reaches_the_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    run.finish()
    assert RecordingChannel.instances[0].allow_insecure is False
    monkeypatch.setenv("CVPILOT_ENV", "dev")
    monkeypatch.setenv("CVPILOT_FORWARD_HOSTS", "hooks.example")
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    run.finish()
    chan = RecordingChannel.instances[0]
    assert chan.allow_insecure is True and chan.allowed_hosts == ("hooks.example",)


# -- isolation: a dead, a slow or a broken alert path never touches training ------------------


def test_a_dead_channel_never_fails_the_run_or_leaks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_channels(monkeypatch, failing=FailingChannel)
    cfg = {**CONFIG, "channels": ["failing"]}
    run = api.init("proj", task="detection", alerts=cfg)
    for step in range(20):
        run.log({"loss": 1.0 / (step + 1)}, step=step)
    run.log({"loss": float("nan")}, step=20)
    run.finish("failed")
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "failed" and view.stats.n_points == 21
    out = events(capsys)
    failed = [e for e in out if e["message"] == "alert_delivery_failed"]
    assert len(failed) == 1 and failed[0]["channel"] == "failing"
    assert LEAK_MARKER not in json.dumps(out)


def test_a_channel_that_never_answers_cannot_slow_log_even_with_its_queue_saturated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fake clock steps past the rate limit on every call, so every NaN is a new alert: the
    channel's queue fills (16) and overflows (the per-run cap is 20), and no call may wait."""
    from support import meta_repro

    monkeypatch.setattr(forwarding, "NOTIFY_CLOSE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(forwarding, "CLOSE_TIMEOUT_S", 0.3)
    use_channels(monkeypatch, blocking=BlockingChannel)
    meta, repro = meta_repro()
    ticks = iter(range(10**6))
    config = parse_alert_config({**CONFIG, "channels": ["blocking"], "rate_limit_s": 1})
    from cvpilot.registry import channels as channel_registry

    channel = channel_registry.get("blocking")()
    handle = runs.start_run(
        meta, repro, store_dir=tmp_path / "s", alerts=config, channels=[channel],
        clock=lambda: float(next(ticks)) * 2.0,
    )  # fmt: skip
    worst = 0.0
    for step in range(60):
        start = time.perf_counter()
        handle.log({"loss": float("nan")}, step=step)
        worst = max(worst, time.perf_counter() - start)
    assert worst < 0.2
    worker = handle._notify._workers[0]  # type: ignore[union-attr]
    assert handle._alerts.stats.queued > 16  # type: ignore[union-attr]
    assert worker.stats.dropped_batches > 0, "the queue never overflowed: the test proves nothing"
    start = time.perf_counter()
    handle.finish("failed")
    assert time.perf_counter() - start < 2.0
    channel.release.set()


def test_an_engine_that_raises_cannot_break_log(monkeypatch: pytest.MonkeyPatch) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("engine bug")

    monkeypatch.setattr(alert_engine.AlertEngine, "on_points", boom)
    monkeypatch.setattr(alert_engine.AlertEngine, "on_finish", boom)
    run.log({"loss": 1.0}, step=0)  # must not raise
    run.finish("failed")
    assert run.status == "failed"


def test_an_abandoned_run_closes_the_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)
    run.abandon()
    assert RecordingChannel.instances[0].calls[-1] == ("close", "running")


# -- the evaluation path: a real comparison against a pinned baseline ------------------------


def _log_eval(run: api.Run, folder: Path, **files: Any) -> None:
    preds, gts, _ = write_eval_files(folder, **files)
    io = tables.get("parquet")()
    run.log_eval(
        load_predictions(preds, "detection", io),
        load_ground_truth(gts, "detection", io),
        step=0,
        class_names=["cat", "dog"],
        n_resamples=0,
    )


EVAL_CONFIG = {
    "channels": ["recording"],
    "rules": [
        {"rule": "eval_regression", "baseline": "tag:prod", "max_drop": {"mAP50_95": 0.5}},
    ],
}


def test_a_degraded_candidate_alerts_and_an_equal_one_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    base = api.init("proj", task="detection")
    _log_eval(base, tmp_path / "base")
    base.finish()
    baselines.set_baseline("proj", "prod", base.run_id, store_dir=tmp_path / "store")
    same = api.init("proj", task="detection", alerts=EVAL_CONFIG)
    _log_eval(same, tmp_path / "same")
    same.finish()
    assert RecordingChannel.instances[0].sent == []
    use_channels(monkeypatch, recording=RecordingChannel)
    bad = api.init("proj", task="detection", alerts=EVAL_CONFIG)
    _log_eval(bad, tmp_path / "bad", broken=10)
    bad.finish()
    (row,) = RecordingChannel.instances[0].sent
    assert row[0] == "eval_regression" and row[1] == 0
    assert "worse than baseline tag:prod: headline_drop on mAP50_95" in row[2]
    assert f"(run {bad.run_id})" in row[2]


def test_a_comparison_that_cannot_be_made_alerts_instead_of_staying_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=EVAL_CONFIG)  # no baseline "prod" exists
    _log_eval(run, tmp_path / "x")
    run.finish()
    (row,) = RecordingChannel.instances[0].sent
    assert row[0] == "eval_regression" and "could not be compared with baseline tag:prod" in row[2]


# -- fix wave: terminal alerts survive a failed flush, summaries, crashes outside a with block -----


def test_a_failed_flush_in_finish_does_not_swallow_the_run_failed_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cvpilot.core.errors import StoreError

    use_channels(monkeypatch, recording=RecordingChannel)
    run = api.init("proj", task="detection", alerts=CONFIG)

    def broken_flush() -> None:
        raise StoreError("disk full")

    healthy_flush = run.handle.flush
    monkeypatch.setattr(run.handle, "flush", broken_flush)
    with pytest.raises(StoreError):
        run.finish("failed")
    monkeypatch.setattr(run.handle, "flush", healthy_flush)
    run.abandon()  # the run is still open after the failed finish: release it
    chan = RecordingChannel.instances[0]
    deadline = time.time() + 5
    while time.time() < deadline and not chan.sent:
        time.sleep(0.02)
    assert [r for r, _, _ in chan.sent] == ["run_failed"]


def test_finish_writes_one_alert_summary_with_queued_suppressed_and_delivered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel, failing=FailingChannel)
    cfg = {**CONFIG, "channels": ["recording", "failing"], "rate_limit_s": 60}
    run = api.init("proj", task="detection", alerts=cfg)
    for step in range(5):
        run.log({"loss": float("nan")}, step=step)
    run.finish("failed")
    (summary,) = [e for e in events(capsys) if e["message"] == "alert_summary"]
    assert summary["queued"] == 2 and summary["suppressed"] == 4  # nan_loss once, run_failed
    assert summary["delivered"] == 2 and summary["delivery_failed"] == 2
    assert summary["dropped"] == 0 and summary["rule_failures"] == 0 and summary["run_id"]


def test_no_alert_summary_without_alerts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = api.init("proj", task="detection")
    run.finish()
    assert not [e for e in events(capsys) if e["message"] == "alert_summary"]


def _crash_run(tmp_path: Path, body: str) -> tuple[Any, int]:
    import subprocess
    import sys

    from notify_support import Listener

    listener = Listener()
    code = (
        "import cvpilot\n"
        "cvpilot.init('p', task='detection', alerts={'channels': ['webhook'],"
        " 'rules': [{'rule': 'run_failed'}]})\n" + body
    )
    env = {
        "PATH": __import__("os").environ["PATH"],
        "CVPILOT_ENV": "dev",
        "CVPILOT_STORE_DIR": str(tmp_path / "s"),
        "CVPILOT_ALERT_WEBHOOK_URL": listener.url + "/hook",
        "GIT_CEILING_DIRECTORIES": str(tmp_path.parent),
    }
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, timeout=120,
        check=False,
    )  # fmt: skip
    time.sleep(0.2)
    received = list(listener.requests)
    listener.close()
    return received, done.returncode


def test_an_uncaught_exception_outside_a_with_block_still_sends_run_failed(tmp_path: Path) -> None:
    received, code = _crash_run(tmp_path, "raise RuntimeError('boom')\n")
    assert code == 1
    assert [r.json()["rule"] for r in received] == ["run_failed"]


def test_a_process_that_simply_ends_without_finishing_sends_nothing(tmp_path: Path) -> None:
    received, code = _crash_run(tmp_path, "pass\n")
    assert code == 0 and received == []
