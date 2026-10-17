"""``cvpilot alerts check``: validate a configuration and replay a recorded run; sends nothing."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import CliResult
from fake_channels import RecordingChannel, use_channels
from notify_support import Listener
from support import write_eval_files

from cvpilot import api
from cvpilot.compare import baselines
from cvpilot.eval.files import load_ground_truth, load_predictions
from cvpilot.notify import _http
from cvpilot.registry import tables

Invoke = Callable[..., CliResult]
CONFIG: dict[str, Any] = {
    "rate_limit_s": 60,
    "rules": [
        {"rule": "nan_loss"},
        {"rule": "no_progress", "patience": 3},
        {"rule": "run_failed"},
    ],
}


@pytest.fixture(autouse=True)
def nothing_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a dry run must not send anything")

    monkeypatch.setattr(_http, "post_json", forbidden)
    for name in ("CVPILOT_ALERTS", "CVPILOT_ALERT_SLACK_URL"):
        monkeypatch.delenv(name, raising=False)


def config_file(tmp_path: Path, data: Any = None, name: str = "alerts.json") -> Path:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(CONFIG if data is None else data))
    return path


def recorded_run(store_root: Path, losses: list[float], status: str = "finished") -> str:
    run = api.init("proj", task="detection", store_dir=store_root)
    for step, value in enumerate(losses):
        run.log({"train/loss": value}, step=step)
    run.finish(status)  # type: ignore[arg-type]
    return run.run_id


def test_a_valid_configuration_is_described_and_nothing_is_sent(
    invoke: Invoke, tmp_path: Path
) -> None:
    res = invoke("alerts", "check", str(config_file(tmp_path)))
    assert res.code == 0, res.err
    assert "config: ok (3 rules, rate limit 60 s)" in res.out
    assert "rule nan_loss" in res.out and "rule no_progress" in res.out
    assert "channels: none (alerts are only logged)" in res.out
    assert res.out.rstrip().endswith("nothing was sent")


def test_the_replay_of_a_recorded_run_lists_what_would_fire(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    run = recorded_run(store_root, [1.0, 0.8, float("nan"), float("nan")], status="failed")
    res = invoke("alerts", "check", str(config_file(tmp_path)), "--project", "proj", "--run", run)
    assert res.code == 0, res.err
    assert f"replay: run {run} (project proj): 4 points, 0 evaluation snapshots, status failed" in (
        res.out
    )
    assert (
        f"would fire: nan_loss at step 2: [proj] Loss train/loss is NaN at step 2 (run {run})."
        in (res.out)
    )
    assert "would fire: run_failed:" in res.out
    assert "would suppress: 1" in res.out  # the second NaN is inside the rate-limit window
    assert not res.event("alert_fired")  # a dry run logs nothing that looks like a real alert
    assert res.out.rstrip().endswith("nothing was sent")


def test_a_healthy_run_would_fire_nothing(invoke: Invoke, store_root: Path, tmp_path: Path) -> None:
    run = recorded_run(store_root, [1.0, 0.8, 0.6, 0.4])
    res = invoke("alerts", "check", str(config_file(tmp_path)), "--project", "proj", "--run", run)
    assert res.code == 0 and "would fire: nothing" in res.out


def test_the_replay_is_deterministic(invoke: Invoke, store_root: Path, tmp_path: Path) -> None:
    run = recorded_run(store_root, [1.0, 1.0, 1.0, 1.0, float("nan")], status="failed")
    args = ("alerts", "check", str(config_file(tmp_path)), "--project", "proj", "--run", run)
    assert invoke(*args).out == invoke(*args).out


def test_a_replay_uses_the_real_comparison_for_the_eval_rule(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    io = tables.get("parquet")()

    def evaluated(folder: str, **files: Any) -> str:
        run = api.init("proj", task="detection", store_dir=store_root)
        preds, gts, _ = write_eval_files(tmp_path / folder, **files)
        run.log_eval(
            load_predictions(preds, "detection", io),
            load_ground_truth(gts, "detection", io),
            step=0,
            class_names=["cat", "dog"],
            n_resamples=0,
        )
        run.finish()
        return run.run_id

    base = evaluated("base")
    baselines.set_baseline("proj", "prod", base, store_dir=store_root)
    bad = evaluated("bad", broken=10)
    cfg = {
        "rules": [
            {"rule": "eval_regression", "baseline": "tag:prod", "max_drop": {"mAP50_95": 0.5}}
        ]
    }
    res = invoke(
        "alerts", "check", str(config_file(tmp_path, cfg)), "--project", "proj", "--run", bad
    )
    assert res.code == 0, res.err
    assert "1 evaluation snapshots" in res.out
    assert "would fire: eval_regression at step 0:" in res.out and "headline_drop" in res.out
    good = invoke(
        "alerts", "check", str(config_file(tmp_path, cfg)), "--project", "proj", "--run", base
    )
    assert "would fire: nothing" in good.out


def test_a_channel_that_cannot_be_built_is_a_configuration_error(
    invoke: Invoke, tmp_path: Path
) -> None:
    cfg = {**CONFIG, "channels": ["slack"]}
    res = invoke("alerts", "check", str(config_file(tmp_path, cfg)))
    assert res.code == 2
    assert "CVPILOT_ALERT_SLACK_URL" in res.human


def test_a_channel_that_can_be_built_is_reported_ready_without_sending(
    invoke: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = Listener()
    try:
        monkeypatch.setenv("CVPILOT_ENV", "dev")
        monkeypatch.setenv("CVPILOT_ALERT_SLACK_URL", server.url + "/services/T/B/SECRETPATH")
        cfg = {**CONFIG, "channels": ["slack"]}
        res = invoke("alerts", "check", str(config_file(tmp_path, cfg)))
        assert res.code == 0, res.err
        assert "channel slack: ready" in res.out
        assert "SECRETPATH" not in res.out + res.err and server.requests == []
    finally:
        server.close()


def test_an_unknown_channel_is_a_configuration_error(
    invoke: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_channels(monkeypatch, recording=RecordingChannel)
    res = invoke(
        "alerts", "check", str(config_file(tmp_path, {**CONFIG, "channels": ["pagerduty"]}))
    )
    assert res.code == 2 and "pagerduty" in res.human


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"rules": []}', '{"rules": [{"rule": "exec", "cmd": "rm -rf /"}]}', ""],
)
def test_an_invalid_configuration_exits_2_naming_the_problem_not_the_value(
    invoke: Invoke, tmp_path: Path, content: str
) -> None:
    res = invoke("alerts", "check", str(config_file(tmp_path, content)))
    assert res.code == 2
    assert res.human.startswith("error:") and "rm -rf" not in res.err
    assert "Traceback" not in res.err


def test_a_missing_file_and_a_symlink_are_refused(invoke: Invoke, tmp_path: Path) -> None:
    assert invoke("alerts", "check", str(tmp_path / "missing.json")).code == 2
    real = config_file(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    assert invoke("alerts", "check", str(link)).code == 2


def test_an_unknown_run_exits_1_and_a_malformed_id_exits_2(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    recorded_run(store_root, [1.0])
    path = str(config_file(tmp_path))
    assert invoke("alerts", "check", path, "--project", "proj", "--run", "0" * 26).code == 1
    assert invoke("alerts", "check", path, "--project", "proj", "--run", "x").code == 2
    assert invoke("alerts", "check", path, "--run", "0" * 26).code == 2  # no project


def test_the_check_is_listed_in_the_help_of_the_alerts_group(invoke: Invoke) -> None:
    res = invoke("alerts", "--help")
    assert res.code == 0 and "check" in res.out
