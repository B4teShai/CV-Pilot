"""Forwarders wired through ``init(forward_to=...)`` and ``CVPILOT_FORWARD_TO``."""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import pytest
import structlog
from fake_forwarders import (
    LEAK_MARKER,
    Blocking,
    Failing,
    FailsToConstruct,
    NeedsSdk,
    Recording,
    use_forwarders,
)

from cvpilot import api, runs
from cvpilot.core.errors import MissingExtraError
from cvpilot.logging import clear_request_context
from cvpilot.registry import UnknownPluginError
from cvpilot.runs import forwarding
from cvpilot.settings import SettingsError, load_settings


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO", "CVPILOT_FORWARD_HOSTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


def test_points_reach_the_forwarder_losslessly_and_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_forwarders(monkeypatch, recording=Recording)
    values = [0.1 + 0.2, 1e-308, -0.0, 123456789.123456789, math.inf, -math.inf, math.nan]
    run = api.init("proj", task="detection", forward_to=["recording"])
    for step, value in enumerate(values):
        run.log({"val/AP/odd name": value, "loss": float(step)}, step=step)
    run.finish()
    (adapter,) = Recording.instances
    assert [c[0] for c in adapter.calls][0] == "open" and adapter.calls[-1] == ("close", "finished")
    got = {(k, s): v for k, s, v in adapter.logged}
    assert len(got) == 2 * len(values)
    for step, value in enumerate(values):
        sent = got[("val/AP/odd name", step)]
        assert (math.isnan(sent) and math.isnan(value)) or sent == value
        assert math.copysign(1.0, sent) == math.copysign(1.0, value)
        assert got[("loss", step)] == float(step)


def test_the_environment_variable_selects_forwarders_and_an_argument_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_forwarders(monkeypatch, recording=Recording, failing=Failing)
    monkeypatch.setenv("CVPILOT_FORWARD_TO", "recording")
    run = api.init("proj", task="detection")
    run.log({"a": 1.0}, step=0)
    run.finish()
    assert [type(i) for i in Recording.instances] == [Recording]
    use_forwarders(monkeypatch, recording=Recording, failing=Failing)
    run = api.init("proj", task="detection", forward_to=["failing"])
    run.finish()
    assert [type(i) for i in Recording.instances] == [Failing]
    use_forwarders(monkeypatch, recording=Recording)
    run = api.init("proj", task="detection", forward_to=[])
    run.finish()
    assert Recording.instances == []


@pytest.mark.parametrize("raw", ["Recording", "a,,b", "a b", "a,a", "x" * 33, "1abc", "a;rm", ",a"])
def test_invalid_forwarder_names_are_settings_errors(raw: str) -> None:
    with pytest.raises(SettingsError, match="CVPILOT_FORWARD_TO"):
        load_settings({"CVPILOT_FORWARD_TO": raw})


@pytest.mark.parametrize(
    "raw", ["*", "Host_One", "a b", "a,,b", ",a", "-x.example", "a" * 300, "https://x"]
)
def test_invalid_allow_listed_hosts_are_settings_errors(raw: str) -> None:
    with pytest.raises(SettingsError, match="CVPILOT_FORWARD_HOSTS"):
        load_settings({"CVPILOT_FORWARD_HOSTS": raw})


def test_forward_hosts_settings_parse_and_reach_the_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert load_settings({}).forward_hosts == ()
    parsed = load_settings({"CVPILOT_FORWARD_HOSTS": " Tracker.Example , api.wandb.ai "})
    assert parsed.forward_hosts == ("tracker.example", "api.wandb.ai")
    use_forwarders(monkeypatch, recording=Recording)
    monkeypatch.setenv("CVPILOT_FORWARD_HOSTS", "tracker.example")
    run = api.init("proj", task="detection", forward_to=["recording"])
    run.finish()
    assert Recording.instances[0].allowed_hosts == ("tracker.example",)


def test_forward_to_settings_parse() -> None:
    assert load_settings({}).forward_to == ()
    assert load_settings({"CVPILOT_FORWARD_TO": ""}).forward_to == ()
    assert load_settings({"CVPILOT_FORWARD_TO": " mlflow , wandb "}).forward_to == (
        "mlflow",
        "wandb",
    )
    with pytest.raises(SettingsError):
        load_settings({"CVPILOT_FORWARD_TO": ",".join(f"f{i}" for i in range(9))})


def test_unknown_forwarder_fails_before_any_run_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    use_forwarders(monkeypatch, recording=Recording)
    with pytest.raises(UnknownPluginError, match="nope"):
        api.init("proj", task="detection", forward_to=["nope"])
    assert runs.list_runs("proj", store_dir=Path.cwd() / "store") == []
    assert api._active is None


def test_a_missing_tracker_package_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    use_forwarders(monkeypatch, needssdk=NeedsSdk)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[needssdk\]"):
        api.init("proj", task="detection", forward_to=["needssdk"])
    assert importlib.metadata.version("pydantic")  # the check is real, not a stub


def test_a_plugin_that_fails_to_construct_is_skipped_and_training_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_forwarders(monkeypatch, recording=Recording, failsctor=FailsToConstruct)
    run = api.init("proj", task="detection", forward_to=["failsctor", "recording"])
    run.log({"a": 1.0}, step=0)
    run.finish()
    assert len(Recording.instances) == 1 and Recording.instances[0].logged == [("a", 0, 1.0)]
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    failed = [e for e in events if e["message"] == "forward_failed"]
    assert len(failed) == 1 and failed[0]["reason"] == "construct_failed"


def test_a_failing_tracker_never_fails_the_run_or_leaks_a_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_forwarders(monkeypatch, failing=Failing)
    run = api.init("proj", task="detection", forward_to=["failing"])
    for step in range(20):
        run.log({"loss": 1.0 / (step + 1)}, step=step)
    run.finish()
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.stats.n_points == 20
    assert LEAK_MARKER not in capsys.readouterr().err
    for path in (tmp_path / "store").rglob("*"):
        if path.is_file() and path.suffix in (".sqlite", ".json", ".parquet", ""):
            assert LEAK_MARKER.encode() not in path.read_bytes(), path


def test_a_hung_tracker_does_not_slow_log_and_finish_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forwarding, "CLOSE_TIMEOUT_S", 0.3)
    use_forwarders(monkeypatch, blocking=Blocking)
    run = api.init("proj", task="detection", forward_to=["blocking"])
    worst = 0.0
    for step in range(300):
        start = time.perf_counter()
        run.log({"loss": float(step)}, step=step)
        run._handle.flush()  # one batch per step: 300 batches against a hung adapter
        worst = max(worst, time.perf_counter() - start)
    assert worst < 0.5
    start = time.perf_counter()
    run.finish()
    assert time.perf_counter() - start < 2.0
    Recording.instances[0].release.set()


def test_the_dev_environment_is_passed_to_adapters_as_allow_insecure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_forwarders(monkeypatch, recording=Recording)
    run = api.init("proj", task="detection", forward_to=["recording"])
    run.finish()
    assert Recording.instances[0].allow_insecure is False
    monkeypatch.setenv("CVPILOT_ENV", "dev")
    use_forwarders(monkeypatch, recording=Recording)
    run = api.init("proj", task="detection", forward_to=["recording"])
    run.finish()
    assert Recording.instances[0].allow_insecure is True


def test_an_abandoned_run_closes_forwarders_with_the_running_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_forwarders(monkeypatch, recording=Recording)
    run = api.init("proj", task="detection", forward_to=["recording"])
    run.log({"a": 1.0}, step=0)
    run.abandon()
    assert Recording.instances[0].calls[-1] == ("close", "running")
