"""Lightning hook: logger and callback against a fake trainer; isolation from the host."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pytest
import structlog
from fake_forwarders import Blocking, Recording, use_forwarders
from fake_frameworks import install_fake_lightning, make_trainer

from cvpilot import api, runs
from cvpilot.core.errors import HookNotFoundError, MissingExtraError
from cvpilot.hooks import lightning as lh
from cvpilot.hooks._guard import MAX_CONSECUTIVE_FAILURES, MAX_POINTS_PER_CALL
from cvpilot.logging import clear_request_context


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    install_fake_lightning(monkeypatch)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


class RecordingSink:
    status = "running"

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.logged, self.finished = fail, [], []
        self.run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    def log(self, points: dict[str, float], *, step: int, epoch: int | None = None) -> None:
        if self.fail:
            raise RuntimeError("store is full")
        self.logged.append((dict(points), step))

    def finish(self, status: str) -> None:
        self.finished.append(status)
        self.status = status


def attached(sink: Any = None, **kw: Any) -> tuple[Any, Any, Any, lh.LightningHook]:
    trainer = make_trainer(**kw)
    sink = sink or RecordingSink()
    hook = lh.LightningHook()
    hook.attach(trainer, sink)
    return trainer, trainer.callbacks[0], trainer.loggers[0], hook


def test_matches_only_a_trainer_of_the_framework() -> None:
    hook = lh.LightningHook()
    assert hook.matches(make_trainer()) and not hook.matches(object()) and not hook.matches("x")


def test_attach_adds_a_subclassed_callback_and_logger_and_detach_removes_them() -> None:
    from fake_frameworks import FRAMEWORK  # noqa: F401

    trainer, callback, logger, hook = attached()
    base_cb = type(callback).__mro__[1]
    base_lg = type(logger).__mro__[1]
    assert base_cb.__name__ == "Callback" and base_lg.__name__ == "Logger"
    assert logger.name == "cvpilot" and logger.experiment is not None and logger.version
    hook.detach()
    assert trainer.callbacks == [] and trainer.loggers == []


def test_scalars_flow_from_the_logger_in_step_order() -> None:
    sink = RecordingSink()
    _, callback, logger, _ = attached(sink)
    callback.on_fit_start(None, None)
    logger.log_metrics({"train/loss": 1.0, "epoch": 0.0}, 0)
    logger.log_metrics({"train/loss": 0.5, "val/AP": 0.3}, 10)
    assert sink.logged == [
        ({"train/loss": 1.0, "epoch": 0.0}, 0),
        ({"train/loss": 0.5, "val/AP": 0.3}, 10),
    ]


def test_repeats_at_one_step_and_older_steps_are_skipped_not_raised() -> None:
    sink = RecordingSink()
    _, _, logger, _ = attached(sink)
    logger.log_metrics({"a": 1.0}, 5)
    logger.log_metrics({"a": 2.0, "b": 3.0}, 5)  # a repeats, b is new
    logger.log_metrics({"a": 9.0}, 3)  # older
    assert sink.logged == [({"a": 1.0}, 5), ({"b": 3.0}, 5)]


def test_only_rank_zero_logs() -> None:
    sink = RecordingSink()
    trainer, callback, logger, _ = attached(sink, rank_zero=False)
    callback.on_fit_start(trainer, None)
    logger.log_metrics({"a": 1.0}, 1)
    callback.on_fit_end(trainer, None)
    assert sink.logged == [] and sink.finished == []


def test_the_last_stage_finishes_and_an_exception_marks_the_run_failed() -> None:
    sink = RecordingSink()
    trainer, callback, _, _ = attached(sink)
    callback.on_fit_end(trainer, None)
    assert sink.finished == []  # fit alone does not end the script: test may follow
    callback.on_teardown(trainer, None, "test")
    assert sink.finished == ["finished"]
    sink2 = RecordingSink()
    trainer, callback, _, _ = attached(sink2)
    callback.on_exception(trainer, None, ValueError("boom"))
    assert sink2.finished == ["failed"]
    callback.on_teardown(trainer, None, "test")  # already closed: ignored, no second finish
    assert sink2.finished == ["failed"]


def test_hostile_metrics_never_raise_and_are_counted() -> None:
    class Bad:
        def __float__(self) -> float:
            raise RuntimeError("tensor on a dead device")

    sink = RecordingSink()
    _, _, logger, _ = attached(sink)
    core = logger._core
    metrics: dict[Any, Any] = {
        "ok": 1.0,
        "nan": math.nan,
        "inf": math.inf,
        3: 1.0,
        None: 1.0,
        "text": "abc",
        "none": None,
        "bool": True,
        "bad": Bad(),
        "ctrl\x00key\n": 2.0,
        "k" * 5000: 3.0,
        "": 1.0,
        "   ": 1.0,
    }
    logger.log_metrics(metrics, 1)
    ((points, step),) = sink.logged
    assert (
        step == 1
        and points["ok"] == 1.0
        and math.isnan(points["nan"])
        and points["inf"] == math.inf
    )
    assert "ctrl_key_" in points and max(len(k) for k in points) <= 256
    assert not core.guard.failures and core.guard.skipped >= 7


@pytest.mark.parametrize("step", [None, -1, True, "x", 1.5, math.nan, math.inf])
def test_bad_steps_are_skipped_without_raising(step: Any) -> None:
    sink = RecordingSink()
    _, _, logger, _ = attached(sink)
    logger.log_metrics({"a": 1.0}, step)
    assert sink.logged == []


def test_points_per_call_are_bounded() -> None:
    sink = RecordingSink()
    _, _, logger, _ = attached(sink)
    logger.log_metrics({f"k{i}": 1.0 for i in range(MAX_POINTS_PER_CALL + 50)}, 1)
    assert len(sink.logged[0][0]) == MAX_POINTS_PER_CALL and logger._core.guard.skipped == 50


def test_a_failing_sink_never_raises_into_the_host_and_is_logged_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    configure_logging("cvpilot-client", Settings(env="prod"))
    sink = RecordingSink(fail=True)
    trainer, callback, logger, _ = attached(sink)
    for step in range(MAX_CONSECUTIVE_FAILURES + 5):
        logger.log_metrics({"a": 1.0}, step)  # must not raise
    callback.on_fit_end(trainer, None)  # a disabled hook ignores every later callback
    core = logger._core
    assert core.guard.disabled and core.guard.failures >= MAX_CONSECUTIVE_FAILURES
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    failed = [e for e in events if e["message"] == "hook_failed"]
    assert (
        len(failed) == 1
        and failed[0]["hook"] == "lightning"
        and failed[0]["error_type"] == "RuntimeError"
    )
    assert "store is full" not in json.dumps(events)
    assert len([e for e in events if e["message"] == "hook_disabled"]) == 1


def test_a_recovering_sink_resets_the_consecutive_counter() -> None:
    class Flaky(RecordingSink):
        def log(self, points: dict[str, float], *, step: int, epoch: int | None = None) -> None:
            if step % 2 == 0:
                raise RuntimeError("flaky")
            super().log(points, step=step, epoch=epoch)

    sink = Flaky()
    _, _, logger, _ = attached(sink)
    for step in range(1, 100):
        logger.log_metrics({"a": 1.0}, step)
    assert not logger._core.guard.disabled and len(sink.logged) > 40


def test_keyboard_interrupt_and_system_exit_are_never_swallowed() -> None:
    class Interrupting(RecordingSink):
        def log(self, points: dict[str, float], *, step: int, epoch: int | None = None) -> None:
            raise KeyboardInterrupt

    _, _, logger, _ = attached(Interrupting())
    with pytest.raises(KeyboardInterrupt):
        logger.log_metrics({"a": 1.0}, 1)


def test_attach_to_something_that_is_not_a_trainer_is_an_error() -> None:
    from cvpilot.hooks.lightning import HookAttachError

    class Odd:
        callbacks = None
        loggers: list[Any] = []

    with pytest.raises(HookAttachError):
        lh.LightningHook().attach(Odd(), RecordingSink())
    with pytest.raises(HookAttachError):
        lh.LightningHook().attach(object(), RecordingSink())


def test_without_the_framework_attach_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    for name in lh.PACKAGES:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[lightning\]"):
        lh.LightningHook().attach(make_trainer(), RecordingSink())


def test_autolog_end_to_end_logs_scalars_finishes_and_announces_the_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda dist: "2.6.6")
    trainer = make_trainer()
    run = api.autolog(trainer, project="proj", task="detection")
    callback, logger = trainer.callbacks[0], trainer.loggers[0]
    callback.on_fit_start(trainer, None)
    for step in range(5):
        logger.log_metrics({"train/loss": 1.0 / (step + 1)}, step)
    callback.on_fit_end(trainer, None)
    callback.on_teardown(trainer, None, "fit")
    callback._core.finish_at_exit()  # a fit-only script is finished when the process exits
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.stats.n_points == 5
    assert view.meta.framework.startswith("lightning==2.6.6")
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    attached_ev = [e for e in events if e["message"] == "hook_attached"]
    assert (
        attached_ev
        and attached_ev[0]["hook"] == "lightning"
        and attached_ev[0]["license_notice"] is None
    )


def test_autolog_of_an_unknown_object_is_hook_not_found_and_no_run_starts(
    tmp_path: Path,
) -> None:
    with pytest.raises(HookNotFoundError, match="lightning"):
        api.autolog(object(), project="proj", task="detection")
    assert runs.list_runs("proj", store_dir=tmp_path / "store") == []
    assert api._active is None


def test_autolog_without_the_package_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as md

    def missing(dist: str) -> str:
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(md, "version", missing)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[lightning\]"):
        api.autolog(make_trainer(), project="proj", task="detection")
    assert api._active is None


def test_a_hung_tracker_never_delays_the_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda dist: "2.6.6")
    use_forwarders(monkeypatch, blocking=Blocking, recording=Recording)
    trainer = make_trainer()
    run = api.autolog(trainer, project="proj", task="detection", forward_to=["blocking"])
    logger = trainer.loggers[0]
    worst = 0.0
    for step in range(200):
        start = time.perf_counter()
        logger.log_metrics({"loss": float(step)}, step)
        run._handle.flush()
        worst = max(worst, time.perf_counter() - start)
    assert worst < 0.5
    Recording.instances[0].release.set()
    trainer.callbacks[0].on_teardown(trainer, None, "test")


def test_a_metric_key_longer_than_the_limit_stays_distinct_from_its_neighbours() -> None:
    from cvpilot.hooks._guard import MAX_KEY_CHARS, metric_key

    a, b = "k" * 300 + "a", "k" * 300 + "b"
    ka, kb = metric_key(a), metric_key(b)
    assert ka != kb and len(ka or "") <= MAX_KEY_CHARS and len(kb or "") <= MAX_KEY_CHARS
    assert metric_key(a) == ka  # deterministic
    assert metric_key("short") == "short"


def test_the_relay_remembers_a_bounded_number_of_keys() -> None:
    from cvpilot.hooks._guard import Guard
    from cvpilot.hooks._sink import MAX_TRACKED_KEYS, ScalarRelay

    class Sink:
        def log(self, points: dict[str, float], *, step: int) -> None: ...

    guard = Guard("t")
    relay = ScalarRelay(Sink(), guard)
    for step in range(1, 3):
        for chunk in range(0, MAX_TRACKED_KEYS + 500, 500):
            relay.log({f"k{chunk + i}": 1.0 for i in range(500)}, step)
    assert len(relay._last) <= MAX_TRACKED_KEYS and guard.skipped > 0
