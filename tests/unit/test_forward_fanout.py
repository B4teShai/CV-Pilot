"""Fan-out isolation: a tracker can be down, slow or hostile and training is unaffected."""

from __future__ import annotations

import io
import json
import threading
import time
from typing import Any

import pytest
import structlog
from fake_forwarders import LEAK_MARKER, Blocking, Failing, FailsToOpen, Recording
from support import batch, meta_repro

from cvpilot.core.schema import RunMeta
from cvpilot.logging import configure_logging
from cvpilot.runs.forwarding import (
    CALL_TIMEOUT_S,
    CLOSE_TIMEOUT_S,
    QUEUE_BATCHES,
    Fanout,
)
from cvpilot.settings import Settings


@pytest.fixture
def logs() -> Any:
    structlog.reset_defaults()
    stream = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="prod"), stream)

    class Reader:
        def records(self) -> list[dict[str, Any]]:
            return [json.loads(x) for x in stream.getvalue().splitlines() if x.startswith("{")]

    yield Reader()
    structlog.reset_defaults()


def make(*adapters: Recording, **kw: Any) -> tuple[Fanout, RunMeta]:
    meta, repro = meta_repro()
    fan = Fanout(list(adapters), meta, repro, **kw)
    fan.start()
    return fan, meta


def test_documented_bounds_are_named_constants() -> None:
    assert QUEUE_BATCHES >= 1 and CALL_TIMEOUT_S > 0 and CLOSE_TIMEOUT_S > 0


def test_delivers_open_points_then_close_in_order() -> None:
    adapter = Recording()
    fan, meta = make(adapter)
    fan.submit(batch(meta, 1))
    fan.submit(batch(meta, 2))
    stats = fan.close("finished")
    assert [c[0] for c in adapter.calls] == ["open", "points", "points", "close"]
    assert adapter.calls[-1] == ("close", "finished")
    assert adapter.logged[0] == ("k0", 1, 1.0) and len(adapter.logged) == 6
    assert stats["recording"].failed_calls == 0 and stats["recording"].dropped_batches == 0


def test_a_failing_adapter_is_counted_logged_once_and_never_raises(logs: Any) -> None:
    bad, good = Failing(), Recording()
    fan, meta = make(bad, good)
    for step in range(5):
        fan.submit(batch(meta, step))  # must not raise
    stats = fan.close("finished")
    assert stats["failing"].failed_calls == 5
    assert len(bad.calls) >= 5  # retried on every batch
    assert len(good.logged) == 15  # the healthy adapter is unaffected
    failed = [r for r in logs.records() if r["message"] == "forward_failed"]
    assert len(failed) == 1
    assert failed[0]["forwarder"] == "failing" and failed[0]["level"] == "WARN"
    assert failed[0]["error_type"] == "RuntimeError" and failed[0]["run_id"] == meta.run_id


def test_exception_text_never_reaches_the_logs(
    logs: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    fan, meta = make(Failing(), FailsToOpen())
    fan.submit(batch(meta, 1))
    fan.close("failed")
    text = json.dumps(logs.records())
    assert LEAK_MARKER not in text and "tracker.invalid" not in text and "Bearer" not in text


def test_a_failed_open_disables_the_adapter_after_one_event(logs: Any) -> None:
    adapter = FailsToOpen()
    fan, meta = make(adapter)
    fan.submit(batch(meta, 1))
    stats = fan.close("finished")
    assert adapter.calls == []  # nothing is sent to an adapter that never opened
    assert stats["failsopen"].disabled_reason == "open_failed"
    failed = [r for r in logs.records() if r["message"] == "forward_failed"]
    assert len(failed) == 1 and failed[0]["reason"] == "open_failed"


def test_a_blocked_adapter_never_blocks_submit_and_the_queue_is_bounded(logs: Any) -> None:
    adapter = Blocking()
    fan, meta = make(adapter, queue_batches=4, close_timeout_s=0.3)
    start = time.perf_counter()
    for step in range(200):
        fan.submit(batch(meta, step))
    assert time.perf_counter() - start < 1.0  # 200 submits against a hung adapter
    stats = fan.close("finished")
    assert stats["blocking"].dropped_batches >= 195 - 1
    dropped = [r for r in logs.records() if r["message"] == "forward_failed"]
    assert len(dropped) == 1 and dropped[0]["reason"] == "queue_full"
    adapter.release.set()


def test_close_returns_within_its_bound_even_if_the_adapter_hangs(logs: Any) -> None:
    adapter = Blocking()
    fan, meta = make(adapter, close_timeout_s=0.3)
    fan.submit(batch(meta, 1))
    time.sleep(0.05)
    start = time.perf_counter()
    stats = fan.close("finished")
    assert time.perf_counter() - start < 1.0
    assert stats["blocking"].abandoned is True
    summary = [r for r in logs.records() if r["message"] == "forward_summary"]
    assert len(summary) == 1 and summary[0]["abandoned"] is True
    adapter.release.set()


def test_an_adapter_stuck_in_a_call_is_marked_stalled_and_stops_receiving(logs: Any) -> None:
    now = [0.0]
    adapter = Blocking()
    fan, meta = make(adapter, call_timeout_s=10.0, clock=lambda: now[0], close_timeout_s=0.2)
    fan.submit(batch(meta, 1))
    assert adapter.entered.wait(2)  # the worker is inside the call
    now[0] = 11.0
    fan.submit(batch(meta, 2))
    stats = fan.close("finished")
    assert stats["blocking"].disabled_reason == "stalled"
    failed = [r for r in logs.records() if r["message"] == "forward_failed"]
    assert len(failed) == 1 and failed[0]["reason"] == "stalled"
    adapter.release.set()


def test_submit_never_raises_even_for_bad_input_or_after_close() -> None:
    fan, meta = make(Recording())
    fan.submit(None)  # type: ignore[arg-type]
    fan.close("finished")
    fan.submit(batch(meta, 1))  # after close: ignored
    fan.close("finished")  # idempotent


def test_worker_logs_carry_the_required_context_fields(logs: Any) -> None:
    fan, meta = make(Failing())
    fan.submit(batch(meta, 1))
    fan.close("finished")
    record = next(r for r in logs.records() if r["message"] == "forward_failed")
    for field in ("@timestamp", "level", "message", "service", "env", "request_id", "logger"):
        assert field in record, field


def test_every_worker_is_a_daemon_thread_and_is_gone_after_close() -> None:
    before = {t.name for t in threading.enumerate()}
    fan, _ = make(Recording())
    new = [t for t in threading.enumerate() if t.name not in before]
    assert new and all(t.daemon for t in new)
    fan.close("finished")
    assert not [t for t in new if t.is_alive()]
