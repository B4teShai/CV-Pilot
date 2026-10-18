"""Alert delivery reuses the forwarder fan-out: bounded, isolated, counted, never blocking."""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest
import structlog
from fake_forwarders import LEAK_MARKER
from support import meta_repro

from cvpilot.core.schema import Alert, Repro, RunMeta, RunStatus
from cvpilot.logging import configure_logging
from cvpilot.runs import forwarding as fw
from cvpilot.settings import Settings

RUN = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def alert(n: int = 1) -> Alert:
    return Alert(rule="nan_loss", project="proj", run_id=RUN, step=n, text=f"alert {n}")


class Chan:
    name = "chan"
    requires: tuple[str, ...] = ()

    def __init__(self, *, allow_insecure: bool = False, allowed_hosts: Sequence[str] = ()) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()

    def open(self, meta: RunMeta, repro: Repro) -> None:
        self.calls.append(("open", meta.run_id))

    def send(self, alerts: Sequence[Alert]) -> None:
        self.entered.set()
        self.release.wait(30)
        self.calls.append(("send", [a.text for a in alerts]))

    def close(self, status: RunStatus) -> None:
        self.calls.append(("close", status))


class Failing(Chan):
    name = "failing"

    def send(self, alerts: Sequence[Alert]) -> None:
        raise OSError(f"https://hooks.invalid/{LEAK_MARKER}")


@pytest.fixture
def logs() -> Any:
    structlog.reset_defaults()
    stream = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="prod"), stream)

    class Reader:
        def records(self) -> list[dict[str, Any]]:
            return [json.loads(x) for x in stream.getvalue().splitlines() if x.startswith("{")]

        text = staticmethod(lambda: stream.getvalue())

    yield Reader()
    structlog.reset_defaults()


def make(*adapters: Chan, **kw: Any) -> fw.Fanout:
    meta, repro = meta_repro()
    fan = fw.notify_fanout(list(adapters), meta, repro, **kw)
    fan.start()
    return fan


def test_alerts_are_delivered_in_order_through_the_channel_method() -> None:
    chan = Chan()
    fan = make(chan)
    fan.submit([alert(1)])
    fan.submit([alert(2)])
    stats = fan.close("failed")
    assert [c[0] for c in chan.calls] == ["open", "send", "send", "close"]
    assert chan.calls[1] == ("send", ["alert 1"]) and chan.calls[-1] == ("close", "failed")
    assert stats["chan"].failed_calls == 0 and stats["chan"].dropped_batches == 0


def test_the_notify_bounds_are_small_and_named() -> None:
    assert fw.NOTIFY_QUEUE <= 32 and fw.NOTIFY_CALL_TIMEOUT_S <= 30
    fan = make(Chan())
    assert fan.queue_batches == fw.NOTIFY_QUEUE and fan.max_queued_points == fw.NOTIFY_QUEUE
    fan.close("finished")


def test_a_failing_channel_is_counted_and_logged_once_by_type_with_alert_events(
    logs: Any,
) -> None:
    bad, good = Failing(), Chan()
    fan = make(bad, good)
    for n in range(5):
        fan.submit([alert(n)])
    stats = fan.close("finished")
    assert stats["failing"].failed_calls == 5 and len(good.calls) == 1 + 5 + 1
    failed = [r for r in logs.records() if r["message"] == "alert_delivery_failed"]
    assert len(failed) == 1 and failed[0]["channel"] == "failing"
    assert failed[0]["error_type"] == "OSError" and failed[0]["n_alerts"] == 1
    assert failed[0]["logger"] == "cvpilot.notify" and failed[0]["level"] == "WARN"
    summary = [r for r in logs.records() if r["message"] == "alert_delivery_summary"]
    assert len(summary) == 1 and summary[0]["failed_alerts"] == 5
    assert not any(r["message"] in ("forward_failed", "forward_summary") for r in logs.records())
    assert LEAK_MARKER not in logs.text()


def test_a_channel_that_never_answers_stalls_only_itself_and_submit_never_waits() -> None:
    slow, good = Chan(), Chan()
    slow.release.clear()
    fan = make(slow, good, call_timeout_s=0.2, close_timeout_s=0.3)
    started = time.perf_counter()
    fan.submit([alert(1)])
    assert slow.entered.wait(5)
    time.sleep(0.3)
    for n in range(2, 60):
        fan.submit([alert(n)])  # a full queue and a stalled channel both drop, never block
    assert time.perf_counter() - started < 1.0
    stats = fan.close("finished")
    slow.release.set()
    assert stats["chan"].disabled_reason == "stalled" or stats["chan"].dropped_batches > 0
    assert len(good.calls) >= 3


def test_a_full_queue_drops_alerts_instead_of_growing() -> None:
    slow = Chan()
    slow.release.clear()
    fan = make(slow, close_timeout_s=0.2)
    fan.submit([alert(0)])
    assert slow.entered.wait(5)
    for n in range(1, 4 * fw.NOTIFY_QUEUE):
        fan.submit([alert(n)])
    stats = fan.close("finished")
    slow.release.set()
    assert stats["chan"].dropped_batches >= 3 * fw.NOTIFY_QUEUE - 2


def test_the_forwarder_defaults_and_events_are_unchanged() -> None:
    assert fw.FORWARD.method == "log_points" and fw.FORWARD.failed_event == "forward_failed"
    assert fw.FORWARD.unit == "points" and fw.FORWARD.label == "forwarder"
    meta, repro = meta_repro()
    assert fw.Fanout([], meta, repro).delivery is fw.FORWARD


def test_channels_resolve_through_their_own_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import EntryPoint

    from cvpilot import registry
    from cvpilot.core.errors import CvpilotError

    points = {"chan": EntryPoint("chan", "test_notify_fanout:Chan", "cvpilot.channels")}
    monkeypatch.setattr(
        registry.channels,
        "_lookup",
        lambda *, group: list(points.values()) if group == "cvpilot.channels" else [],
    )
    monkeypatch.setattr(registry.channels, "_loaded", {})
    (chan,) = fw.resolve_channels(["chan", "chan"])
    assert isinstance(chan, Chan)
    with pytest.raises(CvpilotError, match="cvpilot.channels"):
        fw.resolve_channels(["nope"])


def test_a_stalled_channel_recovers_when_the_call_finishes_and_later_alerts_still_go_out() -> None:
    slow, good = Chan(), Chan()
    slow.release.clear()
    fan = make(slow, good, call_timeout_s=0.2, close_timeout_s=3)
    fan.submit([alert(1)])
    assert slow.entered.wait(5)
    time.sleep(0.35)  # the call has now run longer than the stall limit
    fan.submit([alert(2)])  # dropped and counted: the channel is stuck right now
    slow.release.set()  # the call returns
    deadline = time.time() + 5
    while time.time() < deadline and ("send", ["alert 1"]) not in slow.calls:
        time.sleep(0.02)
    time.sleep(0.2)
    fan.submit([alert(3)])  # a later alert, for example the end-of-run one, is not lost
    stats = fan.close("failed")
    sent = [rows for kind, rows in slow.calls if kind == "send"]
    assert ["alert 1"] in sent and ["alert 3"] in sent and ["alert 2"] not in sent
    assert stats["chan"].dropped_batches == 1 and stats["chan"].disabled_reason is None


def test_forwarders_keep_their_permanent_stall_rule() -> None:
    assert fw.FORWARD.recover is False and fw.NOTIFY.recover is True


def test_delivered_batches_are_counted_separately_from_queued_ones() -> None:
    chan, bad = Chan(), Failing()
    fan = make(chan, bad)
    for n in range(3):
        fan.submit([alert(n)])
    stats = fan.close("finished")
    assert stats["chan"].sent_batches == 3 and stats["failing"].sent_batches == 0
    assert stats["failing"].failed_calls == 3


def test_a_channel_that_fails_to_construct_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import EntryPoint

    from cvpilot import registry
    from cvpilot.core.errors import CvpilotError

    class Broken:
        name = "broken"
        requires: tuple[str, ...] = ()

        def __init__(self, **_kw: Any) -> None:
            raise RuntimeError("plugin bug https://hooks.invalid/" + LEAK_MARKER)

    globals()["Broken"] = Broken
    points = {"broken": EntryPoint("broken", "test_notify_fanout:Broken", "cvpilot.channels")}
    monkeypatch.setattr(
        registry.channels, "_lookup",
        lambda *, group: list(points.values()) if group == "cvpilot.channels" else [],
    )  # fmt: skip
    monkeypatch.setattr(registry.channels, "_loaded", {})
    with pytest.raises(CvpilotError) as info:
        fw.resolve_channels(["broken"])
    assert "broken" in str(info.value) and LEAK_MARKER not in str(info.value)
