"""The alert rules, evaluated with an injected clock: deterministic, bounded, isolated."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from typing import Any

import pytest
from fake_forwarders import LEAK_MARKER

from cvpilot.alerts import engine as eng
from cvpilot.alerts.config import AlertConfig, parse_alert_config
from cvpilot.compare.gate import GateResult, Violation
from cvpilot.core.errors import CompareRefusedError, EvalSetMismatchError
from cvpilot.core.schema import MAX_ALERT_TEXT, Alert

RUN = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Emitted:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, level: str, event: str, **fields: Any) -> None:
        self.rows.append((level, event, fields))

    def named(self, event: str) -> list[dict[str, Any]]:
        return [f for _, e, f in self.rows if e == event]


class Harness:
    def __init__(
        self,
        rules: list[dict[str, Any]],
        check_eval: Callable[[Any, Any], GateResult] | None = None,
        **config: Any,
    ) -> None:
        self.sent: list[Alert] = []
        self.clock = Clock()
        self.emitted = Emitted()
        self.engine = eng.AlertEngine(
            parse_alert_config({"rules": rules, **config}),
            project="proj",
            run_id=RUN,
            deliver=self.sent.append,
            emit=self.emitted,
            clock=self.clock,
            check_eval=check_eval,
        )


def make(rules: list[dict[str, Any]], **kw: Any) -> Harness:
    return Harness(rules, **kw)


# -- nan_loss -------------------------------------------------------------------------------


def test_nan_and_infinite_losses_fire_and_finite_or_other_metrics_do_not() -> None:
    h = make([{"rule": "nan_loss"}], rate_limit_s=1)
    h.engine.on_points({"train/loss": 0.5, "val/mAP": float("nan")}, 0)
    assert h.sent == []
    h.engine.on_points({"train/loss": float("nan")}, 1)
    assert [a.rule for a in h.sent] == ["nan_loss"] and h.sent[0].step == 1
    assert h.sent[0].text == f"[proj] Loss train/loss is NaN at step 1 (run {RUN})."
    h.clock.now += 5
    h.engine.on_points({"Loss": float("inf")}, 2)
    assert h.sent[-1].text == f"[proj] Loss Loss is +infinity at step 2 (run {RUN})."
    h.clock.now += 5
    h.engine.on_points({"x_loss": -math.inf}, 3)
    assert "-infinity" in h.sent[-1].text


def test_the_metric_option_watches_exactly_that_key() -> None:
    h = make([{"rule": "nan_loss", "metric": "a/b"}])
    h.engine.on_points({"train/loss": float("nan")}, 1)
    assert h.sent == []
    h.engine.on_points({"a/b": float("nan")}, 2)
    assert len(h.sent) == 1


# -- loss_spike -----------------------------------------------------------------------------


def test_a_spike_is_a_value_above_factor_times_the_recent_mean() -> None:
    h = make([{"rule": "loss_spike", "factor": 3, "window": 4}])
    for step, value in enumerate([1.0, 1.0, 1.0, 1.0]):
        h.engine.on_points({"loss": value}, step)
    h.engine.on_points({"loss": 3.0}, 4)  # exactly factor x mean: not a spike
    assert h.sent == []
    h.engine.on_points({"loss": 1.0}, 5)
    h.engine.on_points({"loss": 9.0}, 6)
    assert len(h.sent) == 1 and h.sent[0].rule == "loss_spike" and h.sent[0].step == 6
    assert "jumped to 9" in h.sent[0].text


def test_a_spike_needs_a_full_window_and_ignores_non_finite_values_in_it() -> None:
    h = make([{"rule": "loss_spike", "factor": 2, "window": 3}])
    h.engine.on_points({"loss": 1.0}, 0)
    h.engine.on_points({"loss": 100.0}, 1)  # only one value of history: no judgement yet
    assert h.sent == []
    h2 = make([{"rule": "loss_spike", "factor": 2, "window": 3}])
    for step, value in enumerate([1.0, float("nan"), 1.0, float("inf"), 1.0]):
        h2.engine.on_points({"loss": value}, step)
    h2.engine.on_points({"loss": 2.5}, 5)
    assert len(h2.sent) == 1  # the window held 1, 1, 1: NaN and inf never entered it


def test_a_zero_or_negative_mean_never_divides_into_a_spike() -> None:
    h = make([{"rule": "loss_spike", "factor": 2, "window": 2}])
    for step, value in enumerate([0.0, 0.0, 5.0, -1.0, -1.0, 4.0]):
        h.engine.on_points({"loss": value}, step)
    assert h.sent == []


# -- no_progress ----------------------------------------------------------------------------


def test_no_progress_fires_after_patience_values_without_a_new_best() -> None:
    h = make([{"rule": "no_progress", "patience": 3}])
    for step, value in enumerate([1.0, 0.9, 0.9, 0.95]):
        h.engine.on_points({"loss": value}, step)
    assert h.sent == []
    h.engine.on_points({"loss": 0.91}, 4)
    assert len(h.sent) == 1 and h.sent[0].rule == "no_progress" and h.sent[0].step == 4
    assert "has not improved for 3 logged values" in h.sent[0].text


def test_progress_resets_the_count_and_min_delta_decides_what_counts() -> None:
    h = make([{"rule": "no_progress", "patience": 3, "min_delta": 0.1}])
    for step, value in enumerate([1.0, 0.95, 0.93, 0.5, 0.49, 0.48]):
        h.engine.on_points({"loss": value}, step)
    assert h.sent == []  # 0.95 and 0.93 are not 0.1 better, but 0.5 resets the count
    h.engine.on_points({"loss": 0.47}, 6)
    assert len(h.sent) == 1


def test_after_firing_the_count_starts_again_so_the_next_alert_needs_patience_values() -> None:
    h = make([{"rule": "no_progress", "patience": 2}], rate_limit_s=1)
    values = [1.0, 2.0, 2.0, 3.0, 3.0]
    for step, value in enumerate(values):
        h.clock.now += 10
        h.engine.on_points({"loss": value}, step)
    assert [a.step for a in h.sent] == [2, 4]


def test_non_finite_values_do_not_count_as_progress_or_as_stalling() -> None:
    h = make([{"rule": "no_progress", "patience": 2}])
    for step, value in enumerate([1.0, float("nan"), float("nan"), float("nan")]):
        h.engine.on_points({"loss": value}, step)
    assert h.sent == []


# -- rate limit and storms ------------------------------------------------------------------


def test_one_alert_per_rule_and_run_per_window_then_again_after_it() -> None:
    h = make([{"rule": "nan_loss"}], rate_limit_s=60)
    for step in range(10):
        h.engine.on_points({"loss": float("nan")}, step)
    assert len(h.sent) == 1
    assert h.engine.stats.suppressed == 9
    h.clock.now += 59.9
    h.engine.on_points({"loss": float("nan")}, 10)
    assert len(h.sent) == 1
    h.clock.now += 0.2
    h.engine.on_points({"loss": float("nan")}, 11)
    assert len(h.sent) == 2


def test_the_first_suppression_of_a_rule_is_reported_once() -> None:
    h = make([{"rule": "nan_loss"}])
    for step in range(5):
        h.engine.on_points({"loss": float("nan")}, step)
    reports = h.emitted.named("alert_suppressed")
    assert len(reports) == 1 and reports[0]["rule"] == "nan_loss"


def test_a_storm_of_a_million_bad_points_sends_one_alert_in_bounded_time() -> None:
    h = make([{"rule": "nan_loss"}, {"rule": "loss_spike"}, {"rule": "no_progress", "patience": 5}])
    started = time.perf_counter()
    for step in range(200_000):
        h.engine.on_points({"loss": float("nan"), "other": 1.0}, step)
    elapsed = time.perf_counter() - started
    assert len(h.sent) == 1 and elapsed < 10


def test_rules_are_limited_independently() -> None:
    h = make([{"rule": "nan_loss"}, {"rule": "no_progress", "patience": 2}])
    for step, value in enumerate([1.0, 2.0, 3.0, float("nan")]):
        h.engine.on_points({"loss": value}, step)
    assert sorted(a.rule for a in h.sent) == ["nan_loss", "no_progress"]


def test_the_alerts_of_one_run_are_capped_but_the_end_of_a_run_always_gets_through() -> None:
    h = make(
        [{"rule": "nan_loss"}, {"rule": "no_progress", "patience": 2}, {"rule": "run_failed"}],
        rate_limit_s=1,
    )
    for step in range(10 * eng.MAX_ALERTS_PER_RUN):
        h.clock.now += 2
        h.engine.on_points({"loss": float("nan")}, step)
    assert len(h.sent) == eng.MAX_ALERTS_PER_RUN
    assert h.engine.stats.suppressed > 0
    h.engine.on_finish("failed")
    assert h.sent[-1].rule == "run_failed" and len(h.sent) == eng.MAX_ALERTS_PER_RUN + 1
    h.engine.on_finish("failed")  # once per run
    assert len(h.sent) == eng.MAX_ALERTS_PER_RUN + 1


# -- run_failed and run_preempted -----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "rule", "text"),
    [
        ("failed", "run_failed", f"[proj] Run ended as failed (run {RUN})."),
        ("preempted", "run_preempted", f"[proj] Run was preempted (run {RUN})."),
    ],
)
def test_the_end_of_a_run_fires_the_matching_rule(status: str, rule: str, text: str) -> None:
    h = make([{"rule": "run_failed"}, {"rule": "run_preempted"}])
    h.engine.on_finish(status)
    assert [(a.rule, a.text) for a in h.sent] == [(rule, text)]


def test_a_finished_run_and_an_unconfigured_rule_send_nothing() -> None:
    h = make([{"rule": "run_failed"}])
    h.engine.on_finish("finished")
    h.engine.on_finish("preempted")
    assert h.sent == []


# -- isolation: a rule that raises, a delivery that raises -----------------------------------


def test_a_rule_that_raises_is_reported_and_cannot_suppress_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make([{"rule": "loss_spike", "window": 2}, {"rule": "nan_loss"}])
    spike = next(r for r in h.engine._rules if r.name == "loss_spike")

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError(f"bug: Authorization: Bearer {LEAK_MARKER}")

    monkeypatch.setattr(spike, "feed", boom)
    h.engine.on_points({"loss": float("nan")}, 1)  # must not raise
    rules = sorted(a.rule for a in h.sent)
    assert rules == ["loss_spike", "nan_loss"]  # the broken rule is reported, nan_loss still fires
    broken = next(a for a in h.sent if a.rule == "loss_spike")
    assert "could not be evaluated" in broken.text and LEAK_MARKER not in broken.text
    failed = h.emitted.named("alert_rule_failed")
    assert len(failed) == 1 and failed[0]["rule"] == "loss_spike"
    assert failed[0]["error_type"] == "RuntimeError" and LEAK_MARKER not in str(failed)
    assert h.engine.stats.rule_failures == 1
    h.engine.on_points({"loss": 1.0}, 2)  # later calls still work and do not log again
    assert len(h.emitted.named("alert_rule_failed")) == 1


def test_a_failing_delivery_never_raises_and_never_stops_later_alerts() -> None:
    calls: list[Alert] = []

    def deliver(alert: Alert) -> None:
        calls.append(alert)
        raise OSError(f"https://hooks.invalid/{LEAK_MARKER}")

    config = parse_alert_config({"rules": [{"rule": "nan_loss"}, {"rule": "run_failed"}]})
    emitted = Emitted()
    engine = eng.AlertEngine(
        config, project="proj", run_id=RUN, deliver=deliver, emit=emitted, clock=Clock()
    )
    engine.on_points({"loss": float("nan")}, 1)
    engine.on_finish("failed")
    assert [a.rule for a in calls] == ["nan_loss", "run_failed"]
    assert engine.stats.delivery_errors == 2 and LEAK_MARKER not in str(emitted.rows)


def test_hostile_input_to_the_engine_is_counted_not_raised() -> None:
    h = make([{"rule": "nan_loss"}])
    h.engine.on_points({"loss": "x"}, 1)  # type: ignore[dict-item]
    h.engine.on_points({1: 2.0}, 1)  # type: ignore[dict-item]
    h.engine.on_points({"loss": None}, 1)  # type: ignore[dict-item]
    h.engine.on_points({"loss": float("nan")}, -5)
    h.engine.on_points({"loss": float("nan")}, "x")  # type: ignore[arg-type]
    h.engine.on_finish("exploded")
    assert h.sent == [] and h.engine.stats.skipped_points == 5 and h.engine.stats.queued == 0


# -- bounds ---------------------------------------------------------------------------------


def test_the_number_of_watched_keys_and_points_per_call_are_bounded() -> None:
    h = make([{"rule": "no_progress", "patience": 1000}])
    for k in range(eng.MAX_TRACKED_KEYS + 50):
        h.engine.on_points({f"loss_{k}": 1.0}, 0)
    tracked = sum(len(r.state) for r in h.engine._rules if hasattr(r, "state"))
    assert tracked <= eng.MAX_TRACKED_KEYS
    assert len(h.emitted.named("alert_keys_capped")) == 1
    h2 = make([{"rule": "nan_loss"}])
    many = {f"m{i}": 1.0 for i in range(eng.MAX_POINTS_PER_CALL + 10)}
    many["loss"] = float("nan")  # the 1011th key: past the bound
    h2.engine.on_points(many, 0)
    assert h2.sent == [] and len(h2.emitted.named("alert_points_capped")) == 1


# -- the text is constant template plus sanitised values -------------------------------------

HOSTILE_NAMES = [
    "loss\n<!channel> *bold* `code` [x](http://evil.example) @here",
    "loss‮evil​",
    "loss" + "\x1b[31m" + "red",
    "loss&amp;<script>alert(1)</script>",
    "loss" + "x" * 5000,
    "loss$(rm -rf /)`id`;|&",
]


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_values_in_the_message_are_reduced_to_plain_safe_characters(name: str) -> None:
    h = make([{"rule": "nan_loss"}])
    h.engine.on_points({name: float("nan")}, 3)
    (alert,) = h.sent
    text = alert.text
    assert len(text) <= MAX_ALERT_TEXT and text.isprintable() and text.isascii()
    assert text.startswith("[proj] Loss loss") and text.endswith(f"at step 3 (run {RUN}).")
    body = text[len("[proj] ") : -len(f" (run {RUN}).")]
    for bad in "<>&*`[]@\\\n$|;":
        assert bad not in body, (bad, text)


# -- eval_regression ------------------------------------------------------------------------


class FakeSnapshot:
    step = 40
    run_id = RUN
    eval_set_hash = "1" * 64


def failing(kind: str = "headline_drop", subject: str = "mAP50_95", drop: float | None = 2.5):
    result = GateResult(
        passed=False,
        violations=(Violation(kind, subject, "message with " + LEAK_MARKER, drop, 0.5),),
        n_checks=1,
    )
    return lambda spec, snapshot: result


EVAL_RULE = {"rule": "eval_regression", "baseline": "tag:main", "max_drop": {"mAP50_95": 0.5}}


def test_a_failed_gate_against_the_baseline_fires_with_the_worst_violation() -> None:
    h = make([EVAL_RULE], check_eval=failing())
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    (alert,) = h.sent
    assert alert.rule == "eval_regression" and alert.step == 40
    assert alert.text == (
        f"[proj] Evaluation at step 40 is worse than baseline tag:main: headline_drop on "
        f"mAP50_95 (2.5 points below) (run {RUN})."
    )
    assert LEAK_MARKER not in alert.text  # the violation's free-text message is never used


def test_a_passing_gate_sends_nothing() -> None:
    ok = GateResult(passed=True, violations=(), n_checks=1)
    h = make([EVAL_RULE], check_eval=lambda spec, snapshot: ok)
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    assert h.sent == []


def test_an_unverifiable_check_is_reported_as_an_alert_not_skipped() -> None:
    h = make([EVAL_RULE], check_eval=failing("headline_missing", "mAP50_95", None))
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    assert "headline_missing on mAP50_95" in h.sent[0].text
    assert "points below" not in h.sent[0].text


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (CompareRefusedError("x", reason="no_baseline_snapshot"), "no_baseline_snapshot"),
        (EvalSetMismatchError("x", reason="eval_set_hash"), "eval_set_hash"),
        (RuntimeError(LEAK_MARKER), "RuntimeError"),
        (KeyError("tag:main"), "KeyError"),
    ],
)
def test_a_comparison_that_cannot_be_made_is_an_alert_with_a_constant_reason(
    error: Exception, reason: str
) -> None:
    def refuse(spec: Any, snapshot: Any) -> Any:
        raise error

    h = make([EVAL_RULE], check_eval=refuse)
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    (alert,) = h.sent
    assert alert.text == (
        f"[proj] Evaluation at step 40 could not be compared with baseline tag:main "
        f"({reason}) (run {RUN})."
    )
    assert LEAK_MARKER not in alert.text


def test_a_hostile_class_name_in_a_violation_is_sanitised() -> None:
    h = make([EVAL_RULE], check_eval=failing("class_drop", "<!channel>\n*cat*‮", 9.0))
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    text = h.sent[0].text
    assert text.isascii() and text.isprintable() and "<" not in text and "*" not in text


def test_without_the_rule_a_snapshot_is_ignored_and_no_check_is_made() -> None:
    def never(spec: Any, snapshot: Any) -> Any:
        raise AssertionError("must not be called")

    h = make([{"rule": "nan_loss"}], check_eval=never)
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    assert h.sent == []


# -- determinism and construction ------------------------------------------------------------


def test_the_same_inputs_give_the_same_alerts() -> None:
    def run() -> list[str]:
        h = make([{"rule": "nan_loss"}, {"rule": "no_progress", "patience": 2}])
        for step, value in enumerate([1.0, 1.0, 1.0, float("nan")]):
            h.engine.on_points({"loss": value}, step)
        h.engine.on_finish("failed")
        return [a.text for a in h.sent]

    assert run() == run() and len(run()) == 2


def test_the_engine_events_are_logged_with_constant_names() -> None:
    h = make([{"rule": "nan_loss"}])
    h.engine.on_points({"loss": float("nan")}, 7)
    (fired,) = h.emitted.named("alert_fired")
    assert fired == {"rule": "nan_loss", "step": 7, "project": "proj"}
    assert h.emitted.rows[0][0] == "info"


def test_the_engine_accepts_only_a_validated_config() -> None:
    assert isinstance(parse_alert_config({"rules": [{"rule": "nan_loss"}]}), AlertConfig)
    deliver: Callable[[Alert], None] = lambda a: None  # noqa: E731
    with pytest.raises(ValueError):
        eng.AlertEngine(
            {"rules": []},  # type: ignore[arg-type]
            project="proj",
            run_id=RUN,
            deliver=deliver,
        )


# -- fix wave: links, failure keys, emit isolation, suppression reports, overflow, eval gating ----


@pytest.mark.parametrize(
    "name",
    ["loss http://evil.example/x", "loss evil.com/login", "loss www.evil.org", "loss ftp://h"],
)
def test_a_metric_name_cannot_carry_a_clickable_link(name: str) -> None:
    h = make([{"rule": "nan_loss"}])
    h.engine.on_points({name: float("nan")}, 1)
    body = h.sent[0].text
    assert "://" not in body and not re.search(r"\.[A-Za-z]", body.split(" (run ")[0]), body


def test_the_plain_name_of_an_ordinary_metric_is_still_readable() -> None:
    assert eng.plain("train/loss") == "train/loss"
    assert eng.plain("metrics/mAP50-95(B)") == "metrics/mAP50-95(B)"


def test_a_failed_rule_has_its_own_rate_key_and_does_not_use_up_the_real_alerts_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make([{"rule": "loss_spike", "factor": 2, "window": 2}], rate_limit_s=60)
    spike = h.engine._rules[0]
    real = spike.feed

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(spike, "feed", boom)
    h.engine.on_points({"loss": 1.0}, 0)
    assert "could not be evaluated" in h.sent[0].text
    monkeypatch.setattr(spike, "feed", real)  # the bug is fixed; the real rule works at once
    for step, value in enumerate([1.0, 1.0, 9.0], start=1):
        h.engine.on_points({"loss": value}, step)
    assert [a.rule for a in h.sent] == ["loss_spike", "loss_spike"]
    assert "jumped to" in h.sent[1].text


def test_an_emit_function_that_raises_cannot_break_the_engine() -> None:
    sent: list[Alert] = []

    def emit(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("logging is broken")

    engine = eng.AlertEngine(
        parse_alert_config({"rules": [{"rule": "nan_loss"}, {"rule": "run_failed"}]}),
        project="proj", run_id=RUN, deliver=sent.append, emit=emit, clock=Clock(),
    )  # fmt: skip
    engine.on_points({"loss": float("nan")}, 1)
    engine.on_points({"loss": float("nan")}, 2)  # also the suppression report
    engine.on_finish("failed")
    assert [a.rule for a in sent] == ["nan_loss", "run_failed"]


def test_every_kind_of_suppression_is_reported_once_including_the_run_cap() -> None:
    h = make([{"rule": "nan_loss"}, {"rule": "no_progress", "patience": 2}], rate_limit_s=1)
    for step in range(3):
        h.engine.on_points({"loss": float("nan")}, step)  # rate limited
    for step in range(10 * eng.MAX_ALERTS_PER_RUN):
        h.clock.now += 2
        h.engine.on_points({"loss": float("nan")}, 10 + step)  # then the per-run cap
    reasons = sorted((f["rule"], f["reason"]) for f in h.emitted.named("alert_suppressed"))
    assert reasons == [("nan_loss", "rate_limit"), ("nan_loss", "run_cap")]


def test_a_value_too_large_for_a_float_is_skipped_not_fatal() -> None:
    h = make([{"rule": "nan_loss"}])
    huge: dict[str, Any] = {"huge_loss": 10**400, "other_loss": float("nan")}
    h.engine.on_points(huge, 1)
    assert len(h.sent) == 1 and h.engine.stats.skipped_points == 1


def test_an_evaluation_inside_the_rate_limit_window_is_not_compared_again() -> None:
    calls: list[int] = []

    def check(spec: Any, snapshot: Any) -> GateResult:
        calls.append(snapshot.step)
        return GateResult(
            passed=False, violations=(Violation("headline_drop", "m", "x", 2.0, 0.5),)
        )

    h = make([EVAL_RULE], check_eval=check)
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    assert calls == [40] and len(h.sent) == 1 and h.engine.stats.suppressed == 1
    h.clock.now += 10_000
    h.engine.on_eval(FakeSnapshot())  # type: ignore[arg-type]
    assert calls == [40, 40] and len(h.sent) == 2


def test_the_summary_says_what_was_queued_suppressed_and_skipped() -> None:
    h = make([{"rule": "nan_loss"}])
    h.engine.on_points({"loss": float("nan")}, 1)
    h.engine.on_points({"loss": float("nan")}, 2)
    h.engine.on_points({"loss": "x"}, 3)  # type: ignore[dict-item]
    assert h.engine.summary() == {
        "queued": 1, "suppressed": 1, "rule_failures": 0, "submit_errors": 0, "skipped_points": 1,
    }  # fmt: skip
