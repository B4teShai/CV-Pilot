"""The alert rules and the engine that runs them inside the training process.

The engine sees what a run already produces: the scalars of every ``log`` call, the evaluation
snapshots of ``log_eval`` and the final status. It is deterministic (the clock is injected),
evaluates in time proportional to the points of one call, and keeps bounded state: at most
``MAX_TRACKED_KEYS`` watched metric keys per rule, ``MAX_POINTS_PER_CALL`` points looked at per
call, ``MAX_ALERTS_PER_RUN`` alerts per run (the end-of-run alerts are exempt).

Rules (the names of ``cvpilot.core.schema.AlertRule``):

* ``nan_loss``: a loss value that is NaN or infinite.
* ``loss_spike``: a loss above ``factor`` times the mean of the previous ``window`` finite values.
* ``no_progress``: ``patience`` finite values in a row without a new lowest loss (a plateau; a
  hung process that logs nothing cannot be seen from inside the process and is not detected).
* ``eval_regression``: an evaluation snapshot that fails the gate against a pinned baseline, or
  cannot be compared with it. The comparison is ``cvpilot.compare.service.run_gate``, so the
  verdict is the one ``cvpilot gate`` would give.
* ``run_failed`` and ``run_preempted``: the run ended that way.

A loss metric is the exact ``metric`` of the rule, or by default any key containing ``loss``.

Fail safe: a rule that raises is reported, not skipped. The first failure per rule is logged as
``alert_rule_failed`` (the exception *type* only), every failure is counted, and an alert says
that the rule could not be evaluated (subject to the same rate limit). A broken rule never stops
another rule, and nothing here raises into the training loop. A comparison that cannot be made
(no baseline snapshot, a different evaluation set, a failed integrity check) is an alert with a
constant reason, never silence.

Rate limit: one alert per rule and run in ``rate_limit_s`` seconds; a second occurrence inside the
window is counted (``AlertStats.suppressed``), and the first suppression of a rule is logged once
(``alert_suppressed``).

Messages are constant templates with sanitised values: a value is cut to its length limit and
reduced to ``[A-Za-z0-9 _./:()+%=-]`` (anything else becomes ``?``), so a metric key or class name
from data cannot carry markup, a mention, a link, a control character or a shell metacharacter
into Slack, Telegram or a log. Events (constant names): ``alert_fired``, ``alert_suppressed``,
``alert_rule_failed``, ``alert_keys_capped``, ``alert_points_capped``, ``alert_delivery_failed``.
"""

from __future__ import annotations

import itertools
import math
import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cvpilot.alerts.config import (
    AlertConfig,
    EvalRegressionSpec,
    LossSpikeSpec,
    NanLossSpec,
    NoProgressSpec,
)
from cvpilot.core.schema import MAX_ALERT_TEXT, MAX_STEP, Alert, EvalSnapshot
from cvpilot.core.text import safe_text
from cvpilot.logging import get_logger

if TYPE_CHECKING:
    from cvpilot.compare.gate import GateResult

MAX_ALERTS_PER_RUN = 20
"""Most alerts one run sends (end-of-run alerts excepted): a storm that outlasts every rate
limit window is still bounded."""
MAX_TRACKED_KEYS = 16
"""Most metric keys one rule follows; further keys are ignored (reported once)."""
MAX_POINTS_PER_CALL = 1000
"""Most points of one ``log`` call the engine looks at (the hooks' own limit)."""
MAX_VALUE_CHARS = 64
_TERMINAL = {"failed": "run_failed", "preempted": "run_preempted"}
_UNSAFE = re.compile(r"[^A-Za-z0-9 _./:()+%=-]")
_REASON = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,39}")
_KIND = re.compile(r"[a-z][a-z0-9_]{0,39}")

Emit = Callable[..., None]
EvalCheck = Callable[[EvalRegressionSpec, EvalSnapshot], "GateResult"]


def plain(value: object, limit: int = MAX_VALUE_CHARS) -> str:
    """``value`` as text made only of ``[A-Za-z0-9 _./:()+%=-]`` and ``?``, cut to ``limit``."""
    return _UNSAFE.sub("?", safe_text(value, limit))


def _number(value: float) -> str:
    return f"{value:.4g}"


def _is_loss(spec_metric: str | None, key: str) -> bool:
    return key == spec_metric if spec_metric is not None else "loss" in key.lower()


@dataclass
class AlertStats:
    fired: int = 0
    """Alerts handed to the delivery function."""
    suppressed: int = 0
    """Alerts dropped by the rate limit or the per-run cap."""
    rule_failures: int = 0
    delivery_errors: int = 0
    skipped_points: int = 0
    """Points the engine could not read as a finite or non-finite number under a text key."""


@dataclass(frozen=True)
class Finding:
    step: int
    message: str


class _Rule:
    name = ""
    capped = False

    def feed(self, points: Mapping[str, float], step: int) -> Finding | None:
        raise NotImplementedError


class _NanLoss(_Rule):
    name = "nan_loss"

    def __init__(self, spec: NanLossSpec) -> None:
        self._metric = spec.metric

    def feed(self, points: Mapping[str, float], step: int) -> Finding | None:
        for key, value in points.items():
            if not math.isfinite(value) and _is_loss(self._metric, key):
                kind = "NaN" if math.isnan(value) else ("+infinity" if value > 0 else "-infinity")
                return Finding(step, f"Loss {plain(key)} is {kind} at step {step}")
        return None


class _KeyedRule(_Rule):
    """A rule with a small state per watched key."""

    def __init__(self, metric: str | None) -> None:
        self._metric = metric
        self.state: dict[str, Any] = {}

    def _slot(self, key: str, make: Callable[[], Any]) -> Any:
        found = self.state.get(key)
        if found is None:
            if len(self.state) >= MAX_TRACKED_KEYS:
                self.capped = True
                return None
            found = self.state[key] = make()
        return found


class _LossSpike(_KeyedRule):
    name = "loss_spike"

    def __init__(self, spec: LossSpikeSpec) -> None:
        super().__init__(spec.metric)
        self._factor, self._window = spec.factor, spec.window

    def feed(self, points: Mapping[str, float], step: int) -> Finding | None:
        found: Finding | None = None
        for key, value in points.items():
            if not math.isfinite(value) or not _is_loss(self._metric, key):
                continue
            recent = self._slot(key, lambda: deque(maxlen=self._window))
            if recent is None:
                continue
            if found is None and len(recent) == self._window:
                mean = math.fsum(recent) / self._window
                if mean > 0 and value > self._factor * mean:
                    found = Finding(
                        step,
                        f"Loss {plain(key)} jumped to {_number(value)} at step {step}, "
                        f"{_number(value / mean)}x the recent average ({_number(mean)})",
                    )
            recent.append(value)
        return found


class _NoProgress(_KeyedRule):
    name = "no_progress"

    def __init__(self, spec: NoProgressSpec) -> None:
        super().__init__(spec.metric)
        self._patience, self._min_delta = spec.patience, spec.min_delta

    def feed(self, points: Mapping[str, float], step: int) -> Finding | None:
        found: Finding | None = None
        for key, value in points.items():
            if not math.isfinite(value) or not _is_loss(self._metric, key):
                continue
            state = self._slot(key, lambda: [math.inf, 0])  # [best, values since the best]
            if state is None:
                continue
            if value < state[0] - self._min_delta:
                state[0], state[1] = value, 0
                continue
            state[1] += 1
            if state[1] >= self._patience:
                if found is None:
                    found = Finding(
                        step,
                        f"Loss {plain(key)} has not improved for {state[1]} logged values "
                        f"(best {_number(state[0])}, step {step})",
                    )
                state[1] = 0
        return found


def gate_against_baseline(
    spec: EvalRegressionSpec, snapshot: EvalSnapshot, *, project: str, store_dir: Path | None
) -> GateResult:
    """The gate verdict of ``snapshot`` against the baseline of ``spec`` (read from the store).

    Raises:
        CvpilotError: the comparison was refused or a verification failed (see
            ``cvpilot.compare.service.resolve``).
    """
    from cvpilot.compare.service import run_gate  # compare imports runs: resolved at call time

    outcome = run_gate(
        project,
        snapshot.run_id,
        spec.baseline,
        spec.thresholds(),
        eval_set=snapshot.eval_set_hash,
        step=snapshot.step,
        store_dir=store_dir,
    )
    return outcome.result


class AlertEngine:
    """Evaluates the configured rules for one run and hands alerts to ``deliver``.

    ``deliver`` is called on the caller's thread and must not block (the run wires it to a
    bounded queue); it may raise, which is counted. ``emit(level, event, **fields)`` writes the
    engine's events; by default they go to the ``cvpilot.alerts`` logger.
    """

    def __init__(
        self,
        config: AlertConfig,
        *,
        project: str,
        run_id: str,
        deliver: Callable[[Alert], None],
        emit: Emit | None = None,
        clock: Callable[[], float] = time.monotonic,
        store_dir: Path | None = None,
        check_eval: EvalCheck | None = None,
    ) -> None:
        if not isinstance(config, AlertConfig):
            raise ValueError("config must be an AlertConfig (use parse_alert_config)")
        self._config = config
        self._project, self._run_id = project, run_id
        self._deliver = deliver
        self._emit = emit or self._log
        self._clock = clock
        self._check_eval: EvalCheck = check_eval or (
            lambda spec, snapshot: gate_against_baseline(
                spec, snapshot, project=project, store_dir=store_dir
            )
        )
        self.stats = AlertStats()
        self._rules: list[_Rule] = []
        for spec in config.rules:
            if isinstance(spec, NanLossSpec):
                self._rules.append(_NanLoss(spec))
            elif isinstance(spec, LossSpikeSpec):
                self._rules.append(_LossSpike(spec))
            elif isinstance(spec, NoProgressSpec):
                self._rules.append(_NoProgress(spec))
        self._last_sent: dict[str, float] = {}
        self._once: set[str] = set()
        self._n_sent = 0

    # -- plumbing -------------------------------------------------------------------------

    def _log(self, level: str, event: str, **fields: Any) -> None:
        getattr(get_logger("cvpilot.alerts"), level)(event, run_id=self._run_id, **fields)

    def _report_once(self, key: str, level: str, event: str, **fields: Any) -> None:
        if key not in self._once:
            self._once.add(key)
            self._emit(level, event, **fields)

    def _admit(self, rule: str, *, terminal: bool) -> bool:
        now = self._clock()
        last = self._last_sent.get(rule)
        window = self._config.rate_limit_s
        limited = last is not None and now - last < window
        capped = not terminal and self._n_sent >= MAX_ALERTS_PER_RUN
        if limited or capped:
            self.stats.suppressed += 1
            self._report_once(
                f"suppressed:{rule}",
                "info",
                "alert_suppressed",
                rule=rule,
                reason="run_cap" if capped and not limited else "rate_limit",
            )
            return False
        self._last_sent[rule] = now
        self._n_sent += 1
        return True

    def _fire(self, rule: str, step: int | None, message: str, *, terminal: bool = False) -> None:
        if not self._admit(rule, terminal=terminal):
            return
        text = f"[{self._project}] {message} (run {self._run_id})."
        try:
            alert = Alert(
                rule=rule,  # type: ignore[arg-type]  # rule names are the AlertRule literals
                project=self._project,
                run_id=self._run_id,
                step=step,
                text=text[:MAX_ALERT_TEXT],
            )
            self._emit("info", "alert_fired", rule=rule, step=step, project=self._project)
            self._deliver(alert)
            self.stats.fired += 1
        except Exception as exc:  # noqa: BLE001  # delivery must never reach the training loop
            self.stats.delivery_errors += 1
            self._report_once(
                f"delivery:{rule}",
                "warning",
                "alert_delivery_failed",
                rule=rule,
                reason="submit_failed",
                error_type=type(exc).__name__,
            )

    def _rule_failed(
        self, rule: str, exc: BaseException, step: int | None, *, alert: bool = True
    ) -> None:
        self.stats.rule_failures += 1
        self._report_once(
            f"failed:{rule}", "warning", "alert_rule_failed", rule=rule,
            error_type=type(exc).__name__,
        )  # fmt: skip
        if alert:
            reason = type(exc).__name__ if _REASON.fullmatch(type(exc).__name__) else "error"
            self._fire(rule, step, f"Alert rule {rule} could not be evaluated ({reason})")

    # -- events ---------------------------------------------------------------------------

    def on_points(self, points: Mapping[str, float], step: int) -> None:
        """Evaluate the metric rules on the scalars of one ``log`` call. Never raises."""
        try:
            self._on_points(points, step)
        except Exception as exc:  # noqa: BLE001  # the training loop must never see this
            self._rule_failed("engine", exc, None, alert=False)

    def _on_points(self, points: Mapping[str, float], step: int) -> None:
        if not self._rules:
            return
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= MAX_STEP:
            self.stats.skipped_points += len(points)
            return
        clean: dict[str, float] = {}
        for key, value in itertools.islice(points.items(), MAX_POINTS_PER_CALL):
            if (
                isinstance(key, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                clean[key] = float(value)
            else:
                self.stats.skipped_points += 1
        if len(points) > MAX_POINTS_PER_CALL:
            self._report_once("points", "warning", "alert_points_capped", limit=MAX_POINTS_PER_CALL)
        for rule in self._rules:
            try:
                found = rule.feed(clean, step)
            except Exception as exc:  # noqa: BLE001  # a rule that raises is reported, not skipped
                self._rule_failed(rule.name, exc, step)
                continue
            if found is not None:
                self._fire(rule.name, found.step, found.message)
            if rule.capped:
                rule.capped = False
                self._report_once("keys", "warning", "alert_keys_capped", limit=MAX_TRACKED_KEYS)

    def on_eval(self, snapshot: EvalSnapshot) -> None:
        """Run the ``eval_regression`` rule on a stored snapshot. Never raises."""
        spec = self._config.rule("eval_regression")
        if spec is None:
            return
        step = getattr(snapshot, "step", None)
        try:
            result = self._check_eval(spec, snapshot)
        except Exception as exc:  # noqa: BLE001  # a comparison that cannot be made is an alert
            reason = getattr(exc, "reason", None)
            if not (isinstance(reason, str) and _REASON.fullmatch(reason)):
                reason = type(exc).__name__ if _REASON.fullmatch(type(exc).__name__) else "error"
            if not hasattr(exc, "reason"):
                self.stats.rule_failures += 1
                self._report_once(
                    "failed:eval_regression", "warning", "alert_rule_failed",
                    rule="eval_regression", error_type=type(exc).__name__,
                )  # fmt: skip
            self._fire(
                "eval_regression",
                step,
                f"Evaluation at step {step} could not be compared with baseline "
                f"{spec.baseline} ({reason})",
            )
            return
        if result.passed:
            return
        worst = result.violations[0] if result.violations else None
        what = "unverifiable"
        if worst is not None:
            kind = worst.kind if _KIND.fullmatch(worst.kind) else "violation"
            what = f"{kind} on {plain(worst.subject, 40)}"
            if worst.drop_points is not None and math.isfinite(worst.drop_points):
                what += f" ({worst.drop_points:.1f} points below)"
        self._fire(
            "eval_regression",
            step,
            f"Evaluation at step {step} is worse than baseline {spec.baseline}: {what}",
        )

    def on_finish(self, status: str) -> None:
        """Run ``run_failed`` / ``run_preempted`` for a final status. Never raises."""
        rule = _TERMINAL.get(status)
        if rule is None or self._config.rule(rule) is None:
            return
        message = "Run ended as failed" if rule == "run_failed" else "Run was preempted"
        self._fire(rule, None, message, terminal=True)
