"""The alert configuration: a small JSON document, bounded and validated, with no code in it.

A configuration is one object with a list of ``rules``, optional ``channels`` (names of the
channels that deliver) and an optional ``rate_limit_s``. Every rule is an object whose ``rule``
member names it; unknown members, unknown rules, a repeated rule, non-finite numbers, booleans
where a number is expected and out-of-range values are refused, so a typo cannot silently turn a
rule off. There is no template language and nothing is evaluated: message texts are constants of
``cvpilot.alerts.engine``.

JSON was chosen over TOML because Python 3.10 has no TOML reader in its standard library and a
new core dependency was not worth one config file.

The file is read from a descriptor opened without following symbolic links, only if it is a
regular file, and at most ``MAX_CONFIG_BYTES`` are read. Duplicate keys, ``NaN`` and
``Infinity`` are refused. An error names the field and the kind of problem, never the value.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

from cvpilot.compare.gate import ThresholdError, Thresholds
from cvpilot.core.errors import AlertConfigError
from cvpilot.core.schema import ALERT_RULES, is_valid_tag

MAX_CONFIG_BYTES = 16 * 1024
"""Largest configuration file; a full configuration is under 1 KiB."""
MAX_CHANNELS = 8
MAX_METRIC_CHARS = 256
"""The longest metric key (the ``MetricPoint`` limit)."""
MAX_DROP_METRICS = 8
DEFAULT_RATE_LIMIT_S = 300.0
"""One alert per rule and run in this many seconds."""
MAX_RATE_LIMIT_S = 86_400.0
_CHANNEL = r"[a-z][a-z0-9_-]{0,31}"
TAG_PREFIX = "tag:"
"""``eval_regression`` baselines are written ``tag:NAME``, as for ``cvpilot gate``."""


def _number(value: Any) -> Any:
    """A real number: not a boolean, not text (pydantic would coerce both)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number")
    return value


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _metric_name(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not value
        or len(value) > MAX_METRIC_CHARS
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("must be a metric key of 1 to 256 characters without control characters")
    return value


class NanLossSpec(_Spec):
    """Fire when a loss value is NaN or infinite."""

    rule: Literal["nan_loss"]
    metric: str | None = None
    """The exact metric key to watch; by default every key containing ``loss``."""

    _check_metric = field_validator("metric")(_metric_name)


class LossSpikeSpec(_Spec):
    """Fire when a loss is ``factor`` times the mean of the previous ``window`` values."""

    rule: Literal["loss_spike"]
    metric: str | None = None
    factor: float = Field(default=3.0, gt=1.0, le=1000.0, allow_inf_nan=False)
    window: StrictInt = Field(default=20, ge=2, le=1000)

    _check_metric = field_validator("metric")(_metric_name)
    _check_factor = field_validator("factor", mode="before")(_number)


class NoProgressSpec(_Spec):
    """Fire when ``patience`` logged values pass without a new best (lowest) loss."""

    rule: Literal["no_progress"]
    metric: str | None = None
    patience: StrictInt = Field(default=50, ge=2, le=100_000)
    min_delta: float = Field(default=0.0, ge=0.0, le=1e9, allow_inf_nan=False)
    """An improvement smaller than this does not count as progress."""

    _check_metric = field_validator("metric")(_metric_name)
    _check_delta = field_validator("min_delta", mode="before")(_number)


class EvalRegressionSpec(_Spec):
    """Fire when an evaluation snapshot is worse than a pinned baseline by more than ``max_drop``
    points (the gate's thresholds: ``{"mAP50_95": 0.5}``), or cannot be compared with it."""

    rule: Literal["eval_regression"]
    baseline: str
    """``tag:NAME`` of a baseline pinned with ``cvpilot baseline set``."""
    max_drop: dict[str, float] = Field(min_length=1, max_length=MAX_DROP_METRICS)

    @field_validator("baseline")
    @classmethod
    def _baseline(cls, value: str) -> str:
        if not value.startswith(TAG_PREFIX) or not is_valid_tag(value[len(TAG_PREFIX) :]):
            raise ValueError("must be tag:NAME")
        return value

    @field_validator("max_drop", mode="before")
    @classmethod
    def _numbers(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            for number in value.values():
                _number(number)
        return value

    @field_validator("max_drop")
    @classmethod
    def _gate_accepts_them(cls, value: dict[str, float]) -> dict[str, float]:
        try:
            Thresholds(max_drop=value)
        except ThresholdError:
            raise ValueError("must be metric names with limits from 0 up to 100") from None
        return value

    def thresholds(self) -> Thresholds:
        return Thresholds(max_drop=self.max_drop)


class RunFailedSpec(_Spec):
    """Fire when the run ends as ``failed``."""

    rule: Literal["run_failed"]


class RunPreemptedSpec(_Spec):
    """Fire when the run ends as ``preempted``."""

    rule: Literal["run_preempted"]


RuleSpec = Annotated[
    NanLossSpec
    | LossSpikeSpec
    | NoProgressSpec
    | EvalRegressionSpec
    | RunFailedSpec
    | RunPreemptedSpec,
    Field(discriminator="rule"),
]


class AlertConfig(_Spec):
    """A validated alert configuration; see the module docstring."""

    channels: tuple[Annotated[str, Field(pattern=rf"^{_CHANNEL}$")], ...] = Field(
        default=(), max_length=MAX_CHANNELS
    )
    rate_limit_s: float = Field(
        default=DEFAULT_RATE_LIMIT_S, ge=1.0, le=MAX_RATE_LIMIT_S, allow_inf_nan=False
    )
    rules: tuple[RuleSpec, ...] = Field(min_length=1, max_length=len(ALERT_RULES))

    _check_rate = field_validator("rate_limit_s", mode="before")(_number)

    @field_validator("channels", mode="before")
    @classmethod
    def _list(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("must be a list of channel names")
        if len(set(value)) != len(value):
            raise ValueError("a channel is repeated")
        return value

    @field_validator("rules")
    @classmethod
    def _unique(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        names = [spec.rule for spec in value]
        if len(set(names)) != len(names):
            raise ValueError("a rule is repeated")
        return value

    def rule(self, name: str) -> Any:
        """The spec of rule ``name``, or None when it is not configured."""
        return next((spec for spec in self.rules if spec.rule == name), None)


def _known_names() -> frozenset[str]:
    names = {*ALERT_RULES, *AlertConfig.model_fields}
    for model in (
        NanLossSpec, LossSpikeSpec, NoProgressSpec, EvalRegressionSpec, RunFailedSpec,
        RunPreemptedSpec,
    ):  # fmt: skip
        names.update(model.model_fields)
    return frozenset(names)


def _problems(exc: ValidationError) -> str:
    """Field paths and kinds of problem, at most five; never the offending value.

    A path part is shown only when it is a field or rule name of this module or a list index: a
    key that came from the document (an unknown member, a ``max_drop`` metric) is shown as
    ``<key>``, so hostile text cannot ride into an error line.
    """
    known = _known_names()
    seen: list[str] = []
    for err in exc.errors(include_input=False, include_url=False):
        where = ".".join(
            str(part) if isinstance(part, int) or part in known else "<key>" for part in err["loc"]
        )
        line = f"{where or 'config'}: {err['type']}"
        if line not in seen:
            seen.append(line)
    extra = len(exc.errors()) - len(seen[:5])
    return "; ".join(seen[:5]) + (f"; and {extra} more" if extra > 0 else "")


def parse_alert_config(data: object) -> AlertConfig:
    """Validate a mapping (already loaded JSON, or one built in Python).

    Raises:
        AlertConfigError: the configuration is invalid; the message names fields, not values.
    """
    if not isinstance(data, Mapping):
        raise AlertConfigError("the alert configuration must be a JSON object")
    try:
        return AlertConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise AlertConfigError(f"invalid alert configuration: {_problems(exc)}") from None


def _read(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AlertConfigError(f"cannot read the alert configuration: {exc.strerror}") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise AlertConfigError("the alert configuration is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(MAX_CONFIG_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > MAX_CONFIG_BYTES:
        raise AlertConfigError(
            f"the alert configuration is too large (over {MAX_CONFIG_BYTES} bytes)"
        )
    return raw


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def _no_constant(_name: str) -> Any:
    raise ValueError("NaN and Infinity are not valid numbers here")


def load_alert_config(path: str | Path) -> AlertConfig:
    """Read, bound, parse and validate a configuration file.

    Raises:
        AlertConfigError: unreadable, not a regular file, too large, not UTF-8 JSON, has a
            duplicate key or ``NaN``, or fails validation.
    """
    raw = _read(Path(path))
    try:
        data = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_no_constant
        )
    except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
        raise AlertConfigError("the alert configuration is not valid JSON") from None
    return parse_alert_config(data)
