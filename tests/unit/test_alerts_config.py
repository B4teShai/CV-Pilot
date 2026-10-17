"""The alert configuration: a tiny, bounded, validated JSON document with no code in it."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from fake_forwarders import LEAK_MARKER

from cvpilot.alerts.config import (
    MAX_CONFIG_BYTES,
    AlertConfig,
    EvalRegressionSpec,
    LossSpikeSpec,
    NanLossSpec,
    NoProgressSpec,
    load_alert_config,
    parse_alert_config,
)
from cvpilot.core.errors import AlertConfigError

FULL: dict[str, Any] = {
    "channels": ["slack"],
    "rate_limit_s": 120,
    "rules": [
        {"rule": "nan_loss"},
        {"rule": "loss_spike", "factor": 4, "window": 10, "metric": "train/loss"},
        {"rule": "no_progress", "patience": 30, "min_delta": 0.001},
        {"rule": "eval_regression", "baseline": "tag:main", "max_drop": {"mAP50_95": 0.5}},
        {"rule": "run_failed"},
        {"rule": "run_preempted"},
    ],
}


def write(tmp_path: Path, data: object, name: str = "alerts.json") -> Path:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def test_the_documented_example_parses_to_typed_rules() -> None:
    config = parse_alert_config(FULL)
    assert config.channels == ("slack",) and config.rate_limit_s == 120
    kinds = [type(r) for r in config.rules]
    assert kinds == [
        NanLossSpec,
        LossSpikeSpec,
        NoProgressSpec,
        EvalRegressionSpec,
        type(config.rules[4]),
        type(config.rules[5]),
    ]
    spike = config.rules[1]
    assert isinstance(spike, LossSpikeSpec) and (spike.factor, spike.window) == (4.0, 10)
    assert config.rule("no_progress") is config.rules[2] and config.rule("nan_loss") is not None


def test_defaults_are_safe_and_a_rule_list_is_required() -> None:
    config = parse_alert_config({"rules": [{"rule": "nan_loss"}]})
    assert config.channels == () and config.rate_limit_s == 300.0
    with pytest.raises(AlertConfigError):
        parse_alert_config({"channels": ["slack"]})
    with pytest.raises(AlertConfigError):
        parse_alert_config({"rules": []})


def test_a_path_loads_and_a_mapping_is_the_same_as_the_file(tmp_path: Path) -> None:
    assert load_alert_config(write(tmp_path, FULL)) == parse_alert_config(FULL)


BAD: list[tuple[str, Any]] = [
    ("not an object", [1, 2]),
    ("unknown top key", {"rules": [{"rule": "nan_loss"}], "script": "x"}),
    ("unknown rule", {"rules": [{"rule": "gpu_util_below"}]}),
    ("unknown rule key", {"rules": [{"rule": "nan_loss", "exec": "rm -rf /"}]}),
    ("duplicate rule", {"rules": [{"rule": "nan_loss"}, {"rule": "nan_loss"}]}),
    ("rule not an object", {"rules": ["nan_loss"]}),
    ("factor one", {"rules": [{"rule": "loss_spike", "factor": 1}]}),
    ("factor nan", {"rules": [{"rule": "loss_spike", "factor": float("nan")}]}),
    ("factor bool", {"rules": [{"rule": "loss_spike", "factor": True}]}),
    ("factor huge", {"rules": [{"rule": "loss_spike", "factor": 1e9}]}),
    ("window tiny", {"rules": [{"rule": "loss_spike", "window": 1}]}),
    ("window huge", {"rules": [{"rule": "loss_spike", "window": 10**9}]}),
    ("window float", {"rules": [{"rule": "loss_spike", "window": 2.5}]}),
    ("patience zero", {"rules": [{"rule": "no_progress", "patience": 0}]}),
    ("patience huge", {"rules": [{"rule": "no_progress", "patience": 10**9}]}),
    ("min_delta negative", {"rules": [{"rule": "no_progress", "min_delta": -1}]}),
    ("min_delta inf", {"rules": [{"rule": "no_progress", "min_delta": float("inf")}]}),
    ("metric with control char", {"rules": [{"rule": "nan_loss", "metric": "a\nb"}]}),
    ("metric too long", {"rules": [{"rule": "nan_loss", "metric": "m" * 300}]}),
    ("metric empty", {"rules": [{"rule": "nan_loss", "metric": ""}]}),
    ("baseline not a tag", {"rules": [{"rule": "eval_regression", "baseline": "main"}]}),
    ("baseline bad tag", {"rules": [{"rule": "eval_regression", "baseline": "tag:-x"}]}),
    ("baseline tag traversal", {"rules": [{"rule": "eval_regression", "baseline": "tag:../x"}]}),
    ("no max_drop", {"rules": [{"rule": "eval_regression", "baseline": "tag:main"}]}),
    (
        "max_drop hundred",
        {"rules": [{"rule": "eval_regression", "baseline": "tag:m", "max_drop": {"mAP": 100}}]},
    ),
    (
        "max_drop bad name",
        {"rules": [{"rule": "eval_regression", "baseline": "tag:m", "max_drop": {"a b": 1}}]},
    ),
    (
        "max_drop negative",
        {"rules": [{"rule": "eval_regression", "baseline": "tag:m", "max_drop": {"mAP": -1}}]},
    ),
    (
        "max_drop too many",
        {
            "rules": [
                {
                    "rule": "eval_regression",
                    "baseline": "tag:m",
                    "max_drop": {f"m{i}": 1 for i in range(40)},
                }
            ]
        },
    ),
    ("rate zero", {"rate_limit_s": 0, "rules": [{"rule": "nan_loss"}]}),
    ("rate huge", {"rate_limit_s": 10**9, "rules": [{"rule": "nan_loss"}]}),
    ("rate nan", {"rate_limit_s": float("nan"), "rules": [{"rule": "nan_loss"}]}),
    ("channel bad name", {"channels": ["Slack!"], "rules": [{"rule": "nan_loss"}]}),
    ("channel repeated", {"channels": ["a", "a"], "rules": [{"rule": "nan_loss"}]}),
    (
        "too many channels",
        {"channels": [f"c{i}" for i in range(9)], "rules": [{"rule": "nan_loss"}]},
    ),
    ("channels a string", {"channels": "slack", "rules": [{"rule": "nan_loss"}]}),
]


@pytest.mark.parametrize(("label", "data"), BAD, ids=[b[0] for b in BAD])
def test_invalid_configurations_are_refused(label: str, data: Any) -> None:
    with pytest.raises(AlertConfigError):
        parse_alert_config(data)


def test_an_error_names_the_field_and_never_echoes_the_value() -> None:
    secret = LEAK_MARKER
    with pytest.raises(AlertConfigError) as info:
        parse_alert_config({"rules": [{"rule": "nan_loss", "metric": "a\n" + secret}]})
    text = str(info.value)
    assert secret not in text and "metric" in text
    with pytest.raises(AlertConfigError) as info:
        parse_alert_config({"rules": [{"rule": "nan_loss", secret: 1}]})
    assert secret not in str(info.value)
    with pytest.raises(AlertConfigError) as info:
        parse_alert_config({"rules": [{"rule": secret}]})
    assert secret not in str(info.value)


def test_the_file_is_read_bounded_without_following_a_symlink(tmp_path: Path) -> None:
    big = write(tmp_path, '{"rules": [{"rule": "nan_loss"}]}' + " " * MAX_CONFIG_BYTES, "big.json")
    with pytest.raises(AlertConfigError, match="too large"):
        load_alert_config(big)
    real = write(tmp_path, FULL, "real.json")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(AlertConfigError):
        load_alert_config(link)
    with pytest.raises(AlertConfigError):
        load_alert_config(tmp_path)  # a directory
    with pytest.raises(AlertConfigError):
        load_alert_config(tmp_path / "missing.json")


def test_a_fifo_cannot_hang_the_loader(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("no fifo")
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(AlertConfigError):
        load_alert_config(fifo)
    assert time.monotonic() - started < 2


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        '{"rules": [{"rule": "nan_loss"}],}',
        '{"rules": [{"rule": "nan_loss"}], "rules": [{"rule": "run_failed"}]}',  # duplicate key
        '{"rules": [{"rule": "loss_spike", "factor": NaN}]}',
        '{"rules": [{"rule": "loss_spike", "factor": Infinity}]}',
        "[" * 5000 + "]" * 5000,  # nesting bomb inside the size limit
        '{"rules": [{"rule": "nan_loss"}]}\x00',
    ],
)
def test_malformed_documents_are_refused_not_crashed_on(tmp_path: Path, text: str) -> None:
    with pytest.raises(AlertConfigError):
        load_alert_config(write(tmp_path, text))


def test_invalid_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"rules": [{"rule": "nan_loss", "metric": "\xff"}]}')
    with pytest.raises(AlertConfigError):
        load_alert_config(path)


def test_the_loaded_config_is_immutable() -> None:
    config = parse_alert_config(FULL)
    assert isinstance(config, AlertConfig)
    with pytest.raises(Exception):  # noqa: B017  # frozen model
        config.rate_limit_s = 1  # type: ignore[misc]
