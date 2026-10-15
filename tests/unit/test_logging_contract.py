import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cvpilot.logging import REDACTED, clear_request_context, configure_logging, get_logger
from cvpilot.settings import Settings

REQUIRED = {"@timestamp", "level", "message", "service", "env", "request_id", "logger"}


@pytest.fixture
def buf() -> Any:
    stream = io.StringIO()
    yield stream
    clear_request_context()


def setup(buf: io.StringIO, **settings: Any) -> str:
    return configure_logging("cvpilot", Settings(**settings), stream=buf)


def records(buf: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line]


def test_required_fields_present(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("cvpilot.test").info("something_happened")
    (rec,) = records(buf)
    assert REQUIRED <= rec.keys()
    assert rec["service"] == "cvpilot" and rec["env"] == "prod"
    assert rec["logger"] == "cvpilot.test" and rec["level"] == "INFO"
    datetime.fromisoformat(rec["@timestamp"].replace("Z", "+00:00"))


def test_message_is_constant_and_dynamic_values_are_fields(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info("audit_check_completed", check_name="leakage", findings_count=3)
    (rec,) = records(buf)
    assert rec["message"] == "audit_check_completed"
    assert (rec["check_name"], rec["findings_count"]) == ("leakage", 3)


def test_all_keys_are_snake_case(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info("evt", some_field=1, other_field={"nested_key": 2})
    (rec,) = records(buf)
    assert all(re.fullmatch(r"@?[a-z][a-z0-9_]*", k) for k in rec)


def test_request_id_is_uuid_and_unique_per_configure(buf: io.StringIO) -> None:
    first = setup(buf)
    get_logger("x").info("a")
    second = setup(buf)
    get_logger("x").info("b")
    a, b = records(buf)
    assert uuid.UUID(a["request_id"]) and a["request_id"] == first
    assert b["request_id"] == second != first


def test_debug_capped_outside_dev(buf: io.StringIO) -> None:
    setup(buf, env="prod", log_level="DEBUG")
    log = get_logger("x")
    log.debug("hidden")
    log.info("shown")
    assert [r["message"] for r in records(buf)] == ["shown"]


def test_debug_emitted_in_dev(buf: io.StringIO) -> None:
    setup(buf, env="dev", log_level="DEBUG")
    get_logger("x").debug("visible")
    assert [r["message"] for r in records(buf)] == ["visible"]


def test_warning_level_is_uppercase(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").warning("careful")
    assert records(buf)[0]["level"] == "WARNING"


@pytest.mark.parametrize(
    "key",
    [
        "token",
        "password",
        "api_key",
        "jwt",
        "authorization",
        "Authorization",
        "X-Api-Key",
        "access_token",
        "PASSWORD",
    ],
)
def test_sensitive_keys_are_redacted(buf: io.StringIO, key: str) -> None:
    setup(buf)
    get_logger("x").info("evt", **{key: "s3cret"})
    (rec,) = records(buf)
    assert "s3cret" not in buf.getvalue()
    assert REDACTED in rec.values()


def test_nested_and_listed_secrets_are_redacted(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info(
        "evt",
        headers={"Authorization": "Bearer s3cret", "accept": "json"},
        attempts=[{"password": "s3cret"}, ("jwt", {"token": "s3cret"})],
    )
    (rec,) = records(buf)
    assert "s3cret" not in buf.getvalue()
    assert rec["headers"]["accept"] == "json"
    assert rec["headers"]["Authorization"] == REDACTED
    assert rec["attempts"][0] == {"password": REDACTED}
    assert rec["attempts"][1] == ["jwt", REDACTED]


def test_innocent_keys_are_not_redacted(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info("evt", token_count=5, passwords_checked=2)
    (rec,) = records(buf)
    assert (rec["token_count"], rec["passwords_checked"]) == (5, 2)


def test_non_serializable_values_do_not_crash(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info(
        "evt",
        path=Path("/srv/é"),
        when=datetime(2026, 1, 1, tzinfo=timezone.utc),
        raw=b"\xff",
        weird={("tuple", "key"): 1},
    )
    (rec,) = records(buf)
    assert rec["path"].endswith("é")
    assert "é" in buf.getvalue()  # raw output, not escaped
    assert isinstance(rec["raw"], str) and "xff" in rec["raw"]
    assert rec["weird"] == {"('tuple', 'key')": 1}


def test_error_is_logged_once_with_exception(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").error("command_failed", command="audit", exc_info=ValueError("boom"))
    (rec,) = records(buf)
    assert rec["level"] == "ERROR" and "ValueError: boom" in rec["exception"]


def _raise(name: str) -> Any:
    raise AssertionError(f"non-standard JSON constant: {name}")


def test_self_referencing_container_does_not_raise(buf: io.StringIO) -> None:
    setup(buf)
    d: dict[str, Any] = {}
    d["self"] = d
    get_logger("x").info("evt", payload=d)
    (rec,) = records(buf)
    assert "[TRUNCATED]" in buf.getvalue() and rec["message"] == "evt"


def test_deeply_nested_container_is_truncated(buf: io.StringIO) -> None:
    setup(buf)
    deep: Any = "leaf"
    for _ in range(20):
        deep = {"k": deep}
    get_logger("x").info("evt", payload=deep)
    (rec,) = records(buf)
    assert "[TRUNCATED]" in buf.getvalue() and "leaf" not in buf.getvalue()
    assert rec["message"] == "evt"


def test_sensitive_pairs_bytes_keys_and_sets_are_redacted(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info(
        "evt",
        pairs=[("authorization", "Bearer s3cret")],
        bpairs=[(b"token", b"s3cret")],
        bkeys={b"token": "s3cret"},
        fset=frozenset({(("token", "s3cret"),)}),
        aset={("authorization", "s3cret")},
        nested=[{"a": {"token": "s3cret"}}],
    )
    (rec,) = records(buf)
    assert "s3cret" not in buf.getvalue()
    assert rec["pairs"] == [["authorization", REDACTED]]
    assert rec["bpairs"] == [["token", REDACTED]]
    assert rec["bkeys"] == {"token": REDACTED}
    assert rec["fset"] == [[["token", REDACTED]]]
    assert rec["aset"] == [["authorization", REDACTED]]
    assert rec["nested"] == [{"a": {"token": REDACTED}}]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_become_strings(buf: io.StringIO, value: float) -> None:
    setup(buf)
    get_logger("x").info("evt", top=value, nested={"a": [value]})
    line = buf.getvalue().strip()
    rec = json.loads(line, parse_constant=_raise)
    assert rec["top"] == str(value) and rec["nested"]["a"] == [str(value)]


class _Bad:
    def __str__(self) -> str:
        raise RuntimeError("no str")

    __repr__ = __str__


def test_unstringable_object_still_emits_valid_json(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info("evt", bad=_Bad())
    (rec,) = records(buf)
    assert REQUIRED <= rec.keys()
    assert rec["message"] == "log_serialization_failed"
    assert rec["service"] == "cvpilot" and rec["level"] == "INFO"


def test_unstringable_key_does_not_escape(buf: io.StringIO) -> None:
    setup(buf)
    get_logger("x").info("evt", payload={_Bad(): 1})
    lines = buf.getvalue().splitlines()
    assert lines and json.loads(lines[-1])


def test_caller_kwargs_cannot_override_bound_context(buf: io.StringIO) -> None:
    rid = setup(buf)
    get_logger("x").info("e", request_id="evil", service="evil", env="evil")
    (rec,) = records(buf)
    assert (rec["request_id"], rec["service"], rec["env"]) == (rid, "cvpilot", "prod")


def test_clear_request_context_drops_bound_fields(buf: io.StringIO) -> None:
    setup(buf)
    clear_request_context()
    get_logger("x").info("after_clear")
    (rec,) = records(buf)
    assert "request_id" not in rec and "service" not in rec
