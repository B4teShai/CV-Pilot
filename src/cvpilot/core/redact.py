"""Secret redaction for stored run data (``Repro.command_line`` and ``Repro.config``).

Keys are normalised (camelCase to snake_case, hyphens to underscores, lower case) and are
sensitive when they end with a sensitive word, with or without a separator
(``GITHUBTOKEN``, ``accessToken``, ``hf_token``) or with ``<api|access|secret|private|
signing|encryption>_key``. ``token_count``, ``max_tokens``, ``author`` and ``keyboard`` are
not matched. Command lines are redacted by flag (``--token=x``, ``--password x``), by
``NAME=value`` and ``Name: value`` pairs, by bearer credentials, by URL user-info and by
signed-URL query parameters. Redaction is idempotent.

``redact_argv`` is the structural path used for stored command lines: it works on the argument
list, so a value is hidden by its position (``--token VALUE``, ``--token=VALUE``,
``NAME=VALUE``), whatever it contains. The short flags in ``SHORT_SECRET_FLAGS`` (``-p``, as in
``mysql -p``) are treated like sensitive long flags: the next token is redacted unless it
starts with ``-``. This also hides a port after ``-p``; over-redaction is the safe side.

Design note: pattern-based redaction of free-text command lines is best-effort, not a
guarantee; a secret passed positionally or in an unrecognised flag is not found. Users who
cannot accept that risk set ``CVPILOT_STORE_COMMAND_LINE=false`` so only the executable name
and the argument count are stored. Never put secrets on a command line; use the environment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"
TRUNCATED = "[TRUNCATED]"
_MAX_DEPTH = 8
_MAX_REPR = 200
SHORT_SECRET_FLAGS: frozenset[str] = frozenset({"-p"})
_QUERY_SECRET_NAMES = frozenset(
    {"sig", "signature", "x-amz-signature", "x-amz-credential", "key", "code"}
)

_WORDS = (
    "token",
    "password",
    "passwd",
    "pwd",
    "jwt",
    "authorization",
    "auth",
    "secret",
    "credentials?",
    "cookie",
    "session[-_]?id",
    "bearer",
    "(?:api|access|secret|private|signing|encryption)[-_]?key",
)
_SENSITIVE = "(?:" + "|".join(_WORDS) + ")"
_KEY_RE = re.compile(rf".*{_SENSITIVE}", re.DOTALL)  # used with fullmatch on a normalised key
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_NAME = rf"[\w-]*{_SENSITIVE}"
_VALUE = r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|["'].*|\S+)"""
_BEARER = re.compile(r"(?i)\b(bearer|basic)(\s+)(?!\[REDACTED\])[^\s'\"]+")
_FLAG = re.compile(
    rf"(?i)(?P<flag>(?<![\w-])--?{_NAME})(?P<sep>=|\s+)(?P<val>(?!\[REDACTED\]){_VALUE})"
)
_PAIR = re.compile(
    rf"(?i)(?P<name>(?<![\w-]){_NAME})(?P<sep>\s*[:=]\s*)(?P<val>(?!\[REDACTED\]){_VALUE})"
)
_USERINFO = re.compile(r"(?P<scheme>://)(?P<user>[^/\s@:]+)(?P<pw>:[^@\s/]*)?@")
_QUERY = re.compile(
    r"(?i)(?P<lead>[?&](?:sig|signature|x-amz-signature|x-amz-credential|key|code)=)[^&\s'\"]+"
)


def _flag_sub(match: re.Match[str]) -> str:
    sep, val = match["sep"], match["val"]
    if sep.isspace() and val.startswith("-") and not val.startswith(("'", '"')):
        return match.group(0)  # a bare flag followed by another flag, nothing to hide
    return f"{match['flag']}{sep}{REDACTED}"


def _userinfo_sub(match: re.Match[str]) -> str:
    if match["pw"] is None:
        return f"{match['scheme']}{REDACTED}@"
    return f"{match['scheme']}{match['user']}:{REDACTED}@"


def redact_command_line(line: str) -> str:
    """Return ``line`` with secret values replaced by ``[REDACTED]`` (best effort)."""
    out = _BEARER.sub(lambda m: f"{m[1]}{m[2]}{REDACTED}", line)
    out = _FLAG.sub(_flag_sub, out)
    out = _PAIR.sub(lambda m: f"{m['name']}{m['sep']}{REDACTED}", out)
    out = _USERINFO.sub(_userinfo_sub, out)
    return _QUERY.sub(lambda m: f"{m['lead']}{REDACTED}", out)


def _normalise_key(key: object) -> str:
    text = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
    return _CAMEL.sub("_", text).lower().replace("-", "_")


def is_sensitive_key(key: object) -> bool:
    """True when the normalised key ends with a sensitive word."""
    return _KEY_RE.fullmatch(_normalise_key(key)) is not None


def _safe_repr(value: object) -> str:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001  # a hostile __repr__ must not break capture
        text = f"<{type(value).__name__}>"
    return redact_command_line(text[:_MAX_REPR])


def _walk(value: Any, depth: int, path: frozenset[int]) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_command_line(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return REDACTED
    if not isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return _safe_repr(value)
    if depth > _MAX_DEPTH or id(value) in path:
        return TRUNCATED
    path = path | {id(value)}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            name = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)
            sensitive = is_sensitive_key(name) or out.get(name) == REDACTED
            out[name] = REDACTED if sensitive else _walk(v, depth + 1, path)
        return out
    items: list[Any] = []
    for item in value:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            and is_sensitive_key(item[0])
        ):
            items.append([item[0], REDACTED])
        else:
            items.append(_walk(item, depth + 1, path))
    return items


def redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep, JSON-friendly copy with sensitive keys and secret-looking text replaced.

    Bytes become ``[REDACTED]``; objects become a truncated, redacted ``repr``; the input is
    untouched.
    """
    result: dict[str, Any] = _walk(value, 1, frozenset())
    return result


_HEADER = re.compile(r"\s*([\w-]+)\s*:\s*(.+)", re.DOTALL)
_ASSIGN = re.compile(r"([A-Za-z_][\w.-]*)=(.*)", re.DOTALL)
_QUERY_PARAM = re.compile(r"(?P<lead>[?&])(?P<name>[^=&\s]+)=(?P<val>[^&\s]*)")


def _query_sub(match: re.Match[str]) -> str:
    name = match["name"]
    if is_sensitive_key(name) or name.lower() in _QUERY_SECRET_NAMES:
        return f"{match['lead']}{name}={REDACTED}"
    return match.group(0)


def _redact_token(token: str) -> str:
    """Redact credentials inside one value: header pairs, bearer, URL user-info, query."""
    header = _HEADER.fullmatch(token)
    if header and is_sensitive_key(header[1]):
        return f"{header[1]}: {REDACTED}"
    out = _BEARER.sub(lambda m: f"{m[1]}{m[2]}{REDACTED}", token)
    out = _USERINFO.sub(_userinfo_sub, out)
    out = _QUERY_PARAM.sub(_query_sub, out)
    if any(c.isspace() for c in out):
        out = redact_command_line(out)  # free text inside one argument: best effort
    return out


def redact_argv(args: Sequence[str]) -> list[str]:
    """Redact an argument list structurally; returns a new list.

    Rules per token: ``--name=value`` and ``NAME=value`` hide the value when the name is
    sensitive (same logic as :func:`is_sensitive_key`); a bare sensitive flag, or one of
    ``SHORT_SECRET_FLAGS``, hides the next token unless that starts with ``-``; every
    value is also checked for bearer credentials, ``Name: value`` headers with a sensitive
    name, URL user-info and secret query parameters; the executable is token zero and gets
    the same treatment. Idempotent.
    """
    out: list[str] = []
    pending = False
    for token in args:
        if pending and not token.startswith("-"):
            out.append(REDACTED)
            pending = False
            continue
        pending = False
        if token.startswith("-") and "=" in token:
            name, _, value = token.partition("=")
            hidden = is_sensitive_key(name.lstrip("-"))
            out.append(f"{name}={REDACTED if hidden else _redact_token(value)}")
        elif token.startswith("-"):
            pending = token in SHORT_SECRET_FLAGS or is_sensitive_key(token.lstrip("-"))
            out.append(token)
        elif (assign := _ASSIGN.fullmatch(token)) is not None:
            hidden = is_sensitive_key(assign[1])
            out.append(f"{assign[1]}={REDACTED if hidden else _redact_token(assign[2])}")
        else:
            out.append(_redact_token(token))
    return out
