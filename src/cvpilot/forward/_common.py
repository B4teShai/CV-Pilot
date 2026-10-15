"""Helpers shared by the tracker adapters.

Environment handling: an adapter reads the few variables that decide where it talks to and how
it verifies the peer **once, on the training thread, when it is constructed** (``freeze_env``),
checks that snapshot, and later passes the checked values to the SDK explicitly. Reading the
environment again on the worker thread would let something else in the process change it
between the check and the use.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from cvpilot.core.errors import CvpilotError, MissingExtraError
from cvpilot.core.schema import MetricPoint

MAX_URL_CHARS = 2048
"""Longest endpoint accepted from the environment."""
CA_OVERRIDE_VARIABLES = (
    "MLFLOW_TRACKING_SERVER_CERT_PATH",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
"""Variables that replace the trusted certificate authorities; refused outside development."""
_TRUE = {"1", "true", "yes", "on"}


class ForwardConfigError(CvpilotError, ValueError):
    """A tracker endpoint or TLS setting is unsafe; the message never contains the value."""


def freeze_env(names: Sequence[str]) -> dict[str, str | None]:
    """The current value of each variable in ``names`` (None when unset), read once."""
    return {name: os.environ.get(name) for name in names}


def env_flag(env: Mapping[str, str | None], name: str) -> bool:
    return (env.get(name) or "").strip().lower() in _TRUE


def refuse_ca_overrides(env: Mapping[str, str | None], *, allow_insecure: bool) -> None:
    """Refuse variables that swap the trusted certificate authorities, unless dev."""
    if allow_insecure:
        return
    for name in CA_OVERRIDE_VARIABLES:
        if env.get(name):
            raise ForwardConfigError(
                f"{name} replaces the trusted certificate authorities; refused outside "
                "CVPILOT_ENV=dev"
            )


def refuse_insecure_flag(env: Mapping[str, str | None], name: str, *, allow_insecure: bool) -> None:
    """Refuse an SDK switch that turns TLS verification off, unless dev."""
    if env_flag(env, name) and not allow_insecure:
        raise ForwardConfigError(
            f"{name} disables TLS verification; refused outside CVPILOT_ENV=dev"
        )


_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST = re.compile(rf"{_LABEL}(?:\.{_LABEL})*")
_FORBIDDEN_CHARS = frozenset("\\%@?#")


def _refuse_endpoint(what: str, why: str) -> ForwardConfigError:
    return ForwardConfigError(f"{what} {why}")  # never the URL: it may carry credentials


def check_endpoint(
    url: str | None,
    *,
    allow_insecure: bool,
    what: str,
    allowed_hosts: Sequence[str] = (),
    default_host: str | None = None,
) -> str | None:
    """Validate an endpoint and return the **normalised** URL that must be passed to the SDK.

    Accepted: ``https`` (``http`` too in dev) with an ASCII host made of lower-case-able DNS
    labels, no userinfo, no query or fragment, no IPv6 literal, no percent-encoding, backslash,
    whitespace or control character anywhere, and a port of 443 (80 for http), or another port
    only when ``CVPILOT_FORWARD_HOSTS`` lists ``host:port`` (dev may use any port). The host is
    lower-cased and one trailing dot removed; the returned URL is rebuilt from the parsed parts,
    so the string the SDK receives is the string that was checked, not the original text (which
    another parser could read differently).

    With ``allowed_hosts`` the host must equal an entry exactly (no suffix or substring
    matching; an entry with a port allows that port only, one without allows the default port
    only). An unset endpoint means the SDK default: with an allow-list that default must itself be
    listed (``default_host``; None when the default is not a network host, which an allow-list
    cannot vouch for) and None is returned. ``what`` names the environment variable; the URL is
    never put into a message.

    Raises:
        ForwardConfigError: the endpoint is unsafe or not allowed.
    """
    entries = [h.lower() for h in allowed_hosts]
    plain = {e for e in entries if ":" not in e}
    if not url:
        if entries and (default_host is None or default_host.lower() not in plain):
            raise _refuse_endpoint(what, "names no allowed endpoint (CVPILOT_FORWARD_HOSTS is set)")
        return None
    if (
        len(url) > MAX_URL_CHARS
        or not url.isascii()
        or any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c in _FORBIDDEN_CHARS for c in url)
    ):
        raise _refuse_endpoint(what, "is too long or has characters that parsers read differently")
    try:
        parts = urlsplit(url)
    except ValueError:
        raise _refuse_endpoint(what, "is not a valid URL") from None
    scheme = parts.scheme.lower()
    if scheme != "https" and not (scheme == "http" and allow_insecure):
        raise _refuse_endpoint(
            what, "must be an https URL (CVPILOT_ENV=dev also allows http for a local server)"
        )
    netloc = parts.netloc
    if not netloc or "[" in netloc or "]" in netloc:
        raise _refuse_endpoint(what, "must name a DNS host (no IPv6 literal)")
    host_text, _, port_text = netloc.partition(":")
    if ":" in port_text:
        raise _refuse_endpoint(what, "is not a valid URL")
    host = host_text.lower().removesuffix(".")
    if len(host) > 253 or not _HOST.fullmatch(host):
        raise _refuse_endpoint(what, "has an invalid host")
    port: int | None = None
    if port_text:
        if not (port_text.isascii() and port_text.isdigit() and 1 <= int(port_text) <= 65535):
            raise _refuse_endpoint(what, "has an invalid port")
        port = int(port_text)
    if port == (443 if scheme == "https" else 80):
        port = None
    if entries:
        if port is None and host not in plain:
            raise _refuse_endpoint(what, "has a host that is not in CVPILOT_FORWARD_HOSTS")
        if port is not None and f"{host}:{port}" not in entries:
            raise _refuse_endpoint(
                what, "has a port that is not allowed; list host:port in CVPILOT_FORWARD_HOSTS"
            )
    elif port is not None and not allow_insecure:
        raise _refuse_endpoint(
            what, "has a port other than 443; list host:port in CVPILOT_FORWARD_HOSTS to allow it"
        )
    return f"{scheme}://{host}{f':{port}' if port is not None else ''}{parts.path}"


def load_sdk(module: str, extra: str) -> Any:
    """Import a tracker SDK lazily; ``MissingExtraError`` names the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError:
        raise MissingExtraError(
            f"forwarder {extra!r} needs the {module} package: install cvpilot[{extra}]"
        ) from None


def group_by_step(points: Sequence[MetricPoint]) -> list[tuple[int, dict[str, float]]]:
    """Points grouped by step, ascending; a key logged twice at one step keeps the last value."""
    steps: dict[int, dict[str, float]] = {}
    for p in points:
        steps.setdefault(p.step, {})[p.key] = p.value
    return sorted(steps.items())
