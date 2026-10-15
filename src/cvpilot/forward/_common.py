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


def check_endpoint(
    url: str | None,
    *,
    allow_insecure: bool,
    what: str,
    allowed_hosts: Sequence[str] = (),
    default_host: str | None = None,
) -> None:
    """Accept only an ``https`` endpoint (``http`` too in dev) without credentials, on an
    allowed host when ``allowed_hosts`` is set.

    An unset endpoint means the SDK default: with an allow-list that default must itself be
    allowed (``default_host``; None when the default is not a network host, which an allow-list
    cannot vouch for). ``what`` names the environment variable; the URL is never put into a
    message, because it may carry the credentials this check exists to reject.

    Raises:
        ForwardConfigError: the endpoint is unsafe.
    """
    hosts = [h.lower() for h in allowed_hosts]
    if not url:
        if hosts and (default_host is None or default_host.lower() not in hosts):
            raise ForwardConfigError(
                f"CVPILOT_FORWARD_HOSTS is set but {what} names no allowed endpoint"
            )
        return
    if len(url) > MAX_URL_CHARS or any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ForwardConfigError(f"{what} is too long or contains control characters")
    try:
        parts = urlsplit(url)
        userinfo = parts.username is not None or parts.password is not None or "@" in parts.netloc
        host = (parts.hostname or "").lower()
    except ValueError:
        raise ForwardConfigError(f"{what} is not a valid URL") from None
    if userinfo:
        raise ForwardConfigError(f"{what} contains credentials; pass them in the SDK's environment")
    scheme = parts.scheme.lower()
    if scheme != "https" and not (scheme == "http" and allow_insecure):
        raise ForwardConfigError(
            f"{what} must be an https URL (CVPILOT_ENV=dev also allows http for a local server)"
        )
    if not host:
        raise ForwardConfigError(f"{what} has no host")
    if hosts and host not in hosts:
        raise ForwardConfigError(f"the host of {what} is not in CVPILOT_FORWARD_HOSTS")


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
