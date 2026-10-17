"""What the three channels share: where they send is checked once, the secret stays inside."""

from __future__ import annotations

import ssl
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn
from urllib.parse import urlsplit

from cvpilot.core.schema import Alert, Repro, RunMeta, RunStatus
from cvpilot.forward._common import (
    CA_OVERRIDE_VARIABLES,
    ForwardConfigError,
    check_endpoint,
    freeze_env,
    refuse_ca_overrides,
)
from cvpilot.logging import get_logger
from cvpilot.notify import _http


def checked_endpoint(
    url: str | None, *, variable: str, allow_insecure: bool, allowed_hosts: Sequence[str]
) -> str:
    """The normalised https URL of ``url`` (a variable's value), which must be set.

    On top of ``check_endpoint`` (https, no credentials, no query, port 443 unless listed, the
    ``CVPILOT_FORWARD_HOSTS`` allow-list) a production endpoint must name a DNS host: an address
    written as an IP number in any notation, ``localhost`` and ``*.localhost`` are refused here, and
    ``_http`` refuses any name that *resolves* to a non-public address when it connects.

    Raises:
        ForwardConfigError: unset or unsafe; the message names the variable, never the value.
    """
    if not url:
        raise ForwardConfigError(f"{variable} is not set")
    clean = check_endpoint(
        url, allow_insecure=allow_insecure, what=variable, allowed_hosts=allowed_hosts
    )
    if clean is None:  # pragma: no cover  # an empty url was handled above
        raise ForwardConfigError(f"{variable} is not set")
    if not allow_insecure:
        host = urlsplit(clean).hostname or ""
        last = host.rsplit(".", 1)[-1]
        if host == "localhost" or host.endswith(".localhost") or last.isdigit() or (
            last.startswith("0x")
        ):  # fmt: skip
            raise ForwardConfigError(f"{variable} must name a public DNS host, not an address")
    return clean


class HttpChannel:
    """Base of the channels: one https endpoint, read from the environment at construction.

    Subclasses set ``name`` and implement ``_payload``. The endpoint (which may be or contain a
    secret) lives in one private attribute: it is not in ``repr``, not in an error, not in a log
    event, and the object cannot be pickled.
    """

    name = ""
    requires: tuple[str, ...] = ()

    def __init__(
        self,
        url: str,
        *,
        allow_insecure: bool,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if allow_insecure:
            get_logger("cvpilot.notify").warning(
                "insecure_dev_mode", channel=self.name, reason="http to loopback addresses allowed"
            )
        self._url = url
        self._allow_insecure = allow_insecure
        self._headers = dict(headers or {})
        self._tls = ssl.create_default_context() if url.startswith("https://") else None

    @staticmethod
    def _environment(*names: str, allow_insecure: bool) -> dict[str, str | None]:
        """The variables ``names`` read once; a replaced certificate authority is refused."""
        env = freeze_env((*names, *CA_OVERRIDE_VARIABLES))
        refuse_ca_overrides(env, allow_insecure=allow_insecure)
        return env

    def _payload(self, alert: Alert) -> Mapping[str, Any]:
        raise NotImplementedError

    def open(self, meta: RunMeta, repro: Repro) -> None:
        return None

    def send(self, alerts: Sequence[Alert]) -> None:
        for alert in alerts:
            _http.post_json(
                self._url,
                self._payload(alert),
                headers=self._headers,
                allow_insecure=self._allow_insecure,
                tls=self._tls,
            )
            get_logger("cvpilot.notify").info(
                "alert_delivered", channel=self.name, rule=alert.rule, run_id=alert.run_id
            )

    def close(self, status: RunStatus) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{self.name} alert channel>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("an alert channel holds a credential and cannot be pickled")
