"""Generic webhook channel: a JSON POST of the alert to ``CVPILOT_ALERT_WEBHOOK_URL``.

The body is ``{"source": "cvpilot", "rule", "project", "run_id", "step", "text"}``. An optional
``CVPILOT_ALERT_WEBHOOK_TOKEN`` is sent as ``Authorization: Bearer ...`` (only characters that
cannot break a header are accepted); it is never put into the URL.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from cvpilot.core.schema import Alert
from cvpilot.forward._common import ForwardConfigError
from cvpilot.notify._channel import HttpChannel, checked_endpoint

URL_VARIABLE = "CVPILOT_ALERT_WEBHOOK_URL"
TOKEN_VARIABLE = "CVPILOT_ALERT_WEBHOOK_TOKEN"  # noqa: S105  # the variable's name
_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{8,512}")


class WebhookChannel(HttpChannel):
    name = "webhook"

    def __init__(self, *, allow_insecure: bool = False, allowed_hosts: Sequence[str] = ()) -> None:
        env = self._environment(URL_VARIABLE, TOKEN_VARIABLE, allow_insecure=allow_insecure)
        url = checked_endpoint(
            env[URL_VARIABLE],
            variable=URL_VARIABLE,
            allow_insecure=allow_insecure,
            allowed_hosts=allowed_hosts,
        )
        token = env[TOKEN_VARIABLE]
        headers: dict[str, str] = {}
        if token:
            if not _TOKEN.fullmatch(token):
                raise ForwardConfigError(
                    f"{TOKEN_VARIABLE} must be 8 to 512 characters of A-Z a-z 0-9 . _ ~ + / = -"
                )
            headers["Authorization"] = f"Bearer {token}"
        super().__init__(url, allow_insecure=allow_insecure, headers=headers)

    def _payload(self, alert: Alert) -> Mapping[str, Any]:
        return {
            "source": "cvpilot",
            "rule": alert.rule,
            "project": alert.project,
            "run_id": alert.run_id,
            "step": alert.step,
            "text": alert.text,
        }
