"""Slack-style incoming webhook channel: ``{"text": ...}`` to ``CVPILOT_ALERT_SLACK_URL``.

The URL is a secret (its path is the credential): it comes from the environment only and is never
logged. The text is escaped the way Slack documents (``&``, ``<``, ``>``) so nothing in it can
become a mention or a link, and ``mrkdwn`` is turned off so no character is formatting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cvpilot.core.schema import Alert
from cvpilot.notify._channel import HttpChannel, checked_endpoint

URL_VARIABLE = "CVPILOT_ALERT_SLACK_URL"


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SlackChannel(HttpChannel):
    name = "slack"

    def __init__(self, *, allow_insecure: bool = False, allowed_hosts: Sequence[str] = ()) -> None:
        env = self._environment(URL_VARIABLE, allow_insecure=allow_insecure)
        url = checked_endpoint(
            env[URL_VARIABLE],
            variable=URL_VARIABLE,
            allow_insecure=allow_insecure,
            allowed_hosts=allowed_hosts,
        )
        super().__init__(url, allow_insecure=allow_insecure)

    def _payload(self, alert: Alert) -> Mapping[str, Any]:
        return {"text": escape(alert.text), "mrkdwn": False}
