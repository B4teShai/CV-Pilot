"""Telegram Bot API channel: ``sendMessage`` with plain text.

``CVPILOT_ALERT_TELEGRAM_TOKEN`` (the bot token) and ``CVPILOT_ALERT_TELEGRAM_CHAT`` (a numeric
chat id or ``@channelname``) come from the environment only. The Bot API puts the token in the
URL path, so the URL is built here from the fixed host ``api.telegram.org`` and the validated
token, and never leaves this object. No ``parse_mode`` is sent, so Telegram parses nothing in
the text. ``base_url`` exists for tests and notebooks and is accepted only in development.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from cvpilot.core.schema import Alert
from cvpilot.forward._common import ForwardConfigError
from cvpilot.notify._channel import HttpChannel, checked_endpoint

TOKEN_VARIABLE = "CVPILOT_ALERT_TELEGRAM_TOKEN"  # noqa: S105  # the variable's name
CHAT_VARIABLE = "CVPILOT_ALERT_TELEGRAM_CHAT"
API = "https://api.telegram.org"
_TOKEN = re.compile(r"[0-9]{5,16}:[A-Za-z0-9_-]{30,64}")
_CHAT_ID = re.compile(r"-?[0-9]{1,20}")
_CHAT_NAME = re.compile(r"@[A-Za-z][A-Za-z0-9_]{4,31}")


class TelegramChannel(HttpChannel):
    name = "telegram"

    def __init__(
        self,
        *,
        allow_insecure: bool = False,
        allowed_hosts: Sequence[str] = (),
        base_url: str | None = None,
    ) -> None:
        if base_url is not None and not allow_insecure:
            raise ForwardConfigError("a Telegram base_url is accepted only with CVPILOT_ENV=dev")
        env = self._environment(TOKEN_VARIABLE, CHAT_VARIABLE, allow_insecure=allow_insecure)
        token, chat = env[TOKEN_VARIABLE], env[CHAT_VARIABLE]
        if not token:
            raise ForwardConfigError(f"{TOKEN_VARIABLE} is not set")
        if not _TOKEN.fullmatch(token):
            raise ForwardConfigError(f"{TOKEN_VARIABLE} is not a Telegram bot token")
        if not chat or not (_CHAT_ID.fullmatch(chat) or _CHAT_NAME.fullmatch(chat)):
            raise ForwardConfigError(f"{CHAT_VARIABLE} must be a numeric chat id or @channelname")
        self._chat: int | str = int(chat) if _CHAT_ID.fullmatch(chat) else chat
        url = checked_endpoint(
            f"{base_url or API}/bot{token}/sendMessage",
            variable=TOKEN_VARIABLE,
            allow_insecure=allow_insecure,
            allowed_hosts=allowed_hosts,
        )
        super().__init__(url, allow_insecure=allow_insecure)

    def _payload(self, alert: Alert) -> Mapping[str, Any]:
        return {"chat_id": self._chat, "text": alert.text, "disable_web_page_preview": True}
