"""The three channels: endpoints checked like the trackers', credentials from the environment only,
payloads built without interpreting the text."""

from __future__ import annotations

import io
import json
import pickle
from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from notify_support import Listener
from support import meta_repro
from test_forward_hosts import HOSTILE

from cvpilot import registry
from cvpilot.core.schema import Alert
from cvpilot.forward._common import ForwardConfigError
from cvpilot.logging import configure_logging
from cvpilot.notify import _http
from cvpilot.notify.slack import SlackChannel
from cvpilot.notify.telegram import TelegramChannel
from cvpilot.notify.webhook import WebhookChannel
from cvpilot.runs import forwarding as fw
from cvpilot.settings import Settings

RUN = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
SLACK_PATH = "/services/T0000/B0000/SLACKSECRET0123456789"
TG_BOT = "123456789:AAH-secret_TOKEN-0123456789abcdefghijk"  # noqa: S105  # shaped like one
HOOK_BEARER = "whtok-0123456789abcdef"  # noqa: S105
ENV_VARS = (
    "CVPILOT_ALERT_WEBHOOK_URL",
    "CVPILOT_ALERT_WEBHOOK_TOKEN",
    "CVPILOT_ALERT_SLACK_URL",
    "CVPILOT_ALERT_TELEGRAM_TOKEN",
    "CVPILOT_ALERT_TELEGRAM_CHAT",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "MLFLOW_TRACKING_SERVER_CERT_PATH",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def server() -> Iterator[Listener]:
    listener = Listener()
    yield listener
    listener.close()


@pytest.fixture
def logs() -> Iterator[Any]:
    structlog.reset_defaults()
    stream = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="prod"), stream)

    class Reader:
        def text(self) -> str:
            return stream.getvalue()

        def records(self) -> list[dict[str, Any]]:
            return [json.loads(x) for x in stream.getvalue().splitlines() if x.startswith("{")]

    yield Reader()
    structlog.reset_defaults()


def alert(text: str = "[proj] Loss loss is NaN at step 1 (run X).", step: int | None = 1) -> Alert:
    return Alert(rule="nan_loss", project="proj", run_id=RUN, step=step, text=text)


def slack(monkeypatch: pytest.MonkeyPatch, url: str, **kw: Any) -> SlackChannel:
    monkeypatch.setenv("CVPILOT_ALERT_SLACK_URL", url)
    return SlackChannel(**kw)


# -- construction: where it sends is checked once, from the environment -----------------------


@pytest.mark.parametrize(
    ("cls", "variable"),
    [
        (WebhookChannel, "CVPILOT_ALERT_WEBHOOK_URL"),
        (SlackChannel, "CVPILOT_ALERT_SLACK_URL"),
        (TelegramChannel, "CVPILOT_ALERT_TELEGRAM_TOKEN"),
    ],
)
def test_a_channel_without_its_variable_is_a_configuration_error_naming_it(
    cls: type, variable: str
) -> None:
    with pytest.raises(ForwardConfigError, match=variable):
        cls()


@pytest.mark.parametrize("url", HOSTILE)
@pytest.mark.parametrize("cls", [WebhookChannel, SlackChannel])
def test_hostile_urls_are_refused_and_never_echoed(
    monkeypatch: pytest.MonkeyPatch, cls: type, url: str
) -> None:
    variable = "CVPILOT_ALERT_WEBHOOK_URL" if cls is WebhookChannel else "CVPILOT_ALERT_SLACK_URL"
    monkeypatch.setenv(variable, url)
    with pytest.raises(ForwardConfigError) as info:
        cls(allowed_hosts=["allowed.example"])
    assert url not in str(info.value) and variable in str(info.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.example/x",  # http in production
        "https://user:pw@hooks.example/x",
        "https://hooks.example/x?token=abc",
        "https://hooks.example:8443/x",
        "https://127.0.0.1/x",
        "https://10.0.0.5/hook",
        "https://169.254.169.254/latest/meta-data",
        "https://2130706433/x",  # 127.0.0.1 as one integer
        "https://0x7f000001/x",
        "https://0177.0.0.1/x",
        "https://localhost/x",
        "https://app.localhost/x",
        "https://[::1]/x",
    ],
)
def test_production_refuses_http_credentials_ports_and_addresses(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_URL", url)
    with pytest.raises(ForwardConfigError):
        WebhookChannel()


def test_a_good_url_is_accepted_and_the_allow_list_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slack(monkeypatch, "https://hooks.slack.com" + SLACK_PATH)
    slack(monkeypatch, "https://hooks.slack.com" + SLACK_PATH, allowed_hosts=["hooks.slack.com"])
    with pytest.raises(ForwardConfigError, match="CVPILOT_FORWARD_HOSTS"):
        slack(monkeypatch, "https://hooks.slack.com" + SLACK_PATH, allowed_hosts=["other.example"])


def test_http_to_a_local_listener_needs_the_development_setting(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    with pytest.raises(ForwardConfigError):
        slack(monkeypatch, server.url + SLACK_PATH)
    slack(monkeypatch, server.url + SLACK_PATH, allow_insecure=True)


@pytest.mark.parametrize(
    "variable", ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
)
def test_a_replaced_certificate_authority_is_refused_outside_development(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "/tmp/evil-ca.pem")  # noqa: S108
    with pytest.raises(ForwardConfigError, match=variable):
        slack(monkeypatch, "https://hooks.slack.com" + SLACK_PATH)
    slack(monkeypatch, "https://hooks.slack.com" + SLACK_PATH, allow_insecure=True)


def test_the_environment_is_read_once_at_construction(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    channel = slack(monkeypatch, server.url + SLACK_PATH, allow_insecure=True)
    monkeypatch.setenv("CVPILOT_ALERT_SLACK_URL", "http://evil.invalid/x")
    channel.send([alert()])
    assert server.paths() == [SLACK_PATH]


# -- payloads ---------------------------------------------------------------------------------


def test_the_webhook_posts_the_alert_fields_and_the_bearer_token(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_URL", server.url + "/hook")
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_TOKEN", HOOK_BEARER)
    WebhookChannel(allow_insecure=True).send([alert()])
    (got,) = server.requests
    assert got.json() == {
        "source": "cvpilot",
        "rule": "nan_loss",
        "project": "proj",
        "run_id": RUN,
        "step": 1,
        "text": "[proj] Loss loss is NaN at step 1 (run X).",
    }
    assert got.headers["authorization"] == f"Bearer {HOOK_BEARER}"


def test_the_webhook_sends_no_authorization_without_a_token(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_URL", server.url + "/hook")
    WebhookChannel(allow_insecure=True).send([alert(step=None)])
    assert "authorization" not in server.requests[0].headers
    assert server.requests[0].json()["step"] is None


@pytest.mark.parametrize(
    "token", ["short", "has space 0123456789", "line\nbreak0123456789", "x" * 600, "ключ0123456789"]
)
def test_a_webhook_token_that_could_inject_a_header_is_refused(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_URL", "https://hooks.example/x")
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_TOKEN", token)
    with pytest.raises(ForwardConfigError, match="CVPILOT_ALERT_WEBHOOK_TOKEN") as info:
        WebhookChannel()
    assert token not in str(info.value)


def test_the_slack_text_is_escaped_and_never_interpreted(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    channel = slack(monkeypatch, server.url + SLACK_PATH, allow_insecure=True)
    channel.send([alert("a <!channel> <https://evil.example|click> & b > c")])
    (got,) = server.requests
    assert got.path == SLACK_PATH
    assert got.json() == {
        "text": "a &lt;!channel&gt; &lt;https://evil.example|click&gt; &amp; b &gt; c",
        "mrkdwn": False,
    }


def test_telegram_sends_plain_text_to_the_bot_api_path(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_TOKEN", TG_BOT)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_CHAT", "-1001234567890")
    TelegramChannel(allow_insecure=True, base_url=server.url).send([alert("*bold* <b>x</b> _y_")])
    (got,) = server.requests
    assert got.path == f"/bot{TG_BOT}/sendMessage"
    assert got.json() == {
        "chat_id": -1001234567890,
        "text": "*bold* <b>x</b> _y_",
        "disable_web_page_preview": True,
    }
    assert "parse_mode" not in got.json()  # nothing in the text is parsed as markup


def test_a_telegram_channel_name_stays_a_string(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_TOKEN", TG_BOT)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_CHAT", "@my_alert_channel")
    TelegramChannel(allow_insecure=True, base_url=server.url).send([alert()])
    assert server.requests[0].json()["chat_id"] == "@my_alert_channel"


@pytest.mark.parametrize(
    "token",
    ["", "abc", "123:short", "12345678:ok-but/has/slash0123456789abcdefghijkl", TG_BOT + "\n"],
)
def test_a_malformed_telegram_token_is_refused_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_TOKEN", token)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_CHAT", "42")
    with pytest.raises(ForwardConfigError) as info:
        TelegramChannel()
    assert token.strip() not in str(info.value) or token.strip() == ""


@pytest.mark.parametrize("chat", ["", "abc", "1; DROP", "@a", "12.5", "-", "9" * 40, "@bad name"])
def test_a_malformed_telegram_chat_is_refused(monkeypatch: pytest.MonkeyPatch, chat: str) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_TOKEN", TG_BOT)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_CHAT", chat)
    with pytest.raises(ForwardConfigError, match="CVPILOT_ALERT_TELEGRAM_CHAT"):
        TelegramChannel()


def test_the_telegram_endpoint_is_fixed_and_a_base_url_is_for_development_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_TOKEN", TG_BOT)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_CHAT", "42")
    with pytest.raises(ForwardConfigError):
        TelegramChannel(base_url="https://evil.example")
    TelegramChannel()  # the real host
    with pytest.raises(ForwardConfigError, match="CVPILOT_FORWARD_HOSTS"):
        TelegramChannel(allowed_hosts=["hooks.slack.com"])
    TelegramChannel(allowed_hosts=["api.telegram.org"])


# -- secrets never leave the channel ---------------------------------------------------------


def test_a_channel_never_shows_its_endpoint_in_repr_or_pickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_TOKEN", TG_BOT)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_CHAT", "42")
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_URL", "https://hooks.example" + SLACK_PATH)
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_TOKEN", HOOK_BEARER)
    for channel in (TelegramChannel(), WebhookChannel()):
        text = repr(channel) + str(channel)
        assert "SECRET" not in text and TG_BOT not in text and HOOK_BEARER not in text
        with pytest.raises(TypeError):
            pickle.dumps(channel)


def test_delivery_logs_the_channel_and_rule_and_never_the_endpoint(
    monkeypatch: pytest.MonkeyPatch, server: Listener, logs: Any
) -> None:
    channel = slack(monkeypatch, server.url + SLACK_PATH, allow_insecure=True)
    channel.send([alert()])
    delivered = [r for r in logs.records() if r["message"] == "alert_delivered"]
    assert len(delivered) == 1
    assert (delivered[0]["channel"], delivered[0]["rule"]) == ("slack", "nan_loss")
    assert delivered[0]["run_id"] == RUN and delivered[0]["level"] == "INFO"
    assert "SECRET" not in logs.text() and SLACK_PATH not in logs.text()


def test_a_failed_delivery_through_the_fan_out_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch, logs: Any, tmp_path: Any
) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(_http, "_sleep", lambda s: None)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_TOKEN", TG_BOT)
    monkeypatch.setenv("CVPILOT_ALERT_TELEGRAM_CHAT", "42")
    channel = TelegramChannel(allow_insecure=True, base_url=f"http://127.0.0.1:{port}")
    meta, repro = meta_repro()
    fan = fw.notify_fanout([channel], meta, repro)
    fan.start()
    fan.submit([alert()])
    stats = fan.close("finished")
    assert stats["telegram"].failed_calls == 1
    assert "AAH" not in logs.text() and "secret_TOKEN" not in logs.text()
    assert any(r["message"] == "alert_delivery_failed" for r in logs.records())


# -- end to end through the fan-out and the registry ------------------------------------------


def test_every_channel_delivers_the_same_alert_and_a_dead_one_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch, server: Listener
) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    monkeypatch.setattr(_http, "_sleep", lambda s: None)
    monkeypatch.setenv("CVPILOT_ALERT_WEBHOOK_URL", f"http://127.0.0.1:{dead_port}/dead")
    dead = WebhookChannel(allow_insecure=True)
    live = slack(monkeypatch, server.url + SLACK_PATH, allow_insecure=True)
    meta, repro = meta_repro()
    fan = fw.notify_fanout([dead, live], meta, repro)
    fan.start()
    for n in range(3):
        fan.submit([alert(f"[proj] message {n}")])
    stats = fan.close("finished")
    assert stats["webhook"].failed_calls == 3 and stats["slack"].failed_calls == 0
    assert [r.json()["text"] for r in server.requests] == [f"[proj] message {n}" for n in range(3)]


def test_the_first_party_channels_are_registered_without_importing_anything_else() -> None:
    assert registry.channels.names() == ["slack", "telegram", "webhook"]
    for name in ("slack", "telegram", "webhook"):
        factory = registry.channels.get(name)
        assert factory.name == name and factory.requires == ()  # type: ignore[attr-defined]
