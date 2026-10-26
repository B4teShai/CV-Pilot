"""Cloud preemption notice adapters: fixed endpoints, short timeouts, bounded answers."""

from __future__ import annotations

import http.server
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from cvpilot import registry
from cvpilot.remote import spot


class Handler(http.server.BaseHTTPRequestHandler):
    routes: dict[tuple[str, str], tuple[int, bytes, dict[str, str]]] = {}
    seen: list[dict[str, Any]] = []
    delay = 0.0

    def log_message(self, *args: Any) -> None:
        return None

    def _answer(self, method: str) -> None:
        type(self).seen.append({"method": method, "path": self.path, "headers": dict(self.headers)})
        if type(self).delay:
            time.sleep(type(self).delay)
        status, body, headers = type(self).routes.get((method, self.path), (404, b"", {}))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._answer("GET")

    def do_PUT(self) -> None:
        self._answer("PUT")


@pytest.fixture
def server() -> Iterator[str]:
    Handler.routes, Handler.seen, Handler.delay = {}, [], 0.0
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def test_aws_reports_a_notice_with_an_imdsv2_token(server: str) -> None:
    Handler.routes[("PUT", "/latest/api/token")] = (200, b"tok-123", {})
    Handler.routes[("GET", "/latest/meta-data/spot/instance-action")] = (
        200, b'{"action": "terminate", "time": "2026-10-15T10:00:00Z"}', {}
    )  # fmt: skip
    assert spot.AwsSpotNotice(base_url=server).check() is True
    put, get = Handler.seen
    assert "X-aws-ec2-metadata-token-ttl-seconds" in put["headers"]
    assert get["headers"]["X-aws-ec2-metadata-token"] == "tok-123"


def test_aws_without_a_notice_is_false(server: str) -> None:
    Handler.routes[("PUT", "/latest/api/token")] = (200, b"tok", {})
    assert spot.AwsSpotNotice(base_url=server).check() is False  # 404 on instance-action


def test_aws_errors_are_raised_for_the_poller_to_count(server: str) -> None:
    Handler.routes[("PUT", "/latest/api/token")] = (500, b"", {})
    with pytest.raises(OSError):
        spot.AwsSpotNotice(base_url=server).check()
    Handler.routes[("PUT", "/latest/api/token")] = (200, b"t" * 5000, {})
    with pytest.raises(OSError):
        spot.AwsSpotNotice(base_url=server).check()  # an oversized token is not read


def test_gcp_reports_true_only_for_the_word_true(server: str) -> None:
    path = "/computeMetadata/v1/instance/preempted"
    Handler.routes[("GET", path)] = (200, b"TRUE", {})
    assert spot.GcpPreemptionNotice(base_url=server).check() is True
    assert Handler.seen[0]["headers"]["Metadata-Flavor"] == "Google"
    Handler.routes[("GET", path)] = (200, b"FALSE", {})
    assert spot.GcpPreemptionNotice(base_url=server).check() is False
    Handler.routes[("GET", path)] = (200, b"x" * 100_000, {})
    with pytest.raises(OSError):
        spot.GcpPreemptionNotice(base_url=server).check()


def test_redirects_are_not_followed(server: str) -> None:
    Handler.routes[("GET", "/computeMetadata/v1/instance/preempted")] = (
        302, b"", {"Location": "http://example.invalid/steal"}
    )  # fmt: skip
    with pytest.raises(OSError):
        spot.GcpPreemptionNotice(base_url=server).check()
    assert len(Handler.seen) == 1


def test_a_slow_endpoint_times_out_quickly(server: str) -> None:
    Handler.delay = 3.0
    started = time.monotonic()
    with pytest.raises(OSError):
        spot.GcpPreemptionNotice(base_url=server, timeout=0.3).check()
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://169.254.169.254",
        "http://169.254.169.254:8080",
        "http://user:pw@169.254.169.254",
        "http://169.254.169.254/path",
        "http://[::1]:80",
        "ftp://127.0.0.1",
        "",
    ],
)
def test_only_the_fixed_metadata_hosts_are_accepted(url: str) -> None:
    with pytest.raises(ValueError):
        spot.AwsSpotNotice(base_url=url)


def test_timeouts_are_bounded() -> None:
    for bad in (0, -1, 11, float("nan")):
        with pytest.raises(ValueError):
            spot.GcpPreemptionNotice(timeout=bad)


def test_the_adapters_are_registered_by_name() -> None:
    assert registry.spot_notices.names() == ["aws", "gcp"]
    assert registry.spot_notices.get("aws") is spot.AwsSpotNotice


ACTION = ("GET", "/latest/meta-data/spot/instance-action")
TOKEN = ("PUT", "/latest/api/token")


@pytest.mark.parametrize("action", ["terminate", "stop", "hibernate"])
def test_aws_accepts_each_documented_action(server: str, action: str) -> None:
    Handler.routes[TOKEN] = (200, b"tok", {})
    Handler.routes[ACTION] = (200, f'{{"action": "{action}", "time": "t"}}'.encode(), {})
    assert spot.AwsSpotNotice(base_url=server).check() is True


@pytest.mark.parametrize(
    "body",
    [b"", b"<html>captive portal</html>", b"{}", b'{"action": "reboot"}', b'["terminate"]',
     b'{"action": 7}', b"\xff\xfe", b"null"],
)  # fmt: skip
def test_aws_200_without_a_known_action_is_an_error_not_a_notice(server: str, body: bytes) -> None:
    Handler.routes[TOKEN] = (200, b"tok", {})
    Handler.routes[ACTION] = (200, body, {})
    with pytest.raises(OSError, match="instance-action"):
        spot.AwsSpotNotice(base_url=server).check()


@pytest.mark.parametrize("status", [401, 403])
def test_aws_401_and_403_mean_the_service_is_unavailable_not_that_there_is_no_notice(
    server: str, status: int
) -> None:
    from cvpilot.core.errors import NoticeUnavailable

    Handler.routes[TOKEN] = (200, b"tok", {})
    Handler.routes[ACTION] = (status, b"", {})
    with pytest.raises(NoticeUnavailable) as caught:
        spot.AwsSpotNotice(base_url=server).check()
    assert caught.value.status == status
    Handler.routes[TOKEN] = (status, b"", {})
    with pytest.raises(NoticeUnavailable):
        spot.AwsSpotNotice(base_url=server).check()


def test_aws_timeout_is_an_error(server: str) -> None:
    Handler.routes[TOKEN] = (200, b"tok", {})
    Handler.delay = 1.0
    with pytest.raises(OSError):
        spot.AwsSpotNotice(base_url=server, timeout=0.2).check()


def test_gcp_uses_the_link_local_address_so_no_dns_answer_can_redirect_it(server: str) -> None:
    assert spot.GcpPreemptionNotice()._host == "169.254.169.254"
    assert spot.AwsSpotNotice()._host == "169.254.169.254"
    Handler.routes[("GET", "/computeMetadata/v1/instance/preempted")] = (200, b"TRUE", {})
    assert spot.GcpPreemptionNotice(base_url=server).check() is True
    assert Handler.seen[0]["headers"]["Host"] == "metadata.google.internal"
    assert Handler.seen[0]["headers"]["Metadata-Flavor"] == "Google"


def test_the_named_gcp_host_is_still_accepted_when_asked_for_explicitly() -> None:
    assert spot.GcpPreemptionNotice(base_url="http://metadata.google.internal")._host == (
        "metadata.google.internal"
    )


@pytest.mark.parametrize(
    "token",
    [b"abc\r\nX-Injected: 1", b"has space", b"semi;colon", b"caf\xc3\xa9", b"a\x00b", b"tok/en"],
)
def test_a_token_that_is_not_plain_text_is_refused_as_an_oserror_before_it_is_used(
    server: str, token: bytes
) -> None:
    # the token comes from the metadata answer and is sent back as a header value: a CR LF in it
    # would inject a header (http.client raises ValueError, which must not escape the poller)
    Handler.routes[("PUT", "/latest/api/token")] = (200, token, {})
    notice = spot.AwsSpotNotice(base_url=server, timeout=2.0)
    with pytest.raises(OSError, match="token"):
        notice.check()
    assert [s["path"] for s in Handler.seen] == ["/latest/api/token"]  # nothing was sent with it


def test_a_header_value_that_the_client_library_rejects_becomes_an_oserror(
    server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import http.client

    Handler.routes[("PUT", "/latest/api/token")] = (200, b"validtoken", {})
    real = http.client.HTTPConnection.request

    def request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        if method == "GET":
            raise ValueError("Invalid header value")  # what http.client raises for a bad value
        return real(self, method, url, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPConnection, "request", request)
    notice = spot.AwsSpotNotice(base_url=server, timeout=2.0)
    with pytest.raises(OSError, match="ValueError"):
        notice.check()
