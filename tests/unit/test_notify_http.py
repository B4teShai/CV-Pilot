"""The one outbound call of the alert channels: bounded, no redirects, no private addresses."""

from __future__ import annotations

import ipaddress
import socket
import time
from collections.abc import Iterator
from typing import Any

import pytest
from notify_support import Listener

from cvpilot.notify import _http as http

TOKEN_PATH = "/services/T0000/B0000/XXXXSECRETXXXX"  # noqa: S105  # a marker, not a credential


@pytest.fixture
def server() -> Iterator[Listener]:
    listener = Listener()
    yield listener
    listener.close()


@pytest.fixture(autouse=True)
def no_pause(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    pauses: list[float] = []
    monkeypatch.setattr(http, "_sleep", pauses.append)
    return pauses


def post(url: str, **kw: Any) -> None:
    kw.setdefault("allow_insecure", True)
    http.post_json(url, {"text": "hello"}, **kw)


def test_the_documented_bounds_are_named_constants() -> None:
    assert 0 < http.TIMEOUT_S <= 10 and 1 <= http.MAX_ATTEMPTS <= 3
    assert http.MAX_ATTEMPTS * http.TIMEOUT_S + http.RETRY_PAUSE_S < 20  # inside the stall limit


def test_a_post_sends_json_with_the_headers_and_nothing_else(server: Listener) -> None:
    post(server.url + TOKEN_PATH, headers={"Authorization": "Bearer abc12345"})
    (got,) = server.requests
    assert got.path == TOKEN_PATH and got.json() == {"text": "hello"}
    assert got.headers["content-type"] == "application/json"
    assert got.headers["authorization"] == "Bearer abc12345"
    assert got.headers["connection"] == "close" and "cookie" not in got.headers
    assert got.headers["host"] == f"127.0.0.1:{server.port}"


def test_a_redirect_is_refused_and_never_followed(server: Listener) -> None:
    server.mode = "redirect"
    with pytest.raises(http.RedirectRefused):
        post(server.url + "/hook")
    assert server.paths() == ["/hook"]  # one request, no retry, the Location was not requested


@pytest.mark.parametrize(("mode", "error"), [("400", http.EndpointRefused)])
def test_a_client_error_is_refused_without_a_retry(
    server: Listener, mode: str, error: type[Exception]
) -> None:
    server.mode = mode
    with pytest.raises(error):
        post(server.url + "/hook")
    assert len(server.requests) == 1


@pytest.mark.parametrize("mode", ["429", "500", "503"])
def test_a_busy_endpoint_is_retried_once_then_reported(
    server: Listener, mode: str, no_pause: list[float]
) -> None:
    server.mode = mode
    with pytest.raises(http.EndpointUnavailable):
        post(server.url + "/hook")
    assert len(server.requests) == http.MAX_ATTEMPTS and no_pause == [http.RETRY_PAUSE_S]


def test_a_retry_that_succeeds_ends_the_delivery(server: Listener) -> None:
    server.modes = ["503"]
    post(server.url + "/hook")
    assert len(server.requests) == 2


def test_a_closed_port_fails_fast_and_is_retried(no_pause: list[float]) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    started = time.perf_counter()
    with pytest.raises(OSError):  # noqa: PT011
        post(f"http://127.0.0.1:{port}/hook")
    assert time.perf_counter() - started < 2 and len(no_pause) == http.MAX_ATTEMPTS - 1


def test_an_endpoint_that_never_answers_is_abandoned_within_the_bound(server: Listener) -> None:
    server.mode = "slow"
    started = time.perf_counter()
    with pytest.raises(OSError):  # noqa: PT011
        post(server.url + "/hook", timeout_s=0.4)
    elapsed = time.perf_counter() - started
    assert elapsed < http.MAX_ATTEMPTS * 0.4 + 1.5, elapsed


def test_a_response_dripped_forever_hits_the_total_deadline_not_the_per_read_timeout(
    server: Listener,
) -> None:
    server.mode = "drip"  # a byte every 0.1 s resets a per-read timeout of 0.4 s forever
    started = time.perf_counter()
    with pytest.raises((OSError, http.DeliveryError)):
        post(server.url + "/hook", timeout_s=0.5, attempts=1)
    assert time.perf_counter() - started < 3.0


def test_an_oversized_header_is_refused_and_an_oversized_body_is_never_read(
    server: Listener,
) -> None:
    server.mode = "huge_header"
    with pytest.raises(Exception) as info:  # noqa: PT011  # http.client.LineTooLong
        post(server.url + "/hook", attempts=1)
    assert "LineTooLong" in type(info.value).__name__ or isinstance(info.value, OSError)
    server.mode = "huge_body"
    started = time.perf_counter()
    post(server.url + "/hook", attempts=1)  # 2xx: the body is not read, the socket is closed
    assert time.perf_counter() - started < 2


def test_errors_never_carry_the_url_or_its_path() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(OSError) as info:  # noqa: PT011
        post(f"http://127.0.0.1:{port}{TOKEN_PATH}")
    assert "SECRET" not in str(info.value) and TOKEN_PATH not in str(info.value)
    assert "SECRET" not in repr(info.value)


def test_a_header_with_a_line_break_is_refused(server: Listener) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        post(server.url + "/hook", headers={"Authorization": "Bearer a\r\nX-Evil: 1"}, attempts=1)
    assert server.requests == []


# -- the address policy (production path: https only, public addresses only) -----------------

PRIVATE = [
    "127.0.0.1",
    "127.1.2.3",
    "10.1.2.3",
    "172.16.0.1",
    "192.168.1.1",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",  # noqa: S104
    "255.255.255.255",
    "224.0.0.1",
    "::1",
    "::",
    "fe80::1",
    "fc00::1",
    "fd12:3456::1",
    "::ffff:127.0.0.1",
    "::ffff:10.0.0.1",
    "::ffff:169.254.169.254",
    "2002:7f00:0001::",  # 6to4 of 127.0.0.1
    "2002:a9fe:a9fe::",  # 6to4 of 169.254.169.254
    "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
]
PUBLIC = ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8"]


def infos(*addresses: str) -> list[tuple[Any, ...]]:
    out = []
    for address in addresses:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        out.append((family, socket.SOCK_STREAM, 6, "", (address, 443)))
    return out


@pytest.mark.parametrize("address", PRIVATE)
def test_a_private_or_special_address_is_refused_before_any_connection(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: infos(address))

    def forbidden(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("connected to a refused address")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    with pytest.raises(http.AddressRefused):
        http.post_json("https://hooks.example/x", {"text": "t"})


def test_one_private_answer_among_public_ones_refuses_the_whole_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: infos("8.8.8.8", "10.0.0.5"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: 1 / 0)
    with pytest.raises(http.AddressRefused):
        http.post_json("https://hooks.example/x", {"text": "t"})


@pytest.mark.parametrize("address", PUBLIC)
def test_public_addresses_pass_the_policy(address: str) -> None:
    assert http.public_addresses(infos(address)) == [address]
    assert ipaddress.ip_address(address).version in (4, 6)


def test_the_name_is_resolved_once_and_the_connection_goes_to_that_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: list[str] = []
    connected: list[Any] = []

    def getaddrinfo(host: str, port: int, **_k: Any) -> list[tuple[Any, ...]]:
        resolved.append(host)
        return infos("93.184.216.34")

    def create_connection(address: tuple[str, int], timeout: float | None = None, **_k: Any) -> Any:
        connected.append(address)
        raise ConnectionRefusedError

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    with pytest.raises(ConnectionRefusedError):
        http.post_json("https://hooks.example/x", {"text": "t"}, attempts=1)
    assert resolved == ["hooks.example"] and connected == [("93.184.216.34", 443)]  # the pinned IP


def test_development_allows_loopback_but_production_does_not(server: Listener) -> None:
    post(server.url + "/hook")  # allow_insecure=True
    assert len(server.requests) == 1
    with pytest.raises(http.DeliveryError):
        http.post_json(server.url + "/hook", {"text": "t"})  # http in production: refused here too
    assert len(server.requests) == 1


def test_an_unresolvable_name_is_an_ordinary_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_a: Any, **_k: Any) -> Any:
        raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(OSError):  # noqa: PT011
        http.post_json("https://hooks.example/x", {"text": "t"}, attempts=1)
