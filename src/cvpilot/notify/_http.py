"""The one outbound call of the alert channels: a bounded JSON POST over https.

Rules (each has a test in ``tests/unit/test_notify_http.py``):

* **https only**; ``http`` is accepted only with ``allow_insecure`` (``CVPILOT_ENV=dev``), which
  also skips the address policy, so a developer can listen on the loopback interface.
* **Public addresses only.** The host name is resolved once, every address must be public (not
  loopback, private, link-local, shared, multicast, reserved, or an IPv4 address embedded in an
  IPv6 one), and the connection goes to the address that was checked, not to a second lookup:
  a name that later resolves elsewhere cannot reach an internal service. One refused address
  refuses the name.
* **No redirects**: a 3xx answer is an error and ``Location`` is never read.
* **Bounded time**: a total deadline of ``timeout_s`` per attempt, enforced by closing the socket
  (a response dripped one byte at a time cannot outlast it), at most ``MAX_ATTEMPTS`` attempts
  (connection errors, 429 and 5xx are retried once, other answers are not).
* **Bounded reading**: only the status line and headers are read (the standard library bounds
  both); the response body is never read, so a huge answer costs nothing.
* **Constant error text**: a message never contains the URL, its path or a header, because the
  path of a Slack-style webhook or a Telegram bot is a credential.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import ssl
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from cvpilot.core.errors import CvpilotError

TIMEOUT_S = 5.0
"""Total time of one attempt: connect, handshake, request and response headers."""
MAX_ATTEMPTS = 2
RETRY_PAUSE_S = 0.5
MAX_BODY_BYTES = 16 * 1024
MAX_ADDRESSES = 4
"""Most addresses of one name that are tried."""
_sleep = time.sleep


class DeliveryError(CvpilotError, OSError):
    """An alert could not be delivered; the message is constant and never names the endpoint."""


class AddressRefused(DeliveryError):
    """The host name resolved to an address that is not public."""


class RedirectRefused(DeliveryError):
    """The endpoint answered with a redirect, which is never followed."""


class EndpointRefused(DeliveryError):
    """The endpoint rejected the request (not retried), or the URL is not https."""


class EndpointUnavailable(DeliveryError):
    """The endpoint is busy or broke the protocol (retried once)."""


def _public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _public(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            return _public(ip.sixtofour)
        if ip.teredo is not None:
            return False
        if ip in ipaddress.ip_network(
            "64:ff9b::/96"
        ):  # NAT64: the last 32 bits are an IPv4 address
            return _public(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return ip.is_global and not ip.is_multicast


def public_addresses(infos: Sequence[tuple[Any, ...]]) -> list[str]:
    """The distinct addresses of a ``getaddrinfo`` answer, if every one of them is public.

    Raises:
        AddressRefused: there is no address, or any address is not public.
    """
    out: list[str] = []
    for info in infos:
        text = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(text)
        except ValueError:
            raise AddressRefused("the host resolved to something that is not an address") from None
        if not _public(ip):
            raise AddressRefused("the host resolves to an address that is not public")
        if text not in out:
            out.append(text)
    if not out:
        raise AddressRefused("the host did not resolve")
    return out[:MAX_ADDRESSES]


def _connect(
    host: str, port: int, scheme: str, *, allow_insecure: bool, tls: ssl.SSLContext | None,
    deadline: float,
) -> socket.socket:  # fmt: skip
    if allow_insecure:
        targets = [host]
    else:
        targets = public_addresses(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
    last: OSError | None = None
    for target in targets:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            sock = socket.create_connection((target, port), timeout=remaining)
        except OSError as exc:
            last = exc
            continue
        if scheme == "https":
            try:
                sock = (tls or ssl.create_default_context()).wrap_socket(sock, server_hostname=host)
            except BaseException:
                sock.close()
                raise
        return sock
    raise last or TimeoutError("the connection deadline passed")


def _abort(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def _once(
    scheme: str, host: str, port: int, path: str, body: bytes, headers: Mapping[str, str],
    *, allow_insecure: bool, tls: ssl.SSLContext | None, timeout_s: float,
) -> None:  # fmt: skip
    deadline = time.monotonic() + timeout_s
    sock = _connect(host, port, scheme, allow_insecure=allow_insecure, tls=tls, deadline=deadline)
    expired = threading.Event()

    def on_deadline() -> None:
        expired.set()
        _abort(sock)

    timer = threading.Timer(max(0.0, deadline - time.monotonic()), on_deadline)
    timer.daemon = True
    timer.start()
    conn: http.client.HTTPConnection
    if scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout_s, context=tls)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    conn.sock = sock
    try:
        conn.request("POST", path, body, headers=dict(headers))
        status = conn.getresponse().status  # the body is never read
    except http.client.HTTPException:
        raise EndpointUnavailable("the endpoint sent a malformed or oversized answer") from None
    finally:
        timer.cancel()
        conn.close()
        _abort(sock)
    if expired.is_set():  # a cut-off answer reads as a short one: it is a timeout, not a success
        raise TimeoutError("the delivery deadline passed")
    if 200 <= status < 300:
        return
    if 300 <= status < 400:
        raise RedirectRefused("the endpoint answered with a redirect; redirects are not followed")
    if status == 429 or status >= 500:
        raise EndpointUnavailable("the endpoint is busy or failing")
    raise EndpointRefused("the endpoint rejected the alert")


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None = None,
    allow_insecure: bool = False,
    tls: ssl.SSLContext | None = None,
    timeout_s: float = TIMEOUT_S,
    attempts: int = MAX_ATTEMPTS,
) -> None:
    """POST ``payload`` as JSON; return on a 2xx answer.

    ``url`` must already have passed ``cvpilot.forward._common.check_endpoint``; this function
    re-checks only what it relies on (the scheme). ``tls`` is the context to verify the peer with
    (default: the system's, with host name checking).

    Raises:
        DeliveryError: refused (a non-public address, a redirect, a 4xx answer, http outside
            development) or, after the last attempt, busy (429, 5xx, a malformed answer).
        OSError: after the last attempt, the connection failed or the deadline passed.
        ValueError: the body is too large or a header value contains a line break.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme != "https" and not (scheme == "http" and allow_insecure):
        raise EndpointRefused("alerts are delivered over https only")
    host = parts.hostname or ""
    port = parts.port or (443 if scheme == "https" else 80)
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("the alert body is too large")
    sent = {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": "cvpilot-alerts",
        **(headers or {}),
    }
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        if attempt:
            _sleep(RETRY_PAUSE_S)
        try:
            _once(
                scheme, host, port, path, body, sent,
                allow_insecure=allow_insecure, tls=tls, timeout_s=timeout_s,
            )  # fmt: skip
            return
        except (RedirectRefused, EndpointRefused, AddressRefused):
            raise
        except OSError as exc:  # a refused or reset connection, a timeout, EndpointUnavailable
            last = exc
    raise last if last is not None else EndpointUnavailable("no delivery attempt was made")
