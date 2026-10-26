"""Cloud preemption notices: AWS spot interruption and GCP preemption, read from the instance
metadata service.

Optional, off by default, and deliberately small: each adapter talks to one fixed link-local
(or loopback, for tests) host over plain HTTP, which is how metadata services are reached, with
a timeout of at most 10 seconds, no redirects, no credentials, and answers read only up to a few
kilobytes. The URL is never taken from configuration, so there is nothing to point elsewhere.
``check`` returns True when a notice is pending, False when there is none and raises ``OSError``
for anything unexpected (``NoticeUnavailable`` for 401 and 403); the poller (``ckpt/preempt.py``)
retries with backoff and reports it. AWS counts only a JSON answer whose ``action`` is terminate,
stop or hibernate; both adapters use the link-local address, so no DNS answer can redirect them.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import urllib.parse

from cvpilot.core.errors import NoticeUnavailable

MAX_BODY = 4096
MAX_TIMEOUT_S = 10.0
_ALLOWED_HOSTS = {"169.254.169.254", "metadata.google.internal", "127.0.0.1", "localhost"}
_LOOPBACK = {"127.0.0.1", "localhost"}


def _parse(base_url: str) -> tuple[str, int]:
    parts = urllib.parse.urlsplit(base_url)
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        raise ValueError("invalid metadata endpoint port") from None
    if (
        parts.scheme != "http"
        or host not in _ALLOWED_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or (port is not None and host not in _LOOPBACK)
    ):
        raise ValueError("the metadata endpoint must be a fixed metadata host (http, no port)")
    return host, port or 80


class _Metadata:
    def __init__(self, base_url: str, timeout: float) -> None:
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_TIMEOUT_S:
            raise ValueError(f"timeout must be in (0, {MAX_TIMEOUT_S}] seconds")
        if isinstance(timeout, float) and math.isnan(timeout):
            raise ValueError("timeout must be a number")
        self._host, self._port = _parse(base_url)
        self._timeout = float(timeout)

    def _request(self, method: str, path: str, headers: dict[str, str]) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request(method, path, headers=headers)
            response = conn.getresponse()
            body = response.read(MAX_BODY + 1)
        except (http.client.HTTPException, TimeoutError, ValueError) as exc:
            # ValueError: http.client refuses a header value with a control character
            raise OSError(f"metadata request failed: {type(exc).__name__}") from None
        finally:
            conn.close()
        if len(body) > MAX_BODY:
            raise OSError("the metadata answer is too large")
        return response.status, body


_TOKEN = re.compile(rb"[A-Za-z0-9_=-]{1,512}")

INTERRUPTION_ACTIONS = ("terminate", "stop", "hibernate")


def _interruption(body: bytes) -> bool:
    """A 200 from ``instance-action`` is a notice only when it says what AWS will do: a JSON
    object whose ``action`` is terminate, stop or hibernate. Anything else (an empty page, a
    captive portal, another JSON) is an unexpected answer, not a reason to stop training."""
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise OSError("the spot instance-action answer is not JSON") from None
    if not isinstance(document, dict) or document.get("action") not in INTERRUPTION_ACTIONS:
        raise OSError("the spot instance-action answer names no known action")
    return True


class AwsSpotNotice(_Metadata):
    """True when the instance has a pending spot interruption (IMDSv2, token per poll)."""

    name = "aws"

    def __init__(self, *, base_url: str = "http://169.254.169.254", timeout: float = 1.0) -> None:
        super().__init__(base_url, timeout)

    def check(self) -> bool:
        status, token = self._request(
            "PUT", "/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": "60"}
        )
        if status in (401, 403):
            raise NoticeUnavailable(f"metadata token request answered {status}", status)
        if status != 200 or not token or len(token) > 512:
            raise OSError(f"metadata token request answered {status}")
        if not _TOKEN.fullmatch(token):
            # the answer is untrusted text that is sent back as a header value: only the
            # characters IMDS tokens use, so a CR LF cannot inject a header
            raise OSError("the metadata token has characters a token never has")
        status, body = self._request(
            "GET",
            "/latest/meta-data/spot/instance-action",
            {"X-aws-ec2-metadata-token": token.decode("ascii")},
        )
        if status == 200:
            return _interruption(body)
        if status == 404:
            return False
        if status in (401, 403):
            raise NoticeUnavailable(f"spot instance-action answered {status}", status)
        raise OSError(f"spot instance-action answered {status}")


class GcpPreemptionNotice(_Metadata):
    """True when the instance reports ``preempted: TRUE``."""

    name = "gcp"

    def __init__(self, *, base_url: str = "http://169.254.169.254", timeout: float = 1.0) -> None:
        # the link-local address, not the name: no DNS lookup an attacker on the network could
        # answer; the Host header still names the service, as the metadata server expects
        super().__init__(base_url, timeout)

    def check(self) -> bool:
        status, body = self._request(
            "GET",
            "/computeMetadata/v1/instance/preempted",
            {"Metadata-Flavor": "Google", "Host": "metadata.google.internal"},
        )
        if status != 200:
            raise OSError(f"preempted answered {status}")
        return body.strip() == b"TRUE"
