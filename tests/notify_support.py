"""A loopback HTTP listener with hostile behaviours: the only 'network' the alert tests use."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Received:
    def __init__(self, path: str, headers: dict[str, str], body: bytes) -> None:
        self.path, self.headers, self.body = path, headers, body

    def json(self) -> Any:
        return json.loads(self.body)


class Listener:
    """Binds 127.0.0.1 on a free port. ``mode`` decides how every request is answered; ``modes``
    (a list) answers the first requests in order and then falls back to ``mode``."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.modes: list[str] = []
        self.requests: list[Received] = []
        self.release = threading.Event()
        listener = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                listener.requests.append(
                    Received(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
                )
                mode = listener.modes.pop(0) if listener.modes else listener.mode
                try:
                    listener.answer(self, mode)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def answer(self, h: BaseHTTPRequestHandler, mode: str) -> None:
        if mode == "ok":
            h.send_response(200)
            h.send_header("Content-Length", "2")
            h.send_header("Connection", "close")
            h.end_headers()
            h.wfile.write(b"ok")
        elif mode in ("400", "429", "500", "503"):
            h.send_response(int(mode))
            h.send_header("Content-Length", "0")
            h.send_header("Connection", "close")
            h.end_headers()
        elif mode == "redirect":
            h.send_response(302)
            h.send_header("Location", f"{self.url}/followed")
            h.send_header("Content-Length", "0")
            h.send_header("Connection", "close")
            h.end_headers()
        elif mode == "slow":  # never answers
            self.release.wait(30)
        elif mode == "drip":  # a header byte every 0.1 s, for as long as the client listens
            h.wfile.write(b"HTTP/1.1 200 OK\r\n")
            for _ in range(600):
                h.wfile.write(b"X-Drip: a\r\n")
                h.wfile.flush()
                time.sleep(0.1)
        elif mode == "huge_header":
            h.wfile.write(b"HTTP/1.1 200 OK\r\nX-Big: " + b"a" * 300_000 + b"\r\n\r\n")
        elif mode == "huge_body":
            h.send_response(200)
            h.send_header("Content-Length", str(10**9))
            h.end_headers()
            chunk = b"x" * 65536
            for _ in range(400):
                h.wfile.write(chunk)
                h.wfile.flush()
        else:  # pragma: no cover
            raise AssertionError(mode)

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()

    def paths(self) -> list[str]:
        return [r.path for r in self.requests]
