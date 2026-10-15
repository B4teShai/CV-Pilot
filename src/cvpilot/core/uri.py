"""Remote checkpoint locations: a small allow-list grammar, no credentials, no surprises.

A remote is ``<scheme>://<bucket>/<prefix>`` with a scheme from ``REMOTE_SCHEMES``. Everything
that could smuggle something else into a URL is refused: user info (``user:secret@host``), a
query or fragment (signed URLs carry their secret there), percent-encoding, backslashes,
control characters and whitespace, empty or dot path segments, and non-ASCII text. Credentials
come from the storage SDK's own environment variables, never from the location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REMOTE_SCHEMES = frozenset({"s3", "gs", "az", "memory"})
MAX_URI_CHARS = 1024
MAX_SEGMENTS = 32
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,62}")
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=-]{0,127}")


@dataclass(frozen=True)
class RemoteLocation:
    """A validated remote: ``scheme``, ``bucket`` (empty for ``memory``) and ``prefix`` segments."""

    scheme: str
    bucket: str
    prefix: tuple[str, ...]

    @property
    def url(self) -> str:
        tail = "/".join(self.prefix)
        head = f"{self.scheme}://{self.bucket}"
        return f"{head}/{tail}" if tail else head

    def key(self, *parts: str) -> str:
        """``bucket/prefix/parts`` as the storage layer addresses it (no scheme)."""
        pieces = [p for p in (self.bucket, *self.prefix, *parts) if p]
        return "/".join(pieces)


def parse_remote_uri(value: object) -> RemoteLocation:
    """Validate ``value`` as a remote location.

    Raises:
        ValueError: the value is not a string of the allowed form (the message never repeats
            more than 40 characters of it, in case it held a secret).
    """
    if not isinstance(value, str) or not value or len(value) > MAX_URI_CHARS:
        raise ValueError("a remote must be a non-empty string of at most 1024 characters")
    if not value.isascii() or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        raise ValueError("a remote must be ASCII without spaces or control characters")
    if any(c in value for c in "@?#%\\"):
        raise ValueError(
            "a remote must not contain '@', '?', '#', '%' or a backslash: credentials and "
            "signed URLs belong in the environment, not in the location"
        )
    scheme, sep, rest = value.partition("://")
    if not sep or scheme not in REMOTE_SCHEMES:
        raise ValueError(
            f"unsupported remote scheme in {value[:40]!r}; use one of {sorted(REMOTE_SCHEMES)}"
        )
    segments = rest.split("/") if rest else []
    if scheme == "memory":
        bucket = ""
    else:
        if not segments or not _BUCKET.fullmatch(segments[0]):
            raise ValueError(f"invalid bucket name in {value[:40]!r}")
        bucket, segments = segments[0], segments[1:]
    if segments and segments[-1] == "":
        segments = segments[:-1]  # one trailing slash is harmless
    if len(segments) > MAX_SEGMENTS:
        raise ValueError("a remote path has too many segments")
    for segment in segments:
        if not _SEGMENT.fullmatch(segment) or segment in {".", ".."}:
            raise ValueError(f"invalid path segment in {value[:40]!r}")
    if scheme == "memory" and not segments:
        raise ValueError("a memory:// remote needs a path")
    return RemoteLocation(scheme, bucket, tuple(segments))
