"""Out-of-band authentication of checkpoint manifests and pointers (HMAC-SHA256).

Digests and chains inside the checkpoint directory protect against damage, not against someone
who can write the directory: they can recompute all of them. A shared secret they do not have
closes that gap, and unlike the run-store digest it works on a fresh machine (the normal
spot-instance resume), because the key is carried in the environment, not in a database.

``CVPILOT_CKPT_KEY`` (32 to 1024 bytes, read from the environment only) keys the tags. A tag
covers a domain label, the run id and the canonical bytes of the record, so a manifest cannot be
replayed as a pointer or moved to another run. The key is never logged, stored, put in a
manifest or repeated in an error message. Rotation: tags made under an old key fail under the
new one (``auth_invalid``); re-save to re-tag. Files saved before a key existed carry no tag and
are only accepted under the ``any`` policy (as ``local_chain_only``).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
import os
import re
from collections import Counter
from collections.abc import Mapping

from cvpilot.core.errors import CheckpointError

KEY_ENV = "CVPILOT_CKPT_KEY"
MIN_KEY_BYTES = 32
MAX_KEY_BYTES = 1024
MANIFEST_LABEL = b"cvpilot-ckpt-manifest-v1"
POINTER_LABEL = b"cvpilot-ckpt-pointer-v1"
MARKER_LABEL = b"cvpilot-ckpt-marker-v1"
_HEX = re.compile(r"[0-9a-f]{64}")
_HEX_KEY = re.compile(r"[0-9a-fA-F]+")
_B64_KEY = re.compile(r"[A-Za-z0-9+/_-]+={0,2}")
MIN_ENTROPY_BITS = 3.0


def load_key(environ: Mapping[str, str] | None = None) -> bytes | None:
    """The key from ``CVPILOT_CKPT_KEY``, or None when unset or blank.

    Raises:
        CheckpointError: the key is shorter than 32 or longer than 1024 bytes (the message
            never contains the value).
    """
    raw = (os.environ if environ is None else environ).get(KEY_ENV)
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    key = text.encode("utf-8")
    if not MIN_KEY_BYTES <= len(key) <= MAX_KEY_BYTES:
        raise CheckpointError(
            f"{KEY_ENV} must be {MIN_KEY_BYTES} to {MAX_KEY_BYTES} bytes long "
            "(for example the output of: openssl rand -hex 32)"
        )
    if _HEX_KEY.fullmatch(text) and len(text) < 2 * MIN_KEY_BYTES:
        raise CheckpointError(
            f"{KEY_ENV} looks like hex but holds fewer than {MIN_KEY_BYTES} bytes of key "
            f"material: use at least {2 * MIN_KEY_BYTES} hex characters "
            "(openssl rand -hex 32)"
        )
    if "=" in text and _B64_KEY.fullmatch(text):
        try:
            decoded = base64.b64decode(text, altchars=b"-_" if "-" in text or "_" in text else None)
        except (binascii.Error, ValueError):
            decoded = key
        if len(decoded) < MIN_KEY_BYTES:
            raise CheckpointError(
                f"{KEY_ENV} looks like base64 but decodes to fewer than {MIN_KEY_BYTES} bytes "
                "of key material (openssl rand -base64 32)"
            )
    return key


def weakness(key: bytes) -> str | None:
    """Why a key looks guessable (a short repeated pattern, few distinct characters, low Shannon
    entropy per character), or None. A hint, never a refusal: the value is not echoed."""
    text = key.decode("utf-8", "replace")
    n = len(text)
    for size in range(1, n // 2 + 1):
        if n % size == 0 and text == text[:size] * (n // size):
            return "it repeats a short pattern"
    if len(set(text)) < 8:
        return "it uses very few distinct characters"
    counts = Counter(text)
    entropy = -sum(c / n * math.log2(c / n) for c in counts.values())
    if entropy < MIN_ENTROPY_BITS:
        return "its characters are not varied enough to look random"
    return None


STRICTNESS = ("any", "anchored", "authenticated")  # weakest first


def is_weaker(level: str, than: str) -> bool:
    """True when ``level`` demands less proof than ``than`` (unknown names raise ValueError)."""
    for name in (level, than):
        if name not in STRICTNESS:
            raise ValueError(f"require_trust must be one of {', '.join(reversed(STRICTNESS))}")
    return STRICTNESS.index(level) < STRICTNESS.index(than)


def resolve_policy(
    explicit: str | None, configured: str | None, key: bytes | None, *, allow_weaker: bool = False
) -> str:
    """The trust policy in force: the STRICTEST of the explicit argument,
    ``CVPILOT_CKPT_REQUIRE_TRUST`` and the implicit default (``authenticated`` when a key is
    configured, else ``any``). A caller can raise the level, never lower it.

    ``allow_weaker`` (the constructor-level acknowledgement) restores plain precedence:
    explicit, else configured, else the implicit default.
    """
    implicit = "authenticated" if key is not None else "any"
    given = [level for level in (explicit, configured) if level is not None]
    for level in given:
        is_weaker(level, implicit)  # validates the name
    if allow_weaker:
        return given[0] if given else implicit
    best = implicit
    for level in given:
        if is_weaker(best, level):
            best = level
    return best


def tag(key: bytes, label: bytes, run_id: str, data: bytes) -> str:
    """Hex HMAC-SHA256 over ``label``, the run id and ``data`` (NUL separated)."""
    message = label + b"\0" + run_id.encode("ascii", "replace") + b"\0" + data
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify(key: bytes, label: bytes, run_id: str, data: bytes, candidate: object) -> bool:
    """Constant-time check of a tag; anything that is not 64 lower-case hex fails."""
    if not isinstance(candidate, str) or not _HEX.fullmatch(candidate):
        return False
    return hmac.compare_digest(tag(key, label, run_id, data), candidate)
