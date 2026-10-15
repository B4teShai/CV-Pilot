"""Text that came from data (class names, slice values, tags) made safe to print.

Names are validated on input (no ASCII control characters, bounded length), but Unicode format
characters (bidirectional overrides, zero-width joiners) pass that check and can reorder or hide
text in a terminal, a CI log or an HTML report. ``safe_text`` shows every control, format,
surrogate and private-use character as a ``\\uXXXX`` escape and bounds the length, so a name can
never change how the surrounding line reads.
"""

from __future__ import annotations

import unicodedata

MAX_DISPLAY_CHARS = 128
"""Longest name shown in a verdict line or report cell; longer text is cut with an ellipsis."""
_ESCAPED = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}


def safe_text(value: object, limit: int = MAX_DISPLAY_CHARS) -> str:
    """``value`` as printable text with unusual characters escaped, at most ``limit`` characters."""
    text = str(value)
    out: list[str] = []
    for char in text[: limit + 1]:
        if unicodedata.category(char) in _ESCAPED:
            out.append(f"\\u{ord(char):04x}" if ord(char) <= 0xFFFF else f"\\U{ord(char):08x}")
        else:
            out.append(char)
    shown = "".join(out)
    if len(text) > limit:
        return shown[:limit] + "..."
    return shown
