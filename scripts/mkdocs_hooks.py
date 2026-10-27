"""MkDocs hook: let the site render the fenced blocks the doc tests read.

The tests use text after the language (``bash notest``, ``python file=train.py``, ``text expect``)
as markers. The site's Markdown engine only accepts its own attributes there and then shows such a
block as plain inline text, so the hook keeps just the language for the built page. The files on
disk, and the tests that read them, are untouched.
"""

from __future__ import annotations

import re
from typing import Any

# An opening fence with text after the language. A closing fence is a bare run of backticks.
OPENING_FENCE = re.compile(r"^(```[A-Za-z0-9_+-]*)[ \t]+[^\n`]*$", re.MULTILINE)


def on_page_markdown(markdown: str, **_: Any) -> str:
    return OPENING_FENCE.sub(r"\1", markdown)
