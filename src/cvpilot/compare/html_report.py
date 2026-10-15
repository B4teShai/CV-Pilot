"""Static HTML report of a comparison and an optional gate result.

Safety by construction: the page is built from a few element constructors that escape every
string child (``esc`` also runs ``safe_text``, so control and bidirectional characters appear
as ``\\uXXXX``); the only markup that is not escaped is what these constructors emit. There is
no script, no external resource, no link, no attribute that carries data and no inline style
attribute. The page declares a Content-Security-Policy whose only allowance is the one inline
``<style>`` block, by its SHA-256, and ``no-referrer``.

Severity first, then caps: violations come first (worst first), then the headline table, then
per-class tables sorted worst first (unusable rows before measured ones), slices and confusion
cells. Every table is capped by a named constant and says how many rows it left out, so a
hostile snapshot cannot produce an unbounded page and what is cut is the least severe part.
The page contains no timestamp: the same input always gives the same bytes.
"""

from __future__ import annotations

import base64
import hashlib
import html
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

from cvpilot.compare.engine import ClassDelta, Comparison, MetricDelta, SliceDelta, SnapshotRef
from cvpilot.compare.gate import GateResult, Thresholds
from cvpilot.compare.text import MAX_DISPLAY_CHARS, safe_text
from cvpilot.core.errors import CvpilotError

MAX_VIOLATIONS = 100
"""Violations listed in the page, worst first."""
MAX_CLASS_ROWS = 50
"""Rows of each per-class table, worst first."""
MAX_SLICE_ROWS = 100
"""Rows of the slice table, worst first."""
MAX_CELL_ROWS = 30
"""Changed confusion cells listed, largest change first."""
MAX_REPORT_BYTES = 8 * 1024 * 1024
"""Largest report file written."""

_CSS = (
    "body{font:14px/1.4 system-ui,sans-serif;margin:1.5rem;max-width:72rem}"
    "table{border-collapse:collapse;margin:.5rem 0 1.5rem}"
    "th,td{border:1px solid #999;padding:.2rem .5rem;text-align:left}"
    "th{background:#eee}.bad{font-weight:700}.note{color:#444}"
)
_STYLE_HASH = base64.b64encode(hashlib.sha256(_CSS.encode("utf-8")).digest()).decode("ascii")
CSP = f"default-src 'none'; style-src 'sha256-{_STYLE_HASH}'; base-uri 'none'; form-action 'none'"
_TAGS = frozenset(
    {"html", "head", "body", "title", "h1", "h2", "p", "table", "tr", "th", "td", "div", "code"}
)
_CLASSES = frozenset({"bad", "note"})


class Markup(str):
    """Text already safe to embed; only the element constructors below create it."""

    __slots__ = ()


PROSE_CHARS = 2000
"""Longest paragraph; table cells keep the shorter ``safe_text`` default."""


def esc(value: object, limit: int = MAX_DISPLAY_CHARS) -> Markup:
    return Markup(html.escape(safe_text(value, limit), quote=True))


def el(name: str, *children: object, cls: str | None = None) -> Markup:
    if name not in _TAGS or (cls is not None and cls not in _CLASSES):
        raise ValueError("unknown element")  # constants only; a bug, never data
    limit = MAX_DISPLAY_CHARS if name in ("td", "th") else PROSE_CHARS
    inner = "".join(c if isinstance(c, Markup) else esc(c, limit) for c in children)
    attr = f' class="{cls}"' if cls else ""
    return Markup(f"<{name}{attr}>{inner}</{name}>")


def _table(head: list[str], rows: list[list[object]]) -> Markup:
    header = el("tr", *(el("th", h) for h in head))
    body = [el("tr", *(el("td", cell) for cell in row)) for row in rows]
    return el("table", header, *body)


def _num(value: float | None, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec) if value == value and abs(value) != float("inf") else "not finite"


def _more(total: int, shown: int) -> Markup:
    if total <= shown:
        return Markup("")
    return el(
        "p",
        f"{total - shown} more not shown (worst first); the JSON report has them all.",
        cls="note",
    )


def _ref(label: str, ref: SnapshotRef) -> Markup:
    return el(
        "p",
        f"{label}: run {ref.run_id}, step {ref.step}, eval set {ref.eval_set_hash[:12]}, "
        f"{ref.n_images} images, {ref.task}",
    )


def _limits(limits: Mapping[str, float]) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in sorted(limits.items())) or "none"


def _violations(result: GateResult, thresholds: Thresholds | None) -> list[Markup]:
    verdict = "PASS" if result.passed else "FAIL"
    out = [
        el("h2", "Violations"),
        el(
            "p",
            f"Gate {verdict}: {len(result.violations)} violations, {result.n_checks} checks, "
            f"{len(result.skipped)} slices skipped.",
            cls=None if result.passed else "bad",
        ),
    ]
    if thresholds is not None:
        out.append(
            el(
                "p",
                "Limits in points on the 0..100 scale (drop = (baseline - candidate) * 100). "
                f"max-drop: {_limits(thresholds.max_drop)}; per-class: "
                f"{_limits(thresholds.per_class_max_drop)}; slice: "
                f"{_limits(thresholds.slice_max_drop)}; min-images: {thresholds.min_images}.",
                cls="note",
            )
        )
    shown = result.violations[:MAX_VIOLATIONS]
    if shown:
        out.append(
            _table(
                ["check", "subject", "detail"],
                [[f"[{v.kind}]", v.subject, v.message] for v in shown],
            )
        )
    out.append(_more(len(result.violations), len(shown)))
    if result.skipped:
        out.append(el("p", "Skipped slices: " + ", ".join(result.skipped), cls="note"))
    return out


def _headline(rows: tuple[MetricDelta, ...]) -> list[Markup]:
    words = {"overlap": "overlap", "separated": "separated", "unknown": "none stored"}
    body: list[list[object]] = [
        [
            d.name,
            _num(d.baseline),
            _num(d.candidate),
            _num(d.delta_points, "+.2f"),
            words[d.ci_overlap],
        ]
        for d in rows
    ]
    return [
        el("h2", "Headline"),
        _table(["metric", "baseline", "candidate", "delta (points)", "intervals"], body),
    ]


def _class_row(r: ClassDelta) -> list[object]:
    status = {"not_finite": "not finite", "not_applicable": "no ground truth"}.get(
        r.status, r.status
    )
    return [
        r.name,
        _num(r.baseline),
        _num(r.candidate),
        _num(r.delta_points, "+.2f"),
        _num(r.support_baseline, ".0f"),
        status,
    ]


def _classes(cmp: Comparison) -> list[Markup]:
    out: list[Markup] = []
    for metric, rows in cmp.per_class.items():
        shown = rows[:MAX_CLASS_ROWS]
        out += [
            el("h2", f"Classes: {metric}"),
            _table(
                ["class", "baseline", "candidate", "delta (points)", "support", "status"],
                [_class_row(r) for r in shown],
            ),
            _more(len(rows), len(shown)),
        ]
    return out


def _slice_row(r: SliceDelta) -> list[object]:
    return [
        r.label,
        r.metric,
        _num(r.baseline),
        _num(r.candidate),
        _num(r.delta_points, "+.2f"),
        _num(r.n_images_candidate, ".0f"),
        r.status.replace("_", " "),
    ]


def _slices(rows: tuple[SliceDelta, ...]) -> list[Markup]:
    if not rows:
        return []
    shown = rows[:MAX_SLICE_ROWS]
    return [
        el("h2", "Slices"),
        _table(
            ["slice", "metric", "baseline", "candidate", "delta (points)", "images", "status"],
            [_slice_row(r) for r in shown],
        ),
        _more(len(rows), len(shown)),
    ]


def _confusion(cmp: Comparison) -> list[Markup]:
    k = cmp.confusion
    if k is None:
        return []
    shown = k.cells[:MAX_CELL_ROWS]
    return [
        el("h2", "Confusion"),
        el(
            "p",
            f"{k.n_changed_cells} cells changed; missed ground truth {k.missed_delta:+d}, "
            f"spurious detections {k.false_positive_delta:+d} (background is its own class).",
        ),
        _table(
            ["true", "predicted", "baseline", "candidate", "change"],
            [[c.true_name, c.pred_name, c.baseline, c.candidate, f"{c.delta:+d}"] for c in shown],
        ),
        _more(k.n_changed_cells, len(shown)),
    ]


def render_html(
    cmp: Comparison, result: GateResult | None = None, thresholds: Thresholds | None = None
) -> str:
    """The report page for ``cmp`` (and the gate ``result``, listed first when given)."""
    body: list[Markup] = [el("h1", "CV-Pilot compare"), _ref("Candidate", cmp.candidate)]
    body.append(_ref("Baseline", cmp.baseline))
    if result is not None:
        body += _violations(result, thresholds)
    body += _headline(cmp.headline)
    body += _classes(cmp)
    body += _slices(cmp.slices)
    body += _confusion(cmp)
    body.append(
        el(
            "p",
            "Deltas are (candidate - baseline) * 100, metric points on the 0..100 scale. "
            "Intervals are the bootstrap intervals stored in each snapshot.",
            cls="note",
        )
    )
    head = (
        '<meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{CSP}">'
        '<meta name="referrer" content="no-referrer">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>CV-Pilot compare</title><style>{_CSS}</style>"
    )
    return f'<!doctype html>\n<html lang="en"><head>{head}</head>{el("body", *body)}</html>\n'


class ReportWriteError(CvpilotError, OSError):
    """A report file could not be written."""


def write_report(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file, fsync, rename), mode 0644.

    The target must be absent or a regular file that is not a symbolic link; the rename replaces
    a directory entry and never follows one, so a link planted in between is replaced, not
    followed.

    Raises:
        ReportWriteError: too large, a symbolic link, not a regular file, no such directory, or
            an I/O error (no temporary file is left behind).
    """
    data = text.encode("utf-8")
    if len(data) > MAX_REPORT_BYTES:
        raise ReportWriteError(f"the report is too large ({len(data)} bytes)")
    target = Path(path)
    try:
        mode = os.lstat(target).st_mode
    except FileNotFoundError:
        mode = None
    except OSError as exc:
        raise ReportWriteError(f"cannot write {target.name}: {exc.strerror}") from None
    if mode is not None and stat.S_ISLNK(mode):
        raise ReportWriteError(f"{target.name} is a symbolic link; refusing to write through it")
    if mode is not None and not stat.S_ISREG(mode):
        raise ReportWriteError(f"{target.name} is not a regular file")
    parent = target.parent
    if not parent.is_dir():
        raise ReportWriteError(f"directory {parent.name or str(parent)} does not exist")
    tmp = parent / f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(tmp, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ReportWriteError(f"cannot write {target.name}: {exc.strerror or exc}") from None
