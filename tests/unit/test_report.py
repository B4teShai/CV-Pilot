"""HTML and JSON reports: escaping, CSP, severity-first caps and safe writing."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import build_snapshot

from cvpilot.compare import html_report as hr
from cvpilot.compare.confusion import confusion_delta
from cvpilot.compare.engine import compare
from cvpilot.compare.gate import Thresholds, gate
from cvpilot.compare.report import report_json
from cvpilot.core.errors import CvpilotError
from cvpilot.core.schema import EvalSnapshot

EVIL = [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "a&b'c",
    "x‮txt",
    "javascript:alert(1)",
]


def parsed_tags(page: str) -> list[tuple[str, list[str]]]:
    from html.parser import HTMLParser

    found: list[tuple[str, list[str]]] = []

    class Collect(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            found.append((tag, [name for name, _ in attrs]))

    Collect().feed(page)
    return found


def evil_comparison(names: list[str] = EVIL) -> Any:
    per_b = {n: {"AP50_95": 0.9, "support": 5.0} for n in names}
    per_c = {n: {"AP50_95": 0.1, "support": 5.0} for n in names}
    slices_b = {f"camera={n}": {"mAP50_95": 0.9, "n_images": 50.0} for n in names}
    slices_c = {f"camera={n}": {"mAP50_95": 0.1, "n_images": 50.0} for n in names}
    spec = {"camera": "camera"}
    base = build_snapshot(class_names=names, per_class=per_b, slices=slices_b, slice_spec=spec)
    cand = build_snapshot(class_names=names, per_class=per_c, slices=slices_c, slice_spec=spec)
    size = len(names) + 1
    zeros = np.zeros((size, size), dtype=np.int64)
    changed = zeros.copy()
    changed[0, size - 1] = 4
    return compare(cand, base, confusion=confusion_delta(zeros, changed, names))


def thresholds() -> Thresholds:
    return Thresholds(
        max_drop={"mAP50_95": 0.5},
        per_class_max_drop={"AP50_95": 1.0},
        slice_max_drop={"camera=*": 1.0},
        min_images=1,
    )


def test_every_name_is_escaped_and_nothing_executable_is_emitted() -> None:
    cmp = evil_comparison()
    page = hr.render_html(cmp, gate(cmp, thresholds()), thresholds())
    assert "<script" not in page.lower() and "<img" not in page.lower()
    assert "&lt;script&gt;" in page and "&quot;&gt;&lt;img" in page and "a&amp;b&#x27;c" in page
    assert "‮" not in page and "\\u202e" in page
    seen = parsed_tags(page)
    allowed_tags = {"html", "head", "meta", "title", "style", "body", "h1", "h2", "p"}
    allowed_tags |= {"table", "tr", "th", "td"}
    assert {t for t, _ in seen} <= allowed_tags
    assert {a for _, attrs in seen for a in attrs} <= {
        "lang", "charset", "http-equiv", "content", "name", "class",
    }  # fmt: skip
    assert "http://" not in page and "https://" not in page


def test_csp_forbids_everything_but_the_hashed_inline_style() -> None:
    page = hr.render_html(evil_comparison())
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', page)
    assert csp is not None
    policy = csp.group(1)
    assert policy.startswith("default-src 'none'")
    assert "script-src" not in policy and "unsafe" not in policy and "*" not in policy
    css = re.search(r"<style>(.*?)</style>", page, re.S)
    assert css is not None
    digest = base64.b64encode(hashlib.sha256(css.group(1).encode()).digest()).decode()
    assert f"style-src 'sha256-{digest}'" in policy
    assert "base-uri 'none'" in policy and "form-action 'none'" in policy
    assert page.count("<style>") == 1 and "style=" not in page
    assert '<meta name="referrer" content="no-referrer">' in page


def test_report_is_deterministic_and_has_no_timestamp() -> None:
    cmp = evil_comparison()
    result = gate(cmp, thresholds())
    assert hr.render_html(cmp, result, thresholds()) == hr.render_html(cmp, result, thresholds())
    assert report_json(cmp, result, thresholds()) == report_json(cmp, result, thresholds())
    assert not re.search(r"20\d\d-\d\d-\d\dT", hr.render_html(cmp))


def test_violations_come_first_and_are_capped_worst_first(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [f"c{i:03d}" for i in range(150)]
    base = {n: {"AP50_95": 0.9, "support": 5.0} for n in names}
    cand = {n: {"AP50_95": 0.9 - (i + 1) / 1000, "support": 5.0} for i, n in enumerate(names)}
    cmp = compare(
        build_snapshot(class_names=names, per_class=cand),
        build_snapshot(class_names=names, per_class=base),
    )
    result = gate(cmp, Thresholds(per_class_max_drop={"AP50_95": 0.0}, min_images=1))
    assert len(result.violations) == 150
    page = hr.render_html(cmp, result)
    assert page.index("Violations") < page.index("Headline")
    assert f"{150 - hr.MAX_VIOLATIONS} more" in page
    shown = re.findall(r"\[class_drop\]", page)
    assert len(shown) == hr.MAX_VIOLATIONS
    assert "c149" in page and "c000" not in page.split("Headline")[0]  # worst kept, mildest cut


def test_class_table_is_capped_worst_first_and_says_so() -> None:
    names = [f"c{i:03d}" for i in range(300)]
    base = {n: {"AP50_95": 0.9, "support": 5.0} for n in names}
    cand = {n: {"AP50_95": 0.9 - i / 1000, "support": 5.0} for i, n in enumerate(names)}
    cmp = compare(
        build_snapshot(class_names=names, per_class=cand),
        build_snapshot(class_names=names, per_class=base),
    )
    page = hr.render_html(cmp)
    assert f"{300 - hr.MAX_CLASS_ROWS} more" in page
    assert "c299" in page and "c001" not in page


def test_slices_and_confusion_are_capped_with_a_count() -> None:
    n = hr.MAX_SLICE_ROWS + 20
    rows_b = {f"camera=s{i:03d}": {"mAP50_95": 0.9, "n_images": 50.0} for i in range(n)}
    rows_c = {f"camera=s{i:03d}": {"mAP50_95": 0.9 - i / 10000, "n_images": 50.0} for i in range(n)}
    spec = {"camera": "camera"}
    cmp = compare(
        build_snapshot(slices=rows_c, slice_spec=spec),
        build_snapshot(slices=rows_b, slice_spec=spec),
    )
    page = hr.render_html(cmp)
    assert "20 more" in page and f"s{n - 1:03d}" in page and "s000" not in page


def test_nan_values_render_as_unusable_not_as_numbers() -> None:
    good = build_snapshot(metrics={"mAP50_95": 0.6})
    broken = EvalSnapshot.model_construct(**{**good.__dict__, "metrics": {"mAP50_95": math.nan}})
    cmp = compare(broken, good)
    page = hr.render_html(cmp, gate(cmp, Thresholds(max_drop={"mAP50_95": 1.0}, min_images=1)))
    assert "nan" not in page.lower() or "not finite" in page.lower()
    doc = json.loads(report_json(cmp))
    assert doc["headline"][0]["candidate"] is None


def test_json_report_is_canonical_and_parses_strictly() -> None:
    cmp = evil_comparison()
    text = report_json(cmp, gate(cmp, thresholds()), thresholds())
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    assert "NaN" not in text and "Infinity" not in text
    doc = json.loads(text, parse_constant=lambda c: pytest.fail(c))
    assert doc["schema"] == "cvpilot.compare/1" and doc["gate"]["passed"] is False
    assert all(ord(c) < 128 for c in text)


# -- writing -------------------------------------------------------------------------------


def test_write_report_creates_and_replaces_a_regular_file_atomically(tmp_path: Path) -> None:
    target = tmp_path / "cmp.html"
    hr.write_report(target, "one")
    hr.write_report(target, "two")
    assert target.read_text() == "two" and target.stat().st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir()] == ["cmp.html"]


def test_write_report_refuses_symlinks_directories_and_missing_parents(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("keep")
    link = tmp_path / "link.html"
    link.symlink_to(secret)
    with pytest.raises(CvpilotError, match="symbolic link"):
        hr.write_report(link, "x")
    assert secret.read_text() == "keep"
    dangling = tmp_path / "dangling.html"
    dangling.symlink_to(tmp_path / "nowhere")
    with pytest.raises(CvpilotError):
        hr.write_report(dangling, "x")
    with pytest.raises(CvpilotError):
        hr.write_report(tmp_path, "x")
    with pytest.raises(CvpilotError):
        hr.write_report(tmp_path / "no" / "such" / "dir.html", "x")


def test_write_report_enforces_a_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hr, "MAX_REPORT_BYTES", 10)
    with pytest.raises(CvpilotError, match="too large"):
        hr.write_report(tmp_path / "r.html", "x" * 11)
    assert not (tmp_path / "r.html").exists()


def test_write_report_leaves_no_temp_file_when_the_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(CvpilotError):
        hr.write_report(tmp_path / "r.html", "x")
    assert list(tmp_path.iterdir()) == []
