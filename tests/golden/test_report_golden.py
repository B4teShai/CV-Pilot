"""Golden compare reports: the same snapshots always produce these exact bytes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from support import PARAMS, build_snapshot

from cvpilot.compare import html_report as hr
from cvpilot.compare.confusion import confusion_delta
from cvpilot.compare.engine import compare
from cvpilot.compare.gate import Thresholds, gate
from cvpilot.compare.report import report_json

DATA = Path(__file__).parent / "expected"
CAND, BASE = "01ARZ3NDEKTSV4RRFFQ69G5FAV", "01BX5ZZKBKACTAV9WEVGEMMVRY"
NAMES = ["cat", "dog", "<b>bird</b>"]


def fixture() -> tuple[str, str]:
    spec = {"camera": "camera"}
    base = build_snapshot(
        BASE,
        class_names=NAMES,
        step=10,
        metrics={"mAP50_95": 0.600, "mAP50": 0.800, "AR100": 0.700},
        per_class={
            "cat": {"AP50_95": 0.7, "AP50": 0.9, "AR100": 0.8, "support": 30.0},
            "dog": {"AP50_95": 0.6, "AP50": 0.8, "AR100": 0.7, "support": 25.0},
            "<b>bird</b>": {"AP50_95": 0.5, "AP50": 0.7, "AR100": 0.6, "support": 5.0},
        },
        slices={
            "camera=c1": {"mAP50_95": 0.62, "n_images": 60.0},
            "camera=c2": {"mAP50_95": 0.58, "n_images": 40.0},
        },
        slice_spec=spec,
        ci={"mAP50_95": (0.55, 0.65), "mAP50": (0.75, 0.85)},
        params=PARAMS,
    )
    cand = build_snapshot(
        CAND,
        class_names=NAMES,
        step=12,
        metrics={"mAP50_95": 0.585, "mAP50": 0.810, "AR100": 0.690},
        per_class={
            "cat": {"AP50_95": 0.72, "AP50": 0.9, "AR100": 0.8, "support": 30.0},
            "dog": {"AP50_95": 0.55, "AP50": 0.78, "AR100": 0.7, "support": 25.0},
            "<b>bird</b>": {"AP50_95": 0.4, "AP50": 0.6, "AR100": 0.5, "support": 5.0},
        },
        slices={
            "camera=c1": {"mAP50_95": 0.60, "n_images": 60.0},
            "camera=c2": {"mAP50_95": 0.57, "n_images": 40.0},
        },
        slice_spec=spec,
        ci={"mAP50_95": (0.54, 0.63), "mAP50": (0.76, 0.86)},
        params=PARAMS,
    )
    zero = np.zeros((4, 4), dtype=np.int64)
    before, after = zero.copy(), zero.copy()
    for k in range(3):
        before[k, k], after[k, k] = 20, 19
    after[0, 3], after[3, 2] = 2, 3
    th = Thresholds(
        max_drop={"mAP50_95": 1.0},
        per_class_max_drop={"AP50_95": 5.0},
        slice_max_drop={"camera=*": 1.5},
        min_images=30,
    )
    cmp = compare(cand, base, confusion=confusion_delta(before, after, NAMES))
    result = gate(cmp, th)
    return hr.render_html(cmp, result, th), report_json(cmp, result, th)


def check(name: str, text: str) -> None:
    path = DATA / name
    if os.environ.get("CVPILOT_REGENERATE_GOLDEN") == "1":
        path.write_text(text)
    assert path.read_text() == text, f"{name} changed; review the diff, then regenerate"


def test_html_report_matches_the_golden_file() -> None:
    check("report.html", fixture()[0])


def test_json_report_matches_the_golden_file() -> None:
    check("report.json", fixture()[1])
