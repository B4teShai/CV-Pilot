"""Plain-text rendering of a comparison and a gate outcome.

Every line starts with ``gate:`` or two spaces, and every name goes through ``safe_text``: a
class or slice name from data can neither start a line (CI runners read ``::command::`` at the
start of a line as a workflow command) nor hide or reorder text with control or bidirectional
characters.
"""

from __future__ import annotations

from cvpilot.compare.engine import Comparison
from cvpilot.compare.service import GateOutcome
from cvpilot.compare.text import safe_text

TOP_ROWS = 5
"""Worst classes, slices and confusion cells printed by ``compare`` (full data: ``--json``)."""


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def gate_lines(outcome: GateOutcome) -> list[str]:
    result = outcome.result
    base = f"tag:{outcome.tag}" if outcome.tag else f"run {outcome.baseline_run_id}"
    cand = outcome.comparison.candidate
    head = f"run {cand.run_id} step {cand.step} vs baseline {base}"
    if result.passed:
        lines = [
            f"gate: PASS ({result.n_checks} checks, {len(result.skipped)} slices skipped) {head}"
        ]
    else:
        worst = result.violations[0]
        lines = [
            f"gate: FAIL ({len(result.violations)} violations, worst: [{worst.kind}] "
            f"{safe_text(worst.subject)}) {head}"
        ]
        lines += [f"  - [{v.kind}] {safe_text(v.subject)}: {v.message}" for v in result.violations]
    lines += [f"  skipped slice: {safe_text(label)}" for label in result.skipped]
    return lines


def compare_lines(cmp: Comparison) -> list[str]:
    c, b = cmp.candidate, cmp.baseline
    lines = [
        f"  candidate: run {c.run_id} step {c.step} eval_set {c.eval_set_hash[:12]} "
        f"n_images {c.n_images}",
        f"  baseline: run {b.run_id} step {b.step} eval_set {b.eval_set_hash[:12]} "
        f"n_images {b.n_images}",
        "  unit: delta in points on the 0..100 scale (candidate - baseline)",
    ]
    for d in cmp.headline:
        ci = {
            "overlap": "intervals overlap",
            "separated": "intervals separated",
            "unknown": "no intervals",
        }
        lines.append(
            f"  {safe_text(d.name)}: {_value(d.baseline)} -> {_value(d.candidate)} "
            f"({_points(d.delta_points)} points, {ci[d.ci_overlap]})"
        )
    for metric, rows in cmp.per_class.items():
        if metric not in ("AP50_95", "AP", "f1"):
            continue
        for r in rows[:TOP_ROWS]:
            lines.append(
                f"  class {safe_text(r.name)} {metric}: {_value(r.baseline)} -> "
                f"{_value(r.candidate)} ({_points(r.delta_points)} points, "
                f"support {r.support_baseline}, {r.status})"
            )
    for s in cmp.slices[:TOP_ROWS]:
        lines.append(
            f"  slice {safe_text(s.label)} {s.metric}: {_value(s.baseline)} -> "
            f"{_value(s.candidate)} ({_points(s.delta_points)} points, {s.status})"
        )
    if cmp.confusion is not None:
        k = cmp.confusion
        lines.append(
            f"  confusion: {k.n_changed_cells} cells changed, missed {k.missed_delta:+d}, "
            f"spurious {k.false_positive_delta:+d}"
        )
        for cell in k.cells[:TOP_ROWS]:
            lines.append(
                f"  cell {safe_text(cell.true_name)} -> {safe_text(cell.pred_name)}: "
                f"{cell.baseline} -> {cell.candidate} ({cell.delta:+d})"
            )
    return lines
