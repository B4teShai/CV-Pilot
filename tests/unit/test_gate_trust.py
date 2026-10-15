"""Gate trust: nothing a pull request controls may decide the verdict (items 2 to 6 and 8)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from conftest import CliResult
from support import add_snapshot, build_snapshot, make_run, open_project, write_eval_files

from cvpilot.compare.engine import compare
from cvpilot.compare.gate import Thresholds, gate
from cvpilot.core.errors import CompareRefusedError
from cvpilot.core.schema import Baseline
from cvpilot.store import baselines as bl

Invoke = Callable[..., CliResult]
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40


def evaluate_into(
    invoke: Invoke,
    store_root: Path,
    folder: Path,
    *,
    status: str = "finished",
    commit: str | None = None,
    **kw: Any,
) -> str:
    store = open_project(store_root)
    try:
        run = make_run(store, status=status)
    finally:
        store.close()
    preds, gts, classes = write_eval_files(folder, **kw)
    extra = ["--git-commit", commit] if commit else []
    res = invoke(
        "eval", run, "--project", "proj", "--preds", str(preds), "--gts", str(gts),
        "--classes", str(classes), "--slices", "camera=camera", "--allow-closed",
        "--resamples", "0", *extra,
    )  # fmt: skip
    assert res.code == 0, res.err
    return run


@pytest.fixture
def base(invoke: Invoke, store_root: Path, tmp_path: Path) -> str:
    run = evaluate_into(invoke, store_root, tmp_path / "base")
    assert invoke("baseline", "set", "prod", run, "--project", "proj").code == 0
    return run


def gate_args(run: str, *extra: str, baseline: str = "tag:prod") -> list[str]:
    return [
        "gate", run, "--project", "proj", "--baseline", baseline, "--max-drop", "mAP50_95=0.5",
        "--min-images", "20", *extra,
    ]  # fmt: skip


# -- 2: recompute ----------------------------------------------------------------------------


def forge(store_root: Path, honest: str, perfect: str) -> str:
    """A run whose stored predictions are ``honest``'s (bad) but whose metrics are ``perfect``'s."""
    store = open_project(store_root)
    try:
        bad, good = store.list_snapshots(honest)[0], store.list_snapshots(perfect)[0]
        run = make_run(store)
        conf = store.put_artifact(run, store.artifact_path(bad.confusion_sha256), "confusion", {})
        pred = store.put_artifact(
            run, store.artifact_path(bad.predictions_sha256), "predictions", {}
        )
        forged = good.model_copy(
            update={
                "run_id": run,
                "confusion_sha256": conf.sha256,
                "predictions_sha256": pred.sha256,
            }
        )
        store.put_snapshot(forged)
        return run
    finally:
        store.close()


def recompute_args(tmp_path: Path, folder: str) -> list[str]:
    return [
        "--recompute",
        "--gt",
        str(tmp_path / folder / "g.parquet"),
        "--classes",
        str(tmp_path / folder / "c.txt"),
    ]  # noqa: E501


def test_a_forged_snapshot_with_perfect_metrics_passes_without_recompute_and_fails_with_it(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    forged = forge(store_root, bad, base)
    assert invoke(*gate_args(forged)).code == 0  # the documented limit of an unverified gate
    res = invoke(*gate_args(forged, *recompute_args(tmp_path, "base")))
    assert res.code == 1, res.out
    failed = res.event("gate_recompute_failed")
    assert len(failed) == 1 and failed[0]["run_id"] == forged
    assert "gate: PASS" not in res.out and not res.event("gate_passed")


def test_an_honest_run_passes_with_recompute_and_a_regression_still_fails(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    assert invoke(*gate_args(same, *recompute_args(tmp_path, "same"))).code == 0
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    res = invoke(*gate_args(bad, *recompute_args(tmp_path, "bad")))
    assert res.code == 1 and res.event("gate_failed") and not res.event("gate_recompute_failed")


def test_recompute_with_other_ground_truth_or_classes_is_refused(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    write_eval_files(tmp_path / "other", shift_gt=True)
    res = invoke(*gate_args(same, *recompute_args(tmp_path, "other")))
    assert res.code == 1 and res.event("gate_recompute_failed")
    (tmp_path / "names.txt").write_text("cat\nfish\n")
    args = [
        "--recompute",
        "--gt",
        str(tmp_path / "same" / "g.parquet"),
        "--classes",
        str(tmp_path / "names.txt"),
    ]
    res = invoke(*gate_args(same, *args))
    assert res.code == 1 and res.event("gate_recompute_failed")


def test_a_tampered_stored_prediction_artifact_fails_the_digest_before_recomputing(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    store = open_project(store_root)
    try:
        sha = store.list_snapshots(same)[0].predictions_sha256
        path = store.artifact_path(sha)
    finally:
        store.close()
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"x")
    res = invoke(*gate_args(same, *recompute_args(tmp_path, "same")))
    assert res.code == 1 and res.event("snapshot_integrity_failed")


@pytest.mark.parametrize(
    "extra",
    [
        ["--recompute"],
        ["--gt", "g.parquet"],
        ["--classes", "c.txt"],
        ["--recompute", "--gt", "g.parquet"],
    ],
)
def test_recompute_options_must_come_together(
    invoke: Invoke, base: str, tmp_path: Path, extra: list[str]
) -> None:
    (tmp_path / "g.parquet").write_bytes(b"")
    (tmp_path / "c.txt").write_text("a\n")
    assert invoke(*gate_args(base, *extra)).code == 2


def test_the_baseline_is_recomputed_too(invoke: Invoke, store_root: Path, tmp_path: Path) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    good = evaluate_into(invoke, store_root, tmp_path / "good")
    forged = forge(store_root, good, bad)  # a forged *baseline*: good predictions, bad metrics
    assert invoke("baseline", "set", "prod", forged, "--project", "proj").code == 0
    cand = evaluate_into(invoke, store_root, tmp_path / "cand", broken=10)
    assert invoke(*gate_args(cand)).code == 0  # the regression hides behind the low baseline
    res = invoke(*gate_args(cand, *recompute_args(tmp_path, "good")))
    assert res.code == 1 and res.event("gate_recompute_failed")


# -- 3: slice membership -----------------------------------------------------------------------


def test_moving_images_between_slices_changes_the_slice_hash_not_the_eval_hash(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    a = evaluate_into(invoke, store_root, tmp_path / "a")
    b = evaluate_into(invoke, store_root, tmp_path / "b", cameras=["c0"] * 20 + ["c1"] * 20)
    store = open_project(store_root)
    try:
        sa, sb = store.list_snapshots(a)[0], store.list_snapshots(b)[0]
    finally:
        store.close()
    assert sa.eval_set_hash == sb.eval_set_hash
    assert sa.slice_hash and sb.slice_hash and sa.slice_hash != sb.slice_hash


def test_same_eval_hash_with_moved_images_is_refused_end_to_end(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    moved = evaluate_into(invoke, store_root, tmp_path / "m", cameras=["c0"] * 20 + ["c1"] * 20)
    res = invoke(*gate_args(moved, "--slice-max-drop", "camera=*:1.5"))
    assert res.code == 1 and res.event("compare_refused")[0]["reason"] == "slice_hash"
    res = invoke("compare", moved, "--project", "proj", "--baseline", "tag:prod")
    assert res.code == 1 and res.event("compare_refused")[0]["reason"] == "slice_hash"


def test_equal_slice_membership_compares_normally(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    assert invoke(*gate_args(same, "--slice-max-drop", "camera=*:1.5")).code == 0


HASH_A, HASH_B = "a" * 64, "b" * 64
SPEC = {"camera": "camera"}


def slices(n_c1: float) -> dict[str, dict[str, float]]:
    return {"camera=c1": {"mAP50_95": 0.6, "n_images": n_c1}}


def test_a_slice_hash_difference_and_a_one_sided_slice_hash_are_refused() -> None:
    base = build_snapshot(slices=slices(50), slice_spec=SPEC, slice_hash=HASH_A)
    for other in (HASH_B, None):
        cand = build_snapshot(slices=slices(50), slice_spec=SPEC, slice_hash=other)
        with pytest.raises(CompareRefusedError) as exc:
            compare(cand, base)
        assert exc.value.reason == "slice_hash"


def test_per_slice_image_counts_must_match_even_with_equal_hashes() -> None:
    base = build_snapshot(slices=slices(50), slice_spec=SPEC, slice_hash=HASH_A)
    cand = build_snapshot(slices=slices(49), slice_spec=SPEC, slice_hash=HASH_A)
    with pytest.raises(CompareRefusedError) as exc:
        compare(cand, base)
    assert exc.value.reason == "slice_n_images"


def test_snapshots_without_a_slice_hash_fail_a_gate_that_has_slice_thresholds() -> None:
    snap = build_snapshot(slices=slices(50), slice_spec=SPEC)  # slice_hash None on both sides
    cmp = compare(snap, snap)
    with_slices = gate(cmp, Thresholds(slice_max_drop={"camera=*": 1.0}, min_images=1))
    assert [v.kind for v in with_slices.violations] == ["slice_hash_missing"]
    assert gate(cmp, Thresholds(max_drop={"mAP50_95": 1.0}, min_images=1)).passed


# -- 5: who may be the baseline ----------------------------------------------------------------


def test_the_candidate_cannot_be_its_own_baseline(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    run = evaluate_into(invoke, store_root, tmp_path / "x", broken=10)
    res = invoke(*gate_args(run, baseline=run))
    assert res.code == 1 and res.event("compare_refused")[0]["reason"] == "baseline_is_candidate"


@pytest.mark.parametrize("status", ["running", "failed"])
def test_a_baseline_run_must_be_finished(
    invoke: Invoke, store_root: Path, tmp_path: Path, status: str
) -> None:
    store = open_project(store_root)
    try:
        run = make_run(store, status=status)
    finally:
        store.close()
    preds, gts, classes = write_eval_files(tmp_path / status)
    args = ["--preds", str(preds), "--gts", str(gts), "--classes", str(classes), "--allow-closed"]
    assert invoke("eval", run, "--project", "proj", *args, "--slices", "camera=camera").code == 0
    cand = evaluate_into(invoke, store_root, tmp_path / "cand")
    res = invoke(*gate_args(cand, baseline=run))
    assert res.code == 1 and res.event("compare_refused")[0]["reason"] == "baseline_not_finished"


# -- 6: the candidate must be the commit under test ------------------------------------------------


def test_require_commit_accepts_the_matching_commit_and_a_prefix(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    run = evaluate_into(invoke, store_root, tmp_path / "x", commit=COMMIT_A)
    assert invoke(*gate_args(run, "--require-commit", COMMIT_A)).code == 0
    assert invoke(*gate_args(run, "--require-commit", COMMIT_A[:12])).code == 0


def test_an_older_good_run_is_refused_when_a_commit_is_required(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    old_good = evaluate_into(invoke, store_root, tmp_path / "old", commit=COMMIT_B)
    res = invoke(*gate_args(old_good, "--require-commit", COMMIT_A))
    assert res.code == 1 and res.event("compare_refused")[0]["reason"] == "commit_mismatch"


def test_a_candidate_without_provenance_is_refused_when_a_commit_is_required(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    run = evaluate_into(invoke, store_root, tmp_path / "x")
    res = invoke(*gate_args(run, "--require-commit", COMMIT_A))
    assert res.code == 1 and res.event("compare_refused")[0]["reason"] == "provenance_missing"
    assert invoke(*gate_args(run)).code == 0  # the option is what makes it required


@pytest.mark.parametrize("bad", ["abc", "A" * 40, "zz" * 20, "a" * 65, ""])
def test_require_commit_must_be_a_commit_hash(invoke: Invoke, base: str, bad: str) -> None:
    assert invoke(*gate_args(base, "--require-commit", bad)).code == 2


# -- 8: skipping small slices ---------------------------------------------------------------------


def test_skipping_is_capped_by_max_skipped_which_defaults_to_zero() -> None:
    base = {
        "slices": {"camera=c1": slice_row(0.6, 50), "camera=c2": slice_row(0.6, 5)},
        "slice_spec": SPEC,
        "slice_hash": HASH_A,
    }
    cand = {
        "slices": {"camera=c1": slice_row(0.6, 50), "camera=c2": slice_row(0.1, 5)},
        "slice_spec": SPEC,
        "slice_hash": HASH_A,
    }
    cmp = compare(build_snapshot(**cand), build_snapshot(**base))
    default = gate(
        cmp, Thresholds(slice_max_drop={"camera=*": 1.0}, min_images=30, skip_small_slices=True)
    )
    assert [v.kind for v in default.violations] == ["too_many_skipped_slices"]
    assert default.skipped == ("camera=c2",)
    allowed = gate(
        cmp,
        Thresholds(
            slice_max_drop={"camera=*": 1.0}, min_images=30, skip_small_slices=True, max_skipped=1
        ),
    )
    assert allowed.passed and allowed.skipped == ("camera=c2",)


def slice_row(value: float, n: float) -> dict[str, float]:
    return {"mAP50_95": value, "n_images": n}


def test_max_skipped_must_be_a_non_negative_integer() -> None:
    from cvpilot.compare.gate import ThresholdError

    for bad in (-1, True, 1.5):
        with pytest.raises(ThresholdError):
            Thresholds(max_drop={"a": 1.0}, max_skipped=bad)  # type: ignore[arg-type]


def test_a_wildcard_threshold_whose_only_regressed_slice_is_below_min_images_fails(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    cameras = ["big"] * 34 + ["tiny"] * 6
    base = evaluate_into(invoke, store_root, tmp_path / "b", cameras=cameras)
    assert invoke("baseline", "set", "prod", base, "--project", "proj").code == 0
    bad = evaluate_into(invoke, store_root, tmp_path / "x", cameras=cameras, broken=0)
    # regress only the six images of the tiny slice by breaking the last six predictions
    import pyarrow.parquet as pq

    table = pq.read_table(tmp_path / "x" / "p.parquet").to_pydict()
    for i in range(34, 40):
        table["x1"][i], table["y1"][i], table["x2"][i], table["y2"][i] = 120.0, 120.0, 180.0, 180.0
    import pyarrow as pa

    pq.write_table(pa.table(table), tmp_path / "x" / "p2.parquet")
    store = open_project(store_root)
    try:
        run = make_run(store)
    finally:
        store.close()
    res = invoke(
        "eval", run, "--project", "proj", "--preds", str(tmp_path / "x" / "p2.parquet"),
        "--gts", str(tmp_path / "x" / "g.parquet"), "--classes", str(tmp_path / "x" / "c.txt"),
        "--slices", "camera=camera", "--allow-closed", "--resamples", "0",
    )  # fmt: skip
    assert res.code == 0 and bad
    argv = [
        "gate", run, "--project", "proj", "--baseline", "tag:prod", "--slice-max-drop",
        "camera=*:1.5", "--min-images", "30", "--skip-small-slices",
    ]  # fmt: skip
    failed = invoke(*argv)
    assert failed.code == 1 and "[too_many_skipped_slices]" in failed.out
    assert "  skipped slice: camera=tiny" in failed.out
    allowed = invoke(*argv, "--max-skipped", "1")
    assert (
        allowed.code == 0 and "  skipped slice: camera=tiny" in allowed.out
    )  # explicit and visible


# -- 4: the external anchor for the baseline chain -----------------------------------------------


def head(invoke: Invoke) -> str:
    shown = json.loads(invoke("baseline", "show", "prod", "--project", "proj", "--json").out)
    return str(shown["head"]["chain_sha256"])


def test_baseline_show_prints_the_head_sequence_and_chain_digest(invoke: Invoke, base: str) -> None:
    out = invoke("baseline", "show", "prod", "--project", "proj").out
    chain = head(invoke)
    assert f"head_chain_sha256: {chain}" in out and "head_seq: 1" in out and len(chain) == 64


def test_the_anchor_passes_when_it_matches_and_fails_after_a_rollback(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    worse = evaluate_into(invoke, store_root, tmp_path / "w", broken=10)
    anchor_one = head(invoke)
    assert invoke(*gate_args(base, "--expect-baseline-chain", anchor_one)).code == 0
    # a pull request replaces the baseline with a worse run, then the maintainer re-anchors
    assert invoke("baseline", "set", "prod", worse, "--project", "proj", "--replace").code == 0
    anchor_two = head(invoke)
    assert anchor_two != anchor_one
    assert invoke(*gate_args(worse, "--expect-baseline-chain", anchor_two)).code == 0
    # rollback: the newest row (the replacement) is deleted so the old, better pin is current again
    store = open_project(store_root)
    try:
        store._conn.execute("DROP TRIGGER baseline_no_delete")
        store._conn.execute("DELETE FROM baseline WHERE seq=(SELECT max(seq) FROM baseline)")
    finally:
        store.close()
    res = invoke(*gate_args(worse, "--expect-baseline-chain", anchor_two))
    assert res.code == 1 and res.event("baseline_integrity_failed")
    assert invoke(*gate_args(worse)).code == 1  # still a regression against the old pin
    assert invoke(*gate_args(base, "--expect-baseline-chain", anchor_two)).code == 1


def test_the_anchor_is_validated_and_needs_a_tag_baseline(invoke: Invoke, base: str) -> None:
    for bad in ("abc", "G" * 64, "a" * 63, ""):
        assert invoke(*gate_args(base, "--expect-baseline-chain", bad)).code == 2
    res = invoke(*gate_args(base, "--expect-baseline-chain", "a" * 64, baseline=base))
    assert res.code == 2
    res = invoke(
        "compare",
        base,
        "--project",
        "proj",
        "--baseline",
        "tag:prod",
        "--expect-baseline-chain",
        "a" * 64,
    )
    assert res.code == 1 and res.event("baseline_integrity_failed")


def test_the_store_exposes_the_verified_head(tmp_path: Path) -> None:
    store = open_project(tmp_path)
    try:
        assert store.baseline_head() is None
        run = make_run(store)
        add_snapshot(store, run)
        store.put_baseline(
            Baseline(
                project="proj", tag="prod", run_id=run, eval_set_hash="1" * 64, step=0,
                snapshot_sha256=store.snapshot_digest(run, "1" * 64, 0),
                pinned_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
            )
        )  # fmt: skip
        seq, chain = store.baseline_head() or (0, "")
        assert seq == 1 and len(chain) == 64 and bl.GENESIS != chain
    finally:
        store.close()


# -- folded in: event order, identical JSON, refusal reasons, size mismatch ---------------------


def test_no_verdict_is_logged_when_the_report_cannot_be_written(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    link = tmp_path / "link.html"
    link.symlink_to(tmp_path / "elsewhere.html")
    res = invoke(*gate_args(same, "--report", str(link)))
    assert res.code == 1
    assert (
        not res.event("gate_passed")
        and not res.event("gate_failed")
        and res.event("command_failed")
    )
    res = invoke(
        "compare", same, "--project", "proj", "--baseline", "tag:prod", "--report", str(link)
    )
    assert res.code == 1 and not res.event("compare_completed")


def test_the_verdict_is_logged_after_the_output_is_written(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    res = invoke(*gate_args(same, "--report", str(tmp_path / "r.html")))
    assert res.code == 0 and res.event("gate_passed") and (tmp_path / "r.html").exists()


def test_json_output_and_the_json_report_are_byte_identical_and_carry_the_confusion(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    path = tmp_path / "r.json"
    res = invoke(*gate_args(bad, "--json", "--report-json", str(path)))
    assert res.code == 1
    assert res.out.rstrip("\n") == path.read_text()
    assert json.loads(res.out)["confusion"]["n_changed_cells"] > 0
    plain = invoke(*gate_args(bad, "--json"))
    assert plain.out == res.out  # the verdict does not depend on which outputs were asked for


def test_a_candidate_without_the_requested_step_has_its_own_refusal_reason(
    invoke: Invoke, store_root: Path, tmp_path: Path, base: str
) -> None:
    run = evaluate_into(invoke, store_root, tmp_path / "x")
    res = invoke(*gate_args(run, "--step", "99"))
    assert res.code == 1
    event = res.event("compare_refused")[0]
    assert event["reason"] == "no_candidate_snapshot_at_step"
    assert event["candidate_eval_set_hash"] == event["baseline_eval_set_hash"]


def test_a_slice_with_other_images_is_a_size_mismatch_the_gate_fails() -> None:
    import dataclasses

    from cvpilot.compare import engine

    base = build_snapshot(slices=slices(50), slice_spec=SPEC, slice_hash=HASH_A)
    cand = build_snapshot(slices=slices(49), slice_spec=SPEC, slice_hash=HASH_A)
    rows = engine._slices(cand, base)  # compare() refuses first; the row status is the second net
    assert rows[0].status == "size_mismatch"
    cmp = dataclasses.replace(compare(base, base), slices=rows)
    result = gate(cmp, Thresholds(slice_max_drop={"camera=*": 1.0}, min_images=1))
    assert [v.kind for v in result.violations] == ["slice_size_mismatch"]
