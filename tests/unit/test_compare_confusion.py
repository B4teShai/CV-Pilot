"""Confusion deltas including the background class, and the verified artifact loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import H1, add_snapshot, make_run, open_project

from cvpilot.compare import confusion as cf
from cvpilot.core.errors import EvalInputError, SnapshotIntegrityError, StoreError
from cvpilot.registry import tables

NAMES = ["cat", "dog"]


def matrix(cells: dict[tuple[int, int], int]) -> np.ndarray[Any, Any]:
    m = np.zeros((3, 3), dtype=np.int64)
    for (r, c), v in cells.items():
        m[r, c] = v
    return m


def test_delta_names_background_and_separates_misses_from_spurious() -> None:
    base = matrix({(0, 0): 10, (1, 1): 10, (0, 2): 1, (2, 1): 2})
    cand = matrix({(0, 0): 8, (1, 1): 10, (0, 2): 3, (2, 1): 5, (0, 1): 1})
    delta = cf.confusion_delta(base, cand, NAMES)
    assert delta.missed_delta == 2  # ground truth that no detection claimed: [true, C]
    assert delta.false_positive_delta == 3  # detections claiming nothing: [C, pred]
    assert delta.baseline_total == 23 and delta.candidate_total == 27
    first, second = delta.cells[0], delta.cells[1]
    assert (first.true_name, first.pred_name, first.delta) == ("background", "dog", 3)
    assert (second.true_name, second.pred_name, second.delta) == ("cat", "cat", -2)
    names = {(c.true_name, c.pred_name) for c in delta.cells}
    assert ("cat", "background") in names and ("background", "dog") in names
    assert delta.n_changed_cells == 4


def test_identical_matrices_have_no_changed_cells() -> None:
    delta = cf.confusion_delta(matrix({(0, 0): 4}), matrix({(0, 0): 4}), NAMES)
    assert delta.cells == () and delta.n_changed_cells == 0 and delta.missed_delta == 0


def test_cells_are_capped_by_size_of_change_and_counted() -> None:
    n = 40
    base = np.zeros((n + 1, n + 1), dtype=np.int64)
    cand = base.copy()
    for k in range(n):
        cand[k, k] = k + 1
    delta = cf.confusion_delta(base, cand, [f"c{k}" for k in range(n)], max_cells=5)
    assert len(delta.cells) == 5 and delta.n_changed_cells == n
    assert [c.delta for c in delta.cells] == [40, 39, 38, 37, 36]


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(EvalInputError):
        cf.confusion_delta(
            np.zeros((3, 3), dtype=np.int64), np.zeros((4, 4), dtype=np.int64), NAMES
        )


@pytest.fixture
def store(tmp_path: Path) -> Any:
    s = open_project(tmp_path)
    yield s
    s.close()


def test_loader_reads_the_verified_artifact_into_a_dense_matrix(store: Any) -> None:
    run = make_run(store)
    snap = add_snapshot(store, run, class_names=NAMES, confusion=matrix({(0, 0): 4, (1, 2): 7}))
    loaded = cf.load_confusion(store, snap.confusion_sha256, 2)
    assert loaded.shape == (3, 3) and loaded[0, 0] == 4 and loaded[1, 2] == 7


def test_loader_refuses_an_artifact_that_fails_its_digest(store: Any) -> None:
    run = make_run(store)
    snap = add_snapshot(store, run, class_names=NAMES)
    path = store.artifact_path(snap.confusion_sha256)
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(SnapshotIntegrityError):
        cf.load_confusion(store, snap.confusion_sha256, 2)


@pytest.mark.parametrize(
    "cols",
    [
        {"true_class": [3], "pred_class": [0], "count": [1]},  # index above the background
        {"true_class": [0], "pred_class": [-1], "count": [1]},
        {"true_class": [0], "pred_class": [0], "count": [-5]},
        {"true_class": [0], "pred_class": [0], "count": [2**50]},
    ],
)
def test_loader_rejects_hostile_cells(
    store: Any, tmp_path: Path, cols: dict[str, list[int]]
) -> None:
    run = make_run(store)
    path = tmp_path / "evil.parquet"
    tables.get("parquet")().write_columns(
        path, {k: np.asarray(v, dtype=np.int64) for k, v in cols.items()}
    )
    sha = store.put_artifact(run, path, "confusion", {}).sha256
    with pytest.raises(EvalInputError):
        cf.load_confusion(store, sha, 2)


def test_loader_rejects_too_many_rows(store: Any, tmp_path: Path) -> None:
    run = make_run(store)
    path = tmp_path / "many.parquet"
    n = 10  # more rows than the (C+1)^2 = 9 cells of a 2-class matrix
    tables.get("parquet")().write_columns(
        path,
        {
            "true_class": np.zeros(n, dtype=np.int64),
            "pred_class": np.zeros(n, dtype=np.int64),
            "count": np.ones(n, dtype=np.int64),
        },
    )
    sha = store.put_artifact(run, path, "confusion", {}).sha256
    with pytest.raises(EvalInputError):
        cf.load_confusion(store, sha, 2)


def test_loader_size_limit(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    run = make_run(store)
    snap = add_snapshot(store, run, class_names=NAMES)
    monkeypatch.setattr(cf, "MAX_CONFUSION_BYTES", 10)
    with pytest.raises(StoreError, match="too large"):
        cf.load_confusion(store, snap.confusion_sha256, 2)
    assert H1
