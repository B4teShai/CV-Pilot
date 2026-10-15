"""Bounded Parquet reading for evaluation inputs: metadata is checked before any data is read."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cvpilot.core.errors import EvalInputError
from cvpilot.registry import tables
from cvpilot.store import tables as impl

IO = impl.open_tables()
BIG = 1 << 30


def write(path: Path, data: dict[str, object], **kw: object) -> Path:
    pq.write_table(pa.table(data), path, **kw)  # type: ignore[arg-type]
    return path


def read(path: Path, **kw: object) -> dict[str, np.ndarray]:
    args: dict[str, object] = {"required": ["image_id", "cls"], "max_rows": 1000}
    args.update(kw)
    return IO.read_columns(path, **args)  # type: ignore[arg-type]


def test_columns_round_trip_with_types_and_deterministic_bytes(tmp_path: Path) -> None:
    columns = {
        "image_id": np.array(["a", "b", "c"]),
        "cls": np.array([0, 1, 2], dtype=np.int64),
        "score": np.array([0.5, 0.25, 1.0]),
        "flag": np.array([True, False, True]),
    }
    IO.write_columns(tmp_path / "one.parquet", columns)
    IO.write_columns(tmp_path / "two.parquet", columns)
    assert (tmp_path / "one.parquet").read_bytes() == (tmp_path / "two.parquet").read_bytes()
    back = read(tmp_path / "one.parquet", optional=["score", "flag", "missing"])
    assert back["image_id"].tolist() == ["a", "b", "c"] and back["image_id"].dtype.kind == "U"
    assert back["cls"].dtype == np.int64 and back["score"].tolist() == [0.5, 0.25, 1.0]
    assert back["flag"].dtype == bool and "missing" not in back


def test_only_requested_columns_are_returned(tmp_path: Path) -> None:
    path = write(tmp_path / "t.parquet", {"image_id": ["a"], "cls": [1], "extra": [9.0]})
    assert set(read(path)) == {"image_id", "cls"}


def test_nullable_numeric_columns_become_nan_and_others_reject_nulls(tmp_path: Path) -> None:
    path = write(tmp_path / "t.parquet", {"image_id": ["a", "b"], "cls": [1, None]})
    with pytest.raises(EvalInputError, match="nulls"):
        read(path)
    out = read(path, nullable=["cls"])
    assert out["cls"][0] == 1.0 and np.isnan(out["cls"][1])


def test_dictionary_encoded_strings_are_accepted(tmp_path: Path) -> None:
    table = pa.table({"image_id": pa.array(["a", "b", "a"]).dictionary_encode(), "cls": [1, 2, 3]})
    pq.write_table(table, tmp_path / "t.parquet")
    assert read(tmp_path / "t.parquet")["image_id"].tolist() == ["a", "b", "a"]


@pytest.mark.parametrize(
    "data",
    [
        {"image_id": ["a"]},  # missing a required column
        {"image_id": [[1]], "cls": [1]},  # nested type
        {"image_id": [b"a"], "cls": [1]},  # binary
        {"image_id": ["a"], "cls": [{"x": 1}]},  # struct
    ],
)
def test_missing_columns_and_unsupported_types_are_rejected(
    tmp_path: Path, data: dict[str, object]
) -> None:
    with pytest.raises(EvalInputError):
        read(write(tmp_path / "t.parquet", data))


def test_row_count_is_checked_from_metadata_before_reading(tmp_path: Path) -> None:
    path = write(tmp_path / "t.parquet", {"image_id": ["a"] * 11, "cls": [0] * 11})
    assert len(read(path, max_rows=11)["cls"]) == 11
    with pytest.raises(EvalInputError, match="12 rows|11 rows.*limit|limit"):
        read(path, max_rows=10)


def test_file_size_and_uncompressed_size_and_row_groups_are_bounded(tmp_path: Path) -> None:
    path = write(
        tmp_path / "t.parquet",
        {"image_id": [f"id{i}" for i in range(500)], "cls": list(range(500))},
        row_group_size=100,
    )
    read(path, max_rows=500)
    with pytest.raises(EvalInputError, match="file is larger"):
        read(path, max_rows=500, max_file_bytes=10)
    with pytest.raises(EvalInputError, match="uncompressed"):
        read(path, max_rows=500, max_uncompressed_bytes=10)
    with pytest.raises(EvalInputError, match="row groups"):
        read(path, max_rows=500, max_row_groups=4)


def test_a_dictionary_bomb_is_rejected_before_it_is_expanded(tmp_path: Path) -> None:
    ids = pa.array(["x" * 200] + ["y"] * 5000).dictionary_encode()
    pq.write_table(pa.table({"image_id": ids, "cls": [0] * 5001}), tmp_path / "t.parquet")
    with pytest.raises(EvalInputError, match="too large"):
        read(tmp_path / "t.parquet", max_rows=6000, max_string_bytes=100_000)
    with pytest.raises(EvalInputError, match="longer than"):
        read(tmp_path / "t.parquet", max_rows=6000, max_string_length=100)


@pytest.mark.parametrize(
    "blob", [b"", b"PAR1", b"not parquet at all" * 50, b"PAR1" + b"\x00" * 64 + b"PAR1"]
)
def test_malformed_files_raise_the_domain_error(tmp_path: Path, blob: bytes) -> None:
    path = tmp_path / "bad.parquet"
    path.write_bytes(blob)
    with pytest.raises(EvalInputError, match="not a readable Parquet"):
        read(path)


def test_truncated_and_corrupted_files_raise_the_domain_error(tmp_path: Path) -> None:
    good = write(
        tmp_path / "good.parquet", {"image_id": ["a"] * 200, "cls": list(range(200))}
    ).read_bytes()
    for name, blob in {
        "cut": good[: len(good) // 2],
        "footer": good[:-8],
        "flipped": good[:60] + bytes(b ^ 0xFF for b in good[60:200]) + good[200:],
    }.items():
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(blob)
        with pytest.raises(EvalInputError):
            read(path)


def test_paths_that_are_not_regular_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EvalInputError, match="not a file"):
        read(tmp_path)
    with pytest.raises(EvalInputError, match="not a file|cannot"):
        read(tmp_path / "ghost.parquet")


def test_reading_a_large_file_is_fast(tmp_path: Path) -> None:
    n = 200_000
    path = write(
        tmp_path / "big.parquet",
        {"image_id": [f"i{i % 5000}" for i in range(n)], "cls": np.arange(n) % 50},
    )
    start = time.perf_counter()
    out = read(path, max_rows=n)
    assert len(out["cls"]) == n and time.perf_counter() - start < 5


def test_registry_resolves_the_parquet_adapter() -> None:
    assert "parquet" in tables.names()
    assert tables.get("parquet")().read_columns is not None


# -- review fixes: leaf columns, running budget, control characters, error wrapping -------------


def test_a_wide_nested_column_before_the_targets_does_not_shift_the_size_check(
    tmp_path: Path,
) -> None:
    wide = pa.StructArray.from_arrays(
        [pa.array([f"{'x' * 200}{i}{k}" for i in range(2000)]) for k in range(30)],
        names=[f"f{k}" for k in range(30)],
    )
    table = pa.table({"wide": wide, "image_id": [f"i{i}" for i in range(2000)], "cls": [0] * 2000})
    path = tmp_path / "t.parquet"
    pq.write_table(table, path, row_group_size=500)
    out = read(path, max_rows=5000, max_uncompressed_bytes=200_000)  # targets are ~40 KB
    assert len(out["cls"]) == 2000
    with pytest.raises(EvalInputError, match="uncompressed"):
        read(path, max_rows=5000, max_uncompressed_bytes=1_000)


def test_the_running_decoded_budget_holds_even_when_the_footer_understates_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(
        tmp_path / "t.parquet", {"image_id": [f"id{i}" for i in range(5000)], "cls": [1] * 5000}
    )
    monkeypatch.setattr(impl, "_declared_size", lambda *a, **k: 0)  # a forged footer
    with pytest.raises(EvalInputError, match="decoded"):
        read(path, max_rows=10_000, max_uncompressed_bytes=1_000)
    with pytest.raises(EvalInputError, match="rows"):
        read(path, max_rows=100)


def test_control_characters_in_strings_are_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "t.parquet", {"image_id": ["ok", "a\x00b"], "cls": [0, 1]})
    with pytest.raises(EvalInputError, match="control characters"):
        read(path)


def test_unexpected_decoder_errors_become_domain_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path / "t.parquet", {"image_id": ["a"], "cls": [0]})

    def boom(*args: object, **kwargs: object) -> object:
        raise KeyError("x")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", boom)
    with pytest.raises(EvalInputError, match="not a readable Parquet"):
        read(path)


def test_nullable_columns_report_which_values_were_null(tmp_path: Path) -> None:
    path = write(
        tmp_path / "t.parquet", {"image_id": ["a", "b", "c"], "cls": [1.0, None, float("nan")]}
    )
    out = read(path, nullable=["cls"])
    assert out["cls.null"].tolist() == [False, True, False]
    assert np.isnan(out["cls"][2])  # a real NaN is not a null
