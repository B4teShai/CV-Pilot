"""Parquet input and output for evaluation tables (adapter ``parquet`` of the ``tables`` registry).

Reading is bounded by file metadata before any data page is decoded: file size, row count,
number of row groups, decoded size of the requested columns, and, for string columns, the
longest value and the memory of their fixed-width array (a dictionary-encoded column can be tiny
on disk and enormous when expanded). Every failure is an :class:`EvalInputError`; nothing is
unpickled or executed. ``pyarrow`` is imported inside the functions that need it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cvpilot.core.errors import EvalInputError

if TYPE_CHECKING:
    import pyarrow as pa

MAX_FILE_BYTES = 1 << 30
"""Largest Parquet file read for evaluation (1 GiB)."""
MAX_UNCOMPRESSED_BYTES = 2 << 30
"""Largest decoded size of the requested columns (2 GiB), from row-group metadata."""
MAX_ROW_GROUPS = 100_000
"""Most row groups in a file read for evaluation."""
_EMPTY = {"bool": bool, "int": np.int64, "float": np.float64, "string": str}

MAX_COLUMNS = 4096
"""Most columns in the schema of a file read for evaluation."""
MAX_STRING_LENGTH = 256
"""Default longest string value accepted in a string column."""
MAX_STRING_BYTES = 512 * 1024 * 1024
"""Default memory budget of one string column as a fixed-width array (rows x longest x 4)."""


def _fail(path: Path, message: str) -> EvalInputError:
    return EvalInputError(f"{path.name}: {message}")


def _kind(field_type: pa.DataType) -> str | None:
    import pyarrow as pa

    t = field_type
    if pa.types.is_dictionary(t):
        t = t.value_type
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    return None


BATCH_ROWS = 65_536
"""Rows decoded per batch; the decoded-size budget is checked after every batch."""
_CONTROL = r"[\x00-\x1f\x7f]"


def _declared_size(meta: Any, schema: pa.Schema, wanted: Sequence[str]) -> int:
    """Decoded bytes the footer declares for the leaf columns rooted at ``wanted`` columns."""
    total = 0
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        for j in range(group.num_columns):
            column = group.column(j)
            root = column.path_in_schema.split(".", 1)[0]
            if root in wanted:
                total += column.total_uncompressed_size
    return total


def _plain_strings(array: Any, pa: Any, pc: Any, name: str, path: Path, limit: int) -> Any:
    """Validate one string array (or its dictionary) before it is expanded; return plain strings."""
    values = array.dictionary if pa.types.is_dictionary(array.type) else array
    if len(values) and (pc.max(pc.utf8_length(values)).as_py() or 0) > limit:
        raise _fail(path, f"column {name!r} has a value longer than {limit} characters")
    if len(values) and pc.any(pc.match_substring_regex(values, _CONTROL)).as_py():
        raise _fail(path, f"column {name!r} contains control characters")
    return array.dictionary_decode() if pa.types.is_dictionary(array.type) else array


class ParquetTables:
    """Reads and writes columnar tables as Parquet."""

    def read_columns(
        self,
        path: Path,
        *,
        required: Sequence[str],
        optional: Sequence[str] = (),
        nullable: Sequence[str] = (),
        max_rows: int,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
        max_row_groups: int = MAX_ROW_GROUPS,
        max_string_length: int = MAX_STRING_LENGTH,
        max_string_bytes: int = MAX_STRING_BYTES,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Read the ``required`` and any present ``optional`` columns of a Parquet file.

        Numeric columns keep their dtype (nullable ones become float64 with NaN for nulls and
        come with a boolean ``"<name>.null"`` mask), booleans become bool and strings become
        fixed-width ``str`` arrays. Footer metadata only gives a fast first rejection; the
        limits are enforced again on the data actually decoded, batch by batch, because a
        footer can lie. Dictionary values are length- and character-checked before expansion.

        Raises:
            EvalInputError: the path is not a regular file, the file is not valid Parquet, a
                limit is exceeded, a column is missing, nested or of an unsupported type, a
                string holds control characters, or a non-nullable column holds nulls.
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        path = Path(path)
        try:
            stat = path.stat()
        except OSError as exc:
            raise _fail(path, f"cannot read the file: {exc.strerror or exc}") from None
        if not path.is_file():
            raise _fail(path, "not a file")
        if stat.st_size > max_file_bytes:
            raise _fail(path, f"file is larger than the {max_file_bytes} byte limit")
        try:
            return self._read(
                pa, pc, pq, path, required, optional, nullable, max_rows, max_uncompressed_bytes,
                max_row_groups, max_string_length, max_string_bytes,
            )  # fmt: skip
        except EvalInputError:
            raise
        except (pa.ArrowException, OSError, ValueError, KeyError, UnicodeDecodeError) as exc:
            raise _fail(path, f"not a readable Parquet file ({type(exc).__name__})") from None

    def _read(
        self, pa: Any, pc: Any, pq: Any, path: Path, required: Sequence[str],
        optional: Sequence[str], nullable: Sequence[str], max_rows: int, max_bytes: int,
        max_groups: int, max_len: int, max_string_bytes: int,
    ) -> dict[str, np.ndarray[Any, Any]]:  # fmt: skip
        handle = pq.ParquetFile(path)
        meta, schema = handle.metadata, handle.schema_arrow
        if meta.num_rows > max_rows:
            raise _fail(path, f"{meta.num_rows} rows exceed the limit of {max_rows}")
        if meta.num_row_groups > max_groups:
            raise _fail(path, f"{meta.num_row_groups} row groups exceed the limit of {max_groups}")
        if len(schema.names) > MAX_COLUMNS:
            raise _fail(path, f"more than {MAX_COLUMNS} columns")
        present = set(schema.names)
        for name in required:
            if name not in present:
                raise _fail(path, f"missing column {name!r}")
        wanted = [*required, *(n for n in optional if n in present)]
        kinds: dict[str, str] = {}
        for name in wanted:
            kind = _kind(schema.field(name).type)
            if kind is None:
                raise _fail(
                    path, f"column {name!r} has an unsupported type {schema.field(name).type}"
                )
            kinds[name] = kind
        if _declared_size(meta, schema, wanted) > max_bytes:
            raise _fail(path, f"decoded size exceeds the {max_bytes} byte uncompressed limit")
        strings = [n for n in wanted if kinds[n] == "string"]
        reader = pq.ParquetFile(path, read_dictionary=strings) if strings else handle
        parts: dict[str, list[Any]] = {name: [] for name in wanted}
        masks: dict[str, list[Any]] = {name: [] for name in nullable if name in wanted}
        rows = decoded = longest = 0
        for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=wanted, use_threads=False):
            rows += batch.num_rows
            decoded += batch.nbytes
            if rows > max_rows:
                raise _fail(path, f"more than {max_rows} rows decoded")
            if decoded > max_bytes:
                raise _fail(path, f"decoded data exceeds the {max_bytes} byte limit")
            for name in wanted:
                col = batch.column(name)
                if col.null_count and name not in nullable:
                    raise _fail(path, f"column {name!r} contains nulls")
                if kinds[name] == "string":
                    plain = _plain_strings(col, pa, pc, name, path, max_len)
                    longest = max(longest, pc.max(pc.utf8_length(plain)).as_py() or 0)
                    if rows * max(longest, 1) * 4 > max_string_bytes:
                        raise _fail(path, f"column {name!r} is too large as an array")
                    parts[name].append(np.asarray(plain.to_numpy(zero_copy_only=False)).astype(str))
                elif col.null_count:
                    filled = pc.fill_null(col.cast(pa.float64()), float("nan"))
                    parts[name].append(np.asarray(filled.to_numpy()))
                    masks[name].append(np.asarray(col.is_null().to_numpy(zero_copy_only=False)))
                else:
                    parts[name].append(np.asarray(col.to_numpy(zero_copy_only=False)))
                    if name in masks:
                        masks[name].append(np.zeros(len(col), dtype=bool))
        out: dict[str, np.ndarray[Any, Any]] = {}
        for name in wanted:
            out[name] = (
                np.concatenate(parts[name])
                if parts[name]
                else np.zeros(0, dtype=_EMPTY[kinds[name]])
            )
            if name in masks:
                out[f"{name}.null"] = (
                    np.concatenate(masks[name]) if masks[name] else np.zeros(0, dtype=bool)
                )
        return out

    def write_columns(
        self,
        path: Path,
        columns: Mapping[str, np.ndarray[Any, Any]],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        """Write equal-length columns to ``path`` (zstd); the same columns give the same bytes."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({name: pa.array(values) for name, values in columns.items()})
        if metadata:
            table = table.replace_schema_metadata(
                {k.encode(): v.encode() for k, v in metadata.items()}
            )
        pq.write_table(table, path, compression="zstd")


def open_tables() -> ParquetTables:
    """Factory registered as ``cvpilot.tables:parquet``."""
    return ParquetTables()
