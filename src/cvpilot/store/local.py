"""Local run store: SQLite (WAL) for rows, Parquet for metric points, a content-addressed
directory for artifacts. See ADR-0005 and docs/architecture/03-data-model.md.

Single-writer discipline: every statement of one ``LocalRunStore`` goes through one connection
guarded by one lock, writes use ``BEGIN IMMEDIATE``, and a busy timeout covers a second
process. Readers in other processes never block (WAL).

A metric batch becomes visible in this order: Parquet file written to a temp name, fsynced,
renamed into place, then indexed in ``metric_files``. A killed process can therefore leave an
unindexed file or temp file (ignored by readers) but never an indexed partial file.

``pyarrow`` is imported inside the functions that need it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, get_args

from cvpilot.core.errors import (
    DuplicateMetricError,
    InvalidTransitionError,
    MetricOrderError,
    RunClosedError,
    RunNotFoundError,
    SnapshotIntegrityError,
    StoreError,
    StoreNotFoundError,
    StoreVersionError,
)
from cvpilot.core.hashing import sha256_file
from cvpilot.core.schema import (
    ACTIVE_STATUSES,
    Artifact,
    ArtifactKind,
    Baseline,
    EvalSnapshot,
    MetricPoint,
    MetricStats,
    Repro,
    RunMeta,
    RunStatus,
    check_transition,
    is_terminal,
)
from cvpilot.store import additive as _additive
from cvpilot.store import baselines as _baselines
from cvpilot.store import migrations as _migrations
from cvpilot.store import snapshots as _snapshots
from cvpilot.store.paths import require_run_id, require_sha256, resolve_project_dir

if TYPE_CHECKING:
    import pyarrow as pa

DB_NAME = "runs.sqlite"
_BUSY_TIMEOUT_S = 30.0
_PARQUET_COMPRESSION = "zstd"
_COMPACTED = "all.parquet"


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):  # not supported on every platform or filesystem
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class LocalRunStore:
    """One project's store directory. Use :func:`open_store` or :meth:`open`."""

    def __init__(self, path: Path, conn: sqlite3.Connection, schema_version: int) -> None:
        self._path = path
        self._conn = conn
        self._lock = threading.RLock()
        self._steps: dict[str, dict[str, int]] = {}
        self.schema_version = schema_version

    # -- opening ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        project: str,
        root: Path | None = None,
        *,
        create: bool = True,
        migrations: Sequence[_migrations.Migration] | None = None,
    ) -> LocalRunStore:
        """Open (and, with ``create``, initialise or migrate) the store of ``project``."""
        path = resolve_project_dir(root, project)
        db = path / DB_NAME
        if not db.exists():
            if not create:
                raise StoreNotFoundError(f"no run store for project {project!r} at {path}")
            try:
                path.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.close(os.open(db, os.O_CREAT | os.O_RDWR, 0o600))
            except OSError as exc:
                raise StoreError(f"cannot create the run store at {path}: {exc}") from exc
        try:
            conn = sqlite3.connect(
                db, timeout=_BUSY_TIMEOUT_S, isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open {db}: {exc}") from exc
        try:
            version = cls._initialise(conn, db, list(migrations or _migrations.discover()))
            _additive.ensure(conn, path / "evals")
        except BaseException:
            conn.close()
            raise
        for sub in ("metrics", "artifacts", "evals"):
            (path / sub).mkdir(exist_ok=True, mode=0o700)
        return cls(path, conn, version)

    @staticmethod
    def _initialise(
        conn: sqlite3.Connection, db: Path, available: list[_migrations.Migration]
    ) -> int:
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise StoreError(f"{db} does not support WAL mode (journal_mode={mode})")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA recursive_triggers=ON")  # REPLACE fires the delete triggers
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS migrations "
                "(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            conn.execute("COMMIT")
            target = available[-1].version if available else 0
            current = LocalRunStore._read_version(conn)
            if current > target:
                raise StoreVersionError(current, target)
            if current == target:
                return current
            if current > 0:
                LocalRunStore._backup(conn, db, current)
            return LocalRunStore._migrate(conn, available)
        except sqlite3.Error as exc:
            raise StoreError(f"cannot initialise {db}: {exc}") from exc

    @staticmethod
    def _read_version(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _backup(conn: sqlite3.Connection, db: Path, version: int) -> None:
        backup = db.with_name(f"{db.name}.bak-{version}")
        dest = sqlite3.connect(backup)
        try:
            conn.backup(dest)
        finally:
            dest.close()
        os.chmod(backup, 0o600)

    @staticmethod
    def _migrate(conn: sqlite3.Connection, available: list[_migrations.Migration]) -> int:
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = LocalRunStore._read_version(conn)  # another process may have migrated
            for step in available:
                if step.version <= current:
                    continue
                try:
                    step.upgrade(conn)
                except Exception as exc:
                    raise StoreError(
                        f"migration {step.version} ({step.name}) failed: {exc}"
                    ) from exc
                conn.execute(
                    "INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (step.version, step.name, _ts(_now())),
                )
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(step.version),),
                )
                current = step.version
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return current

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- transactions ----------------------------------------------------

    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    yield self._conn
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            except sqlite3.IntegrityError:
                raise
            except sqlite3.Error as exc:
                raise StoreError(f"run store write failed: {exc}") from exc

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            try:
                return self._conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                raise StoreError(f"run store read failed: {exc}") from exc

    def _status_of(self, conn: sqlite3.Connection, run_id: str) -> RunStatus:
        row = conn.execute("SELECT status FROM run_meta WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(f"run {run_id} not found in {self._path}")
        status: RunStatus = row[0]
        return status

    # -- runs ------------------------------------------------------------

    def put_run(self, meta: RunMeta, repro: Repro) -> None:
        if meta.run_id != repro.run_id:
            raise StoreError("meta and repro must share the same run_id")
        if meta.schema_version != self.schema_version:
            raise StoreError(
                f"run schema_version {meta.schema_version} does not match the store's "
                f"{self.schema_version}"
            )
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO run_meta (run_id, project, name, status, parent_run_id, task, "
                    "framework, host, started_at, ended_at, schema_version) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        meta.run_id,
                        meta.project,
                        meta.name,
                        meta.status,
                        meta.parent_run_id,
                        meta.task,
                        meta.framework,
                        meta.host,
                        _ts(meta.started_at),
                        None if meta.ended_at is None else _ts(meta.ended_at),
                        meta.schema_version,
                    ),
                )
                conn.execute(
                    "INSERT INTO repro VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        repro.run_id,
                        repro.git_commit,
                        int(repro.git_dirty),
                        repro.git_diff_sha256,
                        repro.packages_lock_sha256,
                        repro.python_version,
                        repro.cuda,
                        json.dumps(repro.gpu_names),
                        repro.dataset_uri,
                        repro.dataset_hash,
                        json.dumps(repro.config, allow_nan=False),
                        repro.seed,
                        repro.command_line,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc):
                raise StoreError(f"parent run {meta.parent_run_id} does not exist") from exc
            raise StoreError(f"run {meta.run_id} already exists") from exc

    @staticmethod
    def _meta_from_row(row: tuple[Any, ...]) -> RunMeta:
        return RunMeta(
            run_id=row[0],
            project=row[1],
            name=row[2],
            status=row[3],
            parent_run_id=row[4],
            task=row[5],
            framework=row[6],
            host=row[7],
            started_at=datetime.fromisoformat(row[8]),
            ended_at=None if row[9] is None else datetime.fromisoformat(row[9]),
            schema_version=row[10],
        )

    _META_COLUMNS = (
        "run_id, project, name, status, parent_run_id, task, framework, host, "
        "started_at, ended_at, schema_version"
    )

    def get_run(self, run_id: str) -> RunMeta:
        require_run_id(run_id)
        rows = self._query(
            f"SELECT {self._META_COLUMNS} FROM run_meta WHERE run_id=?",  # noqa: S608  # constant
            (run_id,),
        )
        if not rows:
            raise RunNotFoundError(f"run {run_id} not found in {self._path}")
        return self._meta_from_row(rows[0])

    def get_repro(self, run_id: str) -> Repro:
        require_run_id(run_id)
        rows = self._query("SELECT * FROM repro WHERE run_id=?", (run_id,))
        if not rows:
            raise RunNotFoundError(f"run {run_id} not found in {self._path}")
        r = rows[0]
        return Repro(
            run_id=r[0],
            git_commit=r[1],
            git_dirty=bool(r[2]),
            git_diff_sha256=r[3],
            packages_lock_sha256=r[4],
            python_version=r[5],
            cuda=r[6],
            gpu_names=json.loads(r[7]),
            dataset_uri=r[8],
            dataset_hash=r[9],
            config=json.loads(r[10]),
            seed=r[11],
            command_line=r[12],
        )

    def list_runs(self, *, status: RunStatus | None = None) -> list[RunMeta]:
        sql = f"SELECT {self._META_COLUMNS} FROM run_meta"  # noqa: S608  # constant columns only
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status=?"
            params = (status,)
        rows = self._query(sql + " ORDER BY started_at DESC, run_id DESC", params)
        return [self._meta_from_row(r) for r in rows]

    def set_status(self, run_id: str, status: RunStatus, ended_at: datetime) -> None:
        require_run_id(run_id)
        if not is_terminal(status):
            raise InvalidTransitionError(f"run status cannot change to {status}")
        with self._tx() as conn:
            current = self._status_of(conn, run_id)
            check_transition(current, status)
            started = datetime.fromisoformat(
                conn.execute(
                    "SELECT started_at FROM run_meta WHERE run_id=?", (run_id,)
                ).fetchone()[0]
            )
            if ended_at.astimezone(timezone.utc) < started:
                raise StoreError("ended_at is before started_at")
            conn.execute(
                "UPDATE run_meta SET status=?, ended_at=? WHERE run_id=?",
                (status, _ts(ended_at), run_id),
            )

    # -- metrics ---------------------------------------------------------

    @staticmethod
    def _metric_schema() -> pa.Schema:
        import pyarrow as pa

        return pa.schema(
            [
                ("run_id", pa.string()),
                ("key", pa.string()),
                ("step", pa.int64()),
                ("epoch", pa.int64()),
                ("value", pa.float64()),
                ("wall_time", pa.float64()),
            ],
            metadata={b"cvpilot.schema_version": b"1"},
        )

    def _write_parquet(self, target: Path, table: pa.Table) -> None:
        """Write ``table`` to ``target`` atomically: temp file, fsync, rename, fsync dir."""
        import pyarrow.parquet as pq

        tmp = target.with_name(f".{target.name}.tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                pq.write_table(table, handle, compression=_PARQUET_COMPRESSION)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        _fsync_dir(target.parent)

    def _files(self, run_id: str) -> list[tuple[int, str]]:
        rows = self._query(
            "SELECT seq, file FROM metric_files WHERE run_id=? ORDER BY seq", (run_id,)
        )
        return [(r[0], r[1]) for r in rows]

    def _read_file(self, run_id: str, name: str, columns: list[str] | None = None) -> pa.Table:
        import pyarrow as pa
        import pyarrow.parquet as pq

        file = self._path / "metrics" / run_id / name
        if not file.is_file():
            raise StoreError(f"metric file {name} of run {run_id} is missing")
        try:
            return pq.read_table(file, columns=columns)
        except (pa.ArrowException, OSError) as exc:
            raise StoreError(f"cannot read metric file {name} of run {run_id}: {exc}") from exc

    def _last_steps(self, run_id: str) -> dict[str, int]:
        if run_id not in self._steps:
            import pyarrow as pa

            tables = [self._read_file(run_id, f, ["key", "step"]) for _, f in self._files(run_id)]
            result: dict[str, int] = {}
            if tables:
                agg = pa.concat_tables(tables).group_by("key").aggregate([("step", "max")])
                result = dict(
                    zip(
                        agg.column("key").to_pylist(),
                        agg.column("step_max").to_pylist(),
                        strict=True,
                    )
                )
            self._steps[run_id] = result
        return self._steps[run_id]

    def append_points(self, points: Sequence[MetricPoint]) -> str:
        """Append one batch as one Parquet file; return its file name.

        The whole batch is rejected on any violation.

        Raises:
            DuplicateMetricError: a ``(key, step)`` repeats one already stored or in the batch.
            MetricOrderError: a step is lower than the last stored step of its key.
            RunNotFoundError, RunClosedError: the run is unknown or already in a final status.
            StoreError: the files or the index could not be written.
        """
        if not points:
            return ""
        run_id = require_run_id(points[0].run_id)
        if any(p.run_id != run_id for p in points):
            raise ValueError("a batch must belong to one run")
        with self._lock:
            with self._tx() as conn:
                if self._status_of(conn, run_id) not in ACTIVE_STATUSES:
                    raise RunClosedError(
                        f"run {run_id} is finished; metrics can no longer be added"
                    )
            last = dict(self._last_steps(run_id))
            for p in points:
                prior = last.get(p.key)
                if prior is not None and p.step == prior:
                    raise DuplicateMetricError(
                        f"duplicate metric point key={p.key!r} step={p.step}"
                    )
                if prior is not None and p.step < prior:
                    raise MetricOrderError(
                        f"step {p.step} for key {p.key!r} is not greater than the last step {prior}"
                    )
                last[p.key] = p.step
            seq = max((s for s, _ in self._files(run_id)), default=0) + 1
            table = self._to_table(points)
            directory = self._path / "metrics" / run_id
            name = f"{seq:06d}.parquet"
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._write_parquet(directory / name, table)
            except OSError as exc:
                raise StoreError(f"cannot write metric file {name}: {exc}") from exc
            steps = [p.step for p in points]
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO metric_files VALUES (?,?,?,?,?,?,?)",
                    (run_id, seq, name, len(points), min(steps), max(steps), _ts(_now())),
                )
            self._steps[run_id] = last
        return name

    def _to_table(self, points: Sequence[MetricPoint]) -> pa.Table:
        import pyarrow as pa

        return pa.table(
            {
                "run_id": [p.run_id for p in points],
                "key": [p.key for p in points],
                "step": [p.step for p in points],
                "epoch": [p.epoch for p in points],
                "value": [p.value for p in points],
                "wall_time": [p.wall_time for p in points],
            },
            schema=self._metric_schema(),
        )

    def _require_run(self, run_id: str) -> None:
        require_run_id(run_id)
        if not self._query("SELECT 1 FROM run_meta WHERE run_id=?", (run_id,)):
            raise RunNotFoundError(f"run {run_id} not found in {self._path}")

    def read_points(self, run_id: str) -> list[MetricPoint]:
        """All indexed points of a run in write order; unindexed files are ignored."""
        self._require_run(run_id)
        out: list[MetricPoint] = []
        for _, name in self._files(run_id):
            cols = self._read_file(run_id, name).to_pydict()
            out.extend(
                MetricPoint(run_id=r, key=k, step=s, epoch=e, value=v, wall_time=w)
                for r, k, s, e, v, w in zip(
                    cols["run_id"],
                    cols["key"],
                    cols["step"],
                    cols["epoch"],
                    cols["value"],
                    cols["wall_time"],
                    strict=True,
                )
            )
        return out

    def metric_stats(self, run_id: str) -> MetricStats:
        self._require_run(run_id)
        row = self._query(
            "SELECT COUNT(*), COALESCE(SUM(n_points), 0), MIN(min_step), MAX(max_step) "
            "FROM metric_files WHERE run_id=?",
            (run_id,),
        )[0]
        return MetricStats(n_files=row[0], n_points=row[1], min_step=row[2], max_step=row[3])

    def compact_metrics(self, run_id: str) -> None:
        """Merge the batch files of a run into ``all.parquet`` (idempotent)."""
        import pyarrow.parquet as pq

        self._require_run(run_id)
        with self._lock:
            files = self._files(run_id)
            if not files or (len(files) == 1 and files[0][0] == 0):
                return
            directory = self._path / "metrics" / run_id
            target = directory / _COMPACTED
            tmp = directory / f".{_COMPACTED}.tmp"
            try:
                fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    with pq.ParquetWriter(
                        handle, self._metric_schema(), compression=_PARQUET_COMPRESSION
                    ) as writer:
                        for _, name in files:
                            writer.write_table(self._read_file(run_id, name))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, target)
                _fsync_dir(directory)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise StoreError(f"cannot compact metrics of run {run_id}: {exc}") from exc
            stats = self.metric_stats(run_id)
            with self._tx() as conn:
                conn.execute("DELETE FROM metric_files WHERE run_id=?", (run_id,))
                conn.execute(
                    "INSERT INTO metric_files VALUES (?,?,?,?,?,?,?)",
                    (
                        run_id,
                        0,
                        _COMPACTED,
                        stats.n_points,
                        stats.min_step,
                        stats.max_step,
                        _ts(_now()),
                    ),
                )
            for _, name in files:
                if name != _COMPACTED:
                    with contextlib.suppress(OSError):
                        (directory / name).unlink()

    # -- artifacts -------------------------------------------------------

    def artifact_path(self, sha256: str) -> Path:
        require_sha256(sha256)
        return self._path / "artifacts" / sha256[:2] / sha256

    def _ingest(self, source: BinaryIO) -> tuple[str, int]:
        """Copy a stream into the CAS; return its sha256 and size."""
        digest = hashlib.sha256()
        size = 0
        fd, tmp_name = tempfile.mkstemp(dir=self._path / "artifacts", prefix=".tmp-")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := source.read(1 << 20):
                    digest.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            sha = digest.hexdigest()
            dest = self.artifact_path(sha)
            dest.parent.mkdir(exist_ok=True, mode=0o700)
            os.chmod(tmp, 0o444)
            if dest.exists() and self._artifact_intact(dest, sha, size):
                tmp.unlink()
            else:  # new content, or a damaged copy: the verified bytes replace it
                os.replace(tmp, dest)
                _fsync_dir(dest.parent)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        return sha, size

    @staticmethod
    def _artifact_intact(path: Path, sha: str, size: int) -> bool:
        """True when ``path`` has the expected size and SHA-256 (re-hashed, not trusted)."""
        try:
            if path.stat().st_size != size:
                return False
            return sha256_file(path) == sha
        except OSError:
            return False

    def open_artifact(self, sha256: str) -> BinaryIO:
        """Open a stored artifact for reading after verifying its SHA-256.

        Raises:
            StoreError: the address is malformed or the file is missing.
            SnapshotIntegrityError: the stored bytes do not hash to the address.
        """
        path = self.artifact_path(sha256)
        try:
            handle = path.open("rb")
        except FileNotFoundError:
            raise StoreError(f"artifact {sha256[:12]} is missing") from None
        except OSError as exc:
            raise StoreError(f"cannot read artifact {sha256[:12]}: {exc}") from exc
        digest = hashlib.sha256()
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
        if digest.hexdigest() != sha256:
            handle.close()
            raise SnapshotIntegrityError(
                f"artifact {sha256[:12]} failed its integrity check: digest mismatch"
            )
        handle.seek(0)
        return handle

    def _register(
        self, run_id: str, kind: ArtifactKind, meta: Mapping[str, Any], sha: str, size: int
    ) -> Artifact:
        encoded = json.dumps(meta, allow_nan=False, sort_keys=True)
        with self._tx() as conn:
            known = conn.execute(
                "SELECT kind, meta FROM artifact WHERE run_id=? AND sha256=?", (run_id, sha)
            ).fetchone()
            if known is not None:
                same = (
                    known[0] == kind
                    and json.dumps(json.loads(known[1]), allow_nan=False, sort_keys=True) == encoded
                )
                if not same:
                    raise StoreError(
                        f"artifact {sha[:12]} is already registered for run {run_id} "
                        "with a different kind or meta"
                    )
            else:
                conn.execute(
                    "INSERT INTO artifact VALUES (?,?,?,?,?,?,?)",
                    (
                        run_id,
                        sha,
                        kind,
                        f"artifacts/{sha[:2]}/{sha}",
                        size,
                        json.dumps(meta, allow_nan=False),
                        _ts(_now()),
                    ),
                )
        return next(a for a in self.list_artifacts(run_id) if a.sha256 == sha)

    def _check_artifact_args(self, run_id: str, kind: str, meta: Mapping[str, Any]) -> None:
        self._require_run(run_id)
        if kind not in get_args(ArtifactKind):
            raise ValueError(f"unknown artifact kind {kind!r}")
        try:
            json.dumps(meta, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"artifact meta must be JSON-serialisable: {exc}") from None

    def put_artifact(
        self, run_id: str, path: Path, kind: ArtifactKind, meta: Mapping[str, Any]
    ) -> Artifact:
        """Store a file by content address and register it with the run."""
        self._check_artifact_args(run_id, kind, meta)
        source = Path(path).expanduser()
        if not source.stat().st_mode & 0o170000 == 0o100000:
            raise StoreError(f"{source} is not a regular file")
        with source.open("rb") as handle:
            sha, size = self._ingest(handle)
        return self._register(run_id, kind, meta, sha, size)

    def put_artifact_bytes(
        self, run_id: str, data: bytes, kind: ArtifactKind, meta: Mapping[str, Any]
    ) -> Artifact:
        """Store in-memory bytes by content address and register them with the run."""
        import io

        self._check_artifact_args(run_id, kind, meta)
        sha, size = self._ingest(io.BytesIO(data))
        return self._register(run_id, kind, meta, sha, size)

    def list_artifacts(self, run_id: str) -> list[Artifact]:
        self._require_run(run_id)
        rows = self._query(
            "SELECT run_id, sha256, kind, uri, size_bytes, meta, created_at FROM artifact "
            "WHERE run_id=? ORDER BY created_at, sha256",
            (run_id,),
        )
        return [
            Artifact(
                run_id=r[0],
                sha256=r[1],
                kind=r[2],
                uri=r[3],
                size_bytes=r[4],
                meta=json.loads(r[5]),
                created_at=datetime.fromisoformat(r[6]),
            )
            for r in rows
        ]

    # -- eval snapshots (see store/snapshots.py) -------------------------

    def has_snapshot(self, run_id: str, eval_set_hash: str, step: int) -> bool:
        return _snapshots.has_snapshot(self, run_id, eval_set_hash, step)

    def put_snapshot(self, snapshot: EvalSnapshot) -> None:
        """Persist an eval snapshot once; it can never be replaced (invariants 3 and 6).

        Raises:
            RunNotFoundError: unknown run.
            SnapshotExistsError: a snapshot with this key exists.
            StoreError: a referenced artifact is missing or of the wrong kind, the file name
                is taken, or the write failed.
        """
        _snapshots.put_snapshot(self, snapshot)

    def get_snapshot(
        self, run_id: str, eval_set_hash: str, step: int | None = None
    ) -> EvalSnapshot:
        """The snapshot at ``step`` (latest when None), verified against its digest.

        Raises:
            SnapshotNotFoundError: no such snapshot.
            SnapshotIntegrityError: the file, its index row or the hash chain does not match.
            StoreError: the stored file is missing, too large or unreadable.
        """
        return _snapshots.get_snapshot(self, run_id, eval_set_hash, step)

    def list_snapshots(self, run_id: str) -> list[EvalSnapshot]:
        """All verified snapshots of a run ordered by eval set hash then step."""
        return _snapshots.list_snapshots(self, run_id)

    def snapshot_digest(self, run_id: str, eval_set_hash: str, step: int) -> str:
        """SHA-256 of the snapshot file at ``step``, after verifying the file and the chain.

        Raises:
            SnapshotNotFoundError, SnapshotIntegrityError, StoreError: see ``get_snapshot``.
        """
        _snapshots.get_snapshot(self, run_id, eval_set_hash, step)
        digest: str = self._query(
            "SELECT snapshot_sha256 FROM eval_snapshot WHERE run_id=? AND eval_set_hash=? "
            "AND step=?",
            (run_id, eval_set_hash, step),
        )[0][0]
        return digest

    # -- baselines (see store/baselines.py) ------------------------------

    def put_baseline(self, baseline: Baseline, *, replace: bool = False) -> None:
        """Pin a snapshot under a tag; an existing tag needs ``replace`` and stays in the history.

        Raises:
            RunNotFoundError: unknown run.
            BaselineRefusedError: the run is of another project, has no such snapshot, or the
                snapshot digest differs from the stored one.
            BaselineExistsError: the tag is pinned and ``replace`` is false.
            StoreError: too many rows or the write failed.
        """
        _baselines.put_baseline(self, baseline, replace=replace)

    def get_baseline(self, tag: str) -> Baseline:
        """The newest pin of ``tag``, verified (record, chain and the pinned snapshot digest).

        Raises:
            BaselineNotFoundError: the tag is unknown.
            BaselineIntegrityError: a row, the chain or the pinned snapshot does not verify.
        """
        return _baselines.get_baseline(self, tag)

    def list_baselines(self, *, history: bool = False) -> list[Baseline]:
        """Verified pins: newest per tag, or with ``history`` every row in pin order."""
        return _baselines.list_baselines(self, history=history)

    def baseline_head(self) -> tuple[int, str] | None:
        """``(seq, chain digest)`` of the newest baseline row (verified), None when empty."""
        return _baselines.baseline_head(self)


def open_store(project: str, root: Path | None = None, *, create: bool = True) -> LocalRunStore:
    """Factory registered as ``cvpilot.stores:local``."""
    return LocalRunStore.open(project, root, create=create)
