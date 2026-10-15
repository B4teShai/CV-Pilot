# ADR-0005: Local-first run store

**Status:** accepted
**Date:** 2026-10-14

## Context

CV-Pilot records run metadata, reproducibility capture, metric points (tens of thousands per
run), evaluation snapshots with per-class and per-slice tables, and artifacts (predictions,
confusion matrices, HTML reports, config, environment dumps). `compare` and `gate` read two
runs at once; `cvpilot runs ls` and the planned importers read many. Training machines are
often spot instances with no inbound network and sometimes no outbound network for minutes.
The CLI must work on a laptop with no server, and the core package must not require a
database server ([ADR-0003](0003-repo-and-tooling.md)).

Metric points are append-only and column-shaped; metadata is row-shaped and updated
(status transitions, baselines). Artifacts are immutable blobs that several runs may share
(the same predictions file re-registered, the same config).

## Options considered

### Option A: a client that only forwards to the user's tracker

No local state; everything goes to MLflow, W&B or Trackio through their SDKs.

Trade-offs: nothing new to operate; but `gate` would depend on the tracker being reachable
from CI, per-class and per-slice tables would be squeezed into tracker-specific artifact
formats, offline training would lose data, and the project would inherit every tracker's
data model differences. Checkpoint manifests and preemption lineage have no natural home in
a scalar tracker.

### Option B: a required server with a database

Clients stream everything to a FastAPI service backed by Postgres; the CLI queries the
service.

Trade-offs: one query surface and one place to secure; but the CLI stops working offline,
spot jobs lose metrics when the network drops, every user must run a service before the
first run, and the server becomes a mandatory piece of infrastructure with the security
surface that Aim's 2026 reports illustrate.

### Option C: local-first files, optional server, forwarders keep the tracker authoritative

Each project has a directory (default `~/.cvpilot/<project>/`) with `runs.sqlite` in WAL
mode for row-shaped data, Parquet files for metric points, and a content-addressed
directory for artifacts. DuckDB reads SQLite and Parquet together for cross-run queries. A
background thread forwards batches to the optional server with an idempotent `(run_id, seq)`
key; forwarders push scalars to the user's tracker.

Trade-offs: three file formats to keep consistent and a schema to version; in return the
CLI, `gate` and the Python API work with no network, run data is plain files the user owns
and can copy, and the server is a view, not the source of truth. Trackio and MLflow (SQLite
default since 3.7) both validate the local SQLite pattern.

## Decision

Option C.

Rules reviewers can check:

1. The store directory is `$CVPILOT_STORE_DIR/<project>/`, default `~/.cvpilot/<project>/`.
   Its layout is fixed in [../architecture/03-data-model.md](../architecture/03-data-model.md).
2. `runs.sqlite` is opened with `journal_mode=WAL` and `synchronous=NORMAL`; one writer per
   process, readers never block. Tables: `run_meta`, `repro`, `eval_snapshot`, `artifact`,
   `checkpoint_manifest`, `baseline`, `outbox`, `meta` (schema version).
3. Metric points are buffered in memory (at most 10 000 points or 2 s, whichever first) and
   appended as one Parquet file per flush under `metrics/<run_id>/`; `MetricPoint` is unique
   on `(run_id, key, step)`, the writer rejects duplicates in the same batch, and the step
   of a key must be strictly greater than the last stored step of that key (so uniqueness is
   a per-key high-water mark, not a set of every point). Files are
   compacted into one Parquet file per run at `finish()`.
4. Artifacts are stored once per project under `artifacts/<sha256[:2]>/<sha256>`; the
   `artifact` table maps `(run_id, sha256)` to kind and metadata. Checkpoints are not copied
   into the CAS; the manifest records their path and remote URI.
5. Cross-run reads go through DuckDB (`store/query.py`), never through hand-written joins
   over Parquet files in Python.
6. The user's tracker remains authoritative for scalars: every `MetricPoint` is also handed to
   the configured forwarders; CV-Pilot never asks the user to stop logging to their tracker.
7. The server, when configured, receives the same records through `sync/` and stores them in
   its own database. It never holds data the client does not also hold locally, except
   server-side alert state and share links.
8. `pyarrow` (Parquet) is a core dependency, added with the first store implementation;
   `duckdb` (queries) joins the core list in the change that lands `store/query.py`. Both are
   permissively licensed, CPU-only and torch-free. They are imported inside the store
   functions that need them, so `cvpilot --version` and `doctor` do not pay their import
   cost. This amends the core list in ADR-0003. Until `store/query.py` exists, `runs list`
   and `runs show` read SQLite and the Parquet index directly through the store API, which
   is the only departure from rule 5.

## Amendment: eval snapshots

The `eval_snapshot` table is an additive table: `store/additive.py` creates it idempotently when
a store is opened, the schema version stays 1 (03-data-model.md calls a new table a minor
change), and triggers make rows immutable. Evaluation inputs are read through a `tables`
registry (`store/tables.py`) that bounds Parquet reads from file metadata. `numpy` is justified
in ADR-0003.

## Consequences

- Easier: offline training, CI gates from a copied store directory, reproducible
  comparisons from plain files, migration-free adoption next to an existing tracker.
- Harder: schema versioning across three formats; a chaos test tier is needed for
  interrupted Parquet writes (a partial file is ignored by the reader because it lacks the
  footer, and the SQLite row that references it is written only after the file is renamed
  into place).
- Larger core install: the pyarrow and duckdb wheels add roughly 100 MB on disk; the budget
  in [../architecture/01-overview.md](../architecture/01-overview.md) accounts for it.
- Follow-up: `store/` implementation with migrations, `cvpilot import` for Trackio SQLite,
  MLflow and TensorBoard event files, `doctor` checks for store health.

## Links

- [../architecture/03-data-model.md](../architecture/03-data-model.md),
  [../architecture/05-runtime.md](../architecture/05-runtime.md)
- [0001-scope-and-license.md](0001-scope-and-license.md),
  [0006-checkpoint-semantics.md](0006-checkpoint-semantics.md)
