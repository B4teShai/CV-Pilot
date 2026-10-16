# 03 Data model

![Run store data model](../diagrams/png/data-model.png)

All entities are Pydantic models in `core/schema.py` and rows in the local run store of
[ADR-0005](../adr/0005-run-store.md). Built: `RunMeta`, `Repro`, `MetricPoint`, `Artifact`
(models, store tables and tests), `EvalSnapshot`, `Baseline`, `CheckpointManifest`. Types are Python types; `json` means a JSON column validated against
the model on read. Timestamps are timezone-aware, normalised to UTC and stored as ISO 8601.

## Entities

### RunMeta

One training run. A resumed run is a new `RunMeta` whose `parent_run_id` points at the
preempted one.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | `str` | no | ULID, 26 characters, Crockford base32; generated once at `init` |
| `project` | `str` | no | store directory name; `[a-z0-9][a-z0-9-_]{0,63}` |
| `name` | `str` | no | human label; defaults to `<framework>-<started_at date>` |
| `status` | `Literal["running","resumed","finished","failed","preempted"]` | no | see invariant 9 |
| `parent_run_id` | `str` | yes | the preempted run this run continues |
| `task` | `Literal["detection","classification"]` | no | selects the eval engine path |
| `framework` | `str` | no | `"<name>==<version>"`, for example `lightning==2.6.6` |
| `host` | `str` | no | hostname only; never user name or IP |
| `started_at` | `datetime` | no | |
| `ended_at` | `datetime` | yes | set by `finish_run` |
| `schema_version` | `int` | no | store schema the row was written with |

Primary key: `run_id`. Foreign keys: `parent_run_id → RunMeta.run_id`.

### Repro

Reproducibility capture taken once at `init`.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | `str` | no | |
| `git_commit` | `str` | yes | `None` outside a git checkout |
| `git_dirty` | `bool` | no | uncommitted changes present |
| `git_diff_sha256` | `str` | yes | content address of the diff artifact when dirty |
| `packages_lock_sha256` | `str` | no | SHA-256 of `uv.lock`, `poetry.lock` or `pip freeze` output |
| `python_version` | `str` | no | |
| `cuda` | `str` | yes | CUDA runtime and driver, `"12.4 / 550.54"` |
| `gpu_names` | `list[str]` | no | empty on CPU |
| `dataset_uri` | `str` | no | path or URL the training data was read from |
| `dataset_hash` | `str` | no | `dcp:<manifest.version>` or `merkle:<hex>`; see versioning |
| `config` | `json` | no | resolved training config |
| `seed` | `int` | yes | |
| `command_line` | `str` | no | `sys.argv` with secrets redacted by key and by pattern |

Primary key: `run_id`. Foreign keys: `run_id → RunMeta.run_id` (one-to-one).

### MetricPoint

One scalar at one step. Stored in Parquet under `metrics/<run_id>/`; the SQLite table holds
only the Parquet file index.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | `str` | no | |
| `key` | `str` | no | `"<split>/<metric>[/<class or slice>]"`, for example `val/AP/coke_zero_330ml` |
| `step` | `int` | no | global optimiser step, `>= 0`; strictly increasing per `key` within a run |
| `epoch` | `int` | yes | |
| `value` | `float64` | no | NaN and ±inf permitted; NaN triggers `nan_loss` |
| `wall_time` | `float64` | no | Unix seconds at log time |

Primary key: `(run_id, key, step)`. Foreign keys: `run_id → RunMeta.run_id`.

### EvalSnapshot (built)

One evaluation of one run on one evaluation set at one step.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | `str` | no | |
| `eval_set_hash` | `str` | no | content hash of the ground truth and class list; see versioning |
| `step` | `int` | no | `0 <= step < 2**63` |
| `task` | `Literal["detection","classification"]` | no | added field |
| `class_names` | `list[str]` | no | added field; index = class id; unique, at most 2 048 |
| `n_images` | `int` | no | added field |
| `metrics` | `json: dict[str, float]` | no | headline metrics, fractions in [0, 1], finite; undefined ones are absent: `mAP50_95`, `mAP50`, `mAP75`, `mAP_small/medium/large`, `AR1/10/100`, `AR_small/medium/large`; classification: `top1`, `macro_precision`, `macro_recall` |
| `per_class` | `json: dict[str, dict[str, float]]` | no | class name → `{AP50_95, AP50, AR100, support}` (AP keys absent without ground truth); classification `{precision, recall, f1, support}` |
| `slices` | `json: dict[str, dict[str, float]]` | no | `"camera=cam03"` → `{mAP50_95, n_images}` (classification: `top1`) |
| `ci` | `json: dict[str, [lo, hi]]` | no | added field; bootstrap intervals of `mAP50_95`, `mAP50`, `AR100` or `top1`; empty when off |
| `bootstrap` | `json?` | yes | added field; `{n_resamples, seed, level, numpy_version}`, set exactly when `ci` is non-empty (the resampling stream is reproducible within one numpy version) |
| `ci_status` | `"off" \| "computed" \| "not_computed_too_few_images"` | no | added field; `computed` exactly when `ci` is non-empty; fewer than 2 images is recorded, not silently omitted |
| `params` | `json?` | yes | added field; `{iou_thresholds, confusion_iou, confusion_score, max_dets}` so later comparisons can refuse snapshots made with other parameters. `max_dets` is only a field with default 100 of `EvalParams`: the evaluator always uses 100 and `--recompute` refuses a snapshot that records another value. IoU grids and confusion settings are compared with a relative tolerance of 1e-9, so a one-ULP platform difference does not refuse a comparison |
| `provenance` | `json?` | yes | added field; optional `{checkpoint_sha256, git_commit}` supplied by the caller |
| `slice_spec` | `json` | no | added field; slice key to image metadata column, so slice rows are comparable |
| `slice_hash` | `str?` | yes | added field; SHA-256 over every slice column's per-image values in canonical image order (`core/hashing.slice_hash`); null without slices and for snapshots written before it existed. `compare` refuses when it differs, so the same ground truth cannot be sliced differently |
| `confusion_sha256` | `str` | no | artifact holding the `(C+1)×(C+1)` matrix as sparse Parquet rows `(true_class, pred_class, count)`; index `C` is background |
| `predictions_sha256` | `str` | no | artifact holding predictions Parquet: `image_id, cls, score, x1, y1, x2, y2` |

Primary key: `(run_id, eval_set_hash, step)`. Foreign keys: `run_id → RunMeta.run_id`;
`(run_id, confusion_sha256)` and `(run_id, predictions_sha256) → Artifact`.

Integrity (`store/snapshots.py`): the `eval_snapshot` index row holds `snapshot_sha256` (SHA-256 of
the exact JSON bytes) and `chain_sha256` (digest over the previous row of the run, the key and
`snapshot_sha256`). Every read re-hashes the file, compares task, step, `n_images` and both artifact
addresses with the row and recomputes the run's chain; a mismatch raises `SnapshotIntegrityError`
(exit 1, event `snapshot_integrity_failed`). Files are created exclusively with mode 0444 inside the
same `BEGIN IMMEDIATE` transaction as the insert and never replaced. This catches edited files,
swapped artifact addresses and removed or reordered earlier rows. It does not catch someone who can
write both the database and the files (nothing is signed) or the removal of the newest row without an
external anchor; the UPDATE/DELETE triggers and `recursive_triggers` are defence in depth, the digest
is the control. Stores written before this change are not supported (the feature is unreleased).

### Artifact

A file produced by or registered with a run, stored by content address.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | `str` | no | |
| `sha256` | `str` | no | SHA-256 hex of the file bytes; the content address |
| `kind` | `Literal["checkpoint","report","predictions","confusion","config","env","image","diff"]` | no | |
| `uri` | `str` | no | CAS path, or the checkpoint's path/URI for `kind="checkpoint"` |
| `size_bytes` | `int` | no | |
| `meta` | `json` | no | kind-specific; `{}` when empty |
| `created_at` | `datetime` | no | |

Primary key: `(run_id, sha256)`. Foreign keys: `run_id → RunMeta.run_id`.

### CheckpointManifest (built)

Sidecar record written next to every checkpoint file ([ADR-0006](../adr/0006-checkpoint-semantics.md)).

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `run_id` | `str` | no | |
| `step` | `int` | no | |
| `kind` | `Literal["last","best"]` | no | full training state vs weights + EMA |
| `path` | `str` | no | local path relative to the checkpoint dir |
| `uri` | `str` | yes | the intended remote location (validated: no credentials); a `step_N.remote.json` marker records that the upload verified |
| `sha256` | `str` | no | checksum of the checkpoint file |
| `size_bytes` | `int` | no | |
| `format` | `Literal["safetensors","pt"]` | no | `pt` (named for the file suffix) is opt-in and loaded with `weights_only=True` only; safetensors is the default for both kinds |
| `metric_key` | `str` | yes | monitored metric for `best` (set together with `metric_value`) |
| `metric_value` | `float` | yes | finite |
| `metric_mode` | `Literal["max","min"]` | yes | how `best` is ranked (added field) |
| `versions` | `json: dict[str, str]` | no | `cvpilot`, `python`, the run's framework (at most 16 short entries) |
| `git_commit` | `str` | yes | |
| `dataset_hash` | `str` | no | copy of `Repro.dataset_hash` |
| `created_at` | `datetime` | no | informational: order is by `step`, never by time (clock skew is tested) |
| `prev_manifest_sha256` | `str` | no | SHA-256 of the previous manifest file of the same `(run, kind)` (64 zeros for the first) (added field) |
| `prev_chain_sha256` | `str` | no | the previous manifest's `chain_sha256` (added field) |
| `chain_sha256` | `str` | no | `sha256(prev_chain, run_id, kind, step, sha256, size_bytes)` (added field) |
| `schema_version` | `int` | yes | `2` for manifests of this release, absent in older ones (refused under the strict policies); covered by the tag (added field) |
| `chain_version` | `int` | yes | the chain algorithm (`1`: the serializer format is part of the digest) (added field) |
| `tag_alg` | `str` | yes | `hmac-sha256-v1` when tagged (added field) |
| `key_id` | `str` | yes | 16 hex characters identifying the key that tagged it, so a rotated key is told from a missing one (added field) |
| `project` | `str` | yes | project of the writing run, covered by the tag; a restart of another project refuses the manifest (`binding_mismatch`) (added field) |
| `parent_run_id` | `str` | yes | the run that run resumed, covered by the tag; checked in a lineage restore when `project` is set (added field) |
| `chain_reset` | `bool` | no | true when `restart_chain=True` deliberately started a new chain although progress was known (added field) |
| `chain_reset_reason` | `str` | yes | why the chain was reset (added field) |
| `prev_step` | `int` | yes | step of the manifest `prev_manifest_sha256` names, so a predecessor removed by retention (a gap) is told from one that was replaced (added field; absent in manifests written before it existed) |
| `auth_tag` | `str` | yes | HMAC-SHA256 over a domain label, the run id and the canonical record, present when `CVPILOT_CKPT_KEY` was set (added field) |

Primary key: `(run_id, step, kind)`. Foreign key: `run_id → RunMeta.run_id`. The manifest file
is immutable (0444). The run store keeps the digest of each manifest file in the append-only
`checkpoint_manifest` table (`run_id, kind, step, sha256, size_bytes, manifest_sha256,
chain_sha256, record, created_at`) as the external anchor; checkpoints are not registered as
`Artifact` rows (the files are not copied into the content-addressed store), so the `(run_id,
sha256) → Artifact` link of the first sketch does not exist.

The pointer file (`CheckpointPointer`) holds `run_id`, `kind`, `step`, the manifest's relative
name, its digest and chain value, a digest over those fields and the optional tag.

### Baseline (built)

A named pin of one run's snapshot on one evaluation set, used by `compare` and `gate`. The table
is append-only: pinning a tag that exists needs `--replace` and adds a newer row, so the history
stays and a replacement is always visible. The current pin of a tag is its newest row.

| Field | Type | Nullable | Meaning |
|---|---|---|---|
| `project` | `str` | no | must equal the project of the run |
| `tag` | `str` | no | `[a-z0-9][a-z0-9_-]{0,31}` (never starts with `-`, so it cannot be read as an option); `prod`, `staging`, free text |
| `run_id` | `str` | no | |
| `eval_set_hash` | `str` | no | which snapshot of the run is pinned |
| `step` | `int` | no | the pinned step (the latest of the pair unless `--step` is given) |
| `snapshot_sha256` | `str` | no | SHA-256 of the pinned snapshot file when it was pinned |
| `pinned_at` | `datetime` | no | |

Key: `seq` (autoincrement, the order of the chain); lookups use `(tag, seq)`. Foreign keys:
`run_id → RunMeta.run_id`; `(run_id, eval_set_hash, step) → EvalSnapshot`.

Integrity (`store/baselines.py`): the row stores the canonical JSON of the record, its SHA-256
(`record_sha256`) and `chain_sha256 = sha256(previous chain, seq, tag, record_sha256)` over all
rows in `seq` order. Every read re-hashes the record, compares it with the row's columns, walks
the whole chain from the first row (an edited, removed, inserted or reordered row breaks it) and
checks that the pinned snapshot still exists with the digest recorded at pin time; any mismatch
raises `BaselineIntegrityError` (exit 1, event `baseline_integrity_failed`). Pinning reads the
snapshot through `get_snapshot` first, so a baseline never points at a snapshot that fails its
own verification, and the insert re-checks the digest inside the same `BEGIN IMMEDIATE`
transaction. `UPDATE` and `DELETE` triggers are defence in depth. Not covered, as for snapshots:
someone who can write the database and the snapshot files can recompute every digest. Deleting
the newest row is invisible to the store itself and is not harmless: undoing a `--replace` makes
the previous pin current again. The control is an external anchor: `baseline show` prints
`head_seq` and `head_chain_sha256`, and `gate --expect-baseline-chain` compares the newest
row's chain digest with a value kept outside the store. At most 10 000 rows per project.

## Relations

| From | To | Cardinality | Via |
|---|---|---|---|
| RunMeta | Repro | 1 → 0..1 | `Repro.run_id` |
| RunMeta | MetricPoint | 1 → 0..* | `MetricPoint.run_id` |
| RunMeta | EvalSnapshot | 1 → 0..* | `EvalSnapshot.run_id` |
| RunMeta | Artifact | 1 → 0..* | `Artifact.run_id` |
| RunMeta | CheckpointManifest | 1 → 0..* | `CheckpointManifest.run_id` |
| RunMeta | RunMeta (child) | 1 → 0..1 | `RunMeta.parent_run_id` |
| CheckpointManifest | Artifact | 1 → 1 | `(run_id, sha256)` |
| EvalSnapshot | Artifact | 1 → 2 | `confusion_sha256`, `predictions_sha256` |
| Baseline | RunMeta | 0..* → 1 | `Baseline.run_id` |
| Baseline | EvalSnapshot | 1 → 1 | `(run_id, eval_set_hash)` |

## Invariants

Each sentence is a check in `core/schema.py` validators or a store constraint, with the
test that will cover it.

1. `run_id` is a valid ULID (26 Crockford base32 characters) and is generated exactly once
   per `init`; two `init` calls never share a `run_id`.
2. `MetricPoint` is unique on `(run_id, key, step)`; a batch containing a duplicate is
   rejected as a whole with `DuplicateMetricError` naming the key and step. Within a run the
   step of a key must also be strictly greater than the last stored step of that key
   (`MetricOrderError` otherwise); this makes the uniqueness check a per-key high-water mark
   instead of a set of every point, and is enforced twice: at `log()` and by the store. Built.
   The remaining invariants carry their status in the next list.
3. `EvalSnapshot` is unique on `(run_id, eval_set_hash, step)` and immutable (a repeat raises `SnapshotExistsError`; triggers refuse UPDATE and DELETE). Built.
4. `compare` and `gate` require `candidate.eval_set_hash == baseline.eval_set_hash`, equal
   `task`, equal `class_names` and equal `params`; when anything differs, or `params` is missing,
   the command logs `compare_refused` with the reason and both hashes and exits 1 without
   producing a result or report. Built.
5. `Artifact.sha256` equals the SHA-256 of the stored bytes; the CAS path is
   `artifacts/<sha256[:2]>/<sha256>`; identical content is stored once per project.
6. `EvalSnapshot.confusion_sha256` and `predictions_sha256` reference `Artifact` rows of the
   same `run_id` with kinds `confusion` and `predictions`.
7. A `CheckpointManifest` row exists only after the checkpoint file has been `fsync`ed, renamed
   into place and re-read and re-hashed; `LATEST`/`LATEST_BEST` are rewritten last and point at
   a manifest that exists on disk (a pointer left behind by a kill is tolerated: the walk finds
   the newer checkpoint through the chain).
8. For each `(run_id, kind)` the newest verified checkpoint is never deleted; a checkpoint is
   deleted only when a newer one of the same kind has been verified (and uploaded, when a
   remote is configured).
9. `status` transitions are `running → finished | failed | preempted` and
   `resumed → finished | failed | preempted`; `resumed` is the initial status of a run with
   `parent_run_id` set, `running` of a run without. Terminal statuses never change.
10. `parent_run_id` is non-null if and only if the run was created by `resume_from`; the
    parent's status is `preempted` or `failed`; `parent_run_id ≠ run_id`; the parent chain
    is acyclic.
11. `Baseline.run_id` has an `EvalSnapshot` with `Baseline.eval_set_hash` and `Baseline.step`;
    pinning refuses otherwise with exit 1 and `baseline_refused`, as it does for a tag that is
    already pinned (without `--replace`) and for a snapshot that fails verification. Built.
12. `Repro.dataset_hash` starts with `dcp:` when `dataset_uri` contains a DataCopilot
    `manifest.json`, otherwise with `merkle:`; the string after the prefix is non-empty.
13. Sync events are idempotent on `(run_id, seq)`: `seq` increases by 1 per event within a
    run; the server stores the first occurrence and answers 409 to any repeat; the client
    acks on 200 or 409.
14. Every row carries the `schema_version` of the store (`RunMeta.schema_version` for rows;
    Parquet file metadata `cvpilot.schema_version` for metric files).
15. `Repro.command_line` and `Repro.config` contain no value whose key matches the redaction
    list in [errors-and-logging.md](../engineering/errors-and-logging.md); redaction happens
    before the row is written.

Status of the invariants: built and tested: 1, 2, 3, 5, 6, 9, 10 (`resume_from` checks the parent status; the acyclic and `parent_run_id`
checks are schema and foreign-key enforced), 12, 14 (`RunMeta.schema_version` and the Parquet
metadata), 4, 11, 15, 7 and 8 (`tests/unit/test_ckpt_directory.py`, `tests/chaos/`). Planned with their packages: 13.

## On-disk layout

```
$CVPILOT_STORE_DIR/<project>/              default $HOME/.cvpilot/<project>/
  runs.sqlite                               WAL mode; built: run_meta, repro, artifact,
                                            metric_files, meta, migrations,
                                            eval_snapshot, baseline (additive); planned:
                                            checkpoint_manifest (built), outbox
  runs.sqlite-wal, runs.sqlite-shm          SQLite WAL files; present while a writer is open
  metrics/<run_id>/<seq:06d>.parquet        MetricPoint batches in write order (built); a batch
                                            is indexed in metric_files only after its atomic
                                            rename, so a killed writer leaves at most an
                                            unindexed file that readers ignore; compacted to
                                            metrics/<run_id>/all.parquet at finish() (built)
  evals/<run_id>/<eval_set_hash[:12]>_<step:09d>.json
                                            full EvalSnapshot JSON (built; the SQLite row is the index;
                                            a name clash of the 12-character prefix is refused)
  artifacts/<sha256[:2]>/<sha256>           content-addressed files: predictions.parquet,
                                            confusion.parquet, cmp.html, config.json, env.json,
                                            git.diff, gallery images
  (reports are written to the path given to --report and are not stored here)

<checkpoint_dir>/<run_id>/                  CheckpointManager(dir=...)
  last/step_<step:09d>.safetensors          full training state: tensors plus the bounded JSON state
                                            (`.pt` instead when the pt format was opted into)
  last/step_<step:09d>.manifest.json        CheckpointManifest
  last/step_<step:09d>.remote.json          marker: uploaded and verified on the remote
  best/step_<step:09d>.safetensors          weights + EMA
  best/step_<step:09d>.manifest.json
  LATEST                                    the newest `last` manifest: name, digest, chain value
  LATEST_BEST                               the same for `best` (not `BEST`: it would collide with
                                            the `best/` directory on case-insensitive filesystems)
  .lock                                     flock held while saving, pruning or installing
```

The outbox table holds `(run_id, seq, payload_json, created_at, acked_at)`; rows are deleted
7 days after `acked_at`.

## Versioning and compatibility

### Content hashes

- `Artifact.sha256`, `CheckpointManifest.sha256`: SHA-256 over the file bytes, lower-case hex.
- `Repro.dataset_hash`:
  - `dcp:<version>` when `<dataset_uri>/manifest.json` exists and parses as a DataCopilot
    manifest; `version` is copied verbatim and is content-based on the DataCopilot side.
  - `merkle:<hex>` otherwise: list regular files under `dataset_uri` recursively, excluding
    dot-files; sort by POSIX relative path; leaf = `sha256(relpath + "\0" + size + "\0" +
    mtime_ns)`; root = `sha256` over the concatenated leaf digests in sorted order. It
    changes when a file is added, removed, renamed, resized or touched. It is not portable
    across machines that copy files without preserving `mtime`; DataCopilot datasets avoid
    this.
- `EvalSnapshot.eval_set_hash`: SHA-256 over the canonical JSON (images in canonical id order:
  numeric when every id is a canonical non-negative integer, else string order; ids are not
  Unicode-normalised, so NFC and NFD spellings are different ids; boxes closer than 0.005 px hash
  alike)
  `{"class_names": [...], "images": [[image_id, width, height, [[cls, x1, y1, x2, y2, iscrowd], ...]], ...]}`
  with images sorted by `image_id`, annotations sorted lexicographically, coordinates
  rounded to 2 decimals, separators `(",", ":")`, UTF-8. It depends on the annotations and
  class list only, so two machines evaluating the same set agree; it changes when any
  annotation, image set or class name changes and does not change with file paths or
  `mtime`.
- `Repro.packages_lock_sha256`: SHA-256 of the lockfile bytes.

### Schema version

- `meta.schema_version` (and a `migrations` table of applied versions) in `runs.sqlite` and
  `cvpilot.schema_version` in Parquet file metadata; the first released value is `1`
  (built).
- Migrations live in `store/migrations/NNNN_<name>.py`, forward-only, applied when the store
  is opened by a library whose version is newer, after copying `runs.sqlite` to
  `runs.sqlite.bak-<schema_version>`. Parquet files are never rewritten by a migration; a
  reader maps older column sets to the current model.
- A store with a `schema_version` newer than the library supports is refused: the command
  logs `store_version_unsupported` with both versions and exits 1.
- Adding a nullable field or a new table is a minor schema change (same `schema_version`; the
  `eval_snapshot` table is created this way, idempotently at open, see `store/additive.py`;
  readers tolerate it). Renaming or removing a field, or changing a key, increments
  `schema_version` and needs a migration and an ADR note.
- The sync protocol sends `X-CVPilot-Schema: <n>`; the server answers 409 with its own
  version when it cannot store that version, and the client logs `sync_schema_mismatch` at
  `WARN` and keeps the outbox.
