# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/);
versions follow semantic versioning. Before 1.0, minor versions may break the public API.

## [Unreleased]

### Added

- Hooks: `cvpilot.autolog(target, project=..., task=...)` attaches a Lightning logger and callback
  (`cvpilot[lightning]`) or Ultralytics callbacks (`cvpilot[ultralytics]`, AGPL-3.0: the notice is
  in the adapter, in the `hook_attached` event and in the docs; the package is never imported by
  CV-Pilot). Callbacks never raise into the host (a guard counts failures, logs the first as
  `hook_failed` with the error type only and disables the hook after 10 in a row), make no network
  call, read framework objects through narrow views and cap what they take. Registry
  `cvpilot.hooks`.
- Forwarders: `init(forward_to=[...])` or `CVPILOT_FORWARD_TO` sends every logged scalar to MLflow
  (`cvpilot[mlflow]`), W&B (`cvpilot[wandb]`) or Trackio (`cvpilot[trackio]`) through the
  `cvpilot.forwarders` registry. Each forwarder has a daemon worker thread behind a bounded queue:
  a failing, slow or hung tracker is counted and logged once (`forward_failed`,
  `forward_summary`), never raised into or blocking the training loop (`finish` waits at most 5 s).
  Tokens stay in the SDK's own environment; adapters refuse credentials in URLs, plain http outside
  dev and disabled TLS verification. ADR-0008.
- Reports: `--report PATH` (static HTML: every name escaped, no script or link, CSP limited to its
  hashed inline style, severity-first with row caps) and `--report-json PATH` (canonical JSON, no
  timestamps) on `compare` and `gate`; atomic writes that refuse symbolic links; golden tests.
- `cvpilot compare RUN --baseline tag:NAME|RUN_ID` and `cvpilot gate RUN --baseline ... --max-drop
  NAME=POINTS --per-class-max-drop NAME=POINTS --slice-max-drop "KEY=*:POINTS" --min-images N`.
  Deltas and limits are points on the 0..100 scale (a drop is `(baseline - candidate) * 100`;
  improvements never fail). The gate fails closed: absent or non-finite metrics, absent,
  mismatched or too small slices, too few images, thresholds that gated nothing, snapshots or
  baselines that fail verification, and a different `eval_set_hash`, task, class names or
  evaluation parameters (`compare_refused`, exit 1) all fail. Events `compare_completed`,
  `compare_refused`, `gate_passed`, `gate_failed`, `slice_skipped`. ADR-0007.
- Baselines: `cvpilot baseline set|list|show` pin a verified snapshot under a tag. The `baseline`
  table is append-only (`--replace` adds a newer pin and keeps the history), each row carries a
  digest and a hash chain, and every read re-verifies the record, the chain and the pinned
  snapshot digest. Events `baseline_set`, `baseline_refused`, `baseline_integrity_failed`.
- Eval engine: `cvpilot.log_eval` / `Run.log_eval` and `cvpilot eval RUN_ID --preds P --gts G
  --classes C` (alias `eval-file`). COCO-protocol detection metrics (101-point AP, small/medium/large ranges,
  per-class AP/AR, confusion with background), top-1 classification, slices on image metadata
  (at most 256 values per key), seeded bootstrap intervals over images, canonical `eval_set_hash`,
  immutable `EvalSnapshot` keyed `(run_id, eval_set_hash, step)` with predictions and confusion as
  content-addressed Parquet artifacts; `runs show` lists snapshots; event `eval_completed`.
  Parity with pycocotools 2.0.11 is exact on 300 randomized sets and the edge cases
  (tolerance 1e-4). `benchmarks/eval_throughput.py`: 5 000 images and 100 000 detections in 0.7 s,
  1 000 bootstrap resamples +2.1 s, 140 MB peak.
- Bounded Parquet input (`store/tables.py`, registry `tables`): file size, rows, row groups,
  decoded bytes and string widths are checked from metadata before reading.
- Run capture: `cvpilot.init`, `log`, `log_artifact`, `finish` and the `Run` handle. Metric
  points are validated per call, buffered (at most 10 000 points or 2 s) and written as
  batched Parquet files; a failed write keeps the points. Steps must be strictly increasing
  per metric key. Runs left `running` by a dead process can be marked `failed`
  (`runs.recover_run`); `init(resume_from=...)` continues a `preempted` or `failed` run.
- Local run store: SQLite in WAL mode with `schema_version` and a `migrations` table, forward
  migrations with a backup, Parquet metric files indexed after an atomic rename, compaction at
  `finish`, a content-addressed artifact directory, validated store paths and run ids.
- Domain: `RunMeta`, `Repro`, `MetricPoint`, `Artifact`, the `MetricSink` and `RunStore`
  protocols, monotonic ULID run ids, dataset Merkle hash and plain-file hash.
- Reproducibility capture: git commit, dirty flag and diff of tracked changes, lockfile or
  installed-set hash, Python, host, CUDA and GPU names (optional `pynvml`, else `nvidia-smi`,
  else none) and the command line, redacted per argument. Works outside git and without a GPU.
- CLI: `runs list` (alias `ls`), `runs show`, `repro`; events `run_started`, `run_resumed`,
  `metrics_flushed`, `run_finished`; one `error:` line plus one event for expected failures.
- Settings: `CVPILOT_PROJECT`, `CVPILOT_STORE_DIR`, `CVPILOT_STORE_COMMAND_LINE`.
- Entry-point registries (`cvpilot.stores`, `cvpilot.capturers`) and layering tests for ADR-0004.
- Tests: unit, golden (100 000-point round trip, `repro` script round trip) and chaos
  (SIGKILL of a writer mid-batch); `benchmarks/log_readback.py`, `benchmarks/install_size.sh`.

### Changed

- New core dependency: `numpy` (eval math, imported lazily so the CLI start-up is unchanged);
  `pycocotools` is a dev-only dependency for the parity tests. ADR-0003, ADR-0005 and the
  overview are amended; `faster-coco-eval` is not added.
- New core dependency: `pyarrow` (imported lazily, store only). ADR-0003, ADR-0005 and the
  overview constraints are amended; DuckDB stays out until cross-run queries exist.

### Security

- Snapshot integrity: per-snapshot SHA-256 and a per-run hash chain verified on every read,
  exclusive 0444 snapshot files written in the insert transaction, artifacts re-hashed before
  deduplication and read-only, triggers on `artifact` and `eval_snapshot`, `recursive_triggers`.
- Eval inputs: Parquet decoded size and string checks run on the decoded batches, leaf-column size
  mapping, bounded confusion matrix, slice work and memory budgets, `--max-memory-mb`, class file
  read without following symlinks, control characters and lone surrogates rejected, closed runs
  refused without `--allow-closed`, scrubbed tracebacks and error lines.
- Eval semantics: score ties across images follow the numeric image id, default class names come
  from the ground truth only, snapshots record parameters, provenance, slice spec and `ci_status`.

- Secret redaction for stored command lines and configs: by key (camelCase and run-together
  names included), by argument position, bearer credentials, URL user-info and signed query
  parameters. Best effort for free text; `CVPILOT_STORE_COMMAND_LINE=false` stores no arguments.
