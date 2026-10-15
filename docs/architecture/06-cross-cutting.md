# 06 Cross-cutting concerns

The contracts live in [../engineering/](../engineering/README.md); this document adds only
what is specific to CV-Pilot.

## Logging

Contract: [errors-and-logging.md](../engineering/errors-and-logging.md) and
[ADR-0002](../adr/0002-logging.md). Service names: `cvpilot` for the CLI (built),
`cvpilot-client` for the in-process API (built) and hooks (planned), `cvpilot-server` for the server
(planned). `request_id` is a UUID per CLI invocation (built); in the training process it is
`<run_id>:<seq>` of the event being processed (built; `seq` counts the run's events); in the server it is the HTTP request id.
Operational logs are separate from run metrics: nothing in the run store is derived from
logs and no metric value is logged as text.

Events this project emits (constant `message` values; fields are `snake_case`):

| Event | Level | Fields | Emitter | Status |
|---|---|---|---|---|
| `doctor_completed` | INFO | `status`, `package_version`, `python_version` | `cli/` | built |
| `command_failed` | ERROR | `command`, `exc_info` | `cli/` boundary | built |
| `run_started`, `run_finished`, `run_resumed` | INFO | `run_id`, `project`, `status`, `parent_run_id`, `task`, `n_points`, `duration_ms` | `runs/` | built (`run_finished` carries the final `status`, so `run_failed` and `run_preempted` are not separate events) |
| `hook_attached`, `hook_missing` | INFO / ERROR | `hook`, `framework`, `extra` | `api.py` | planned |
| `metrics_flushed` | DEBUG | `run_id`, `n_points`, `file` | `runs/` | built |
| `metrics_rejected` | ERROR | `run_id`, `key`, `step` | `runs/`, `store/` | planned (today `DuplicateMetricError` is raised to the caller) |
| `store_write_failed`, `store_version_unsupported` | ERROR | `command`, `store_version`, `library_version`, `exc_info` | CLI boundary, from `StoreError` and `StoreVersionError` (the store raises, it does not log) | built |
| `eval_completed` | INFO | `run_id`, `step`, `eval_set_hash`, `n_images`, `n_classes`, `n_predictions`, `task`, `duration_ms` | `eval/` | built |
| `compare_completed`, `compare_refused` | INFO / ERROR | `candidate_run_id`, `baseline_run_id`, `candidate_eval_set_hash`, `baseline_eval_set_hash`, `seed` | `compare/` | planned |
| `gate_passed`, `gate_failed`, `slice_skipped` | INFO / ERROR / WARN | `run_id`, `baseline_tag`, `n_violations`, `worst_metric`, `worst_delta`, `slice`, `n_images` | `gate` | planned |
| `baseline_refused` | ERROR | `project`, `tag`, `run_id`, `eval_set_hash` | `cli/` | planned |
| `checkpoint_saved`, `checkpoint_verified`, `checkpoint_restored`, `checkpoint_uploaded` | INFO | `run_id`, `step`, `kind`, `size_bytes`, `sha256_prefix`, `is_best`, `uri`, `duration_ms` | `ckpt/`, `remote/` | planned |
| `checkpoint_skipped_corrupt`, `checkpoint_upload_failed`, `checkpoint_verify_failed`, `preemption_save_incomplete` | WARN | `run_id`, `step`, `kind`, `reason` | `ckpt/` | planned |
| `checkpoint_save_failed` | ERROR | `run_id`, `step`, `exc_info` | `ckpt/` | planned |
| `checkpoint_restore_empty` | INFO | `run_id`, `dir`, `remote` | `ckpt/` | planned |
| `preemption_signal_received` | INFO | `run_id`, `signal`, `grace_seconds` | `ckpt/` | planned |
| `forward_failed` | WARN | `forwarder`, `run_id`, `n_points`, `reason` | `forward/` | planned |
| `sync_batch_sent`, `sync_batch_failed`, `sync_outbox_full`, `sync_schema_mismatch` | DEBUG / WARN | `run_id`, `seq_from`, `seq_to`, `status_code`, `retry_in_s`, `outbox_size` | `sync/` | planned |
| `alert_fired`, `alert_delivered`, `alert_delivery_failed` | INFO / INFO / WARN | `rule`, `run_id`, `channel`, `reason` | `alerts/`, `notify/` | planned |
| `nvml_unavailable` | INFO | `reason` | `capture/` | planned (capture currently falls back to `nvidia-smi` or none without logging) |
| `request_completed`, `request_failed` | INFO / ERROR | `method`, `path`, `status_code`, `duration_ms`, `project`, `exc_info` | `server/` (access log and exception handler) | planned |

Rules specific to this project: `sha256` values are logged as `sha256_prefix` (12
characters); tokens, webhook URLs, chat ids and SMTP URLs are never fields; image paths from
datasets are logged only as counts.

## Configuration keys

Contract: settings are environment variables validated at start
([standards.md](../engineering/standards.md)). The full key table with defaults and status
is in [05-runtime.md](05-runtime.md). Built today: `CVPILOT_ENV`, `CVPILOT_LOG_LEVEL`, `CVPILOT_PROJECT`, `CVPILOT_STORE_DIR`,
`CVPILOT_STORE_COMMAND_LINE`.

## Errors and exit codes

Contract: one boundary in `cli/app.py`; `0 / 1 / 2 / 130`. Per command family:

| Family | Exit 1 when | Exit 2 when | Notes |
|---|---|---|---|
| `--version`, `doctor` (built) | an unexpected exception | invalid `CVPILOT_*` value | |
| `runs list`, `runs show`, `repro` (built) | run not found, store unreadable or newer than the library (`store_version_unsupported`) | unknown option, missing `--project` with no `CVPILOT_PROJECT`, invalid project or malformed run id, invalid `CVPILOT_*` | a well-formed but unknown run id is exit 1; expected failures print one `error:` line on `stderr` plus one event |
| `baseline`, `import` (planned) | baseline refused, importer parse error | unknown option, missing `--project` with no `CVPILOT_PROJECT` | |
| `eval` (built) | run not found, input tables or files fail validation (including slice cardinality and malformed Parquet), snapshot exists, write error | bad `--slices` syntax, missing file, malformed run id, number out of range | parity is tested, not checked at run time |
| `compare` (planned) | refused (`eval_set_hash` mismatch), report cannot be written | bad `--baseline` syntax | refusal is a data problem, so it is 1, not 2 |
| `gate` (planned) | any violation, refused comparison, run or baseline missing | bad threshold syntax | CI treats any non-zero as blocking; `1` with `gate_failed` is the regression signal |
| `ckpt verify|restore|ls` (planned) | corrupt or missing checkpoint found, nothing to restore | path is not a checkpoint directory | |
| `sync` (planned) | server unreachable after retries in `--once` mode, auth rejected | `CVPILOT_SERVER_URL` set without `CVPILOT_TOKEN` | |
| `server` (planned) | bind failure, database migration failure | `[server]` extra missing, invalid database URL | |

Exceptions raised by the Python API (`MissingExtraError`, `DuplicateMetricError`,
`StoreError`, `EvalSetMismatchError`, `CheckpointCorruptError`) are specific and name the
offending input; they are never swallowed by the library.

## Security and threat model

Assets: run data and reports (may reveal model quality and dataset content), checkpoints
(model weights), project tokens, channel secrets, the server host, the training machine.

Trust boundaries: training process ↔ server (HTTPS over the public network); CI ↔ server;
browser ↔ server; server ↔ notification channels; CV-Pilot ↔ third-party plugins; CV-Pilot
↔ checkpoint files from disk or a bucket; CV-Pilot ↔ dataset files.

| Threat | Where | Mitigation | Status |
|---|---|---|---|
| Unauthenticated access to run data (Aim's 2026 reports) | server | every route except `/healthz` and valid share links requires a project token; tokens are random 32 bytes, stored as SHA-256; auth-on-every-route test and schemathesis fuzzing | planned |
| Token leakage through logs or captured command lines | client, CI | key-based redaction in `logging.py` (`*token*`, `authorization`, ...); `Repro.command_line` redacted per argument (`redact_argv`) and `Repro.config` by key before storage; `CVPILOT_STORE_COMMAND_LINE=false` stores no arguments | built |
| Transport interception | client ↔ server | TLS terminated by a reverse proxy; the server refuses to start with `CVPILOT_SERVER_URL` scheme `http` unless `CVPILOT_ENV=dev` | planned |
| Arbitrary code execution from a checkpoint | `ckpt/`, hooks | `torch.load(weights_only=True)` always; `safetensors` for `best`; a file without a manifest is never loaded; checksum before load | planned |
| Pickle or code execution from dataset or prediction files | `eval/`, `store/` | only Parquet and JSON files are read; no `pickle`, no `eval`, no `yaml.load` without `SafeLoader`; Parquet metadata (file size, rows, row groups, decoded bytes, string widths) is checked before data is read; eval limits are named constants with boundary tests | built (eval, Parquet tables), planned (images) |
| Server-side request forgery through webhook URLs | `notify/`, server | webhook and Slack URLs come from the environment only, never from API requests; outbound requests resolve the host and reject private and link-local ranges unless `CVPILOT_ENV=dev` | planned |
| Denial of service on ingest or websocket | server | per-token rate limit (600 requests/min), batch size cap (1 000 points, 1 MB), websocket backpressure with 100-message queue per client | planned |
| Tampered snapshot or artifact files | `store/` | SHA-256 per snapshot file and per-run hash chain verified on every read, row fields compared with the file, exclusive 0444 files written in the insert transaction, artifacts re-hashed before deduplication and readable through `open_artifact`; not a signature: a writer of both database and files can recompute | built |
| Resource exhaustion from hostile eval inputs | `store/tables.py`, `eval/`, `core/` | metadata and decoded-batch size limits (the footer is not trusted), dictionary values checked before expansion, control characters rejected, class-agnostic confusion capped at 300 detections per image with chunked IoU, slice work budget, estimated-memory check and worker scaling, `--max-memory-mb` (`RLIMIT_AS`, Linux only; elsewhere only the input limits apply) | built |
| Secrets, home paths and ids in logs and error lines | `logging.py`, `cli/` | formatted tracebacks and error lines are scrubbed (home directory, bearer tokens, `name=value` credentials); validation messages refer to ids by length and digest | built |
| Evaluating into a finished run | `eval/` | refused unless `allow_closed` / `--allow-closed` | built |
| Path traversal in artifact or checkpoint paths | `store/`, `remote/`, server | paths are resolved and checked against the store or checkpoint root before read or write (store: built, with run ids validated as ULIDs and project names by pattern before any path use); the server never accepts file paths from clients, only content and hashes | built (store), planned (checkpoints, server) |
| Malicious or buggy third-party plugin | registries | plugins load only from installed distributions' entry points; no dynamic import from configuration strings; plugin exceptions are caught at the registry boundary and logged with the plugin name | planned |
| Telegram `/stop` abuse | server | commands accepted only from allow-listed chat ids; `/stop` requires a confirmation reply within 60 s | planned |
| Share-link enumeration | server | 32-byte random tokens, expiry ≤ 7 days, read-only, revocable | planned |
| Dependency compromise | all | committed lockfile, `uv sync --locked` in CI, pinned actions, dependency scanning in CI | built (lockfile), planned (scanning) |
| Credential exposure in `doctor` | CLI | `doctor` reports whether `CVPILOT_TOKEN` is set, never its value | planned |
| Copyleft contamination of the distribution through the Ultralytics extra | `hooks/` | `ultralytics` is declared only in the `[ultralytics]` extra and imported only inside `hooks/ultralytics.py`; an import-linter contract forbids it elsewhere; no code is vendored or derived; the AGPL-3.0 notice is shown in `--help` and the runtime docs | planned |

Redaction of `Repro.command_line` and `Repro.config` is best effort, not a guarantee. Keys
are matched after camelCase and hyphen normalisation by suffix (`accessToken`, `GITHUBTOKEN`,
`PRIVATE_KEY`, `session_id`, `cookie`; not `token_count`, `max_tokens`, `author`); command
lines are matched by flag, `NAME=value` pairs, bearer credentials, URL user-info and
signed-URL query parameters; config strings get the same pattern pass, bytes are replaced and
objects are stored as a truncated redacted `repr`. A secret passed positionally or under an
unrecognised flag is not found. Never put secrets on a command line. Where that risk is not
acceptable, `CVPILOT_STORE_COMMAND_LINE=false` stores only the executable name and the number
of arguments (see [05-runtime.md](05-runtime.md)).

A `SECURITY.md` with a disclosure address is part of the first server release.

## Licensing and extras policy

Contract: [dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md).
Project specifics:

| Integration or model | Classification | Reason |
|---|---|---|
| Lightning, HF Trainer, MMEngine/MMDetection, RF-DETR hooks | extras of this package | Apache-2.0 upstream |
| Ultralytics hook (`hooks/ultralytics.py`) | `[ultralytics]` extra, lazy import inside the adapter, AGPL-3.0 notice in `--help` and [05-runtime.md](05-runtime.md) | AGPL-3.0; never core, never vendored, never imported from `core/` or application code |
| MLflow, W&B SDK, Trackio, TensorBoard forwarders | extras | Apache-2.0 / MIT upstream |
| numpy | core, added with `eval/` (ADR-0003) | BSD |
| faster-coco-eval, safetensors, fsspec | core; added in the change that lands `ckpt/`; `faster-coco-eval` is no longer planned (the engine computes COCO metrics itself) | Apache-2.0 / Apache-2.0 / BSD |
| pycocotools | test dependency only (parity tier, pinned 2.0.11), never imported from `src/` | BSD |
| supervision (diff gallery rendering) | dependency of `compare/`, imported lazily | MIT |
| FastAPI, Uvicorn, SQLAlchemy, asyncpg | `[server]` extra | MIT / BSD / MIT / Apache-2.0 (psycopg is not used: LGPL is outside the allow-list) |
| `httpx` | `[alerts]` extra | BSD |
| pyarrow, duckdb | core (ADR-0005) | Apache-2.0 / MIT |
| Model weights (YOLO, DETR, backbones) | never redistributed | the user's framework downloads them; CV-Pilot records their hash in `Repro.config` only |
| Non-commercial or no-derivative licensed code or weights | forbidden | allow-list |

## Observability

What a user can inspect:

- `cvpilot doctor` (built): prints `ok`, logs `doctor_completed` with package and Python
  versions. Planned checks, each reported as a field on `doctor_completed` and as a line on
  `stdout`: store directory writable and its `schema_version`; SQLite WAL support; installed
  extras and their versions; `CVPILOT_SERVER_URL` reachable and the token accepted (value
  never printed); NVML available; fsspec backend for `CVPILOT_CHECKPOINT_REMOTE` reachable;
  checkpoint directory `LATEST` verifies.
- `cvpilot runs list|show --json` (built): run metadata, repro record, metric counts and
  artifacts; planned: last eval snapshot, baseline tags.
- `cvpilot ckpt ls|verify` (planned): every manifest with its verification result.
- Direct queries: `runs.sqlite` with any SQLite client; `metrics/*.parquet` with DuckDB or
  pandas; the embedded `cvpilot.schema_version` metadata in each Parquet file.
- Server (planned): `/healthz` (unauthenticated liveness), `/api/v1/runs`, the live view at
  `/ui/runs/{run_id}` with curves, GPU utilisation, latest eval and compare reports; the
  access log as JSON on `stderr` with `request_completed` events.
- Structured logs on `stderr` from every process, filterable by `run_id`, `request_id` and
  `message`.
