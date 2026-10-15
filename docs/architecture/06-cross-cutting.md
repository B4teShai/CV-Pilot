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
| `hook_attached`, `hook_missing` | INFO / ERROR | `hook`, `framework`, `extra`, `license_notice` (null unless the framework is copyleft) | `api.py` (`autolog`) | built |
| `hook_no_epochs` | WARN | `hook`, `reason` (the framework trained without calling the hook) | `hooks/` (YOLO) | built |
| `hook_failed`, `hook_disabled` | WARN | `hook`, `callback`, `error_type` (type only, never the text); `hook`, `failures` | `hooks/` (the guard around every callback) | built; `hook_failed` once per hook |
| `metrics_flushed` | DEBUG | `run_id`, `n_points`, `file` | `runs/` | built |
| `metrics_rejected` | ERROR | `run_id`, `key`, `step` | `runs/`, `store/` | planned (today `DuplicateMetricError` is raised to the caller) |
| `store_write_failed`, `store_version_unsupported` | ERROR | `command`, `store_version`, `library_version`, `exc_info` | CLI boundary, from `StoreError` and `StoreVersionError` (the store raises, it does not log) | built |
| `eval_completed` | INFO | `run_id`, `step`, `eval_set_hash`, `n_images`, `n_classes`, `n_predictions`, `task`, `duration_ms` | `eval/` | built |
| `compare_completed` | INFO | `candidate_run_id`, `baseline_run_id`, `candidate_eval_set_hash`, `baseline_eval_set_hash`, `candidate_step`, `baseline_step` | `compare/` | built |
| `compare_refused` | ERROR | `reason` (`eval_set_hash`, `task`, `class_names`, `params`, `params_missing`, `slice_hash`, `slice_n_images`, `class_support`, `no_candidate_snapshot`, `no_baseline_snapshot`, `no_candidate_snapshot_at_step`, `baseline_is_candidate`, `baseline_not_finished`, `commit_mismatch`, `provenance_missing`, `no_baseline`), `candidate_run_id`, `baseline_run_id`, `candidate_eval_set_hash`, `baseline_eval_set_hash` | `cli/` boundary, from `CompareRefusedError` | built |
| `gate_recompute_failed` | ERROR | `run_id`, `what` (the first thing that differed) | `cli/` boundary, from `RecomputeMismatchError` | built |
| `gate_passed`, `gate_failed` | INFO / ERROR | `run_id`, `baseline_tag` (null for a run-id baseline), `n_violations`, and for a failure `worst_metric`, `worst_kind`, `worst_delta` (points, null when unverifiable) | `compare/` | built |
| `slice_skipped` | WARN | `run_id`, `slice` | `compare/`, only with `--skip-small-slices` | built |
| `baseline_set` | INFO | `project`, `tag`, `run_id`, `eval_set_hash`, `step`, `replaced` | `compare/` | built |
| `baseline_refused` | ERROR | `project`, `tag`, `run_id`, `eval_set_hash` | `cli/` boundary, from `BaselineRefusedError` | built |
| `baseline_integrity_failed` | ERROR | `command` | `cli/` boundary, from `BaselineIntegrityError` | built |
| `checkpoint_saved`, `checkpoint_verified`, `checkpoint_restored`, `checkpoint_uploaded` | INFO | `run_id`, `step`, `kind`, `size_bytes`, `sha256_prefix`, `is_best`, `uri`, `duration_ms` | `ckpt/`, `remote/` | planned |
| `checkpoint_skipped_corrupt`, `checkpoint_upload_failed`, `checkpoint_verify_failed`, `preemption_save_incomplete` | WARN | `run_id`, `step`, `kind`, `reason` | `ckpt/` | planned |
| `checkpoint_save_failed` | ERROR | `run_id`, `step`, `exc_info` | `ckpt/` | planned |
| `checkpoint_restore_empty` | INFO | `run_id`, `dir`, `remote` | `ckpt/` | planned |
| `preemption_signal_received` | INFO | `run_id`, `signal`, `grace_seconds` | `ckpt/` | planned |
| `forward_failed` | WARN | `forwarder`, `run_id`, `n_points`, `reason` (`exception`, `open_failed`, `queue_full`, `stalled`, `base_exception`, `step_regressed`, `construct_failed`, `submit_failed`), `error_type` (the exception class only, never its text) | `runs/forwarding.py` | built; logged once per forwarder |
| `forward_skipped_rank` | INFO | `forwarders` (the names that were not started) | `api.py` (`init`, when `RANK` or `LOCAL_RANK` is not 0) | built |
| `metrics_dropped` | ERROR | `run_id`, `n_points`, `error_type` (a batch that failed with a data error and was dropped instead of re-queued) | `runs/` | built |
| `forward_summary` | WARN | `forwarder`, `run_id`, `failed_calls`, `failed_points`, `dropped_batches`, `dropped_points`, `adapter_dropped_points`, `disabled_reason`, `abandoned` | `runs/forwarding.py` at finish, only when something went wrong | built |
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
| `baseline set|list|show` (built) | refused (no such snapshot or eval set, ambiguous eval set, tag already pinned without `--replace`), unknown tag, run not found, snapshot or baseline fails integrity verification | unknown option, missing `--project`, malformed tag, run id or eval set, step out of range | a pin is append-only; `list` of a project without a store is empty and exit 0 |
| `import` (planned) | importer parse error | unknown option, missing `--project` with no `CVPILOT_PROJECT` | |
| `eval` (built) | run not found, input tables or files fail validation (including slice cardinality and malformed Parquet), snapshot exists, write error | bad `--slices` syntax, missing file, malformed run id, number out of range | parity is tested, not checked at run time |
| `compare` (built) | refused (`eval_set_hash`, task, class names or parameters differ, no candidate snapshot), run or baseline missing, snapshot or baseline fails verification | bad `--baseline` syntax or run id, no project | refusal is a data problem, so it is 1, not 2 |
| `gate` (built) | any violation (exceeded drop, absent or non-finite metric, absent or too small slice, too few images, unmatched threshold), refused comparison, run or baseline missing, any verification failure, any unexpected exception | bad threshold syntax, no threshold, `--min-images` below 1, bad `--baseline` | CI treats any non-zero as blocking; `1` with `gate_failed` is the regression signal; no code path exits 0 without `gate_passed` |
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
| Arbitrary code execution from a checkpoint | `ckpt/`, hooks | safetensors is the default and has no executing load path; `.pt` is opt-in (`allow_pt_format` or `CVPILOT_CKPT_ALLOW_PT=1`), needs the framework at 2.6 or newer (read from package metadata before anything is imported), is zip-pre-scanned (member count, names, sizes, compression ratio, one `data.pkl` whose opcodes may name only tensor storage, dtype and ordered-dict globals) and is loaded only with `weights_only=True` from the verified bytes; a file without a manifest is never loaded; residual risk: a flaw in the framework's own restricted unpickler | built |
| Forged, replaced or rolled-back checkpoint in a directory other people can write | `ckpt/` | see "Who can write the checkpoint directory" below: HMAC tags with `CVPILOT_CKPT_KEY`, the run-store digest, the trust policy `CVPILOT_CKPT_REQUIRE_TRUST`, a trust level on every restore; fails closed with `checkpoint_untrusted` | built |
| Pickle or code execution from dataset or prediction files | `eval/`, `store/` | only Parquet and JSON files are read; no `pickle`, no `eval`, no `yaml.load` without `SafeLoader`; Parquet metadata (file size, rows, row groups, decoded bytes, string widths) is checked before data is read; eval limits are named constants with boundary tests | built (eval, Parquet tables), planned (images) |
| Server-side request forgery through webhook URLs | `notify/`, server | webhook and Slack URLs come from the environment only, never from API requests; outbound requests resolve the host and reject private and link-local ranges unless `CVPILOT_ENV=dev` | planned |
| Denial of service on ingest or websocket | server | per-token rate limit (600 requests/min), batch size cap (1 000 points, 1 MB), websocket backpressure with 100-message queue per client | planned |
| Tampered snapshot or artifact files | `store/` | SHA-256 per snapshot file and per-run hash chain verified on every read, row fields compared with the file, exclusive 0444 files written in the insert transaction, artifacts re-hashed before deduplication and readable through `open_artifact`; not a signature: a writer of both database and files can recompute | built |
| A report path used to overwrite another file (the store, a runner file such as `$GITHUB_ENV`, a workspace file) | `compare/html_report.py` | strict parent resolution, root containment (`--report-dir`, `--allow-outside`), refusal inside the store and on runner files, symlinks refused, a file the tool did not create is replaced only with `--force` | built |
| Memory or run state carried into forked children and distributed ranks | `runs/forwarding.py`, `runs/lifecycle.py`, `api.py` | forwarder queues bounded in points, forwarders only on rank zero, `os.register_at_fork` orphans the fan-out and the run in children | built |
| Script or markup injection through class, slice or tag names in reports and logs | `compare/html_report.py`, `compare/render_text.py` | every string child is HTML-escaped and passed through `safe_text` (control, format and private-use characters shown as `\uXXXX`); the page has no script, link or data-carrying attribute and a CSP with only a hashed inline style; text lines start with `gate:` or two spaces so a name cannot become a CI workflow command; JSON is escaped by the serializer (ASCII only, no `NaN`) | built |
| A stored comparison, report or gate result trusted in place of a recomputation | `compare/service.py` | `gate` and `compare` never read a stored result; they recompute from snapshots verified on read, a tag from a verified baseline row and confusion from digest-verified artifacts | built |
| Tampered baseline row or re-pointed baseline | `store/baselines.py` | record digest, store-wide chain, pinned snapshot digest verified on every read; append-only table with triggers | built |
| Resource exhaustion from hostile eval inputs | `store/tables.py`, `eval/`, `core/` | metadata and decoded-batch size limits (the footer is not trusted), dictionary values checked before expansion, control characters rejected, class-agnostic confusion capped at 300 detections per image with chunked IoU, slice work budget, estimated-memory check and worker scaling, `--max-memory-mb` (`RLIMIT_AS`, Linux only; elsewhere only the input limits apply) | built |
| Secrets, home paths and ids in logs and error lines | `logging.py`, `cli/` | formatted tracebacks and error lines are scrubbed (home directory, bearer tokens, `name=value` credentials) and made printable with `safe_text` (no control or bidirectional characters); validation messages refer to ids by length and digest | built |
| Evaluating into a finished run | `eval/` | refused unless `allow_closed` / `--allow-closed` | built |
| Path traversal in artifact or checkpoint paths | `store/`, `remote/`, server | paths are resolved and checked against the store or checkpoint root before read or write (store: built, with run ids validated as ULIDs and project names by pattern before any path use); the server never accepts file paths from clients, only content and hashes | built (store), planned (checkpoints, server) |
| Malicious or buggy third-party plugin | registries | plugins load only from installed distributions' entry points; no dynamic import from configuration strings; plugin exceptions are caught at the registry boundary and logged with the plugin name | planned |
| Telegram `/stop` abuse | server | commands accepted only from allow-listed chat ids; `/stop` requires a confirmation reply within 60 s | planned |
| Share-link enumeration | server | 32-byte random tokens, expiry ≤ 7 days, read-only, revocable | planned |
| Tracker adapter leaking credentials, weakening TLS or being pointed elsewhere | `forward/*` | adapters read no credential, take no token argument, accept only https endpoints (http in dev; every other scheme refused), refuse credentials in URLs, SDK switches that disable TLS verification and certificate-authority overrides outside dev, enforce `CVPILOT_FORWARD_HOSTS` when set (whole-host equality on a structurally validated, normalised URL; ports need a `host:port` entry; the normalised URL and an explicit W&B `base_url` are what the SDK receives), read and freeze the relevant environment once on the training thread and pass it to the SDK explicitly, and never log exception text or a URL (a test scans the adapter sources for credential variable names and for verification switches); Trackio cannot be verified and refuses to run with an allow-list (ADR-0008) | built |
| A tracker outage, hang or slow call stalling training | `runs/forwarding.py` | bounded queue with `put_nowait`, one daemon thread per forwarder, stall timeout (30 s), `close` bounded at 5 s, every adapter exception counted and logged once, never raised | built |
| Shell injection through GitHub Action inputs, branch names or pull request titles | `action.yml`, `.github/workflows/` | inputs reach the script only through `env:` and are used only inside double quotes as single argv entries, options are passed as `--name=value` and the run id after `--`, no third-party action is used, shipped workflows pin actions to full SHAs, declare read-only permissions and never use `pull_request_target`; `tests/unit/test_action.py` parses the YAML, rejects any `${{ ... }}` in a `run:` block, unquoted variables, unpinned actions and write permissions, and runs the real script against hostile values | built |
| Gate decided by data a pull request controls (forged metrics, moved slice images, rolled-back or self-referencing baseline, older good run) | `compare/service.py`, `compare/recompute.py`, `compare/engine.py` | see the trust model above: recomputation, `slice_hash`, commit binding, baseline rules, the external chain anchor | built |
| Dependency compromise | all | committed lockfile, `uv sync --locked` in CI, actions pinned to full SHAs, dependency scanning in CI | built (lockfile, pinned actions), planned (scanning) |
| Credential exposure in `doctor` | CLI | `doctor` reports whether `CVPILOT_TOKEN` is set, never its value | planned |
| Copyleft contamination of the distribution through the Ultralytics extra | `hooks/` | the framework is declared only in the `[ultralytics]` extra and never imported by this package (the hook gets the model through `add_callback`); a layering test parses every module and fails on an import anywhere but `hooks/ultralytics.py`, and on a top-level one there; no code is vendored or derived; the AGPL-3.0 notice is in the adapter, in the `hook_attached` event and in the runtime docs | built |
| A hook callback raising into, or stalling, the host training loop | `hooks/` | every callback body runs under a guard that catches `Exception` (never `KeyboardInterrupt`), counts, logs once and disables the hook after 10 consecutive failures; no network in callbacks; values converted defensively and capped; a hung forwarder cannot delay a callback | built |

### Who can write the checkpoint directory

Digests, the manifest chain and the `LATEST` pointer detect damage (kill during a write, a
truncated upload, a flipped bit). They do not detect an adversary who can write the directory:
that adversary can write a new manifest, a data file with the matching SHA-256, a new pointer
and a consistent chain. Three levels of trust, reported on every restore (`trust_level` in
`checkpoint_restored`, `Restored.trust`) and selectable with `CVPILOT_CKPT_REQUIRE_TRUST`
(`any`, `anchored`, `authenticated`; CLI `--require-trust`):

| Level | Proof | Works without a run store | Holds against a writer of the directory |
|---|---|---|---|
| `authenticated` | HMAC-SHA256 tag (key `CVPILOT_CKPT_KEY`, 32 to 1024 bytes, environment only) over a domain label, the run id and the canonical bytes of the manifest (and of the pointer) | yes (the spot-instance resume case) | yes, unless the key leaks |
| `anchored` | the digest of the manifest file is in the run store (`checkpoint_manifest`, append-only) | no | yes, unless the store is writable too |
| `local_chain_only` | the directory's own chain and pointer | yes | no |

Rules: under `authenticated` and `anchored` every candidate needs its own proof; nothing is
vouched for by a neighbour's chain, so a forged newer manifest cannot launder an older one. If
candidates exist but none reaches the required level, restore raises `checkpoint_untrusted`
and loads nothing; it never falls back to a lower level. A store lookup that fails is reported
as `anchor_error` and fails closed under `anchored`; it is not treated as "no store". A tag that
is present but wrong rejects the candidate under every policy. With a key set, a pointer
without a valid tag is ignored. The default policy is `any`, because a laptop or a private
node resuming its own directory has no adversary: for any directory that is shared, restored
from a bucket or writable by others, set a key and `CVPILOT_CKPT_REQUIRE_TRUST=authenticated`
(the GitHub Action and the tutorial do). Key rotation: tags made under the old key fail under
the new one (`auth_invalid`); unsigned files from before the key are accepted only under `any`.
The key is never logged, stored, written to a manifest or included in an error message.

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

## Trust model for CI gates

`cvpilot gate` decides whether a pull request may merge, so the question is who controls its
inputs. The gate is a trust boundary **only** under the conditions below; with fewer of them it
protects against accidents (a stale file, a typo, an honest bug), not against a contributor who
wants it to pass.

What a contributor can do to an unprotected setup, and the control that stops it:

| Attack | Why it works without the control | Control |
|---|---|---|
| Store metrics that were never measured | `put_snapshot` and `record_eval` are public API; a snapshot with the baseline's `eval_set_hash`, perfect metrics and valid digests verifies | `gate --recompute --gt GT --classes FILE`: the stored predictions (SHA-256 verified) are re-evaluated on ground truth from a trusted place with the recorded parameters, and every stored metric, per-class and slice row, the slice membership hash and the confusion matrix must match (tolerance 1e-9). Not covered: predictions that were forged consistently; produce them in the trusted job (below) and bind them with `--require-commit` |
| Slice the same images differently | `eval_set_hash` covers annotations and classes, not image metadata, so a regressed camera can move into a bigger slice | `slice_hash` (digest of each slice column's per-image values) is stored in the snapshot, must be equal on both sides (refusal `slice_hash`), per-slice `n_images` must match (`slice_n_images`), and a gate with slice thresholds fails when a snapshot has no `slice_hash` |
| Use an old good run | any run with the right eval set passes | `--require-commit SHA` against `provenance.git_commit` (the candidate must have been evaluated with `--git-commit`; absent provenance is refused) |
| Be the baseline | `--baseline <run id>` accepted the candidate itself, a running or a failed run | refused: the candidate is never its own baseline, a baseline run must be `finished`, and the Action accepts only `tag:` baselines unless `allow-run-baseline` is set |
| Roll back a `--replace` | deleting the newest baseline row makes the previous pin current and the chain still verifies | `--expect-baseline-chain SHA256`: the chain digest of the newest row (`cvpilot baseline show` prints `head_seq` and `head_chain_sha256`) kept in a protected variable; any removed, added or re-ordered row changes it |
| Hide a regression in a small slice | `--skip-small-slices` skipped it silently | `--max-skipped N` (default 0): skipped slices are counted against a cap, listed in the output and in the Action summary, and exceeding the cap fails |
| Change the code that runs | the workflow, `action.yml`, the CI scripts and the installed cvpilot all came from the pull request | the rules below |

The rules for a CI gate (the README example follows them and `tests/unit/test_action.py` lints it):

1. Install cvpilot from a pinned release with hashes (`pip install --require-hashes`), never from
   the pull request.
2. Take the baseline store and the ground truth from the base branch or a protected artifact,
   not from the pull request. The recommended flow is: a trusted script from the base branch copies
   the baseline store, runs `cvpilot eval --git-commit` on the pull request's predictions with the
   trusted ground truth, and gates the resulting run. The pull request then supplies only
   predictions (bounded Parquet, parsed with size and row limits).
3. Gate with `--recompute` (trusted ground truth), `--require-commit` (the commit under test) and
   `--expect-baseline-chain` (from a protected variable).
4. Protect `.github/` and `action.yml` with CODEOWNERS or required workflow rulesets; check out
   the base commit (`ref: github.event.pull_request.base.sha`, `persist-credentials: false`) and
   use the action from that checkout.
5. Trigger on `pull_request` with `permissions: contents: read`; never `pull_request_target`.

Residual risks, stated plainly: nothing is signed, so anyone who can write both the store and the
files can recompute every digest (the anchor and the recomputation are what remove that); a
maintainer who can edit the protected variable or the base branch can change the baseline;
`--recompute` verifies metrics against the stored predictions, not that the predictions came from
the model under test.

## Licensing and extras policy

Contract: [dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md).
Project specifics:

| Integration or model | Classification | Reason |
|---|---|---|
| Lightning, HF Trainer, MMEngine/MMDetection, RF-DETR hooks | extras of this package | Apache-2.0 upstream |
| Ultralytics hook (`hooks/ultralytics.py`) | `[ultralytics]` extra, lazy import inside the adapter, AGPL-3.0 notice in the adapter docstring, `license_notice`, the `hook_attached` event and [05-runtime.md](05-runtime.md); the package is never imported by this project | AGPL-3.0; never core, never vendored, never imported from `core/` or application code |
| MLflow, W&B SDK, Trackio, TensorBoard forwarders | extras | Apache-2.0 / MIT upstream |
| numpy | core, added with `eval/` (ADR-0003) | BSD |
| faster-coco-eval, safetensors, fsspec | core; added in the change that lands `ckpt/`; `faster-coco-eval` is no longer planned (the engine computes COCO metrics itself) | Apache-2.0 / Apache-2.0 / BSD |
| pycocotools | test dependency only (parity tier, pinned 2.0.11), never imported from `src/` | BSD |
| supervision (diff gallery rendering) | planned with the gallery; not a dependency today (reports use the standard library) | MIT |
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
