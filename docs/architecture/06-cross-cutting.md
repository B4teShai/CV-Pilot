# 06 Cross-cutting concerns

The contracts live in [engineering](../engineering/README.md). This page adds what is specific to
CV-Pilot.

## Logging

Contract: [errors and logging](../engineering/errors-and-logging.md) and
[ADR-0002](../adr/0002-logging.md). Service names are `cvpilot` for the CLI and `cvpilot-client` for
the in-process API and hooks. `request_id` is a UUID per CLI invocation, and in a training process
it is `<run_id>:<seq>` of the event being processed. Logs are separate from run metrics: nothing in
the store is derived from logs, and no metric value is logged as text.

Rules: `sha256` values are logged as a 12-character `sha256_prefix`. Tokens, webhook URLs and chat
ids are never fields. Image paths are logged only as counts. Exception text from adapters is never
logged, only its type.

The main events (fixed `message`, `snake_case` fields):

| Event | Level | Meaning |
|---|---|---|
| `doctor_completed`, `command_failed` | INFO, ERROR | The CLI finished, or its error boundary caught something. |
| `run_started`, `run_resumed`, `run_finished`, `run_preempted` | INFO | The run lifecycle. |
| `hook_attached`, `hook_missing`, `hook_failed`, `hook_disabled` | INFO, ERROR, WARN | A hook was attached, was not found, raised in a callback, or was switched off. |
| `metrics_flushed`, `metrics_dropped` | DEBUG, ERROR | A Parquet batch was written, or a batch failed on bad data and was dropped. |
| `store_write_failed`, `store_version_unsupported` | ERROR | The store could not be written, or is newer than the library. |
| `eval_completed` | INFO | A snapshot was stored. |
| `compare_completed`, `compare_refused` | INFO, ERROR | A comparison ran, or was refused with a `reason`. |
| `gate_passed`, `gate_failed`, `gate_recompute_failed`, `slice_skipped` | INFO, ERROR, WARN | The gate verdict and its special cases. |
| `baseline_set`, `baseline_refused`, `baseline_integrity_failed` | INFO, ERROR | Baseline pinning and its integrity check. |
| `checkpoint_saved`, `checkpoint_verified`, `checkpoint_uploaded`, `checkpoint_restored` | INFO | A checkpoint step succeeded. `restored` carries `trust_level`. |
| `checkpoint_skipped_corrupt`, `checkpoint_untrusted` | WARN, ERROR | A candidate was refused, with a `reason`. |
| `checkpoint_save_failed`, `checkpoint_upload_failed` | ERROR, WARN | A save or an upload failed. |
| `preemption_signal_received`, `preemption_save_incomplete` | INFO, WARN | A stop request arrived, or the save missed the grace window. |
| `forward_failed`, `forward_summary` | WARN | A tracker call failed (once per forwarder), and the totals at finish. |
| `alert_fired`, `alert_suppressed`, `alert_delivery_failed`, `alert_summary` | INFO, WARN | Alert activity; never the message text, URL or credential. |
| `insecure_dev_mode` | WARN | A channel was built under `CVPILOT_ENV=dev`. |

## Configuration and exit codes

Settings are environment variables checked at start ([standards](../engineering/standards.md)). The
table is in the [settings reference](../reference/settings.md). One boundary in `cli/app.py` turns
errors into exit codes `0`, `1`, `2` and `130`, and the per-command meaning is on the same page.
Expected failures print one `error:` line on `stderr` and log one event. Python API errors
(`MissingExtraError`, `DuplicateMetricError`, `StoreError`, `CheckpointCorruptError`, ...) name the
offending input and are never swallowed.

## Security and threat model

Assets: run data and reports (they can reveal model quality and dataset content), checkpoints
(weights), channel secrets and the training machine. Trust boundaries: CV-Pilot and checkpoint files
from a disk or bucket, dataset and prediction files, CI and pull request code, and alert endpoints.

| Threat | Mitigation |
|---|---|
| Secrets in logs or captured command lines | Key-based redaction in logging; `command_line` redacted per argument and `config` by key; `CVPILOT_STORE_COMMAND_LINE=false` stores no arguments. |
| Code execution from a checkpoint | safetensors by default, with no executing loader. `pt` is opt-in, strictly scanned and loaded with `weights_only=True` ([Trust model](../guide/trust.md)). |
| Forged, replaced or rolled-back checkpoint in a shared folder | HMAC tags, the run-store digest, a trust policy and a trust level on every restore. Fails closed. |
| Code execution from dataset or prediction files | Only Parquet and JSON are read. Metadata, rows and decoded sizes are checked before data is read. No `pickle`, no unsafe YAML. |
| Request forgery through alert URLs | Addresses come from the environment only; https, public addresses, resolved once, no redirects. |
| Alert storms and text injection | One alert per rule per window and 20 per run. Constant templates; values reduced to plain characters. |
| Tampered snapshot, artifact or baseline files | SHA-256 per file and a hash chain verified on every read. Not a signature: a writer of the database and files can recompute. |
| A report path that overwrites another file | Strict path resolution, containment in `--report-dir`, no symlinks, and `--force` to replace a file cvpilot did not create. |
| Markup or workflow-command injection through class, slice or tag names | Every string is HTML-escaped and passed through `safe_text`. The page has no script or link, and text lines start with `gate:` or two spaces. |
| A stored result trusted instead of recomputed | `gate` and `compare` recompute from snapshots verified on read. |
| Resource exhaustion from hostile evaluation inputs | Size limits before expansion, capped confusion work, `--max-memory-mb` (Linux). |
| Path traversal | Paths are resolved against the store or checkpoint root. Run ids are checked as ULIDs and project names by pattern. |
| A tracker adapter leaking credentials or weakening TLS | Adapters read no credential, accept only https, refuse TLS-disabling switches, and honour `CVPILOT_FORWARD_HOSTS`. |
| A tracker outage stalling training | Bounded queues, one daemon thread each, a 30 s stall timeout, `close` bounded at 5 s. |
| Shell injection through Action inputs | Inputs reach the script only through `env:` and double quotes; workflows pin actions to full SHAs and use read-only permissions. |
| A hook raising into the host | A guard catches `Exception`, counts it and disables the hook after 10 failures in a row. |
| Copyleft contamination | Ultralytics is only an extra, never imported, and a test fails on any other mention. |
| Dependency compromise | A committed lockfile, `uv sync --locked` in CI, pinned actions. |

Redaction of `command_line` and `config` is best effort. Keys are matched by suffix after
normalisation (`accessToken`, `PRIVATE_KEY`; not `token_count`), and command lines by flag,
`NAME=value`, bearer credentials and signed-URL parameters. A secret passed positionally or under an
unrecognised flag is not found, so never put secrets on a command line.

## Trust model for CI gates

`cvpilot gate` decides whether a pull request may merge, so the question is who controls its inputs.
The gate is a trust boundary only under the conditions below. With fewer of them it protects against
accidents, not against a contributor who wants it to pass.

| Attack | Control |
|---|---|
| Store metrics that were never measured | `gate --recompute --gt GT --classes FILE` re-evaluates the stored predictions on trusted ground truth. Every stored metric, per-class and slice row, and the confusion matrix must match (tolerance 1e-9). |
| Slice the same images differently | `slice_hash` must match on both sides, per-slice image counts must match, and a gate with slice limits fails when it is missing. |
| Use an old good run | `--require-commit SHA` against the snapshot's `git_commit`. Missing provenance is refused. |
| Be the baseline | A candidate is never its own baseline, a baseline run must be `finished`, and the Action accepts only `tag:` baselines. |
| Roll back a `--replace` | `--expect-baseline-chain SHA256`: the newest row's chain value, kept in a protected variable. |
| Hide a regression in a small slice | `--max-skipped N` (default 0) caps skipped slices, which are listed in the output. |
| Change the code that runs | The rules below. |

Rules for a CI gate (the [CI guide](../guide/gating-ci.md) example follows them, and
`tests/unit/test_action.py` lints it):

1. Install cvpilot from a pinned release with hashes, never from the pull request.
2. Take the baseline store and ground truth from the base branch or a protected artifact. A trusted
   script copies the store and runs `cvpilot eval --git-commit` on the pull request's predictions.
   The pull request then supplies only predictions.
3. Gate with `--recompute`, `--require-commit` and `--expect-baseline-chain`.
4. Protect `.github/` and `action.yml` with CODEOWNERS, and check out the base commit with
   `persist-credentials: false`.
5. Trigger on `pull_request` with `contents: read`, never `pull_request_target`.

Residual risk: nothing is signed, so anyone who can write both the store and its files can recompute
every digest, and the anchor and the recomputation are what remove that. A maintainer who can edit
the protected variable or the base branch can change the baseline. `--recompute` checks metrics
against the stored predictions, not that the predictions came from the model under test.

## Licensing

Contract: [dependencies and licensing](../engineering/dependencies-and-licensing.md).

| Item | Classification |
|---|---|
| Lightning, Hugging Face, MMEngine, RF-DETR hooks | Extras. Apache-2.0 upstream. |
| Ultralytics hook | The `[ultralytics]` extra, with an AGPL-3.0 notice in the adapter, in `license_notice` and in the `hook_attached` event. Never core, never vendored. |
| MLflow, W&B, Trackio forwarders | Extras. Apache-2.0 or MIT upstream. |
| numpy, pyarrow | Core. BSD, Apache-2.0. |
| safetensors, fsspec | Optional extras, imported lazily. |
| pycocotools | Test dependency only. |
| Model weights | Never redistributed. The framework downloads them and CV-Pilot records their hash. |
| Non-commercial or no-derivative code or weights | Forbidden. |

## Observability

- `cvpilot doctor` prints `ok` and logs `doctor_completed`.
- `cvpilot runs list|show --json` prints run metadata, the repro record, metric counts and artifacts.
- `cvpilot ckpt list|verify|prune` shows every manifest with its verification result and trust level.
- You can query `runs.sqlite` with any SQLite client and `metrics/*.parquet` with DuckDB or pandas.
- Structured logs on `stderr` are filterable by `run_id`, `request_id` and `message`.
