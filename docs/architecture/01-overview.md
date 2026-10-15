# 01 Overview

## Purpose

CV-Pilot is a Python package that sits beside a team's existing experiment tracker and adds
what trackers do not have for detection and classification work: evaluation snapshots with
per-class and per-slice tables, regression gates against a pinned baseline that fail a CI
job, prediction-diff reports that show which images changed, checkpoints that are
integrity-checked and resumable after preemption, and alerts from a small optional server.
Scalars are forwarded to MLflow, W&B or Trackio; run data stays in local files the user owns.

Built today: the CLI (`--version`, `doctor`, `runs list`, `runs show`, `repro`, `eval`), settings,
logging, the run schema, run capture (`init`, `log`, `log_artifact`, `finish`, crash and
resume handling), the local run store and the eval engine (`log_eval`: COCO-style detection
metrics, top-1 classification, slices, bootstrap intervals, immutable snapshots). Everything else
in this document set (hooks, compare, gate, checkpoints, forwarders, sync, alerts, server) is
planned and marked as such.

## Users

| Persona | What they do with CV-Pilot | Entry point |
|---|---|---|
| CV engineer training detectors or classifiers | Adds `autolog()` or a callback to the training script, logs eval snapshots, resumes after a spot kill, runs `compare` locally | Python API, CLI |
| Reviewer of a model change (tech lead, release owner) | Reads the gate result and the HTML prediction diff on a pull request, pins the production baseline | CLI (`baseline`, `compare`), HTML report, server UI |
| CI maintainer | Wires `cvpilot gate` into the pipeline and relies on its exit codes | CLI exit codes |
| On-call for long training jobs | Receives `nan_loss`, `no_progress`, `preempted`, `resumed`, `new_best` alerts on Telegram or Slack; opens the live view on a phone | Server, notification channels |
| Platform engineer | Runs the server container behind a TLS proxy, mounts `/checkpoint` to a bucket, issues per-project tokens | Docker image, env vars |

## Scope

In scope:

- Run store: metadata, reproducibility capture, metric points, eval snapshots, artifacts.
- Automatic capture from Ultralytics, Lightning, HF Trainer, MMEngine, RF-DETR and plain
  PyTorch, each an extra of this package (`cvpilot[ultralytics]` carries an AGPL-3.0
  notice).
- Eval engine: COCO-style mAP with per-class AP/AR, confusion matrix including background,
  slices on any metadata key, bootstrap confidence intervals over images.
- `compare` and `gate` with thresholds on headline, per-class and per-slice deltas; HTML
  report with a prediction-diff gallery.
- Checkpoint manager with atomic writes, manifests, verified restore, preemption handling,
  remote mirror.
- Forwarders to MLflow, W&B, Trackio and TensorBoard.
- Optional server (FastAPI, websocket live view, htmx UI) and alert rules with Telegram,
  Slack, webhook and SMTP channels; client-side rule evaluation without a server.
- Importers for Trackio SQLite, MLflow and TensorBoard event files.

Explicit non-goals:

- A general experiment tracker UI (curve dashboards, sweeps, model registry). Sweeps are
  delegated to Optuna and the registry to MLflow.
- Dataset management, annotation or dataset QA. That is DataCopilot; CV-Pilot only records
  its `manifest.version`.
- Training orchestration or job scheduling. SkyPilot and SLURM stay in charge; CV-Pilot
  provides recipes and signal handlers.
- Segmentation, keypoint or tracking metrics in the first eval engine release.
- Multi-tenant SaaS, SSO, billing.

## Quality attributes

Numbers are budgets. A budget becomes a measurement when the named script exists and its
result is recorded here; until then the row is a commitment, not a claim.

| Attribute | Budget | Reference input | How verified |
|---|---|---|---|
| Core install size | ≤ 200 MB on disk for `cvpilot` plus core dependencies (typer, click, pydantic, structlog, numpy, pyarrow, duckdb, faster-coco-eval, safetensors, fsspec) | clean `uv venv`, Python 3.12, Linux x86_64 | `benchmarks/install_size.sh`; CI dependency check. Measured on macOS arm64, Python 3.12, the current core set (typer, click, pydantic, structlog, pyarrow, numpy; duckdb, faster-coco-eval, safetensors, fsspec not added): 164 MB, no torch. `faster-coco-eval` is not added: the engine computes COCO metrics itself and pycocotools is a test-only dependency |
| Core install time | ≤ 30 s with a warm wheel cache, ≤ 120 s cold | same | same script |
| CLI startup | `cvpilot --version` ≤ 200 ms p95; `cvpilot doctor` ≤ 500 ms p95 | laptop, Python 3.12 | `benchmarks/startup.sh` with hyperfine (planned) |
| Logging overhead in the training loop | `cvpilot.log()` of 50 scalars ≤ 0.2 ms p95; end-to-end throughput loss < 2 % | Lightning toy detector on coco8, 20 epochs, CPU | `benchmarks/overhead.py` (planned); run-store part measured by `benchmarks/log_readback.py`: p95 0.17 ms per `log()` of 50 scalars, 100 000 points in 0.5 s, macOS arm64, Python 3.13 |
| Eval parity | per-class AP and mAP within 1e-4 of pycocotools | synthetic corpus (no COCO val2017 fixture exists): 300 randomized sets (1 to 24 images, 1 to 6 classes, crowd boxes, tied scores), 40 sets through `evaluate`, 40 with unpadded integer ids and identical or quantised scores, 40 with 8 to 20 classes, 25 weighted-bootstrap-versus-duplicated-image comparisons, and 13 hand-built edge cases (no detections, no ground truth, both empty, crowd-only, crowd-heavy with ties, identical scores, single class and image, empty images, a class without ground truth, 160 detections in one group with a forced multi-chunk matcher, duplicated ground-truth boxes, boundary areas, IoU exactly equal to twelve thresholds) | `tests/parity/test_coco_parity.py` (built; default suite, about 40 s). Measured against pycocotools 2.0.11: maximum absolute difference 0 on the 12 summary metrics and on per-class AP, AP50 and AR100 (test tolerance 1e-12) |
| Eval throughput | ≤ 10 s for 5 000 images and 100 000 detections on 4 CPU cores; bootstrap with 1 000 resamples ≤ 5 s extra | same fixture scaled | `benchmarks/eval_throughput.py` (built). Measured on macOS arm64, 12 cores, Python 3.10, 5 000 images, 50 classes, 30 028 boxes, 100 000 detections: evaluation 0.67 s; 1 000 bootstrap resamples +2.1 s with 4 threads (+5.0 s with 1); 10 slices +0.07 s; `cvpilot eval` end to end 3.4 s |
| Eval memory ceiling | ≤ 2 GB RSS for the input above | same | same script, peak RSS. Measured: 140 MB in process (122 MB after evaluation), 198 MB for the CLI child |
| Client memory ceiling | ≤ 200 MB RSS added to the training process; metric buffer bounded at 10 000 points | Lightning toy detector | `benchmarks/overhead.py` |
| Determinism | same store contents and seed produce byte-identical `EvalSnapshot` JSON and `compare` report (bootstrap seeded, seed stored in the report) | any two runs of the parity fixture | `tests/unit/test_eval_engine.py`, `test_eval_bootstrap.py`, `test_eval_record.py`, `test_cli_eval.py` (built for snapshots: identical bytes for identical input and seed, also across bootstrap worker counts); `tests/golden/test_report_golden.py` (planned for `compare`) |
| Round-trip losslessness | every `MetricPoint` forwarded to a tracker is readable back with equal `key`, `step` and float64 `value`; store export/import of a project reproduces identical tables | 3 runs × 10 000 points | `tests/golden/test_forward_roundtrip.py` (planned), `tests/golden/test_store_roundtrip.py` (built: 100 000 points read back identical) |
| Checkpoint integrity | 0 corrupted restores in 1 000 simulated kills during save or upload | `tests/chaos/` harness | `tests/chaos/test_restore_latest_valid.py` (planned, nightly) |
| Checkpoint save overhead | ≤ 5 % over `torch.save` of the same state on local SSD | 500 MB state dict | `benchmarks/ckpt_save.py` (planned) |
| Sync never blocks training | `log()` latency independent of server reachability; outbox persists across restarts; retries with exponential backoff 1 s to 60 s | server down for 10 min during a run | `tests/unit/test_sync_outbox.py` (planned) |
| Server ingest | ≤ 50 ms p95 per batch of 1 000 metric points on SQLite; ≤ 20 ms on Postgres | single server container, 2 vCPU | `benchmarks/server_ingest.py` (planned) |

## Constraints

| Constraint | Source |
|---|---|
| Apache-2.0 for every file, including the server; copyleft packages only as lazily imported extras, never vendored | [ADR-0001](../adr/0001-scope-and-license.md), [dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md) |
| Core is CPU-only and torch-free; its runtime dependencies are `typer`, `click`, `pydantic`, `structlog`, `numpy` (eval math) and `pyarrow` (lazy, store only); PyTorch, framework SDKs, FastAPI and tracker SDKs are extras | [ADR-0003](../adr/0003-repo-and-tooling.md), [ADR-0005](../adr/0005-run-store.md) |
| Python ≥ 3.10; CI on 3.10 to 3.13 | [standards.md](../engineering/standards.md) |
| No database server required: SQLite WAL locally; Postgres optional for the server only | [ADR-0005](../adr/0005-run-store.md) |
| Four layers, downward imports, registries for adapters | [ADR-0004](../adr/0004-layering-and-dependency-rule.md) |
| Structured JSON logs; secrets never logged | [ADR-0002](../adr/0002-logging.md), [errors-and-logging.md](../engineering/errors-and-logging.md) |

## Context view

![System context](../diagrams/png/context.png)

| External system | Direction | Protocol / format | Why |
|---|---|---|---|
| Training frameworks (Ultralytics, Lightning, HF Trainer, MMEngine, RF-DETR, plain PyTorch) | framework → CV-Pilot | in-process callbacks: `on_fit_epoch_end`, `on_validation_end`, `on_save`, `TrainerCallback`, MMEngine hook | source of metrics, predictions and checkpoint events |
| User's tracker (MLflow, W&B, Trackio, TensorBoard) | CV-Pilot → tracker | tracker SDK calls: `log_metric`, `wandb.log`, event files | tracker stays authoritative for scalars |
| Object storage (S3, GCS, any fsspec backend) | CV-Pilot → storage | fsspec `put`/`get`, multipart where supported | checkpoint mirror and artifact export |
| Notification channels (Telegram Bot API, Slack incoming webhook, generic webhook, SMTP) | CV-Pilot → channel | HTTPS JSON POST; SMTP | alert delivery |
| Scheduler / cloud (SkyPilot, SLURM, spot instances) | scheduler → CV-Pilot | SIGTERM/SIGUSR1; HTTP GET to instance-metadata preemption endpoints | triggers save-before-kill and marks the run `preempted` |
| CI runner (GitHub Actions, GitLab CI, any shell) | CI → CV-Pilot | process invocation of `cvpilot gate`; exit code | blocks a merge on regression |
| GPU telemetry (NVML via pynvml) | CV-Pilot → NVML | library calls every 10 s | `gpu_util_below` alert, utilisation curves |
| DataCopilot (sibling project) | CV-Pilot → DataCopilot files | reads `manifest.json` at `dataset_uri` | records `manifest.version` as `Repro.dataset_hash` |
| Reviewer's browser | browser → CV-Pilot server | HTTPS, websocket (`/ws/runs/{run_id}`) | live curves, eval and compare reports |

## Container view

![Containers](../diagrams/png/containers.png)

| Container | Responsibility | Technology | Status |
|---|---|---|---|
| `cvpilot` CLI | `--version`, `doctor`, `runs list`, `runs show`, `repro`, `eval`; planned: `compare`, `gate`, `baseline`, `ckpt`, `sync`, `import`, `server` | Typer, single error boundary | built (`--version`, `doctor`, `runs`, `repro`, `eval`) |
| Python API | `cvpilot.init/log/log_artifact/log_eval/finish`, `Run` handle; planned: `autolog`, `resume_path` | pure Python | built (`init`, `log`, `log_artifact`, `log_eval`, `finish`) |
| Framework hooks | adapters per framework | framework SDKs as extras | planned |
| Core library | Pydantic schemas, invariants, ULIDs, redaction, hashing, metric math | pydantic v2, numpy | built |
| Eval engine | mAP, per-class AP/AR, confusion with background, slices, bootstrap CIs | numpy (pycocotools only in tests) | built |
| Compare + gate | deltas with CIs, thresholds, HTML report with prediction-diff gallery, exit codes | numpy, supervision for rendering, Jinja2 templates | planned |
| Checkpoint manager | atomic save, manifest, verified restore, retention, preemption handler, remote mirror | safetensors, torch (extra), fsspec | planned |
| Forwarders | push scalars and artifacts to MLflow, W&B, Trackio, TensorBoard | tracker SDKs (extras) | planned |
| Sync client | offline-first outbox, background thread, HTTPS batches with `(run_id, seq)` | stdlib `urllib`, `ssl`, `threading` | planned |
| Alert rules | `nan_loss`, `loss_spike`, `no_progress`, `gpu_util_below`, `eval_regression_vs_baseline`, `new_best`, `failed`, `preempted`, `resumed`; evaluated client-side and server-side | pure Python | planned |
| Local run store | `~/.cvpilot/<project>/`: SQLite WAL, Parquet, content-addressed artifacts; planned: DuckDB queries | sqlite3, pyarrow | built (duckdb queries planned) |
| Server | REST ingest, websocket live view, htmx + uPlot UI, token auth, share links, alert delivery | FastAPI, Uvicorn, SQLite or Postgres; separate Docker image via `[server]` extra | planned |
