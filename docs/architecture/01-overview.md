# 01 Overview

## Purpose

CV-Pilot is a Python package that sits beside a team's existing experiment tracker and adds
what trackers do not have for detection and classification work: evaluation snapshots with
per-class and per-slice tables, regression gates against a pinned baseline that fail a CI
job, prediction-diff reports that show which images changed, checkpoints that are
integrity-checked and resumable after preemption, and alerts from a small optional server.
Scalars are forwarded to MLflow, W&B or Trackio; run data stays in local files the user owns.

Built today: the CLI (`--version`, `doctor`, `runs list`, `runs show`, `repro`, `eval`, `baseline`,
`compare`, `gate`), settings, logging, the run schema, run capture (`init`, `log`, `log_artifact`,
`finish`, crash and resume handling, `autolog`), the local run store, the eval engine (`log_eval`:
COCO-style detection metrics, top-1 classification, slices, bootstrap intervals, immutable
snapshots), append-only baselines, compare and the fail-closed gate with HTML and JSON reports,
forwarders to MLflow, W&B and Trackio, the Lightning and Ultralytics hooks, and a composite
GitHub Action. Everything else in this document set (checkpoints, sync, alerts, server, the
other hooks, TensorBoard, the prediction-diff gallery) is planned and marked as such.

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
  remote mirror (built in 0.1.0).
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
| Compare and gate cost | `cvpilot gate` end to end including HTML and JSON reports ≤ 3 s and ≤ 300 MB RSS on a 5 000-image snapshot (set in this phase) | 5 000 images, 50 classes, 10 slices, 100 000 detections, two snapshots | `benchmarks/compare_throughput.py` (built). Measured on macOS arm64, Python 3.10, median of 5: pin baseline 0.005 s; resolving and verifying both snapshots and confusion artifacts 0.006 s; compare plus gate under 1 ms; both reports 0.003 s (HTML 22 KB, JSON 41 KB); `cvpilot gate` in a child process 0.27 s (max 0.29 s), child peak RSS 81 MB. Snapshots hold aggregates, so the cost does not grow with the image count; the 5 000-image evaluation that creates them is measured by the eval rows |
| Eval memory ceiling | ≤ 2 GB RSS for the input above | same | same script, peak RSS. Measured: 140 MB in process (122 MB after evaluation), 198 MB for the CLI child |
| Client memory ceiling | ≤ 200 MB RSS added to the training process; metric buffer bounded at 10 000 points | Lightning toy detector | `benchmarks/overhead.py` |
| Determinism | same store contents and seed produce byte-identical `EvalSnapshot` JSON and `compare` report (bootstrap seeded, seed stored in the report) | any two runs of the parity fixture | `tests/unit/test_eval_engine.py`, `test_eval_bootstrap.py`, `test_eval_record.py`, `test_cli_eval.py` (built for snapshots: identical bytes for identical input and seed, also across bootstrap worker counts); `tests/golden/test_report_golden.py` (built: the HTML and JSON reports; `compare` does no resampling, so the report has no seed) |
| Round-trip losslessness | every `MetricPoint` forwarded to a tracker is readable back with equal `key`, `step` and float64 `value`; store export/import of a project reproduces identical tables | 3 runs × 10 000 points | `tests/unit/test_forward_integration.py` (built: float64 values, including `NaN`, `inf` and `-0.0`, reach a forwarder with equal key and step; tracker-side key sanitising is the only deviation), `tests/golden/test_store_roundtrip.py` (built: 100 000 points read back identical) |
| Checkpoint integrity | 0 corrupted restores after a kill during save or upload; every checkpoint restore returns is complete and has exactly the content that was saved | `tests/chaos/` harness | built: `tests/chaos/test_ckpt_sigkill.py` (SIGKILL at random delays and at four chosen points inside a save, restore returns a whole checkpoint and the debris is cleaned by the next save), `test_ckpt_hostile.py` (full disk by file-size limit, three concurrent saver processes, a file replaced right after it was read, symlink swaps, orphan temp files, clock skew, a forged coherent directory), `test_ckpt_signals.py` (real SIGTERM and SIGUSR1), `test_ckpt_kill_points.py` (1 000 seeded kills: 200 at each of four places inside a save and 200 at a random moment; after each, a fresh process restored the expected checkpoint byte for byte and saved the next step) and `test_ckpt_truncated_upload.py` (200 uploads to an in-memory filesystem cut after a random number of bytes; none was restored, the newest completed one was). Local disk only; no power loss |
| Checkpoint save overhead | Withdrawn: the 5 % rule over `torch.save` cannot be met while the digest and the read-back exist. Measured instead (see the result) | real model states from notebook 04: tiny Lightning classifier, YOLO11n weights (10.7 MB), ResNet-18 weights (44.8 MB) | `benchmarks/ckpt_save.py` (`profile_state`, called by notebook 04). **Result** (macOS arm64, local SSD, Python 3.13, `keep_last=3`, against `torch.save` of the same state written and `fsync`ed): ResNet-18 weights `save_now` 52.71 ms against 16.67 ms (3.2 times; 3.2 to 3.5 over repeated runs); YOLO11n weights 19.34 ms against 12.05 ms (1.6 times; 1.5 to 2.0 over repeated runs because the framework's own save takes only about 12 ms); the tiny classifier 4.48 ms against 0.44 ms, dominated by fixed costs (retention, pointer, manifest, locking). Two SHA-256 passes (while writing and when the file is read back) are 56 % and 36 % of a ResNet-18 and a YOLO11n save; for ResNet-18 the exclusive phase times are 40.3 % writing with hashing, 33.2 % the read-back, 12.0 % serialising, 5.7 % retention, 1.4 % the pointer, 6.5 % unattributed. Median of 9 interleaved, warmed-up rounds `fsync` is not the cost on this machine (macOS `fsync` flushes to the drive cache; a Linux server will differ). At the default interval of 10 minutes a ResNet-18 save is 0.009 % of the wall time ([ADR-0010](../adr/0010-real-checkpoint-files-and-mmengine.md)) |
| Sync never blocks training | `log()` latency independent of server reachability; outbox persists across restarts; retries with exponential backoff 1 s to 60 s | server down for 10 min during a run | `tests/unit/test_sync_outbox.py` (planned) |
| Server ingest | ≤ 50 ms p95 per batch of 1 000 metric points on SQLite; ≤ 20 ms on Postgres | single server container, 2 vCPU | `benchmarks/server_ingest.py` (planned) |

## Constraints

| Constraint | Source |
|---|---|
| Apache-2.0 for every file, including the server; copyleft packages only as lazily imported extras, never vendored | [ADR-0001](../adr/0001-scope-and-license.md), [dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md) |
| Core is CPU-only and torch-free; its runtime dependencies are `typer`, `click`, `pydantic`, `structlog`, `numpy` (eval math) and `pyarrow` (lazy, store only); PyTorch, framework SDKs, FastAPI and tracker SDKs are extras | [ADR-0003](../adr/0003-repo-and-tooling.md), [ADR-0005](../adr/0005-run-store.md) |
| Python ≥ 3.10; CI on 3.10 to 3.14 | [standards.md](../engineering/standards.md) |
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
| `cvpilot` CLI | `--version`, `doctor`, `runs list`, `runs show`, `repro`, `eval`, `baseline`, `compare`, `gate`, `ckpt list|verify|prune`; planned: `sync`, `import`, `server` | Typer, single error boundary | built (`--version`, `doctor`, `runs`, `repro`, `eval`, `baseline`, `compare`, `gate`, `ckpt`) |
| Python API | `cvpilot.init/log/log_artifact/log_eval/finish/autolog/resume_path`, `CheckpointManager`, `Run` handle | pure Python | built (frozen by `tests/unit/test_api_surface.py`) |
| Framework hooks | adapters per framework: Lightning (logger and callback), Ultralytics, Hugging Face and MMEngine (callbacks and hooks), `autolog` | framework SDKs as extras | built (Lightning, Ultralytics scalars and per-class AP, Hugging Face, MMEngine scalars and metrics, RF-DETR scalars and metrics); plain PyTorch planned |
| Core library | Pydantic schemas, invariants, ULIDs, redaction, hashing, metric math | pydantic v2, numpy | built |
| Eval engine | mAP, per-class AP/AR, confusion with background, slices, bootstrap CIs | numpy (pycocotools only in tests) | built |
| Compare + gate | deltas with the stored CIs, fail-closed thresholds, recomputation on trusted ground truth, HTML and JSON reports, exit codes (the prediction-diff gallery is planned) | numpy and the standard library (`html`; no template engine or rendering dependency) | built |
| Checkpoint manager | atomic save, manifest, verified restore, retention, preemption handler, remote mirror, HMAC authentication and a trust policy | safetensors and fsspec (optional extras `[safetensors]` and `[remote]`), the framework only for the opt-in `pt` format | built |
| Forwarders | push scalars to MLflow, W&B, Trackio (TensorBoard planned) on worker threads behind bounded queues | tracker SDKs (extras), `runs/forwarding.py` | built (MLflow, W&B, Trackio; artifacts and TensorBoard planned) |
| Sync client | offline-first outbox, background thread, HTTPS batches with `(run_id, seq)` | stdlib `urllib`, `ssl`, `threading` | planned |
| Alert rules | `nan_loss`, `loss_spike`, `no_progress`, `eval_regression`, `run_failed`, `run_preempted`, evaluated in the training process (ADR-0009); planned: `gpu_util_below`, `new_best`, `resumed` and server-side evaluation | pure Python | built (client side) |
| Local run store | `~/.cvpilot/<project>/`: SQLite WAL, Parquet, content-addressed artifacts; planned: DuckDB queries | sqlite3, pyarrow | built (duckdb queries planned) |
| Server | REST ingest, websocket live view, htmx + uPlot UI, token auth, share links, alert delivery | FastAPI, Uvicorn, SQLite or Postgres; separate Docker image via `[server]` extra | planned |
