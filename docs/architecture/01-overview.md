# 01 Overview

## Purpose

CV-Pilot adds what experiment trackers lack for detection and classification work: scored
snapshots per class and per slice, a gate that fails a CI job when a model gets worse, and
checkpoints that are verified and resumable after preemption. Metrics can also go to MLflow, W&B or
Trackio. Run data stays in local files you own.

## Users

| Who | What they do | Entry point |
|---|---|---|
| CV engineer | Adds `autolog()`, logs evaluations, resumes after a kill | Python API, CLI |
| Reviewer or release owner | Reads the gate result, pins the production baseline | `baseline`, `compare` |
| CI maintainer | Wires `cvpilot gate` into the pipeline | Exit codes, GitHub Action |
| On-call for long jobs | Gets an alert when the loss turns NaN or a job is preempted | Slack, Telegram, webhook |

## Scope

In scope:

- A run store: metadata, how to reproduce a run, metric points, evaluation snapshots, artifacts.
- Hooks for Lightning, Hugging Face, Ultralytics, MMEngine and RF-DETR, each an extra.
- An evaluation engine: COCO-style mAP, per-class AP, confusion matrix, slices and bootstrap
  intervals.
- `compare` and `gate` with limits on headline, per-class and per-slice drops.
- A checkpoint manager: atomic writes, verified restore, preemption handling, remote mirror.
- Forwarders to MLflow, W&B and Trackio, and alert rules with webhook, Slack and Telegram channels.

Not in scope:

- A tracker UI, sweeps or a model registry. Use Optuna and MLflow.
- Dataset management or annotation. That is [DataCopilot](https://github.com/B4teShai/DataCopilot);
  CV-Pilot only records its dataset version.
- Job scheduling. SkyPilot and SLURM stay in charge.
- Segmentation, keypoint or tracking metrics.
- Multi-tenant hosting, SSO or billing.

What exists today is on the [status page](../status.md).

## What must hold

| Quality | Budget |
|---|---|
| Core install | Under 200 MB, CPU only, no deep-learning framework. |
| Logging overhead | `log()` of 50 scalars under 0.2 ms at p95. |
| Evaluation parity | Per-class AP and mAP within 1e-4 of pycocotools. |
| Evaluation cost | 5,000 images and 100,000 detections in under 10 s on 4 cores. |
| Gate cost | Under 3 s and 300 MB for a 5,000-image snapshot. |
| Determinism | The same store and seed give byte-identical snapshots and reports. |
| Checkpoint integrity | A kill during a save or upload never produces a corrupt restore. |
| Save cost | Measured, not budgeted ([ADR-0010](../adr/0010-real-checkpoint-files-and-mmengine.md)). |

Each budget has a script in `benchmarks/` or a test that checks it.

## Constraints

| Constraint | Why |
|---|---|
| Apache-2.0 everywhere; copyleft only as lazily imported extras | [ADR-0001](../adr/0001-scope-and-license.md) |
| Core is CPU-only; its dependencies are `typer`, `click`, `pydantic`, `structlog`, `numpy`, `pyarrow` | [ADR-0003](../adr/0003-repo-and-tooling.md) |
| Python 3.10 or newer, tested on 3.10 to 3.14 | [standards](../engineering/standards.md) |
| No database server: SQLite locally | [ADR-0005](../adr/0005-run-store.md) |
| Four layers, imports point downward | [ADR-0004](../adr/0004-layering-and-dependency-rule.md) |
| Structured JSON logs, no secrets | [ADR-0002](../adr/0002-logging.md) |

## Context

![CV-Pilot and the people and tools around it](../diagrams/png/context.png)

| Outside | Direction | How |
|---|---|---|
| Training frameworks | Framework calls CV-Pilot | In-process callbacks |
| Your tracker | CV-Pilot sends metrics | The tracker's own SDK |
| Object storage (S3, GCS, Azure) | CV-Pilot copies checkpoints | fsspec |
| Slack, Telegram, webhook | CV-Pilot sends alerts | HTTPS POST |
| Scheduler or cloud | Tells CV-Pilot to stop | SIGTERM, instance metadata |
| CI runner | Runs `cvpilot gate` | Process and exit code |
| DataCopilot | CV-Pilot reads one file | `dataset.dcp/manifest.json` |
