# ADR-0001: Scope and license

**Status:** accepted
**Date:** 2026-10-14

## Context

Scalar experiment tracking is a commodity. MLflow (Apache-2.0, SQLite by default since 3.7)
and Trackio (open source, `import trackio as wandb`, local SQLite) are free and self-hosted;
W&B is a proprietary SaaS now inside CoreWeave Forge; Neptune's hosted service was shut down
on 2026-03-05 after its acquisition and deleted customer data; Aim's last release is from
May 2025 and its remote server shipped unauthenticated; ClearML's client is Apache-2.0 but its
server is SSPL. TensorBoard is maintenance-only.

None of these tools answers the question a detection or classification team asks before a
model ships: *did the new run regress on the classes and slices we care about, and can I
see the images?* None of them makes checkpoints integrity-checked and resumable after a
spot preemption (SkyPilot relaunches jobs but restarts from scratch unless the code
checkpoints correctly; Lightning's `save_last=True` does not verify the file it loads).

Constraints that apply:

- The core package must install on CPU-only machines and must not import PyTorch
  ([ADR-0003](0003-repo-and-tooling.md)).
- The license allow-list in
  [../engineering/dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md)
  keeps copyleft packages (AGPL-3.0 Ultralytics) out of the core and forbids vendoring
  their code; they may be an optional extra imported lazily inside the adapter that needs
  them, with a notice that using the extra puts the user's combined work under that license.
- The sibling project DataCopilot produces a dataset `manifest.version`; runs must be able
  to record which dataset version they trained and evaluated on.

## Options considered

### Option A: build another general experiment tracker

A new run store, dashboards, artifact registry and sweeps, competing with MLflow, W&B and
Trackio on their own ground.

Trade-offs: full control of the data model; but the market is crowded and partly free, the
dashboard surface is enormous, and users would have to migrate their history. Every hour
spent on generic UI is an hour not spent on CV semantics.

### Option B: a thin plugin for one tracker (for example an MLflow plugin)

Register CV metrics and reports as MLflow artifacts and ship an MLflow UI plugin.

Trade-offs: smallest surface; but it locks the project to one tracker's release cycle and
plugin API, cannot run offline without that tracker's server, and leaves checkpoint
semantics, preemption handling and alerts out of scope because they are not tracker
concerns.

### Option C: a CV-native layer on top of any tracker

An installable package with a local-first run store, framework hooks, an eval engine with
regression gates and prediction diffs, a resume-safe checkpoint manager, forwarders that
keep the user's tracker authoritative for scalars, and an optional small server for live
monitoring and alerts.

Trade-offs: more components than B and a hook per framework to maintain; in return the
project owns the one lane nobody covers (per-class and per-slice regression gates in CI,
verified resume) and stays neutral across trackers.

### License options

- **Apache-2.0 for everything, including the server.** Open server, no surprise for
  self-hosters; forgoes a proprietary upsell.
- **Open-core: Apache-2.0 client, restrictive or proprietary server.** ClearML's SSPL model.
  Funds a company; but SSPL is not OSI-approved, blocks adoption in many organisations, and
  the server here is small enough that an open one is the better moat.

## Decision

Option C with Apache-2.0 for everything.

Rules reviewers can check:

1. CV-Pilot does not implement dashboards for scalar curves beyond the live view needed for
   monitoring a running job. Scalars are forwarded to the user's tracker through
   `forward/`; the tracker stays the system of record for scalars.
2. Every feature is one of: regression gate, prediction diff, eval snapshot, checkpoint
   semantics, preemption handling, alert, or the capture needed to make those reproducible.
   A feature outside that list needs a new ADR.
3. The run store is local files the user owns (SQLite + Parquet + content-addressed files;
   [ADR-0005](0005-run-store.md)). A server is optional and never required for the CLI or
   the Python API to work.
4. Every framework hook, including Ultralytics, is an extra of this package
   (`cvpilot[ultralytics]`, `cvpilot[lightning]`, `cvpilot[hf]`, `cvpilot[mmdet]`). The
   `ultralytics` package (AGPL-3.0) is declared only in the `[ultralytics]` extra and is
   imported only inside `hooks/ultralytics.py`, never at module top level and never from
   `core/` or an application package. `cvpilot autolog --help`, the hook's docstring and
   [../architecture/05-runtime.md](../architecture/05-runtime.md) state that using the
   extra makes the user's combined work subject to AGPL-3.0. No Ultralytics code is
   vendored or derived from; the distribution itself stays Apache-2.0.
5. `Repro.dataset_hash` stores the DataCopilot `manifest.version` when the dataset is a
   DataCopilot dataset, otherwise a Merkle hash over the dataset folder. The two formats are
   distinguishable by prefix ([../architecture/03-data-model.md](../architecture/03-data-model.md)).
6. Every source file, including the server, is Apache-2.0; `LICENSE` and `NOTICE` cover the
   whole repository.

## Consequences

- Easier: adoption by teams that already have a tracker; no migration; the core wheel stays
  small and CPU-only.
- Harder: one adapter per framework and per tracker to keep green against upstream API
  churn; a version-matrix CI job per hook is required once the first hook exists.
- Forbidden: vendoring or deriving from AGPL code (only a lazy import inside the adapter);
  a required server; storing run data only on a server.
- Follow-up: ADR-0005 (run store), ADR-0006 (checkpoint semantics), the plugin contract in
  [../architecture/02-components.md](../architecture/02-components.md).

## Links

- [../architecture/01-overview.md](../architecture/01-overview.md)
- [0003-repo-and-tooling.md](0003-repo-and-tooling.md),
  [0004-layering-and-dependency-rule.md](0004-layering-and-dependency-rule.md)
- [../engineering/dependencies-and-licensing.md](../engineering/dependencies-and-licensing.md)
