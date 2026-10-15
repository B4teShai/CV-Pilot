# Command line

This page is generated from the Typer application by `scripts/gen_cli_reference.py`; a test
fails when it is out of date. Exit codes: `0` success, `1` failure (a regression, a failed
verification, a missing or unreadable input), `2` usage or configuration error, `130`
interrupted. Every command logs structured JSON events on `stderr`; results go to `stdout`.


## `cvpilot repro`

```text
cvpilot repro [OPTIONS] {run_id}
```

Print a bash script that reproduces a run (stdout). Exit 1 if the run does not exist.

| Argument | Type | Description |
|---|---|---|
| `run_id` | str | Run id (ULID). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |

## `cvpilot eval`

```text
cvpilot eval [OPTIONS] {run_id}
```

Evaluate Parquet files into a run and store an immutable snapshot.

| Argument | Type | Description |
|---|---|---|
| `run_id` | str | Run id (ULID). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--preds`, `--predictions` | file |  | Predictions Parquet. Required. |
| `--gts`, `--ground-truth` | file |  | Ground-truth Parquet. Required. |
| `--classes` | file |  | Class names: JSON list or lines. Required. |
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--step` | int range | `0` | Training step of the predictions. |
| `--slices` | str |  | KEY=COLUMN of a ground-truth metadata column; repeatable. Repeatable. |
| `--seed` | int range | `0` | Bootstrap seed. |
| `--resamples` | int range | `1000` | Bootstrap resamples (0 = off). |
| `--max-memory-mb` | int range | `8192` | Address-space ceiling in MB (Linux; 0 = off). |
| `--allow-closed` | flag |  | Allow finished, failed or preempted runs. |
| `--checkpoint-sha256` | str |  | SHA-256 of the checkpoint. |
| `--git-commit` | str |  | Commit of the code. |
| `--json` | flag |  | Print JSON instead of text. |

## `cvpilot compare`

```text
cvpilot compare [OPTIONS] {run_id}
```

Compare a run with a baseline (deltas, intervals, worst classes, slices, confusion).

| Argument | Type | Description |
|---|---|---|
| `run_id` | str | Run id (ULID). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--baseline` | str |  | tag:NAME of a pinned baseline, or a baseline run id. Required. |
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--eval-set` | str |  | Eval set hash or prefix (8+ hex). |
| `--step` | int range |  | Candidate snapshot step (default: latest). |
| `--require-commit` | str |  | Refuse a candidate not produced from this commit (7+ hex); eval needs --git-commit. |
| `--expect-baseline-chain` | str |  | Chain digest of the baseline table kept outside the store (see baseline show). |
| `--report` | file |  | Write a static HTML report (no script, strict CSP). |
| `--report-json` | file |  | Write the canonical JSON report. |
| `--report-dir` | directory |  | Directory reports must stay inside (default: the current directory). |
| `--allow-outside` | flag |  | Allow reports outside --report-dir. |
| `--force` | flag |  | Replace a report file that cvpilot did not create. |
| `--json` | flag |  | Print JSON instead of text. |

## `cvpilot gate`

```text
cvpilot gate [OPTIONS] {run_id}
```

Gate a run against a baseline; exit 1 on a regression or any unverifiable check.

| Argument | Type | Description |
|---|---|---|
| `run_id` | str | Run id (ULID). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--baseline` | str |  | tag:NAME of a pinned baseline, or a baseline run id. Required. |
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--eval-set` | str |  | Eval set hash or prefix (8+ hex). |
| `--step` | int range |  | Candidate snapshot step (default: latest). |
| `--max-drop` | str |  | NAME=POINTS: largest allowed drop of a headline metric. Repeatable. |
| `--per-class-max-drop` | str |  | NAME=POINTS: largest drop of any class. Repeatable. |
| `--slice-max-drop` | str |  | "KEY=VALUE:POINTS" or "KEY=*:POINTS" (every value). Repeatable. |
| `--min-images` | int range | `30` | Fewest images in the eval set and in a slice. |
| `--skip-small-slices` | flag |  | List slices below --min-images as skipped instead of failing them (see --max-skipped). |
| `--max-skipped` | int range | `0` | Most slices --skip-small-slices may skip (default 0). |
| `--recompute` | flag |  | Re-evaluate the stored predictions on --gt and fail unless every stored metric matches. Use it in CI: without it the gate trusts whatever the store says. |
| `--gt` | file |  | Trusted ground-truth Parquet for --recompute. |
| `--classes` | file |  | Class names file for --recompute. |
| `--require-commit` | str |  | Refuse a candidate not produced from this commit (7+ hex); eval needs --git-commit. |
| `--expect-baseline-chain` | str |  | Chain digest of the baseline table kept outside the store (see baseline show). |
| `--report` | file |  | Write a static HTML report (no script, strict CSP). |
| `--report-json` | file |  | Write the canonical JSON report. |
| `--report-dir` | directory |  | Directory reports must stay inside (default: the current directory). |
| `--allow-outside` | flag |  | Allow reports outside --report-dir. |
| `--force` | flag |  | Replace a report file that cvpilot did not create. |
| `--json` | flag |  | Print JSON instead of text. |

## `cvpilot doctor`

```text
cvpilot doctor [OPTIONS]
```

Check that the installation works.

## `cvpilot runs`

```text
cvpilot runs [OPTIONS] COMMAND [ARGS]...
```

Inspect recorded runs.

### `cvpilot runs list`

```text
cvpilot runs list [OPTIONS]
```

List runs of a project, newest first. Exit 0 (an empty project is not an error).

| Option | Type | Default | Description |
|---|---|---|---|
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--status` | one of `running`, `resumed`, `finished`, `failed`, `preempted` |  | Only this status. |
| `--json` | flag |  | Print JSON instead of text. |

### `cvpilot runs show`

```text
cvpilot runs show [OPTIONS] {run_id}
```

Show one stored run. Exit 1 if the run does not exist.

| Argument | Type | Description |
|---|---|---|
| `run_id` | str | Run id (ULID). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--json` | flag |  | Print JSON instead of text. |

## `cvpilot baseline`

```text
cvpilot baseline [OPTIONS] COMMAND [ARGS]...
```

Pin and inspect baselines.

### `cvpilot baseline set`

```text
cvpilot baseline set [OPTIONS] {tag} {run_id}
```

Pin a verified snapshot of RUN_ID under TAG. Exit 1 when refused.

| Argument | Type | Description |
|---|---|---|
| `tag` | str | Baseline tag. |
| `run_id` | str | Run id (ULID). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--eval-set` | str |  | Eval set hash or prefix (8+ hex characters); optional when the run has one. |
| `--step` | int range |  | Snapshot step (default: the latest). |
| `--replace` | flag |  | Pin a tag that is already pinned (keeps history). |

### `cvpilot baseline list`

```text
cvpilot baseline list [OPTIONS]
```

List the verified baselines of a project (exit 0 when there are none).

| Option | Type | Default | Description |
|---|---|---|---|
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--history` | flag |  | Include replaced pins. |
| `--json` | flag |  | Print JSON instead of text. |

### `cvpilot baseline show`

```text
cvpilot baseline show [OPTIONS] {tag}
```

Show a baseline and its headline metrics, verifying both. Exit 1 for an unknown tag.

| Argument | Type | Description |
|---|---|---|
| `tag` | str | Baseline tag. |

| Option | Type | Default | Description |
|---|---|---|---|
| `--project` | str |  | Project name (default: $CVPILOT_PROJECT). |
| `--json` | flag |  | Print JSON instead of text. |

## `cvpilot ckpt`

```text
cvpilot ckpt [OPTIONS] COMMAND [ARGS]...
```

List, verify and prune checkpoints.

### `cvpilot ckpt list`

```text
cvpilot ckpt list [OPTIONS] {directory}
```

List the checkpoints of every run (or one run), newest first, judging manifests only.

Nothing is hashed: use ``verify`` for that. Exit 0 (an empty directory is not an error);
exit 2 when the path is not a directory.

| Argument | Type | Description |
|---|---|---|
| `directory` | path | Checkpoint directory (holds one folder per run). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--run` | str |  |  |
| `--kind` | one of `last`, `best`, `all` | `all` | Which checkpoints. |
| `--project` | str |  | Project whose run store anchors the digests. |
| `--require-trust` | one of `any`, `anchored`, `authenticated` |  | any, anchored or authenticated (see the help above). |
| `--json` | flag |  | Print JSON instead of text. |

### `cvpilot ckpt verify`

```text
cvpilot ckpt verify [OPTIONS] {directory}
```

Verify every checkpoint: size, structure, SHA-256, chain and the trust policy.

Prints one line per checkpoint on stdout. Exit 0 when everything verifies; exit 1 when a
checkpoint is corrupt, truncated, forged or untrusted, or when nothing verifies at all;
exit 2 when the path is not a directory or the options are invalid.

| Argument | Type | Description |
|---|---|---|
| `directory` | path | Checkpoint directory (holds one folder per run). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--run` | str |  |  |
| `--kind` | one of `last`, `best`, `all` | `all` | Which checkpoints. |
| `--project` | str |  | Project whose run store anchors the digests. |
| `--require-trust` | one of `any`, `anchored`, `authenticated` |  | any, anchored or authenticated (see the help above). |
| `--json` | flag |  | Print JSON instead of text. |

### `cvpilot ckpt prune`

```text
cvpilot ckpt prune [OPTIONS] {directory}
```

Delete checkpoints beyond ``--keep-last`` and ``--keep-best``.

Only checkpoints that verify are counted or deleted, the newest valid checkpoint of each
kind is never deleted and at least one is always kept (``--keep-*`` of 0 is refused). Exit
0; exit 1 when a saver holds the directory lock; exit 2 on a usage error.

| Argument | Type | Description |
|---|---|---|
| `directory` | path | Checkpoint directory (holds one folder per run). |

| Option | Type | Default | Description |
|---|---|---|---|
| `--run` | str |  | Run to prune (required). Required. |
| `--keep-last` | int range | `3` |  |
| `--keep-best` | int range | `2` |  |
| `--mode` | one of `max`, `min` | `max` | Best means max or min metric. |
| `--dry-run` | flag |  | Only say what would go. |
| `--project` | str |  | Project whose run store anchors the digests. |
| `--require-trust` | one of `any`, `anchored`, `authenticated` |  | any, anchored or authenticated (see the help above). |
