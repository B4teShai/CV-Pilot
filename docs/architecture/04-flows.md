# 04 Flows

Two flows, end to end. Exit codes: `0` ok, `1` the command ran and failed (a failing gate or a
refused comparison included), `2` usage or configuration error, `130` interrupted
([contract](../engineering/errors-and-logging.md)). Event names are the fixed `message` values in
[06 Cross-cutting](06-cross-cutting.md).

## Flow A: hook, log, eval, gate

Purpose: record metrics and an evaluation snapshot from a training framework without changing the
loop, then fail a CI job when the new run is worse than the pinned baseline.

Needs: a run started by `cvpilot.init()` or `cvpilot.autolog()`; predictions and ground truth at the
end of a validation epoch; for the gate, a pinned baseline and a CI runner that can read the store.

![Flow A: from a framework callback to a CI verdict](../diagrams/png/flow-a.png)

1. The framework fires a callback. The hook converts its scalars to `MetricPoint`s and calls
   `sink.log(points, step=...)`.
2. `runs/` validates the batch, appends it to the store, offers it to the forwarders and runs the
   `nan_loss`, `loss_spike` and `no_progress` alert rules.
3. At the end of validation the hook calls `sink.log_eval(preds, gts, step=...)`.
4. `eval/` hashes the evaluation set, matches detections, and writes the predictions, the confusion
   matrix and the snapshot. It logs `eval_completed`.
5. In CI, `cvpilot gate RUN --baseline tag:prod ...` loads the baseline and the candidate snapshot.
   Each is checked against its digest and hash chain.
6. If the eval set hash, task, class names or evaluation settings differ, the comparison is refused
   and nothing else runs.
7. Otherwise `compare/` computes the deltas, using the intervals stored in the snapshots, and `gate`
   checks the limits.
8. The command prints a one-line verdict and the violations, then exits.

How it fails:

| Failure | What happens | Exit | Event |
|---|---|---|---|
| Extra not installed, or no hook matches | `autolog` raises `MissingExtraError` or `HookNotFoundError`; no run starts | Python error | `hook_missing` |
| A callback fails inside the framework | Caught and counted; the hook is disabled after 10 failures in a row | none | `hook_failed`, `hook_disabled` |
| A forwarder is down, slow or hung | Counted, logged once, queue drops batches; training is never blocked | none | `forward_failed` |
| Store write error | `log()` raises `StoreError`; your loop decides | none | `store_write_failed` |
| Run, baseline or project not found | One line on `stderr` | 1 | `command_failed` |
| Bad option, bad threshold syntax, no threshold | Usage error | 2 | none |
| Eval set, task, classes, settings or slices differ; baseline is the candidate or unfinished; wrong commit | Refused, no verdict, no report | 1 | `compare_refused` |
| `--recompute` finds a stored result that differs | No verdict | 1 | `gate_recompute_failed` |
| A snapshot, baseline row or chain fails its check, or `--expect-baseline-chain` differs | No verdict | 1 | `snapshot_integrity_failed`, `baseline_integrity_failed` |
| A metric or slice is missing or not finite, too few images, or a threshold gated nothing | Violation listed | 1 | `gate_failed` |
| A slice is smaller than `--min-images` | Violation, or skipped with `--skip-small-slices` up to `--max-skipped` | 1 | `gate_failed`, `slice_skipped` |
| A drop exceeds its limit | Violations printed, worst first | 1 | `gate_failed` |
| Everything holds (gains never fail) | Verdict printed | 0 | `gate_passed` |
| Ctrl-C | Partial output discarded | 130 | none |

Events in order: `run_started`, `hook_attached`, `metrics_flushed`, `eval_completed`,
`compare_completed`, then `gate_passed` or `gate_failed`, then `run_finished`. Evaluating into a
finished run needs `--allow-closed`.

## Flow B: save, preemption, restore

Purpose: a job killed at any moment resumes from the newest checkpoint that is complete and
verified, locally or from a bucket, with its history recorded. A missing, rolled-back or forged
checkpoint is never mistaken for a fresh start.

Needs: a `CheckpointManager` created inside the training process, a folder on local disk, and for
lineage resume a run store and folder that survive the restart. On a node with neither, use
`restore(discover=True)` with a remote. With `CVPILOT_CKPT_KEY` set the trust level defaults to
`authenticated`, and a call can only tighten it.

![Flow B: an atomic save, a preemption and a verified restore](../diagrams/png/flow-b.png)

Saving:

1. `maybe_save` decides: interval passed, `best` improved, or a preemption flag is set.
2. The state is split into tensors and a JSON part and checked against the bounds a restore applies.
   Under the folder lock, leftovers of a killed save are removed. The data goes to a private temp
   file, is hashed while written, and is synced.
3. The manifest is written the same way, renamed into place, then the data file is renamed. The
   folder is synced.
4. The data is re-read and re-hashed, the manifest digest is recorded in the run store, and only
   then is `LATEST` rewritten. The events are `checkpoint_saved` and `checkpoint_verified`.
5. With a remote, the data goes to a `.part` object, size and SHA-256 are read back, it is renamed,
   and the manifest goes last. A failed upload is logged and retried by a later save.
6. Retention deletes valid checkpoints beyond `keep_last` and `keep_best`, never the newest valid
   one, and with a remote only once the newest is confirmed there.

Preemption:

7. `SIGTERM`, `SIGUSR1` or a polled cloud notice sets a flag. Nothing else runs in the handler. The
   next `maybe_save` logs `preemption_signal_received`, saves, marks the run `preempted` (event
   `run_preempted`) as soon as the checkpoint verifies, then uploads with the time left. Past
   `grace_seconds` it logs `preemption_save_incomplete`.

Restoring:

8. `restore()` walks this run's folder, then the runs it resumed, nearest first. A candidate must
   have a manifest that parses, matches its name and chains, reach the required trust level, not be
   older than the rollback floor, and be a regular file of the recorded size and SHA-256. Others are
   skipped with `checkpoint_skipped_corrupt` or `checkpoint_untrusted`.
9. With nothing valid locally, the remote is consulted through the same checks. With
   `discover=True`, the newest remote runs are listed and tried best step first.
10. If a floor is known and nothing reaches it, restore raises `CheckpointUntrustedError`
    (`rolled_back` or `checkpoint_missing`). Under `any` it logs `checkpoint_progress_lost`. Only
    when nothing is known does it return `None`.
11. The trainer gets the state and applies the random state itself with `apply_rng(state["rng"])`.
    The child run (`init(resume_from=...)`) has `parent_run_id` and status `resumed`. The events are
    `checkpoint_restored` (with `trust_level`) and `run_resumed`.

How it fails:

| Failure | What happens | Event |
|---|---|---|
| Kill during the temp write | The temp file has no manifest and is invisible; the next save deletes it | none |
| Kill after the manifest rename, before the data rename | Manifest without data is skipped as `data_missing` and deleted later | `checkpoint_skipped_corrupt` |
| Kill after the renames, before `LATEST` moves | `LATEST` names the previous checkpoint; the new one is found through the chain | none |
| Kill during upload | The remote manifest is absent or mismatched; a later save uploads again | `checkpoint_upload_failed` |
| Disk full, or a state a restore would refuse (such as a bfloat16 tensor) | `CheckpointError`; nothing written, previous checkpoint untouched | `checkpoint_save_failed` |
| Nothing known to have existed | `restore()` returns `None`: a fresh start | `checkpoint_restore_empty` |
| A floor is known and no valid checkpoint reaches it | `CheckpointUntrustedError`; under `any`, `None` plus a warning | `checkpoint_untrusted`, `checkpoint_progress_lost` |
| The remote lists checkpoints and none passes | `CheckpointUntrustedError`, not a fresh start | `checkpoint_untrusted` |
| A save is rejected (forged step above the trusted ones) | The step goes on the quarantine list; no file moves | `checkpoint_quarantined` |
| A call asks for weaker trust than configured | Ignored | `checkpoint_trust_downgrade_ignored` |
| A `pt` file fails the strict scan | Not returned; the reason is logged | `resume_point_skipped` |
| `cvpilot ckpt verify` finds a corrupt or untrusted file | Listed on `stdout`, exit 1 | `checkpoint_verify_failed` |
| `cvpilot ckpt` on a path that is not a checkpoint folder | Usage error, exit 2 | none |
| The metadata service of a cloud notice fails | Retried with backoff; the thread says why it ends | `notice_poller_degraded`, `preemption_poller_stopped` |

Events in order: `checkpoint_saved`, `checkpoint_verified`, `checkpoint_uploaded`,
`preemption_signal_received`, `run_preempted`, then `checkpoint_skipped_corrupt` or
`checkpoint_untrusted`, `checkpoint_restored`, `run_resumed`. Tests: `tests/unit/test_ckpt_*.py` and
`tests/chaos/`.
