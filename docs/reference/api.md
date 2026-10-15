# Python API

The public API is the set of names in `__all__` of `cvpilot`, `cvpilot.ckpt`, `cvpilot.eval`,
`cvpilot.deprecation` and the exception classes of `cvpilot.core.errors`. A test
(`tests/unit/test_api_surface.py`) snapshots their names and signatures, so a change cannot reach
a release unnoticed. Everything else is internal and may change in any release.

## Recording

```python notest
import cvpilot

run = cvpilot.init(
    "proj", task="detection", config={"lr": 0.01}, seed=0, resume_from=None, forward_to=None
)  # a Run; also a context manager
cvpilot.log({"train/loss": 0.4}, step=10)  # steps strictly increase per key
snapshot = cvpilot.log_eval(preds, gts, step=10, slices={"camera": "camera"})
cvpilot.log_artifact("report.html", kind="report")
cvpilot.finish()  # "finished", "failed" or "preempted"
run = cvpilot.autolog(trainer_or_model, project="proj", task="detection")
```

`init` raises `RunAlreadyActiveError` while another run of the process is active; `Run` can be
used as a context manager (`failed` on an exception). `PredictionTable` and `AnnotationTable`
come from `cvpilot.eval`.

## Checkpoints

```python notest
from cvpilot.ckpt import CheckpointManager, find_resume_point

path = cvpilot.resume_path("proj")  # newest verified file to resume from, or None
point = find_resume_point("proj")  # the same, with the run it came from
ckpt = CheckpointManager(run, dir="/checkpoint")
ckpt.maybe_save(step, model=model, optimizer=opt, metrics={"mAP50": 0.7})
ckpt.save_now(step, model=model)  # immediately, any step that advances
restored = ckpt.restore()  # Restored(state, manifest, run_id, skipped, trust)
state = ckpt.restore_latest_valid()  # the state only, or None
ckpt.adopt("last.pt", step)  # bring a framework-written file under verification
ckpt.verify(manifest)
ckpt.list()
ckpt.preempted()
ckpt.close()
```

Errors are `CheckpointError` and its subclasses (`CheckpointCorruptError`,
`CheckpointUntrustedError`, `CheckpointFormatError`, `CheckpointBusyError`,
`CheckpointStepError`, `CheckpointExistsError`, `CheckpointPathError`), all `CvpilotError`s: the
command line prints one `error:` line and one structured event for them.

## Stability and deprecation

Before 1.0 a minor version may break the API, but never without notice: a public name is removed
only after a release in which it warned (`CvpilotDeprecationWarning`, shown by default). The
helper is `cvpilot.deprecation.deprecated(since="0.2.0", removal="0.4.0", alternative=...)`; it
refuses a removal that is not in a later minor version than the warning, registers the name, and a
test fails when a registered name is due for removal in the version being released. The
CHANGELOG (in the repository root) lists every deprecation
and removal.

## Events

| Event | Level | When |
|---|---|---|
| `run_started`, `run_resumed`, `run_finished`, `run_preempted` | info | run lifecycle (`run_preempted` carries the signal and whether a checkpoint was saved) |
| `checkpoint_saved`, `checkpoint_verified`, `checkpoint_restored`, `checkpoint_uploaded` | info | `restored` carries `trust_level` |
| `checkpoint_skipped_corrupt`, `checkpoint_verify_failed`, `checkpoint_upload_failed`, `preemption_save_incomplete` | warning | a file was refused or a transfer failed |
| `checkpoint_untrusted` | warning, error | a checkpoint lacks the required proof; `error` when restore fails closed |
| `checkpoint_save_failed` | error | a save raised to the caller |
| `preemption_signal_received`, `checkpoint_restore_empty` | info | |
| `gate_passed`, `gate_failed`, `compare_refused`, `baseline_set` | info, error | gate and baseline commands |
