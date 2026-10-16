# Checkpoints and resume

A checkpoint is only useful if you can prove it is complete when the job comes back. The
checkpoint manager writes atomically, hashes and records what it wrote, restores only what
verifies, keeps the newest valid file of each kind whatever else is deleted, and treats a
preemption signal as a request to save, not a reason to crash. The design is in
[ADR-0006](../adr/0006-checkpoint-semantics.md); the failure modes are in
[Flow B](../architecture/04-flows.md).

## Use it

```python notest
import cvpilot
from cvpilot.ckpt import CheckpointManager, apply_rng

run = cvpilot.init("fridge-sku", task="detection")
ckpt = CheckpointManager(
    run,
    dir="/checkpoint",  # or CVPILOT_CHECKPOINT_DIR
    remote="s3://ml-ckpt/fridge/",  # optional, or CVPILOT_CHECKPOINT_REMOTE
    keep_last=3,
    keep_best=2,
    monitor="val/mAP50_95",
    mode="max",
    every_n_minutes=20,
)
state = ckpt.restore_latest_valid()  # None on a fresh start
start = 0
if state is not None:
    model.load_state_dict(state["model"])
    apply_rng(state["rng"])  # restoring never sets your generators: you do (see Limits)
    start = state["dataloader_state"]["step"] + 1
for step in range(start, total_steps):
    ...
    ckpt.maybe_save(
        step,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        rng=True,
        dataloader_state={"step": step, **sampler.state_dict()},
        metrics={"val/mAP50_95": m},
    )
    if ckpt.preempted():
        break  # the run is already marked preempted
```

`model`, `optimizer`, `scheduler`, `scaler` and `ema` may be mappings or objects with a
`state_dict()`. Tensors (NumPy arrays and framework tensors) go into a
[safetensors](https://github.com/huggingface/safetensors) file; everything else (numbers, strings,
lists, dicts, RNG state) is stored as bounded JSON inside it, so restoring never executes code.
`last` checkpoints hold the full state; `best` checkpoints hold the model and an optional EMA,
written only when the monitored metric improves. `maybe_save` saves when the interval elapsed
(`every_n_steps` or `every_n_minutes`, default every 10 minutes), when `best` improved and when a
preemption is signalled.

## What a save does

Data to a private temp file (`O_EXCL`, mode 0600, `fsync`), then its manifest, then both renamed
into place, the directory `fsync`ed, the data re-read and re-hashed, the manifest digest recorded in
the run store, and only then the `LATEST` pointer rewritten. A `SIGKILL` at any instant leaves
either the previous state or a complete new checkpoint; `tests/chaos/` kills real processes at
random moments and at four chosen points inside a save. A concurrent saver waits on a directory
lock. A full disk raises `CheckpointError` to your loop and leaves the previous checkpoint
untouched.

## What a save refuses

A save only ever adds. Its step must exceed the **floor**: the highest step of any self-consistent
manifest in the directory (trusted or not, and every step on the quarantine list), the run store's highest
step for the run and kind, a tag-valid pointer, `min_step` and the newest step the remote lists. A
rejected or forged entry therefore still counts, so deleting the newest checkpoint or planting a
lower-step forgery cannot make the next save look like a healthy continuation.

* **Quarantine is a list.** A manifest the policy does not trust whose step is above every trusted
  one (a planted `step_999999`) is recorded in the run's `QUARANTINE` list and logged as
  `checkpoint_quarantined`; no file is moved or deleted, and the save fails with
  `CheckpointStepError` saying so. The step keeps counting toward the floor until you decide.
  `cvpilot ckpt quarantine DIR --run ID` lists the entries and `--clear` only releases them
  (`checkpoint_quarantine_cleared`); saving resumes. Removing the files is a separate, confirmed
  command: `cvpilot ckpt trash DIR --run ID --yes` moves the listed files to a trash directory and
  refuses without `--yes`. Key and evidence checks run before anything is recorded, so a wrong or
  missing key never writes to the list.
* **Chain.** A new manifest chains to the newest *trusted* checkpoint. With none, but a floor above
  zero (or a rejected higher entry), the save fails closed with `CheckpointChainError`
  (`chain_unavailable`). `CheckpointManager(restart_chain=True)` (never an environment setting)
  starts a new chain on purpose: the manifest records `chain_reset: true` and the reason inside the
  tagged record, `checkpoint_chain_reset` is logged and `ckpt verify` shows it in every later run.
  Genesis is only for the very first save of a run with no floor.
* **Never replaced.** Data and manifest files are created exclusively and given their final name
  with a hard link, which cannot overwrite (a rename would): a save at an existing step is refused
  (`CheckpointExistsError`), or is a no-op logged as `checkpoint_already_saved` when the existing
  checkpoint verifies and the bytes are identical. `adopt` and the remote install go through the
  same rule, a marker is never rewritten in place, and the pointer only moves forward (a stale
  process cannot move it back: `checkpoint_pointer_regression_refused`). Retention never deletes the
  checkpoint the pointer names. Predecessors the retention removed leave a gap that `prev_step`
  explains, so they need not be kept.

## What a restore does

It walks back from the newest manifest. A checkpoint is skipped, counted and logged
(`checkpoint_skipped_corrupt`, with a reason) unless its manifest parses, matches its name and run,
chains, is vouched for at the required trust level, and its file is a regular file (never a
symlink) of exactly the recorded size whose structure parses and whose SHA-256 matches. The bytes
that were hashed are the bytes that are parsed: the file is read once. A file without a manifest
is never loaded. When nothing verifies, `restore_latest_valid()` returns `None`.

A resumed run restores from its own directory first, then from the runs it resumed, nearest first,
and from the remote when nothing local verifies. `cvpilot.resume_path("proj")` and
`cvpilot.ckpt.find_resume_point` tell a restarted script which run and file to resume from.

### A restart, end to end

A restarted script gets a **new run id**, so it finds its predecessor through the run store and the
checkpoint directory: both must live on storage that survives the restart (a persistent mount).
This is the recipe the README runs twice in a test:

```python notest
import cvpilot
from cvpilot.ckpt import CheckpointManager, find_resume_point
from cvpilot.runs import recover_run

point = find_resume_point("fridge-sku")  # newest run of this project that can be resumed, or None
if point is not None and point.needs_recover:  # the earlier process was killed outright
    recover_run("fridge-sku", point.run_id)
run = cvpilot.init(
    "fridge-sku", task="detection", resume_from=None if point is None else point.run_id
)
ckpt = CheckpointManager(run, every_n_steps=1)  # directory: CVPILOT_CHECKPOINT_DIR
restored = ckpt.restore()  # None on a fresh start; this run's directory, then the runs it resumed
start = 0 if restored is None else restored.manifest.step + 1
print(f"starting at step {start}")
for step in range(start, start + 3):
    ckpt.maybe_save(step, model={"w": [float(step)]})
    if ckpt.preempted():
        break  # SIGTERM or SIGUSR1 set a flag; the run is already marked preempted
```

### A fresh node with only the bucket

When neither the run store nor the checkpoint directory survived, set `remote=` (or
`CVPILOT_CHECKPOINT_REMOTE`) and ask for discovery:

```python notest
restored = ckpt.restore(discover=True)  # or ckpt.restore_latest_valid(discover=True)
```

Only if the run's own lineage yields nothing does it list the newest 20 run prefixes on the remote
(at most 1000 are read), orders the candidates by step (the newer run wins a tie) and fetches each
through the same gate as any remote checkpoint: policy, size caps, rollback floor, then the
SHA-256, before anything is installed. `restore_latest_valid(run_ids=[...])` names the runs to
search explicitly. Discovery cannot tell whose job a run prefix belongs to: use a bucket prefix per
job, and `CVPILOT_CKPT_KEY` with `authenticated` so that only checkpoints written with your key are
accepted.

A resumed run compares a new monitored value with the best of its whole lineage (this run and the
runs it resumed, for the same metric and mode), so a `best` checkpoint is only written when it beats
what the ancestors already hold.

## Retention

`keep_last` and `keep_best` (defaults 3 and 2) apply after every save and count only checkpoints
that verify. The newest valid checkpoint of each kind is never deleted, and with a remote nothing
is deleted until the newest checkpoint is confirmed uploaded and verified there.
`cvpilot ckpt prune` applies the same rules by hand.

## Preemption

`CheckpointManager` installs handlers for `SIGTERM` and `SIGUSR1` (configurable). A handler only
stores two attributes and sets a flag: no lock, no logging, no I/O, nothing that can raise into
your loop. The next `maybe_save` saves inside the grace window (`grace_seconds`, default 30),
logs `preemption_signal_received` and `run_preempted`, and marks the run `preempted`. Past the
window it logs `preemption_save_incomplete` and the previous checkpoint is the resume point. Any
failure on this path (not only checkpoint errors: a `state_dict` that raises `RuntimeError`, a store
that cannot be written) is logged and never reaches your loop. The run is marked `preempted` as soon
as the local checkpoint is verified; the upload then gets only what is left of the grace window
(`preemption_upload_deferred` when it is cut short or the window is spent) and retention runs last.

Platform notices are optional and off by default: `notice_source="aws"` or `"gcp"` polls the
instance metadata service (the fixed link-local address, so no DNS answer can redirect it; short
timeout) from a daemon thread. AWS counts only a JSON answer whose `action` is `terminate`, `stop`
or `hibernate`; any other 200 is an error, not a notice. The poller never gives up: a failing
service (timeouts, 401 or 403 from a disabled metadata service) is retried with exponential
backoff up to five minutes and logged as `notice_poller_degraded` and `notice_poller_recovered`, and
`preemption_poller_stopped` says why the thread ended (`notice`, `stopped` or `crashed`). There is
no Azure notice source: use the signal path there.

Handlers are installed after everything that can fail has succeeded (an unknown notice source or an
invalid interval leaves your handlers untouched). A handler you had installed for the same signal
keeps running after the flag is set, is reported once as `signal_handler_displaced` and is put back
by `close()`; a default or ignored disposition is never chained.

## Remote mirror

`remote="s3://bucket/prefix"` (extra `cvpilot[remote]` plus the backend package) uploads after the
local save verified: data to a `.part` object, resumed from the remote prefix when it equals the
local prefix, size and SHA-256 read back before the rename, the manifest last, retries with
backoff inside one deadline. TLS is on, timeouts are bounded, credentials come from the backend's
own environment and never from the location (a location with user info, a query or a signed URL
is refused), and nothing about the path or the backend's messages reaches the logs. An upload that
fails is logged (`checkpoint_upload_failed`) and caught up by a later save; it never stops
training.

## Frameworks that write their own files

`CheckpointManager.adopt(path, step)` brings a file a framework wrote (Lightning's `last.ckpt`, a
YOLO `last.pt`) under verification, as a `pt` checkpoint. That format needs an explicit opt-in
(`allow_pt_format=True` or `CVPILOT_CKPT_ALLOW_PT=1`) and a recent framework: see
[Trust model](trust.md). `cvpilot.hooks.lightning.resume_arguments(point)` returns
`{"ckpt_path": ...}` and the YOLO hook's `resume_arguments(point)` returns the weights path and
`{"resume": True}`. They accept only a `pt` checkpoint that `cvpilot.resume_path` (or
`find_resume_point`) returned in this process, as the path or the `ResumePoint`: never `"last"`, a
relative name, an arbitrary path or a safetensors file (the default format is CheckpointManager's
own and no framework can resume from it; restore it with `CheckpointManager.restore` into your
model). Limits, stated plainly: the strict archive scan accepts plain state dictionaries, so a
Lightning `last.ckpt` is adopted only if its archive scans clean, and an Ultralytics `last.pt`
(which pickles the model object) is refused with `unsafe_archive`; when a `pt` file is skipped,
`resume_point_skipped` and `resume_point_unavailable` say why instead of returning a bare `None`.
The real-library behaviour is checked by `pytest -m real_libs`, which needs those libraries
installed and is not part of the default run.

## Command line

`cvpilot ckpt list DIR`, `verify DIR` and `prune DIR --run ID`; see the
[reference](../reference/cli.md). `verify` exits `1` when anything is corrupt, truncated, forged or
untrusted.

## Limits

`pt` archives over 4 GiB are refused (zip64); restoring holds the whole file in memory once
(`max_checkpoint_bytes`, default 8 GiB); the manager supports POSIX platforms only (it needs
directory descriptors); `restore` returns NumPy arrays for safetensors checkpoints (convert with
`torch.from_numpy` or your framework's equivalent).

Tensor dtypes: float16, float32, float64, the signed and unsigned integer types and bool
round-trip. **bfloat16, the float8 types and quantised tensors have no NumPy equivalent**, so a save
that meets one raises `CheckpointError` naming the tensor and its dtype; cast it first (for example
`tensor.float()`) and cast back after restoring. Integers in the non-tensor state must be below
2**64 in magnitude, the state at most 100 000 nodes and 32 levels deep, and its JSON at most 1 MiB;
anything a restore would refuse is refused at save.

Random state: `rng=True` stores the Python, NumPy (legacy global generator) and, if the framework
is already imported, its CPU generator. **Restoring does not touch your generators**: call
`cvpilot.ckpt.apply_rng(state["rng"])` yourself after restoring (it validates everything first and
changes nothing if the state is malformed). Not covered: CUDA generators, per-worker data loader
generators and `numpy.random.Generator` objects you created yourself; keep their seeds or states
in `dataloader_state`.
