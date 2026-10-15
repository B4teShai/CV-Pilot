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
from cvpilot.ckpt import CheckpointManager

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
for step in range(start, total_steps):
    ...
    ckpt.maybe_save(
        step,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        rng=True,
        dataloader_state=sampler.state_dict(),
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
window it logs `preemption_save_incomplete` and the previous checkpoint is the resume point. A
failing save on this path never raises.

Platform notices are optional and off by default: `notice_source="aws"` or `"gcp"` polls the
instance metadata service (fixed host, short timeout, bounded errors) from a daemon thread.

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
[Trust model](trust.md). `cvpilot.hooks.lightning.resume_arguments(path)` returns
`{"ckpt_path": ...}` and the YOLO hook's `resume_arguments(path)` returns the weights path and
`{"resume": True}`; both refuse relative paths and the string `"last"`.

## Command line

`cvpilot ckpt list DIR`, `verify DIR` and `prune DIR --run ID`; see the
[reference](../reference/cli.md). `verify` exits `1` when anything is corrupt, truncated, forged or
untrusted.

## Limits

`pt` archives over 4 GiB are refused (zip64); restoring holds the whole file in memory once
(`max_checkpoint_bytes`, default 16 GiB); the manager supports POSIX platforms only (it needs
directory descriptors); `restore` returns NumPy arrays for safetensors checkpoints (convert with
`torch.from_numpy` or your framework's equivalent).
