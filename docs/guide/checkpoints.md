# Checkpoints and resume

`CheckpointManager` saves your training state so that a killed job can continue where it stopped.
It writes each [checkpoint](concepts.md#glossary) so that a kill at any instant leaves either the
previous one or a complete new one, and it restores only a checkpoint that verifies. A preemption
signal makes it save at once instead of crashing. Design: [ADR-0006](../adr/0006-checkpoint-semantics.md).
Security: [Trust model](trust.md).

## Save and restore

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
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state["model"].items()})
    apply_rng(state["rng"])  # restoring never sets your generators: you do
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

`model`, `optimizer`, `scheduler`, `scaler` and `ema` can be mappings or objects with a
`state_dict()`. Tensors go into a [safetensors](https://github.com/huggingface/safetensors) file and
everything else (numbers, strings, lists, dicts, random state) becomes bounded JSON inside it, so
a restore never runs code.

- `last` checkpoints hold the full state. `best` checkpoints hold the model and an optional EMA, and
  are written only when the monitored metric improves.
- `maybe_save` saves when the interval has passed (`every_n_steps` or `every_n_minutes`, default
  10 minutes), when `best` improved and when a preemption was signalled.
- `keep_last` and `keep_best` (default 3 and 2) delete older ones after a save. The newest valid
  checkpoint of each kind is never deleted, and with a remote nothing is deleted until the newest
  one is confirmed uploaded. `cvpilot ckpt prune` applies the same rules by hand.

## Resume after a restart

A restarted script gets a **new run id**. It finds its predecessor through the run store and the
checkpoint folder, so both must be on storage that survives the restart. A test runs this recipe
twice, as a killed job and its restart:

```python notest
import cvpilot
from cvpilot.ckpt import CheckpointManager, find_resume_point
from cvpilot.runs import recover_run

point = find_resume_point("fridge-sku")  # newest run of this project that can be resumed, or None
if point is not None and point.needs_recover:  # the earlier process was killed outright
    recover_run("fridge-sku", point.run_id)  # needs the lease: CVPILOT_CHECKPOINT_DIR or force=True
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

```bash notest
export CVPILOT_PROJECT=fridge-sku CVPILOT_STORE_DIR="$PWD/state/store" CVPILOT_CHECKPOINT_DIR="$PWD/state/ckpt"
python restart.py   # prints: starting at step 0
python restart.py   # prints: starting at step 3, and logs checkpoint_restored step=2 kind=last
```

The second process is a new run that resumes the first. A resumed run compares a new best metric
with the best of its whole chain of ancestors.

### A fresh node with only the bucket

If neither the store nor the folder survived, set `remote=` and ask for discovery:

```python notest
restored = ckpt.restore(discover=True)  # or ckpt.restore_latest_valid(discover=True)
```

When the run's own history yields nothing, it lists the newest 20 run prefixes in the bucket and
checks each candidate before installing it. Discovery cannot tell whose job a prefix belongs to, so
use one bucket prefix per job, and a key with `authenticated` trust so that only your checkpoints
are accepted.

## What a save guarantees

- **Atomic.** Data goes to a private temp file, is hashed and synced, then renamed into place with
  its [manifest](concepts.md#glossary). The data is read back and re-hashed, and only then does the
  `LATEST` pointer move.
- **Only adds.** A save must use a step above the [rollback floor](concepts.md#glossary) and never
  replaces an existing file. A full disk raises `CheckpointError` and leaves the previous
  checkpoint untouched.
- **Refuses a forged step.** A checkpoint the trust policy rejects, with a step above every trusted
  one, goes on the [quarantine](concepts.md#glossary) list and the save fails with
  `CheckpointStepError`. `cvpilot ckpt quarantine DIR --run ID` lists it and `--clear` releases it.
  `cvpilot ckpt trash DIR --run ID --yes` moves the files away.
- **Chained.** A new manifest links to the newest trusted one. With none and a floor above zero, the
  save fails with `CheckpointChainError`. `CheckpointManager(restart_chain=True)` starts a new
  chain on purpose and records that it did.

## What a restore does

It walks back from the newest checkpoint and skips, counts and logs
(`checkpoint_skipped_corrupt`, with a reason) any that fails a check: the manifest parses and
chains, the [trust level](concepts.md#glossary) is high enough, and the file is a regular file of
the recorded size whose SHA-256 matches. The bytes that were hashed are the bytes that are parsed.
A file without a manifest is never loaded. With nothing valid, `restore_latest_valid()` returns
`None`. If something proves a later step existed, it raises `CheckpointUntrustedError` instead of
starting over ([details](trust.md#nothing-left-is-not-a-fresh-start)).

A resumed run restores from its own folder, then from the runs it resumed (nearest first), then
from the remote. `cvpilot.resume_path("proj")` and `cvpilot.ckpt.find_resume_point` tell a
restarted script which run and file to use.

## Preemption

The manager handles `SIGTERM` and `SIGUSR1`. The handler only sets a flag, so nothing can raise into
your loop. The next `maybe_save` saves inside the grace window (`grace_seconds`, default 30), marks
the run `preempted`, then uploads with the time left. Past the window it logs
`preemption_save_incomplete` and the previous checkpoint is the resume point. `ckpt.preempted()` is
true only after this has happened, so `maybe_save` then `if ckpt.preempted(): break` never leaves
without the save.

`notice_source="aws"` or `"gcp"` also polls the instance metadata service for a preemption notice
from a background thread. It is off by default and retries with backoff. Azure has no notice source,
so use the signal there. A handler you installed for the same signal keeps running and is put back
by `close()`.

## Remote mirror

`remote="s3://bucket/prefix"` (needs `cvpilot[remote-s3]` or its siblings) uploads after the local
save verified: data first, size and SHA-256 read back, the manifest last. TLS is on, timeouts are
bounded, and credentials come from the backend's own environment, never from the URL. A failed
upload is logged (`checkpoint_upload_failed`) and retried by a later save. It never stops training.
How well it has been tested is on the [status page](../status.md).

## Frameworks that write their own files

`CheckpointManager.adopt(path, step)` brings a framework's file under verification as a `pt`
checkpoint. That format is pickle-based, so it needs `allow_pt_format=True` or
`CVPILOT_CKPT_ALLOW_PT=1` and a strict scan ([details](trust.md#the-pickle-based-pt-format)).
`cvpilot.hooks.lightning.resume_arguments(point)` returns `{"ckpt_path": ...}` for a point that
`resume_path` or `find_resume_point` returned in this process.

| File | Result |
|---|---|
| Lightning `last.ckpt` | Accepted and resumed with identical weights. |
| Ultralytics `last.pt` | Refused: it pickles the whole model object. Resume it with Ultralytics. |
| MMEngine `epoch_N.pth` | Refused: it pickles a class. Resume it with MMEngine. |
| Hugging Face `Trainer` folder | Refused: a folder, not a file. Save with `CheckpointManager` instead. |

A `pt` file the scan refuses is reported (`resume_point_skipped`), not returned as `None`. Facts
behind the table: [ADR-0010](../adr/0010-real-checkpoint-files-and-mmengine.md).

## Limits

- POSIX only. A restore holds the whole file in memory once (`max_checkpoint_bytes`, default 8 GiB);
  `pt` archives over 4 GiB are refused.
- Restore returns NumPy arrays for safetensors checkpoints. Convert with `torch.from_numpy`.
- Tensor types float16, float32, float64, the integer types and bool round-trip. **bfloat16, the
  float8 types and quantised tensors have no NumPy equivalent**, so a save raises `CheckpointError`
  naming the tensor. Cast first (`tensor.float()`) and cast back after restoring.
- Non-tensor state: integers below 2**64, at most 100 000 nodes, 32 levels and 1 MiB of JSON.
  Anything a restore would refuse is refused at save.
- A save costs about 1.5 to 3.5 times a plain `torch.save`, because it hashes the file twice
  ([measured](../adr/0010-real-checkpoint-files-and-mmengine.md)).
- Random state: `rng=True` stores the Python, NumPy (legacy global) and framework CPU generators.
  **Restoring does not touch your generators**: call `cvpilot.ckpt.apply_rng(state["rng"])`
  yourself. It checks everything first and changes nothing if the state is malformed. CUDA
  generators, per-worker loader generators and your own `numpy.random.Generator` objects are not
  covered: keep their seeds in `dataloader_state`.

## Command line

`cvpilot ckpt list DIR`, `verify DIR` and `prune DIR --run ID`: see the
[reference](../reference/cli.md). `verify` exits `1` for anything corrupt, truncated, forged or
not trusted enough.
