# Tutorial: a synthetic run from log to resume

This tutorial walks through the whole loop on generated data: record a run, evaluate it, pin
a baseline, gate a candidate, survive a preemption and a damaged checkpoint, and turn on
authenticated checkpoints. It needs no GPU, no dataset and no network. **Every command on this
page is executed by `tests/docs/test_tutorial.py`**, and the lines under "Expected" are checked
against the real output, so what you read here is what the code does.

Run the commands in an empty directory, in one shell, from the top.

## 1. Check the install

```bash
cvpilot --version
cvpilot doctor
```

```text expect
cvpilot 0.
ok
```

`doctor` prints `ok` on `stdout` and one JSON log line on `stderr`. Everything cvpilot logs is
structured JSON on `stderr`; results go to `stdout`.

## 2. Choose where things live

```bash
export CVPILOT_PROJECT=demo
export CVPILOT_STORE_DIR="$PWD/store"
export CVPILOT_CHECKPOINT_DIR="$PWD/ckpt"
```

The store holds run rows, metrics and evaluation snapshots; the checkpoint directory holds one
folder per run. Neither needs to exist yet.

## 3. Generate a synthetic evaluation set

A detection set of 60 images with two classes and three cameras, the ground truth, one set of
good predictions and one in which a third of the images are missed.

```python file=make_data.py
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

n = 60
ids = [f"img{i:03d}" for i in range(n)]
classes = [i % 2 for i in range(n)]
box = (10.0, 10.0, 50.0, 50.0)
far = (120.0, 120.0, 180.0, 180.0)


def predictions(broken: int) -> pa.Table:
    boxes = [far if i < broken else box for i in range(n)]
    return pa.table(
        {
            "image_id": ids,
            "cls": classes,
            "score": [0.9] * n,
            "x1": [b[0] for b in boxes],
            "y1": [b[1] for b in boxes],
            "x2": [b[2] for b in boxes],
            "y2": [b[3] for b in boxes],
        }
    )


pq.write_table(
    pa.table(
        {
            "image_id": ids,
            "cls": pa.array(classes, type=pa.int64()),
            "x1": [box[0]] * n,
            "y1": [box[1]] * n,
            "x2": [box[2]] * n,
            "y2": [box[3]] * n,
            "width": [200] * n,
            "height": [200] * n,
            "camera": [f"cam{i % 3}" for i in range(n)],
        }
    ),
    "gts.parquet",
)
pq.write_table(predictions(0), "preds_good.parquet")
pq.write_table(predictions(20), "preds_bad.parquet")
Path("classes.txt").write_text("cat\ndog\n")
print("wrote gts.parquet, preds_good.parquet, preds_bad.parquet, classes.txt")
```

```bash
python make_data.py
```

```text expect
wrote gts.parquet
```

## 4. A training script that checkpoints and resumes

`train.py` is a stand-in for a training loop. It looks for a checkpoint to resume from, logs a
metric per step, and hands its state to a `CheckpointManager`. `--preempt-at N` sends the
process a `SIGTERM` at step N, which is what a scheduler or a spot instance does.

```python file=train.py
import argparse
import os
import signal
from pathlib import Path

import numpy as np

import cvpilot
from cvpilot.ckpt import CheckpointManager, find_resume_point
from cvpilot.runs import recover_run

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--preempt-at", type=int)
parser.add_argument("--fresh", action="store_true", help="ignore earlier checkpoints")
args = parser.parse_args()

CKPT = Path(os.environ["CVPILOT_CHECKPOINT_DIR"])
point = None if args.fresh else find_resume_point("demo", checkpoint_dir=CKPT)
if point is not None and point.needs_recover:  # the earlier process was killed outright
    recover_run("demo", point.run_id)
run = cvpilot.init(
    "demo",
    task="detection",
    config={"steps": args.steps},
    seed=0,
    resume_from=None if point is None else point.run_id,
)
Path("run_id.txt").write_text(run.run_id)
ckpt = CheckpointManager(run, dir=CKPT, every_n_steps=10, keep_last=2)

state = ckpt.restore_latest_valid()  # None on a fresh start
weights = np.zeros(4) if state is None else state["model"]["w"]
start = 0 if state is None else state["dataloader_state"]["step"] + 1
if state is not None:
    print(f"resumed from step {start - 1} of run {point.run_id}")

for step in range(start, args.steps):
    weights = weights + 0.1
    cvpilot.log({"train/loss": 1.0 / (step + 1)}, step=step)
    if step == args.preempt_at:
        os.kill(os.getpid(), signal.SIGTERM)  # the handler only sets a flag
    saved = ckpt.maybe_save(step, model={"w": weights}, dataloader_state={"step": step})
    if ckpt.preempted():
        print(f"preempted: checkpoint saved at step {saved.step if saved else None}")
        break
else:
    run.finish()
    print(f"finished at step {args.steps - 1}")
ckpt.close()
```

## 5. Get preempted

```bash
python train.py --steps 30 --preempt-at 14
```

```text expect
preempted: checkpoint saved at step 14
```

The handler only set a flag. The next `maybe_save` wrote a checkpoint outside its normal
interval, marked the run `preempted` and returned; the loop saw `ckpt.preempted()` and stopped.

```bash
cvpilot runs list
cvpilot ckpt list "$CVPILOT_CHECKPOINT_DIR"
```

```text expect
preempted
000000014
000000010
```

Two checkpoints exist: step 10 from the interval and step 14 from the preemption.

## 6. Damage the newest checkpoint

Truncate the newest file, as a kill during a write or a half-finished upload would.

```bash
RUN1=$(cat run_id.txt)
NEWEST="$CVPILOT_CHECKPOINT_DIR/$RUN1/last/step_000000014.safetensors"
python -c "import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.chmod(0o644); p.write_bytes(p.read_bytes()[:100])" "$NEWEST"
cvpilot ckpt verify "$CVPILOT_CHECKPOINT_DIR" --project demo || echo "verify exit code: $?"
```

```text expect
size_mismatch
verify exit code: 1
```

`verify` hashes every file and exits 1 for anything that is corrupt, truncated, forged or not
trusted enough. Nothing was loaded.

```bash
python -c "import cvpilot; print(cvpilot.resume_path('demo'))"
```

```text expect
step_000000010.safetensors
```

`resume_path` skipped the damaged file and returned the previous valid one. It never returns the
string `"last"` and never a file it has not verified.

## 7. Resume

```bash
python train.py --steps 30
```

```text expect
resumed from step 10 of run
finished at step 29
```

The new run is a child of the preempted one (`parent_run_id`) and continues its step timeline.

```bash
RUN2=$(cat run_id.txt)
cvpilot runs show "$RUN2" | grep -E "^(status|parent_run_id)"
```

```text expect
status: finished
parent_run_id: 01
```

## 8. Evaluate and pin a baseline

```bash
cvpilot eval "$RUN2" --preds preds_good.parquet --gts gts.parquet --classes classes.txt \
  --slices camera=camera --step 29 --allow-closed --json > /dev/null
cvpilot baseline set prod "$RUN2"
```

```text expect
baseline_set
```

These runs have finished, so evaluating into them needs the explicit `--allow-closed`; inside a
training script you would call `run.log_eval(...)` while the run is open. The snapshot is
immutable and the pin is append-only: replacing a tag needs `--replace` and
keeps the old pin in the history.

## 9. Gate a regression, then a good run

A candidate whose predictions miss a third of the images:

```bash
python train.py --steps 30 --fresh
RUN3=$(cat run_id.txt)
cvpilot eval "$RUN3" --preds preds_bad.parquet --gts gts.parquet --classes classes.txt \
  --slices camera=camera --step 29 --allow-closed --json > /dev/null
cvpilot gate "$RUN3" --baseline tag:prod --max-drop mAP50_95=1 --min-images 30 \
  || echo "gate exit code: $?"
```

```text expect
gate: FAIL
gate exit code: 1
```

A candidate that matches the baseline:

```bash
python train.py --steps 30 --fresh
RUN4=$(cat run_id.txt)
cvpilot eval "$RUN4" --preds preds_good.parquet --gts gts.parquet --classes classes.txt \
  --slices camera=camera --step 29 --allow-closed --json > /dev/null
cvpilot gate "$RUN4" --baseline tag:prod --max-drop mAP50_95=1 --min-images 30
```

```text expect
gate: PASS
```

The gate fails closed: a missing metric, a different evaluation set, a snapshot or baseline that
fails its integrity check, or too few images all exit 1. Read [Gating in CI](guide/gating-ci.md)
before using it on pull requests.

## 10. Authenticate the checkpoints

Digests and chains inside a checkpoint directory catch damage. They do not stop someone who can
write the directory from forging a consistent set of files. A key that only you hold does:

```bash
export CVPILOT_CKPT_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export CVPILOT_CKPT_REQUIRE_TRUST=authenticated
python train.py --steps 30 --fresh
RUN5=$(cat run_id.txt)
cvpilot ckpt verify "$CVPILOT_CHECKPOINT_DIR" --run "$RUN5"
cvpilot ckpt verify "$CVPILOT_CHECKPOINT_DIR" --run "$RUN4" || echo "verify exit code: $?"
```

```text expect
authenticated
auth_missing
verify exit code: 1
```

The checkpoints just written carry an HMAC tag and verify as `authenticated`. The earlier run
has none, so under `authenticated` it is refused (`auth_missing`) instead of being trusted. Use
this for any directory that is shared, restored from a bucket or writable by others. See
[Trust model](guide/trust.md).

## What you used

| Step | Feature | Reference |
|---|---|---|
| 5, 7 | preemption, resume, run lineage | [Checkpoints and resume](guide/checkpoints.md) |
| 6 | verified restore, `ckpt verify` | [Command line](reference/cli.md) |
| 8, 9 | snapshots, baselines, the gate | [Gating in CI](guide/gating-ci.md) |
| 10 | authenticated checkpoints | [Trust model](guide/trust.md) |
