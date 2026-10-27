# Tutorial: from a first run to a resume

You will record a run, get preempted, resume, pin a baseline, fail a bad model at the gate and
sign your checkpoints. It needs no GPU, no dataset and no network. A test runs every command on
this page, and the lines marked "expect" are checked against the real output. Words you do not know
are in the [glossary](guide/concepts.md#glossary).

Run the commands in an empty directory, in one shell, from the top.

## 1. Check the install

```bash
cvpilot --version
cvpilot doctor
```

```text expect
cvpilot {*}
ok
```

```text expect stderr
json: message=doctor_completed status=ok
```

Results go to `stdout`. Everything cvpilot logs is one JSON line per event on `stderr`.

## 2. Choose where things live

```bash
export CVPILOT_PROJECT=demo
export CVPILOT_STORE_DIR="$PWD/store"
export CVPILOT_CHECKPOINT_DIR="$PWD/ckpt"
```

The store holds runs, metrics and evaluation snapshots. The checkpoint directory holds one folder
per run. Neither needs to exist yet.

## 3. Generate a synthetic evaluation set

A detection set of 60 images, two classes and three cameras: the ground truth, good predictions,
and predictions that miss a third of the images.

```python file=make_data.py
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

n = 60
ids = [f"img{i:03d}" for i in range(n)]
cls = pa.array([i % 2 for i in range(n)], pa.int64())
near, far = (10.0, 10.0, 50.0, 50.0), (120.0, 120.0, 180.0, 180.0)


def table(broken: int, **extra: list) -> pa.Table:
    boxes = [far if i < broken else near for i in range(n)]
    cols = {"image_id": ids, "cls": cls, **extra}
    for j, name in enumerate(["x1", "y1", "x2", "y2"]):
        cols[name] = [b[j] for b in boxes]
    return pa.table(cols)


cameras = [f"cam{i % 3}" for i in range(n)]
pq.write_table(table(0, width=[200] * n, height=[200] * n, camera=cameras), "gts.parquet")
pq.write_table(table(0, score=[0.9] * n), "preds_good.parquet")
pq.write_table(table(20, score=[0.9] * n), "preds_bad.parquet")
Path("classes.txt").write_text("cat\ndog\n")
print("wrote gts.parquet, preds_good.parquet, preds_bad.parquet, classes.txt")
```

```bash
python make_data.py
```

```text expect
wrote gts.parquet, preds_good.parquet, preds_bad.parquet, classes.txt
```

## 4. A training script that checkpoints and resumes

`train.py` stands in for a training loop. It looks for a checkpoint to resume from, logs a metric
per step and hands its state to a `CheckpointManager`. `--preempt-at N` sends the process a
`SIGTERM` at step N, as a scheduler or spot instance does.

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

```text expect stderr
json: message=preemption_signal_received signal=SIGTERM
json: message=checkpoint_saved step=14 kind=last
json: message=run_preempted signal=SIGTERM step=14 saved=True
```

The signal handler only set a flag. The next `maybe_save` wrote an extra checkpoint, marked the
run `preempted` and returned, and the loop stopped.

```bash
cvpilot runs list
cvpilot ckpt list "$CVPILOT_CHECKPOINT_DIR"
```

```text expect
RUN_ID                      STATUS     STARTED                    NAME
{*}  preempted  {*}
{*}  last  000000014  valid    anchored
{*}  last  000000010  valid    anchored
```

Step 10 came from the interval and step 14 from the preemption. Both are `anchored`.

## 6. Damage the newest checkpoint

Truncate the newest file, as a kill during a write or a half-finished upload would.

```bash
RUN1=$(cat run_id.txt)
NEWEST="$CVPILOT_CHECKPOINT_DIR/$RUN1/last/step_000000014.safetensors"
python -c "import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.chmod(0o644); p.write_bytes(p.read_bytes()[:100])" "$NEWEST"
cvpilot ckpt verify "$CVPILOT_CHECKPOINT_DIR" --project demo || echo "verify exit code: $?"
```

```text expect
{*}  last  000000014  corrupt  size_mismatch
{*}  last  000000010  valid    anchored
verify exit code: 1
```

`verify` hashes every file and exits 1 for anything corrupt, truncated, forged or not trusted
enough. It loads nothing.

```bash
python -c "import cvpilot; print(cvpilot.resume_path('demo'))"
```

```text expect
{*}/ckpt/{*}/last/step_000000010.safetensors
```

```text expect stderr
json: message=checkpoint_skipped_corrupt step=14 reason=size_mismatch
```

`resume_path` skipped the damaged file and returned the previous valid one. It only returns a file
it has verified.

## 7. Resume

```bash
python train.py --steps 30
```

```text expect
resumed from step 10 of run {*}
finished at step 29
```

The new run is a child of the preempted one (`parent_run_id`) and continues its steps.

```bash
RUN2=$(cat run_id.txt)
cvpilot runs show "$RUN2" | grep -E "^(status|parent_run_id)"
```

```text expect
status: finished
parent_run_id: {*}
```

## 8. Evaluate and pin a baseline

```bash
cvpilot eval "$RUN2" --preds preds_good.parquet --gts gts.parquet --classes classes.txt \
  --slices camera=camera --step 29 --allow-closed --json > /dev/null
cvpilot baseline set prod "$RUN2"
```

```text expect
pinned prod: run {*} eval_set {*} step 29
```

```text expect stderr
json: message=eval_completed step=29 task=detection n_images=60
json: message=baseline_set tag=prod replaced=False
```

These runs are finished, so `eval` needs `--allow-closed`; in a training script you call
`run.log_eval(...)` while the run is open. A snapshot is never edited, and replacing a pin needs
`--replace` and keeps the old one.

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
finished at step 29
gate: FAIL (1 violations, worst: [headline_drop] mAP50_95) run {*} step 29 vs baseline tag:prod
  - [headline_drop] mAP50_95: mAP50_95 dropped {*} points (limit 1)
gate exit code: 1
```

```text expect stderr
json: message=gate_failed n_violations=1 worst_metric=mAP50_95 worst_kind=headline_drop
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
finished at step 29
gate: PASS (1 checks, 0 slices skipped) run {*} step 29 vs baseline tag:prod
```

```text expect stderr
json: message=gate_passed n_violations=0
```

The gate is fail-closed: anything it cannot check exits 1. Read [Gating in CI](guide/gating-ci.md)
before using it on pull requests.

## 10. Authenticate the checkpoints

Hashes inside a checkpoint folder catch damage, but not someone who can write the folder and forge
a consistent set of files. A secret key that only you hold does:

```bash
export CVPILOT_CKPT_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export CVPILOT_CKPT_REQUIRE_TRUST=authenticated
python train.py --steps 30 --fresh
RUN5=$(cat run_id.txt)
cvpilot ckpt verify "$CVPILOT_CHECKPOINT_DIR" --run "$RUN5"
cvpilot ckpt verify "$CVPILOT_CHECKPOINT_DIR" --run "$RUN4" || echo "verify exit code: $?"
```

```text expect
finished at step 29
{*}  last  000000020  valid    authenticated
{*}  last  000000010  valid    authenticated
{*}  last  000000020  corrupt  auth_missing
{*}  last  000000010  corrupt  auth_missing
verify exit code: 1
```

The new checkpoints carry an HMAC tag and verify as `authenticated`. The earlier run has none, so
it is refused (`auth_missing`). Use this for any folder that is shared, restored from a bucket or
writable by others. See [Trust model](guide/trust.md).

## What you used

| Step | Feature | Reference |
|---|---|---|
| 5, 7 | preemption and resume | [Checkpoints and resume](guide/checkpoints.md) |
| 6 | verified restore, `ckpt verify` | [Command line](reference/cli.md) |
| 8, 9 | snapshots, baselines, the gate | [Gating in CI](guide/gating-ci.md) |
| 10 | authenticated checkpoints | [Trust model](guide/trust.md) |
