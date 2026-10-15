# CV-Pilot

Regression gates, remote monitoring and resume-safe checkpoints for vision models, on top of
the experiment tracker you already use.

## Status

Built and tested: run capture, the local run store and the eval engine.

- Python API: `cvpilot.init`, `log`, `log_artifact`, `finish` (and `Run` as a context manager)
  record a run (environment, git state, dataset hash, metrics) into `~/.cvpilot/<project>/`:
  SQLite in WAL mode for run rows, batched Parquet files for metric points, a
  content-addressed directory for artifacts. A process killed with SIGKILL leaves a readable
  store; the run stays `running` until it is marked `failed`.
- Eval: `Run.log_eval(preds, gts, step=...)` and `cvpilot eval RUN_ID --preds p.parquet --gts g.parquet --classes names.txt` store an immutable snapshot (COCO-style mAP, per-class AP, confusion, slices, bootstrap intervals); see below.
- CLI: `cvpilot --version`, `doctor`, `runs list`, `runs show RUN_ID`, `repro RUN_ID`, `eval RUN_ID`, with the
  exit-code contract of [docs/engineering/errors-and-logging.md](docs/engineering/errors-and-logging.md).
- Settings: `CVPILOT_ENV`, `CVPILOT_LOG_LEVEL`, `CVPILOT_PROJECT`, `CVPILOT_STORE_DIR`,
  `CVPILOT_STORE_COMMAND_LINE`; structured JSON logging on `stderr`.

Everything under "Planned usage" below is **not implemented**: no hooks,
`compare`, `gate`, checkpoint manager, forwarders, server or alerts. The architecture
documents mark each part as built or planned and describe only the built parts as behaviour
that exists. See [CHANGELOG.md](CHANGELOG.md).

## Install (from source)

Requires Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repository-url> cv-pilot
cd cv-pilot
uv sync
uv run cvpilot --version
uv run cvpilot doctor
```

`doctor` prints `ok` on `stdout` and one JSON log line (`doctor_completed`) on `stderr`.
Exit codes: `0` success, `1` failure, `2` usage or configuration error, `130` interrupted.

## Record and inspect a run

```python
# train.py
import cvpilot

with cvpilot.init(task="detection", config={"lr": 0.01}, seed=0) as run:
    for step in range(3):
        cvpilot.log({"train/loss": 1.0 / (step + 1)}, step=step)
```

```bash
export CVPILOT_PROJECT=demo CVPILOT_STORE_DIR=$PWD/store   # default store: ~/.cvpilot
uv run python train.py              # logs run_started and run_finished on stderr
uv run cvpilot runs list            # RUN_ID  STATUS  STARTED  NAME
uv run cvpilot runs show <RUN_ID>   # add --json for the full record
uv run cvpilot repro <RUN_ID>       # prints a bash script: checkout, stored diff, command
```

Notes:

- `init` works outside a git repository and without a GPU: absent facts are recorded as none.
- A step must be strictly greater than the last step logged for the same metric key.
- `log` buffers up to 10 000 points and writes one Parquet file when the buffer is full or
  2 s have passed; `finish` flushes and compacts them.
- Secrets in the config and on the command line are redacted before storage (best effort for
  free text). Set `CVPILOT_STORE_COMMAND_LINE=false` to store only the executable name and
  the number of arguments; `repro` then cannot re-run the command.
- Exit codes of `runs` and `repro`: `1` for a well-formed run id that is not in the store,
  `2` for a missing project, an invalid project or a malformed run id.
- `benchmarks/log_readback.py` logs 100 000 points and reads them back;
  `benchmarks/install_size.sh` checks that the core install carries no torch.

## Evaluate predictions

Parquet files: predictions `image_id, cls, score, x1, y1, x2, y2`; ground truth `image_id, cls,
x1, y1, x2, y2` plus optional `iscrowd`, `width`, `height` and metadata columns (a row with a
null `cls` declares an image without objects); one class name per line (or a JSON list).

```bash
uv run cvpilot eval <RUN_ID> --preds p.parquet --gts g.parquet --classes names.txt \
  --slices camera=camera --step 10 --seed 0 --resamples 1000      # add --json for the snapshot
uv run cvpilot runs show <RUN_ID>                                  # lists the snapshot
uv run python benchmarks/eval_throughput.py                        # timing and peak memory
```

Exit codes of `eval`: `1` invalid inputs, a closed run (add `--allow-closed` to evaluate into a
finished run), an unknown run, a snapshot whose digest does not match, or an existing snapshot
(snapshots are immutable per set and step); `2` usage errors (bad `--slices`, a step or seed out
of range). `--max-memory-mb` caps the address space on Linux. Optional `--checkpoint-sha256` and
`--git-commit` are stored with the snapshot. Values are fractions in [0, 1].

## GitHub Action

`action.yml` is a composite action around `cvpilot gate`. It expects `cvpilot` to be installed in
the job's Python and the project store to be on the runner (a checked-out or mounted copy).
Inputs reach the script only through `env:` and are used only inside double quotes, so a branch
name or a pull request title can not become shell code; the action uses no third-party action.
Give the job read-only permissions:

```yaml
on: pull_request
permissions:
  contents: read
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - uses: ./
        with:
          run-id: 01ARZ3NDEKTSV4RRFFQ69G5FAV
          baseline: tag:prod
          project: demo
          store-dir: ci-store
          max-drop: |
            mAP50_95=0.5
          slice-max-drop: |
            camera=*:1.5
          min-images: "30"
```

`tests/unit/test_action.py` parses `action.yml`, every workflow and the example above and
rejects any `${{ ... }}` expression inside a `run:` script, an unpinned third-party action,
write permissions, unquoted variables, and runs the real script with hostile input values.

## Planned usage

Nothing in this section runs yet. The examples fix the public interface that the planned
packages will implement; see [docs/architecture/02-components.md](docs/architecture/02-components.md)
for signatures and status.

### Planned: automatic capture from a training framework

Hooks are extras: `cvpilot[ultralytics]`, `cvpilot[lightning]`, `cvpilot[hf]`,
`cvpilot[mmdet]`; forwarders `cvpilot[mlflow]`, `cvpilot[wandb]`, `cvpilot[trackio]`;
`cvpilot[alerts]` and `cvpilot[server]`. The `ultralytics` package is AGPL-3.0: it is
imported only inside the Ultralytics hook, and using `cvpilot[ultralytics]` makes your
combined work subject to AGPL-3.0. Every other extra and the core stay Apache-2.0.

```python
from ultralytics import YOLO
import cvpilot

model = YOLO("yolo11s.pt")
cvpilot.autolog(model, project="fridge-sku", forward_to=["mlflow"])
model.train(data="fridge.yaml", epochs=100, imgsz=1024)
```

```python
import lightning as L
import cvpilot
from cvpilot.hooks.lightning import CVPilotCallback, CVPilotLogger

trainer = L.Trainer(
    logger=CVPilotLogger(project="fridge-sku"),
    callbacks=[
        CVPilotCallback(
            eval_task="detection",
            class_names=CLASSES,
            slices={"camera": "meta.camera_id"},
        )
    ],
)
trainer.fit(model, dm, ckpt_path=cvpilot.resume_path("fridge-sku"))  # None on a fresh start
```

### Evaluation snapshots in a plain training loop

`init`, `log`, `log_eval` and `finish` are built; passing `PredictionTable` and `AnnotationTable`
(from `cvpilot.eval`) instead of raw arrays is what the planned hooks will do for you.

```python
import cvpilot

run = cvpilot.init(project="fridge-sku", task="detection", config=cfg)
for step, batch in enumerate(loader):
    loss = train_step(batch)
    cvpilot.log({"train/loss": loss}, step=step)
snapshot = cvpilot.log_eval(preds, gts, step=step, slices={"camera": "camera_id"})
cvpilot.finish()
```

### Planned: resume-safe checkpoints

```python
from cvpilot.ckpt import CheckpointManager

ckpt = CheckpointManager(
    run,
    dir="/checkpoint",
    remote="s3://ml-ckpt/fridge-sku/",
    keep_last=3,
    keep_best=2,
    monitor="val/mAP50_95",
    mode="max",
    every_n_minutes=20,
)
state = ckpt.restore_latest_valid()  # None on a fresh start
for step in range(state["step"] if state else 0, total_steps):
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
```

### Planned: compare and gate in CI

```bash
cvpilot baseline set prod 01J9... --eval-set fridge_val_v7
cvpilot compare 01JA... --baseline tag:prod --report cmp.html
cvpilot gate 01JA... --baseline tag:prod \
  --max-drop mAP50_95=0.5 --per-class-max-drop AP50_95=2.0 \
  --slice-max-drop "camera=*:1.5" --min-images 30
cvpilot repro 01JA...          # prints a reproduction script
cvpilot ckpt verify /checkpoint/01JA...
```

`gate` exits `0` when every threshold holds, `1` when any threshold is exceeded or the
comparison is refused (different `eval_set_hash`), `2` on a usage error.

## Architecture

![System context](docs/diagrams/png/context.png)

![Containers](docs/diagrams/png/containers.png)

- [docs/architecture/README.md](docs/architecture/README.md): the full document set
  (overview, components, data model, flows, runtime, cross-cutting concerns).
- [docs/engineering/README.md](docs/engineering/README.md): coding standards, testing,
  errors and logging, dependencies and licensing, workflow.
- [docs/adr/](docs/adr/): architecture decision records.

## License

Apache-2.0, including the server. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
