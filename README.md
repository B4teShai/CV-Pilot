# CV-Pilot

CV-Pilot stops you from shipping a worse vision model and from losing training progress.

## What is it?

A new model can quietly get worse on one class or one camera while the average score looks fine,
and a long training job can die and lose hours of progress. Trackers such as MLflow, Weights & Biases
and Trackio record experiments. CV-Pilot adds a per-class regression gate and crash-safe
checkpoints, and forwards its metrics to your tracker. It records each training run, scores
detection and classification per class, compares every new model with your current best and fails
your CI build if something got worse. A preempted or killed job resumes where it stopped.
Everything runs on your machine; no server is needed.

![CV-Pilot records your training, compares the new model with your best one and tells CI pass or fail](https://raw.githubusercontent.com/B4teShai/CV-Pilot/main/docs/diagrams/png/overview.png)

## Quick start

CV-Pilot itself is not on PyPI, so install it from a clone (Python 3.10 or newer; its dependencies
come from PyPI):

```bash notest
git clone https://github.com/B4teShai/CV-Pilot.git && cd CV-Pilot
uv sync && source .venv/bin/activate
# without uv: python3 -m venv .venv && . .venv/bin/activate && pip install -e .
```

To use it in your own project, pin the commit you reviewed (`git rev-parse HEAD` in a clone):
`uv add git+https://github.com/B4teShai/CV-Pilot --rev <full-commit-sha>` or
`pip install "cvpilot @ git+https://github.com/B4teShai/CV-Pilot@<full-commit-sha>"`. Extras and
local paths are in the [install guide](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/install.md).

Record a run from your training code:

```python file=train.py
import cvpilot

run = cvpilot.init(project="demo", task="detection", config={"lr": 0.01})
for step in range(3):
    cvpilot.log({"train/loss": 1.0 / (step + 1)}, step=step)
cvpilot.finish()
print(run.run_id)
```

Then list and inspect it. The commands also print log events as JSON lines on `stderr`; add
`2>/dev/null` to hide them. The table below is `stdout`, and `{*}` stands for a value that differs on
every run:

```bash
RUN=$(python train.py)    # prints the run id; log lines go to stderr
cvpilot runs list --project demo
cvpilot runs show "$RUN" --project demo
```

```text expect
RUN_ID                      STATUS     STARTED                    NAME
{*}  finished   {*}  python-{*}
project: demo
status: finished
task: detection
config: {"lr": 0.01}
n_points: 3
steps: 0..2
```

`runs show` prints more fields than this.

## What you can do

- Record runs and reproduce them: `cvpilot.init`, `cvpilot.log`, `cvpilot repro`.
- Measure detection and classification quality per class: `cvpilot eval`, `cvpilot.log_eval`. They
  read Parquet files of predictions and ground-truth boxes ([format](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/gating-ci.md)).
- Compare a new model with your current best and fail CI on a drop: `cvpilot gate`, `cvpilot compare`.
- Run that check on every pull request with the GitHub Action.
- Send metrics to MLflow, W&B or Trackio: `cvpilot.init(forward_to=["mlflow"])`.
- Hook into Lightning, Hugging Face, Ultralytics, MMEngine and RF-DETR training: `cvpilot.autolog`.
- Get a message when the loss turns NaN, stalls or a model regresses: `cvpilot.init(alerts=...)`, `cvpilot alerts check`.
- Save crash-proof checkpoints and resume after a kill: `CheckpointManager`, `cvpilot ckpt`.

## Learn more

- [Tutorial](https://github.com/B4teShai/CV-Pilot/blob/main/docs/tutorial.md): a full run from logging to a failed gate to a resume.
- [Concepts](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/concepts.md): every term in plain words.
- [Gating in CI](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/gating-ci.md): fail a pull request that makes the model worse.
- [How it works](https://github.com/B4teShai/CV-Pilot/blob/main/docs/architecture/README.md): the architecture in one picture.
- [Checkpoints](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/checkpoints.md) and [trust](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/trust.md): resuming safely.
- [Settings](https://github.com/B4teShai/CV-Pilot/blob/main/docs/reference/settings.md) and exit codes.
- [Project status](https://github.com/B4teShai/CV-Pilot/blob/main/docs/status.md): what works today.
- [Contributing](https://github.com/B4teShai/CV-Pilot/blob/main/CONTRIBUTING.md) and the [changelog](https://github.com/B4teShai/CV-Pilot/blob/main/CHANGELOG.md).

## License

Apache-2.0 ([LICENSE](https://github.com/B4teShai/CV-Pilot/blob/main/LICENSE)). The optional `cvpilot[ultralytics]` extra uses the AGPL-3.0 Ultralytics package, which then applies to your combined work.
