# CV-Pilot

CV-Pilot stops you from shipping a worse vision model and from losing training progress.

## What is it?

A new model can quietly get worse on one class or one camera while the average score looks fine,
and a long training job can die and lose hours of progress. CV-Pilot works next to the experiment
tracker you already use (MLflow, Weights & Biases, Trackio). It records each training run, measures
detection and classification quality per class, compares every new model with your current best and
fails your CI build if something got worse. It also saves crash-proof checkpoints, so a preempted
or killed job resumes where it stopped. Everything runs on your machine from Python or the command
line; no server is needed.

![CV-Pilot records your training, compares the new model with your best one and tells CI pass or fail](https://raw.githubusercontent.com/B4teShai/CV-Pilot/main/docs/diagrams/png/overview.png)

## Quick start

Install it (Python 3.10 or newer):

```bash notest
pip install cvpilot
```

Record a run from your training code:

```python file=train.py
import cvpilot

run = cvpilot.init(project="demo", task="detection", config={"lr": 0.01})
for step in range(3):
    cvpilot.log({"train/loss": 1.0 / (step + 1)}, step=step)
cvpilot.finish()
print(run.run_id)
```

Then list and inspect it:

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

`{*}` stands for a value that differs on every run, and `runs show` prints more fields than this.

## What you can do

- Record runs and reproduce them: `cvpilot.init`, `cvpilot.log`, `cvpilot repro`.
- Measure detection and classification quality per class: `cvpilot eval`, `cvpilot.log_eval`.
- Compare a new model with your current best and fail CI on a drop: `cvpilot gate`, `cvpilot compare`.
- Run that check on every pull request with the GitHub Action.
- Send metrics to MLflow, W&B or Trackio: `cvpilot.init(forward_to=["mlflow"])`.
- Hook into Lightning, Hugging Face, Ultralytics and MMEngine training: `cvpilot.autolog`.
- Get a message when the loss turns NaN, stalls or a model regresses: `cvpilot.init(alerts=...)`, `cvpilot alerts check`.
- Save crash-proof checkpoints and resume after a kill: `CheckpointManager`, `cvpilot ckpt`.

## Learn more

- [Tutorial](https://github.com/B4teShai/CV-Pilot/blob/main/docs/tutorial.md): a full run from logging to a failed gate to a resume.
- [User guide](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/concepts.md): concepts, install, gating in CI.
- [How it works](https://github.com/B4teShai/CV-Pilot/blob/main/docs/architecture/README.md): the architecture.
- [Checkpoints](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/checkpoints.md) and [trust](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/trust.md): resuming safely.
- [Settings](https://github.com/B4teShai/CV-Pilot/blob/main/docs/reference/settings.md) and exit codes.
- [What is built and what is planned](https://github.com/B4teShai/CV-Pilot/blob/main/docs/status.md).
- [Contributing](https://github.com/B4teShai/CV-Pilot/blob/main/CONTRIBUTING.md) and the [changelog](https://github.com/B4teShai/CV-Pilot/blob/main/CHANGELOG.md).

## License

Apache-2.0 ([LICENSE](https://github.com/B4teShai/CV-Pilot/blob/main/LICENSE)). The optional `cvpilot[ultralytics]` extra uses the AGPL-3.0 Ultralytics package, which then applies to your combined work.
