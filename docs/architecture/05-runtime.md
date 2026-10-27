# 05 Runtime

Where CV-Pilot code runs, what it reads and writes, and where configuration and secrets live.

## Process model

| Where | What runs | Threads and subprocesses |
|---|---|---|
| Your training process | Your framework, the hook, `api.py`, `runs/`, `eval/`, `ckpt/`, alert rules, forwarders, store and capture | The main thread does the training, the hook callbacks and the alert rules (microseconds per `log`). One daemon thread per forwarder and per alert channel. One optional daemon thread polls a cloud preemption notice. No subprocess except `git`, once at `init`. |
| The `cvpilot` CLI | One command per invocation | Single-threaded, except `eval`, whose bootstrap uses up to 4 worker threads. |
| A CI runner | `cvpilot gate` as a step, or the composite action `action.yml` that wraps it | The exit code gates the job. |

The hook never blocks training on I/O. `log()` appends to a buffer of at most 10,000 points. The
buffer is written on the caller's thread when it is full or 2 seconds have passed: one Parquet write
of tens of milliseconds. A failed write keeps the points and raises `StoreError`. Forwarders and
alert channels own all network I/O, behind bounded queues ([ADR-0008](../adr/0008-forwarders.md)).

Signal handlers for `SIGTERM` and `SIGUSR1` only set a flag.

## Where things live

| Node | Runs | Writes |
|---|---|---|
| Your machine | CLI, notebooks | The run store, reports |
| GPU machine (spot, SLURM, workstation) | The training process with hooks and the checkpoint manager | The run store, the checkpoint folder, object storage, trackers, alert channels |
| CI runner | `cvpilot gate` or `compare --report` | A report, uploaded by the CI job |
| Object storage | Nothing | The checkpoint mirror, `<remote>/<run_id>/{last,best}/...` |

| Data | Default location | Override |
|---|---|---|
| Run store | `~/.cvpilot/<project>/` | `CVPILOT_STORE_DIR` or `init(store_dir=...)` |
| Checkpoints | None; one folder per run, mode 0700 | `CVPILOT_CHECKPOINT_DIR` |
| Checkpoint mirror | None | `CVPILOT_CHECKPOINT_REMOTE` (`s3://`, `gs://`, `az://`, no credentials in the URL) |
| Reports | Only the `--report` and `--report-json` paths | Those options |
| Logs | `stderr` | Your process supervisor |

The folder layout is in [03 Data model](03-data-model.md).

## Configuration

All configuration is environment variables with the `CVPILOT_` prefix, read once at start by
`settings.py`. An invalid value is reported on `stderr` with exit 2 before logging starts. An
explicit function argument or CLI option beats the variable, which beats the default. There is no
configuration file. Every variable is in the [settings reference](../reference/settings.md). Two
more are read: `RANK` and `LOCAL_RANK`, set by distributed launchers. A non-zero rank starts no
forwarder and sends no alerts.

## Secrets

| Secret | May live in | Must never appear in |
|---|---|---|
| `CVPILOT_CKPT_KEY` | The process environment, a CI secret | Logs, manifests, errors, the run store |
| Alert URLs and tokens | The process environment, read once when the channel is built | Logs, alert text, `repr`, the run store, the alert config file |
| Tracker credentials (`MLFLOW_TRACKING_TOKEN`, `WANDB_API_KEY`, ...) | The environment, read by the tracker SDK only | CV-Pilot settings, logs, the run store, URLs |
| Cloud credentials | The backend's own mechanism (instance role, `AWS_*`) | CV-Pilot settings, logs, manifests |

Logging also redacts keys ending in `token`, `password`, `api_key`, `jwt` and `authorization`. That is
a second line of defence: alert channels never put a URL or token into a field at all.

## Extras and hardware

The evaluation engine runs on CPU with the core package. Checkpoints need `cvpilot[safetensors]`.
Verifying a checkpoint (size, structure, SHA-256, chain) needs no extra, and the framework is needed
only for the opt-in `pt` format. The extras are listed on the [install page](../guide/install.md#extras).
GPU hooks need a GPU for training and nothing else from CV-Pilot.
