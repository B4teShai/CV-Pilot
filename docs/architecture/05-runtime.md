# 05 Runtime

![Runtime and deployment](../diagrams/png/runtime.png)

## Process model

| Where | What runs | Threads and subprocesses |
|---|---|---|
| Training process (built: `api.py`, `runs/`, store and capture in-process; planned: hooks, `eval/`, `ckpt/`, forwarders, alerts) | the framework, the hook adapter, `api.py`, `runs/`, `eval/`, `ckpt/`, client-side `alerts/`, forwarders | main thread: training and the hook callbacks; `cvpilot-sync` daemon thread: outbox → HTTPS; `cvpilot-nvml` daemon thread: NVML sample every 10 s; signal handlers for SIGTERM/SIGUSR1 set a flag only. No subprocess except `git` once at `init`. |
| `cvpilot` CLI (built: `--version`, `doctor`, `runs`, `repro`) | one command per invocation, single-threaded | none; planned `compare` uses a `ThreadPoolExecutor` of `min(4, cpu_count)` for bootstrap resamples |
| Server container (planned) | Uvicorn with one FastAPI worker per CPU, websocket broadcaster, alert delivery worker | each worker is a process; alert delivery is an asyncio task; SQLite requires one worker, Postgres allows several |
| CI runner (planned) | `cvpilot gate` as a step; exit code gates the job | same as CLI |

The hook never blocks training on I/O: `log()` appends to an in-memory buffer bounded at
10 000 points; the store flush happens on the caller's thread when the buffer is full or 2 s
have passed since the last flush, checked on each `log()` call (one Parquet write, tens of
milliseconds, built; a failed write keeps the points and raises `StoreError`); forwarders are called after the flush and their failures are logged,
not raised; the sync thread owns all network I/O.

## Deployment nodes

| Node | Runs | Reads | Writes |
|---|---|---|---|
| Developer machine | `cvpilot` CLI (built), Python API in notebooks (planned) | local run store, checkpoint dirs, optional server over HTTPS | local run store, reports |
| GPU training machine (spot, SLURM node, workstation) | training process with hooks, sync thread, NVML sampler, checkpoint manager | dataset, DataCopilot `manifest.json`, NVML, preemption metadata endpoint | local run store, `/checkpoint`, object storage (via fsspec), tracker, server (via sync), channels (client-side alerts) |
| CI runner | `cvpilot gate`, `cvpilot compare --report` | project store (checked-out copy or mounted bucket) or server snapshots over HTTPS | report artifact uploaded by the CI job |
| Object storage (S3, GCS, any fsspec backend) | nothing | — | checkpoint mirror (`<remote>/<run_id>/{last,best}/...`), exported artifacts |
| Server host (optional) | `cvpilot-server` Docker image behind a TLS reverse proxy; SQLite file volume or Postgres | ingest from sync clients, snapshots for CI and browsers | server DB, alert deliveries, share links |
| Notification channels | Telegram Bot API, Slack, webhook endpoints, SMTP relay | — | — |

## Storage locations and defaults

| Data | Default location | Override | Notes |
|---|---|---|---|
| Run store (built) | `$HOME/.cvpilot/<project>/` | `CVPILOT_STORE_DIR` or `init(store_dir=...)` | layout in [03-data-model.md](03-data-model.md); one directory per project |
| Checkpoints | `CheckpointManager(dir=...)`, no default | `CVPILOT_CHECKPOINT_DIR` | a mounted bucket path on SkyPilot |
| Checkpoint mirror | none | `CheckpointManager(remote=...)` or `CVPILOT_CHECKPOINT_REMOTE` | fsspec URL |
| Compare reports | the `--report` and `--report-json` paths only; nothing is written to the store | `--report`, `--report-json` | static HTML (no script, strict CSP), at most 8 MiB; the prediction-diff gallery is planned |
| Server database | `/data/cvpilot.sqlite` inside the container | `CVPILOT_SERVER_DATABASE_URL` | `sqlite:///...` or `postgresql://...` |
| Logs | `stderr` | none | collected by the process supervisor or container runtime |

## Configuration

All configuration is environment variables with the `CVPILOT_` prefix, read once at process
start by `settings.py`; an invalid value is reported on `stderr` with exit 2 before logging
is configured. Precedence, highest first: explicit function argument or CLI option; environment
variable; default. There is no configuration file.

| Variable | Type / values | Default | Used by | Status |
|---|---|---|---|---|
| `CVPILOT_ENV` | `dev`, `staging`, `prod` | `prod` | logging (`env` field, DEBUG only in `dev`) | built |
| `CVPILOT_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | logging | built |
| `CVPILOT_PROJECT` | project name `[a-z0-9][a-z0-9_-]{0,63}` | none | `init()` and CLI commands when the project is omitted | built |
| `CVPILOT_STORE_DIR` | directory path, expanded and resolved; not a file or `/` | `$HOME/.cvpilot` | store, CLI | built |
| `CVPILOT_STORE_COMMAND_LINE` | `true`, `false` (also `1/0`, `yes/no`, `on/off`) | `true` | `Repro.command_line` capture; `false` stores only the executable name and the argument count | built (`settings.py` parses it; `api.init` passes it to `capture_repro`; tests cover both values) |
| `CVPILOT_SERVER_URL` | `https://host[:port]` | none (sync disabled) | sync client, `gate --from-server` | planned |
| `CVPILOT_TOKEN` | per-project token | none | sync client, CLI calls to the server; sent as `Authorization: Bearer` | planned, secret |
| `CVPILOT_CHECKPOINT_DIR` | directory path | none | `CheckpointManager` default `dir` | planned |
| `CVPILOT_CHECKPOINT_REMOTE` | fsspec URL | none | `CheckpointManager` default `remote` | planned |
| `CVPILOT_FORWARD_TO` | comma-separated forwarder names | empty | `init()` default `forward_to` | planned |
| `CVPILOT_ALERT_CHANNELS` | comma-separated channel names | empty | client-side alert delivery | planned |
| `CVPILOT_SERVER_DATABASE_URL` | SQLAlchemy URL | `sqlite:////data/cvpilot.sqlite` | server only | planned |
| `CVPILOT_SERVER_TOKEN_HASHES` | path to a file of `project:sha256(token)` lines | `/data/tokens` | server only | planned |
| `CVPILOT_TELEGRAM_BOT_TOKEN`, `CVPILOT_TELEGRAM_CHAT_ID` | string | none | Telegram channel | planned, secret |
| `CVPILOT_SLACK_WEBHOOK_URL`, `CVPILOT_WEBHOOK_URL` | URL | none | Slack and generic webhook channels | planned, secret |
| `CVPILOT_SMTP_URL` | `smtp[s]://user:pass@host:port` | none | SMTP channel | planned, secret |

## Secrets

| Secret | May live in | Must never appear in |
|---|---|---|
| `CVPILOT_TOKEN` (project token) | environment of the training process and CI (CI secret store), the `Authorization` header over TLS | logs, `Repro.command_line`, `Repro.config`, reports, the run store, URLs, `doctor` output |
| Server token hashes | the file named by `CVPILOT_SERVER_TOKEN_HASHES` (SHA-256 of the token only) | the server database in clear text, logs, HTTP responses |
| Telegram bot token, Slack and webhook URLs, SMTP URL | environment of the process that delivers alerts (server, or training process for client-side delivery) | logs, alert messages, share links, the run store |
| Cloud credentials for fsspec | the backend's own mechanism (instance role, `AWS_*`, `GOOGLE_APPLICATION_CREDENTIALS`) | CV-Pilot configuration, logs, manifests |
| Share links | server database (random 32-byte token, expiry ≤ 7 days) | logs |

Key-based redaction in `logging.py` already covers keys ending in `token`, `password`,
`api_key`, `jwt`, `authorization`. The alerts package adds the suffixes `webhook_url`,
`smtp_url` and `chat_id` to the redaction list before any channel is registered. Free-text
fields are never used for secrets.

## Optional extras

| Extra | Unlocks | Dependencies added | Hardware |
|---|---|---|---|
| `cvpilot[lightning]` | Lightning logger and callback hook | `lightning`, `torch` | GPU recommended for training; CPU works for tests |
| `cvpilot[hf]` | HF `Trainer` callback hook | `transformers`, `torch` | same |
| `cvpilot[mmdet]` | MMEngine hook and vis backend | `mmengine`, `mmdet`, `torch` | same |
| `cvpilot[mlflow]`, `cvpilot[wandb]`, `cvpilot[trackio]` | forwarder to that tracker | the tracker SDK | none |
| `cvpilot[alerts]` | Telegram, Slack and generic webhook channels (SMTP needs no extra) | `httpx` | none |
| `cvpilot[server]` | `cvpilot server`, websocket live view, htmx UI, share links | `fastapi`, `uvicorn[standard]`, `websockets`, `sqlalchemy`, `asyncpg` (Postgres optional) | 1 vCPU, 512 MB RAM for SQLite mode |
| `cvpilot[ultralytics]` | Ultralytics callbacks (`autolog(YOLO(...))`), per-class AP and confusion capture, verified `resume=` | `ultralytics` (AGPL-3.0), `torch` | GPU recommended |

Notice for `cvpilot[ultralytics]`: `ultralytics` is AGPL-3.0. It is imported only inside
`hooks/ultralytics.py` when the hook is attached; no Ultralytics code is vendored. Installing
and using this extra makes the user's combined work subject to AGPL-3.0; `cvpilot autolog
--help` and the hook docstring repeat this notice. The core package and every other extra
stay Apache-2.0.

The eval engine runs on CPU with the core package (`numpy`, `pyarrow`; `pycocotools` only for
parity tests); checkpoint verification adds `safetensors` and `fsspec`; `torch`
is needed only to load `kind="last"` checkpoints and arrives with any framework extra.
GPU telemetry requires an NVIDIA driver and `pynvml`; without them the sampler logs
`nvml_unavailable` once at `INFO` and stays idle.
