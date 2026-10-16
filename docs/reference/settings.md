# Settings

CV-Pilot reads its settings from environment variables with the `CVPILOT_` prefix, once at
process start. Nothing here is required: the defaults work on a laptop. The full table, including
the settings of parts that are not built yet, is in
[Runtime and configuration](../architecture/05-runtime.md).

| Setting | Values | Default | What it does |
|---|---|---|---|
| `CVPILOT_ENV` | `dev`, `staging`, `prod` | `prod` | Sets the `env` field of the logs; debug logging and plain http to a tracker are allowed only in `dev` |
| `CVPILOT_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | How much is logged to `stderr` (one JSON object per line) |
| `CVPILOT_PROJECT` | a name such as `fridge-sku` | none | The project used when a command or `init` leaves it out |
| `CVPILOT_STORE_DIR` | a directory | `~/.cvpilot` | Where runs, metrics and evaluation snapshots are stored |
| `CVPILOT_STORE_COMMAND_LINE` | `true`, `false` | `true` | `false` stores only the program name and the number of arguments, so `cvpilot repro` cannot re-run the command |
| `CVPILOT_FORWARD_TO` | `mlflow`, `wandb`, `trackio` (comma separated) | empty | Send metrics to these trackers; same as `init(forward_to=[...])` |
| `CVPILOT_FORWARD_HOSTS` | host names (comma separated) | empty | Only these hosts may be tracker endpoints |
| `CVPILOT_CHECKPOINT_DIR` | a directory | none | Where `CheckpointManager` writes checkpoints |
| `CVPILOT_CHECKPOINT_REMOTE` | `s3://`, `gs://` or `az://` URL | none | Also copy checkpoints to this bucket |
| `CVPILOT_CKPT_KEY` | a secret, 32 to 1024 bytes | none | Signs checkpoints so a forged one is refused; see [Trust model](../guide/trust.md) |
| `CVPILOT_CKPT_REQUIRE_TRUST` | `any`, `anchored`, `authenticated` | `any` | The weakest proof a restore accepts; use `authenticated` for a shared or bucket-backed directory |
| `CVPILOT_CKPT_MIN_STEP` | a step number | none | Refuse a checkpoint older than this step |
| `CVPILOT_CKPT_ALLOW_PT` | `true`, `false` | `false` | Allow the pickle-based `pt` checkpoint format |

Tracker tokens stay in the tracker's own environment variables. CV-Pilot refuses credentials in
URLs, plain http outside `CVPILOT_ENV=dev` and disabled TLS verification.

## Extras

`pip install cvpilot` has no deep-learning framework. Add what you use: `cvpilot[safetensors]`
(checkpoints), `[remote-s3]`, `[remote-gcs]`, `[remote-azure]` or `[remote]` (copy checkpoints to a
bucket), `[lightning]`, `[ultralytics]` (the Ultralytics package is AGPL-3.0, see
[Install](../guide/install.md)), `[mlflow]`, `[wandb]`, `[trackio]`.

## Exit codes

Every command uses the same codes: `0` success, `1` failure, `2` usage or configuration error,
`130` interrupted. The full contract is in
[Errors and logging](../engineering/errors-and-logging.md), and the generated
[command line reference](cli.md) lists each command.

| Command | `1` means | `2` means |
|---|---|---|
| `runs show`, `repro` | a well-formed run id that is not in the store | a missing or invalid project, a malformed run id |
| `eval` | invalid inputs, a closed run (add `--allow-closed` to evaluate into a finished run), an unknown run, a snapshot whose digest does not match, or an existing snapshot (snapshots are immutable per set and step) | a bad `--slices`, a step or seed out of range |
| `gate` | the model regressed, or anything could not be verified (see [Gating in CI](../guide/gating-ci.md)) | no threshold given |

`cvpilot doctor` prints `ok` on `stdout` and one JSON log line (`doctor_completed`) on `stderr`.
