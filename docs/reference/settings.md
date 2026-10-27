# Settings

CV-Pilot reads its settings from environment variables with the `CVPILOT_` prefix, once at
process start. Nothing is required: the defaults work on a laptop. An explicit function argument or
command option beats the variable, and the variable beats the default.

| Setting | Values | Default | What it does |
|---|---|---|---|
| `CVPILOT_ENV` | `dev`, `staging`, `prod` | `prod` | The `env` field of the logs. Debug logging and plain http are allowed only in `dev`. |
| `CVPILOT_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` | How much is logged to `stderr` (one JSON object per line) |
| `CVPILOT_PROJECT` | a name such as `fridge-sku` | none | The project used when a command or `init` leaves it out |
| `CVPILOT_STORE_DIR` | a directory | `~/.cvpilot` | Where runs, metrics and evaluation snapshots are stored |
| `CVPILOT_STORE_COMMAND_LINE` | `true`, `false` | `true` | `false` stores only the program name and the number of arguments, so `cvpilot repro` cannot re-run the command |
| `CVPILOT_FORWARD_TO` | `mlflow`, `wandb`, `trackio` (comma separated) | empty | Send metrics to these trackers; same as `init(forward_to=[...])` |
| `CVPILOT_FORWARD_HOSTS` | host names (comma separated) | empty | Only these hosts may be tracker endpoints |
| `CVPILOT_ALERTS` | a file | none | The alert configuration (JSON) used when `init` is not given `alerts=`; see [Alerts](../guide/alerts.md) |
| `CVPILOT_CHECKPOINT_DIR` | a directory | none | Where `CheckpointManager` writes checkpoints |
| `CVPILOT_CHECKPOINT_REMOTE` | `s3://`, `gs://` or `az://` URL | none | Also copy checkpoints to this bucket (`memory://` is accepted only with `CVPILOT_ENV=dev`: it keeps nothing after the process ends) |
| `CVPILOT_CKPT_KEY` | a secret, 32 to 1024 bytes | none | Signs checkpoints so a forged one is refused; see [Trust model](../guide/trust.md) |
| `CVPILOT_CKPT_REQUIRE_TRUST` | `any`, `anchored`, `authenticated` | `any` | The weakest proof a restore accepts; use `authenticated` for a shared or bucket-backed directory |
| `CVPILOT_CKPT_MIN_STEP` | a step number | none | Refuse a checkpoint older than this step |
| `CVPILOT_CKPT_ALLOW_PT` | `true`, `false` | `false` | Allow the pickle-based `pt` checkpoint format |

Alert channels read their address and credential from `CVPILOT_ALERT_WEBHOOK_URL`,
`CVPILOT_ALERT_WEBHOOK_TOKEN`, `CVPILOT_ALERT_SLACK_URL`, `CVPILOT_ALERT_TELEGRAM_TOKEN` and
`CVPILOT_ALERT_TELEGRAM_CHAT`: environment only, https only, never logged. Tracker tokens stay in
the tracker's own variables. Extras are listed on the [install page](../guide/install.md#extras).

## Exit codes

Every command uses the same codes: `0` success, `1` failure, `2` usage or configuration error,
`130` interrupted. The generated [command line reference](cli.md) lists each command, and the
contract is in [Errors and logging](../engineering/errors-and-logging.md).

| Command | `1` means | `2` means |
|---|---|---|
| `runs show`, `repro` | A well-formed run id that is not in the store. | A missing or invalid project, or a malformed run id. |
| `eval` | Invalid inputs, a closed run (add `--allow-closed`), an unknown run, a bad snapshot digest or an existing snapshot. | A bad `--slices`, or a step or seed out of range. |
| `alerts check` | The run to replay cannot be found or read. | An invalid configuration, a channel that cannot be built, or a missing project. |
| `gate` | The model regressed, or anything could not be checked ([Gating in CI](../guide/gating-ci.md)). | No threshold given. |
| `ckpt verify` | A corrupt, truncated, forged or untrusted checkpoint. | Not a checkpoint folder. |

`cvpilot doctor` prints `ok` on `stdout` and logs `doctor_completed` on `stderr`.
