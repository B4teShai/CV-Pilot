# Install

CV-Pilot needs Python 3.10 or newer. The core install is small and CPU-only: it pulls in
`click`, `numpy`, `pyarrow`, `pydantic`, `structlog` and `typer`, never a deep-learning
framework, a web framework or a database server.

## From source

```bash notest
git clone <repository-url> cv-pilot
cd cv-pilot
uv sync
uv run cvpilot --version
uv run cvpilot doctor
```

`uv sync` also installs the documentation and test tools (`docs` and `dev` groups). A published
package is pending: nothing has been uploaded to any index, and the maintainer decides when.

## Extras

Each extra is imported lazily inside the code that needs it, and the error names the extra to
install.

| Extra | Adds | Notes |
|---|---|---|
| `cvpilot[safetensors]` | the `safetensors` library | needed to save and load checkpoints (the default format) |
| `cvpilot[remote]` | `fsspec` | mirror checkpoints to object storage; add the backend (`s3fs`, `gcsfs`, `adlfs`) |
| `cvpilot[lightning]` | Lightning | logger and callback for `Trainer` |
| `cvpilot[ultralytics]` | Ultralytics | callbacks for YOLO models; **AGPL-3.0**, see below |
| `cvpilot[mlflow]`, `[wandb]`, `[trackio]` | the tracker SDK | forward scalars to a tracker |

The Ultralytics package is licensed AGPL-3.0. CV-Pilot never imports it from its own code and
stays Apache-2.0, but installing the extra and using the hook makes the combined work you
distribute or serve subject to that license.

## Check the install

`cvpilot doctor` prints `ok` on `stdout` and one structured log line on `stderr`; exit code `0`.
Exit codes everywhere: `0` success, `1` failure, `2` usage or configuration error, `130`
interrupted.

## Where data goes

| Setting | Default | Holds |
|---|---|---|
| `CVPILOT_STORE_DIR` | `~/.cvpilot` | one folder per project: run rows, metrics, snapshots, artifacts |
| `CVPILOT_CHECKPOINT_DIR` | none | one folder per run: checkpoints and their manifests |
| `CVPILOT_PROJECT` | none | the project name when a command or `init` omits it |

All settings are listed in [Runtime and configuration](../architecture/05-runtime.md).
