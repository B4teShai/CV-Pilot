# Install

CV-Pilot needs Python 3.10 or newer. The core install is small and CPU-only: it pulls in
`click`, `numpy`, `pyarrow`, `pydantic`, `structlog` and `typer`, never a deep-learning
framework, a web framework or a database server.

CV-Pilot is installed from its git repository. It is not on PyPI or any other package index, so
`pip install cvpilot` does not work; every command below works without one.

## From a clone

With [uv](https://docs.astral.sh/uv/):

```bash notest
git clone https://github.com/B4teShai/CV-Pilot.git
cd CV-Pilot
uv sync
uv run cvpilot --version
uv run cvpilot doctor
```

`uv sync` creates `.venv` and also installs the documentation and test tools (`docs` and `dev`
groups); `source .venv/bin/activate` makes `cvpilot` and `python` the ones in that environment.

With plain `pip` (no uv):

```bash notest
git clone https://github.com/B4teShai/CV-Pilot.git
cd CV-Pilot
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/cvpilot --version
.venv/bin/cvpilot doctor
```

`pip install .` (without `-e`) copies the package into the environment instead of linking the
clone. `scripts/check_clone.sh` does this in a temporary directory, from a fresh clone of
the current commit and a fresh virtual environment, and then runs the README quick start.

## Use it in your own project

Point your project's installer at the repository, pinned to a full commit: a branch URL installs
whatever the branch holds on the day, and nothing here is signed or released. Take the commit from
a clone you reviewed (`git rev-parse HEAD`) and put its 40 characters where `<full-commit-sha>`
stands:

```bash notest
uv add git+https://github.com/B4teShai/CV-Pilot --rev <full-commit-sha>               # a uv project
pip install "cvpilot @ git+https://github.com/B4teShai/CV-Pilot@<full-commit-sha>"    # a virtual environment
pip install -e ../CV-Pilot                                                            # a local clone, editable
```

`uv add` records the commit it resolved in your `uv.lock`; with `pip`, keep the SHA in your
requirements file. A local clone is only as good as the checkout in it.

## Extras

Each extra is imported lazily inside the code that needs it, and the error names the extra to
install. From a clone, add extras like this:

```bash notest
uv sync --extra safetensors --extra lightning      # with uv
pip install -e '.[safetensors,lightning]'          # with pip
pip install "cvpilot[safetensors] @ git+https://github.com/B4teShai/CV-Pilot@<full-commit-sha>"  # in your project
uv add "cvpilot[safetensors] @ git+https://github.com/B4teShai/CV-Pilot" --rev <full-commit-sha>  # in a uv project
```

When an error message says `pip install "cvpilot[hf]"`, it names the extra: install that extra in
one of the ways above.

| Extra | Adds | Notes |
|---|---|---|
| `cvpilot[safetensors]` | the `safetensors` library | needed to save and load checkpoints (the default format) |
| `cvpilot[remote-s3]`, `[remote-gcs]`, `[remote-azure]` | `fsspec` and `s3fs`, `gcsfs` or `adlfs` | mirror checkpoints to that kind of object storage |
| `cvpilot[remote]` | `fsspec` only | for a backend you install yourself |
| `cvpilot[lightning]` | Lightning | logger and callback for `Trainer` |
| `cvpilot[hf]` | Hugging Face `transformers` with PyTorch | callback for the `Trainer`; see [Alerts](alerts.md) |
| `cvpilot[ultralytics]` | Ultralytics | callbacks for YOLO models; **AGPL-3.0**, see below |
| `cvpilot[mmengine]` | MMEngine | hook for the MMEngine `Runner`: scalars, `val/` and `test/` metrics, the end of the run |
| `cvpilot[rfdetr]` | RF-DETR (Apache-2.0) and its training dependencies | hook for an RF-DETR model's `train`: losses and COCO metrics, the end of the run; the framework's `roboflow` upload client is left out |
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
