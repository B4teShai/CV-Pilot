# Install

CV-Pilot needs Python 3.10 or newer. The core install is small and CPU-only. It never pulls in a
deep-learning framework, a web framework or a database server.

It is installed from its git repository. It is not on PyPI, so `pip install cvpilot` does not work.

## From a clone

```bash notest
git clone https://github.com/B4teShai/CV-Pilot.git
cd CV-Pilot
uv sync
uv run cvpilot doctor
```

`doctor` prints `ok`. `uv sync` creates `.venv` and also installs the test and documentation tools;
`source .venv/bin/activate` makes `cvpilot` and `python` the ones in that environment.

Without uv:

```bash notest
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/cvpilot doctor
```

## In your own project

Pin a full commit. A branch URL installs whatever the branch holds on the day, and nothing here is
signed or released. Take the commit from a clone you reviewed (`git rev-parse HEAD`) and put its 40
characters where `<full-commit-sha>` stands:

```bash notest
uv add git+https://github.com/B4teShai/CV-Pilot --rev <full-commit-sha>               # a uv project
pip install "cvpilot @ git+https://github.com/B4teShai/CV-Pilot@<full-commit-sha>"    # a virtual environment
pip install -e ../CV-Pilot                                                            # a local clone, editable
```

`uv add` records the commit in your `uv.lock`; with `pip`, keep the SHA in your requirements file.

## Extras

An extra adds one optional library, imported only when the code that needs it runs. The error
message names the extra to install:

```bash notest
uv sync --extra safetensors --extra lightning      # from a clone, with uv
pip install -e '.[safetensors,lightning]'          # from a clone, with pip
pip install "cvpilot[safetensors] @ git+https://github.com/B4teShai/CV-Pilot@<full-commit-sha>"  # in your project
uv add "cvpilot[safetensors] @ git+https://github.com/B4teShai/CV-Pilot" --rev <full-commit-sha>  # in a uv project
```

| Extra | Adds | Use |
|---|---|---|
| `cvpilot[safetensors]` | `safetensors` | Save and load checkpoints (the default format). |
| `cvpilot[remote-s3]`, `[remote-gcs]`, `[remote-azure]` | `fsspec` and the backend | Copy checkpoints to object storage. |
| `cvpilot[remote]` | `fsspec` only | The same, with a backend you install yourself. |
| `cvpilot[lightning]` | Lightning | Logger and callback for `Trainer`. |
| `cvpilot[hf]` | `transformers` with PyTorch | Callback for the Hugging Face `Trainer`. |
| `cvpilot[ultralytics]` | Ultralytics | Callbacks for YOLO models. **AGPL-3.0**, see below. |
| `cvpilot[mmengine]` | MMEngine | Hook for the MMEngine `Runner`. |
| `cvpilot[rfdetr]` | RF-DETR | Hook for an RF-DETR model's `train`. |
| `cvpilot[mlflow]`, `[wandb]`, `[trackio]` | The tracker SDK | Forward metrics to a tracker. |

The Ultralytics package is AGPL-3.0. CV-Pilot never imports it and stays Apache-2.0, but installing
the extra and using the hook makes the combined work you distribute or serve subject to that
license.

## Where data goes

By default runs go to `~/.cvpilot` and checkpoints go nowhere until you set `CVPILOT_CHECKPOINT_DIR`.
Every setting and the exit codes are on the [settings page](../reference/settings.md).
