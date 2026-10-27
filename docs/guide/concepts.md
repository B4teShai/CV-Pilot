# Concepts

Every term the docs use is defined once, in the glossary below. Other pages link here instead of
repeating a definition.

## Glossary

### Runs

| Term | Plain meaning |
|---|---|
| Run | One training or evaluation job: an id, a project and a status (`running`, `resumed`, `finished`, `failed`, `preempted`). |
| Project | A name that groups runs. Each project has its own folder in the store. |
| Store | The folder holding your runs (`CVPILOT_STORE_DIR`, default `~/.cvpilot`): plain files and one SQLite database. |
| `recover_run` | The call that marks a run `failed` after its process died without finishing. It refuses unless it can show the process is gone. |
| Preempted | The scheduler or cloud asked the job to stop. CV-Pilot saves a checkpoint and marks the run `preempted`. |
| Forwarder | A background thread that copies metrics to MLflow, W&B or Trackio, so a slow tracker never blocks training. |
| Event | One JSON log line on `stderr` with a fixed `message` such as `run_started`. Secrets are never logged. |

### Evaluation and gating

| Term | Plain meaning |
|---|---|
| Evaluation set | The images and ground-truth boxes you score a model on. |
| Eval set hash | A fingerprint of the ground truth and class list. Equal hashes mean two runs scored the same data. |
| Snapshot | The saved result of scoring one run on one evaluation set at one step: mAP, per-class AP, confusion matrix. Never edited. |
| Slice | The images that share a metadata value, such as `camera=cam3`, scored on their own. |
| Bootstrap interval | A range around a metric from re-scoring random resamples of the images. It tells noise from a real change. |
| Baseline | A named pin (such as `prod`) to a snapshot you trust. Pins are only added, never overwritten. |
| Gate | `cvpilot gate`: compares a run with a baseline and exits `1` if the run is worse. |
| Fail-closed | When something cannot be checked (missing metric, different eval set, failed integrity check) the gate fails instead of passing. |
| Hash chain | Each record stores a hash that covers the record before it. Editing, deleting or reordering an earlier record breaks the chain. |

### Checkpoints and trust

| Term | Plain meaning |
|---|---|
| Checkpoint | A saved training state (weights, optimizer, random state) and its manifest. |
| Manifest | A small file beside a checkpoint: its size, SHA-256, a hash-chain link and, with a key, an HMAC tag. |
| HMAC tag | A short code made from the manifest and your secret key. Only a holder of the key can produce it. |
| LATEST pointer | A file naming the newest checkpoint. A shortcut that is checked, never trusted alone. |
| Trust level | How a restored checkpoint was vouched for: `authenticated`, `anchored` or `local_chain_only`. |
| `authenticated` | The manifest has an HMAC tag made with your key (`CVPILOT_CKPT_KEY`). The strongest level. |
| `anchored` | The manifest's hash is also recorded in the run store, so deleting or swapping the file is noticed. |
| `local_chain_only` | Only the checkpoint folder's own hash chain vouches for it. Catches damage, not a person who can write the folder. This is the default. |
| Rollback floor | The newest step known to have been saved (from the store, the pointer or `CVPILOT_CKPT_MIN_STEP`). Anything older is refused. |
| Quarantine | A list of refused checkpoint steps. Nothing is moved or deleted; the steps still raise the floor. |

### Internals

| Term | Plain meaning |
|---|---|
| ULID | The 26-character run id. Ids sort by creation time. |
| WAL | SQLite's write-ahead log mode. Readers never block the writer, and a killed process leaves a readable database. |
| Content-addressed | Stored under the hash of its bytes: identical files are kept once, and a changed file is noticed. |
| Lease | A lock a live run holds on its checkpoint folder. `recover_run` never marks a run failed while it is held. |
| Liveness | Whether a run's process is still alive. If that is unknown, `recover_run` refuses unless you pass `force=True`. |
| Debris | Temporary or half-written files left by a killed save. The next save removes them. |

## Recording a run

- `init` works outside a git repository and without a GPU: missing facts are stored as none.
- A step must be larger than the last step logged for the same metric.
- `log` buffers points and writes them in batches; `finish` flushes and compacts them.
- Secrets in the config and command line are redacted before storage (best effort).
  `CVPILOT_STORE_COMMAND_LINE=false` stores only the program name, and `cvpilot repro` then cannot
  re-run the command.
- `cvpilot repro RUN` prints a bash script: check out the commit, apply the stored diff, run the
  command.
- A well-formed run id that is not in the store exits `1`; a malformed id or missing project
  exits `2` ([exit codes](../reference/settings.md#exit-codes)).

## Datasets prepared with DataCopilot

[DataCopilot](https://github.com/B4teShai/DataCopilot) turns a dataset into a versioned
`dataset.dcp` folder. Pass it as `dataset_uri` and the run records that version instead of hashing
every file:

```python notest
import cvpilot

run = cvpilot.init("fridge-sku", task="detection", dataset_uri="/data/fridge.dcp")
```

`cvpilot runs show RUN` then prints `dataset_hash: dcp:<version>`. `dcp:<version>` means the
version DataCopilot gave the dataset, copied from `dataset.dcp/manifest.json` and not checked
against the files. `merkle:<hex>` means CV-Pilot hashed the folder's file names, sizes and times
itself, because no usable version was there. The two packages do not depend on each other.

## Trackers and training frameworks

```python notest
import cvpilot

run = cvpilot.init("fridge-sku", task="detection", forward_to=["mlflow"])  # or CVPILOT_FORWARD_TO
run = cvpilot.autolog(trainer, project="fridge-sku", task="detection")  # Lightning, YOLO, ...
```

`autolog` picks the hook that matches the object: Lightning `Trainer`, Hugging Face `Trainer`,
Ultralytics `YOLO`, MMEngine `Runner` or an RF-DETR model. Each needs its [extra](install.md#extras).
Forwarders and hooks never raise into training. A tracker that is down is counted and logged once.
Tracker tokens stay in the tracker's own environment variables, and CV-Pilot refuses credentials
in URLs, plain http outside `CVPILOT_ENV=dev` and disabled TLS verification.
