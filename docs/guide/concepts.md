# Concepts

**Run.** One training or evaluation job: an id (a ULID), a project, a status (`running`,
`resumed`, `finished`, `failed`, `preempted`), the captured environment (`Repro`), and metric
points. A run that continues a preempted or failed one is a *new* run whose `parent_run_id` points
at it; the step timeline continues, the history stays separate.

**Store.** `CVPILOT_STORE_DIR/<project>/`: SQLite in WAL mode for rows, batched Parquet for
metric points, a content-addressed directory for artifacts. A process killed with `SIGKILL`
leaves a readable store; its run stays `running` until `recover_run` marks it `failed`.

**Snapshot.** An immutable evaluation of one run on one evaluation set at one step: COCO-style
mAP, per-class AP, confusion, slices and bootstrap intervals, keyed by
`(run, eval set hash, step)`. The evaluation set hash covers the ground truth and class list, so
two machines evaluating the same set agree.

**Baseline.** A named, append-only pin of a snapshot (`prod`, `staging`). Replacing a tag keeps the
old pin in the history; every read re-verifies digests and a hash chain.

**Gate.** `compare` shows the differences; `gate` decides. It fails closed: a missing or
non-finite metric, a different evaluation set, a snapshot that fails its integrity check, too few
images or a threshold that gated nothing all exit `1`. See [Gating in CI](gating-ci.md).

**Checkpoint.** A file with a *manifest* (size, SHA-256, a chain value linking it to the previous
manifest, optionally an HMAC tag) and pointers (`LATEST`). A restore walks back from the newest
and loads the first that verifies. See [Checkpoints and resume](checkpoints.md).

**Trust level.** How a restored checkpoint was vouched for: `authenticated` (HMAC key),
`anchored` (a digest in the run store) or `local_chain_only` (the directory's own chain). See
[Trust model](trust.md).

**Events.** Everything cvpilot logs is one JSON object per line on `stderr` with a constant
`message` (`run_started`, `checkpoint_saved`, `checkpoint_skipped_corrupt`, `gate_failed`, ...) and
the details in separate fields; secrets are redacted by key and never logged.

## Recording a run: what to expect

- `init` works outside a git repository and without a GPU: absent facts are recorded as none.
- A step must be strictly greater than the last step logged for the same metric key.
- `log` buffers up to 10 000 points and writes one Parquet file when the buffer is full or 2 s
  have passed; `finish` flushes and compacts them.
- Secrets in the config and on the command line are redacted before storage (best effort for free
  text). Set `CVPILOT_STORE_COMMAND_LINE=false` to store only the executable name and the number of
  arguments; `cvpilot repro` then cannot re-run the command. `repro RUN` prints a bash script:
  checkout, the stored diff and the command.
- `runs` and `repro` exit `1` for a well-formed run id that is not in the store and `2` for a
  missing project, an invalid project or a malformed run id (see [Settings](../reference/settings.md)).

## Datasets prepared with DataCopilot

[DataCopilot](https://github.com/B4teShai/DataCopilot) converts and audits a dataset into a
versioned `dataset.dcp` directory. Pass that directory as `dataset_uri` and the run records the
dataset by its version instead of hashing every file:

```bash notest
# in a DataCopilot clone (uv sync once): prints the version, for example 3b87778c...
uv run datacopilot ingest data/annotations.json --format coco -o /data/fridge.dcp
```

```python notest
import cvpilot

run = cvpilot.init("fridge-sku", task="detection", dataset_uri="/data/fridge.dcp")
```

`cvpilot runs show RUN` then prints `dataset_hash: dcp:<version>`, the same version DataCopilot
printed, so a gate result, a checkpoint manifest and a reproduction script all name the exact
dataset version the model was trained on. CV-Pilot reads only the `version` field of
`dataset.dcp/manifest.json` (at most 64 MiB, parsed as JSON, nothing executed); without a readable
version it falls back to a `merkle:` hash of the directory's files. The two packages do not depend
on each other: install both in the same environment, or run DataCopilot from its own clone.
`tests/unit/test_api.py` checks this with a manifest copied from a real `datacopilot ingest` run.

## Send metrics to a tracker and hook into a framework

```python notest
import cvpilot

run = cvpilot.init("fridge-sku", task="detection", forward_to=["mlflow"])  # or CVPILOT_FORWARD_TO
# Lightning (extra cvpilot[lightning]): the Trainer gets a logger and a callback
run = cvpilot.autolog(trainer, project="fridge-sku", task="detection")
# Ultralytics (extra cvpilot[ultralytics]): see the license note in the install guide
run = cvpilot.autolog(yolo_model, project="fridge-sku", task="detection")
# MMEngine (extra cvpilot[mmengine]): a hook on the Runner (call runner.train() and runner.test() after)
run = cvpilot.autolog(runner, project="fridge-sku", task="detection")
# RF-DETR (extra cvpilot[rfdetr]): call model.train(...) after; the run ends when it returns
run = cvpilot.autolog(model, project="fridge-sku", task="detection")
```

Forwarders (`cvpilot[mlflow]`, `[wandb]`, `[trackio]`) run on their own threads behind bounded
queues: a tracker that is down, slow or hung is counted and logged once and never raises into or
blocks training (`finish` waits at most 5 s). Tokens stay in the tracker SDK's own environment
variables; CV-Pilot refuses credentials in URLs, plain http outside `CVPILOT_ENV=dev` and disabled
TLS verification. Hook callbacks never raise into the framework. The `ultralytics` package is
AGPL-3.0: using `cvpilot[ultralytics]` makes your combined work subject to AGPL-3.0; CV-Pilot never
imports it. These integrations are tested with fake trainers and injected fake SDK modules; smoke
tests against the real libraries exist behind `pytest -m real_libs`, and notebooks 03, 04 and 05
ran the Hugging Face, Lightning, Ultralytics, MMEngine and RF-DETR hooks on real training runs. Design: [ADR-0008](../adr/0008-forwarders.md).

## Layers

Interface (`cli/`, the public API) uses application packages (`runs/`, `eval/`, `compare/`,
`ckpt/`), which use the domain (`core/`: schemas, hashing, errors) and reach infrastructure
(`store/`, `hooks/`, `forward/`, `remote/`) only through registries. Optional dependencies are
imported inside the function that needs them. [ADR-0004](../adr/0004-layering-and-dependency-rule.md)
has the rule and a test enforces it.
