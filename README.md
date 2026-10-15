# CV-Pilot

Regression gates, remote monitoring and resume-safe checkpoints for vision models, on top of
the experiment tracker you already use.

## Status

Alpha. Built and tested: run capture, the local run store, the eval engine, baselines, compare and
gate, reports, tracker forwarders, Lightning and Ultralytics hooks, a GitHub Action, and the checkpoint manager
(atomic saves, verified restore, retention, preemption handling, object-storage mirror, HMAC
authentication and a trust policy).

- Python API (frozen by a snapshot test): `cvpilot.init`, `log`, `log_artifact`, `log_eval`,
  `finish`, `autolog`, `resume_path` and `CheckpointManager`. `Run` is a context manager.
- CLI: `cvpilot --version`, `doctor`, `runs list|show`, `repro`, `eval`, `baseline`, `compare`,
  `gate`, `ckpt list|verify|prune`, with the exit-code contract of
  [docs/engineering/errors-and-logging.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/engineering/errors-and-logging.md). The full
  reference is generated from the code: [docs/reference/cli.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/reference/cli.md).
- Settings: `CVPILOT_ENV`, `CVPILOT_LOG_LEVEL`, `CVPILOT_PROJECT`, `CVPILOT_STORE_DIR`,
  `CVPILOT_STORE_COMMAND_LINE`, `CVPILOT_FORWARD_TO`, `CVPILOT_FORWARD_HOSTS`,
  `CVPILOT_CHECKPOINT_DIR`, `CVPILOT_CHECKPOINT_REMOTE`, `CVPILOT_CKPT_KEY`,
  `CVPILOT_CKPT_REQUIRE_TRUST`, `CVPILOT_CKPT_ALLOW_PT`; structured JSON logging on `stderr`.
- Documentation: a MkDocs site (`mkdocs.yml`, `docs/`) with a user guide, a tutorial whose every
  command is run by a test ([docs/tutorial.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/tutorial.md)) and the architecture set.

Not built: the server, alert rules and channels, sync, and hooks for Hugging Face, MMEngine and
RF-DETR. The architecture documents mark each part as built or planned. See
[CHANGELOG.md](https://github.com/B4teShai/CV-Pilot/blob/main/CHANGELOG.md).

## Install

Requires Python 3.10 or newer.

```bash notest
pip install cvpilot        # or: uv add cvpilot
cvpilot --version
cvpilot doctor
```

To work on CV-Pilot itself, clone [the repository](https://github.com/B4teShai/CV-Pilot) and run
`uv sync` (needs [`uv`](https://docs.astral.sh/uv/)).

`doctor` prints `ok` on `stdout` and one JSON log line (`doctor_completed`) on `stderr`.
Exit codes: `0` success, `1` failure, `2` usage or configuration error, `130` interrupted.
Extras: `cvpilot[safetensors]` (checkpoints), `[remote]` (object-storage mirror), `[lightning]`,
`[ultralytics]` (AGPL-3.0, see below), `[mlflow]`, `[wandb]`, `[trackio]`.

## Record and inspect a run

The commands in this section are run by `tests/docs/test_readme.py`. They assume the virtualenv
is active (otherwise prefix with `uv run`).

```python file=train.py
import cvpilot

with cvpilot.init(task="detection", config={"lr": 0.01}, seed=0) as run:
    for step in range(3):
        cvpilot.log({"train/loss": 1.0 / (step + 1)}, step=step)
```

```bash
export CVPILOT_PROJECT=demo CVPILOT_STORE_DIR="$PWD/store"   # default store: ~/.cvpilot
python train.py                     # logs run_started and run_finished on stderr
cvpilot runs list                   # RUN_ID  STATUS  STARTED  NAME
RUN=$(cvpilot runs list --json | python -c "import json, sys; print(json.load(sys.stdin)[0]['run_id'])")
cvpilot runs show "$RUN"            # add --json for the full record
cvpilot repro "$RUN" | head -n 3    # prints a bash script: checkout, stored diff, command
```

```text expect
RUN_ID                      STATUS     STARTED                    NAME
{*}  finished   {*}  python-{*}
status: finished
#!/usr/bin/env bash
# Reproduction script for CV-Pilot run {*} (project demo).
```

```text expect stderr
json: message=run_started project=demo status=running
json: message=run_finished project=demo status=finished n_points=3
```

Notes:

- `init` works outside a git repository and without a GPU: absent facts are recorded as none.
- A step must be strictly greater than the last step logged for the same metric key.
- `log` buffers up to 10 000 points and writes one Parquet file when the buffer is full or
  2 s have passed; `finish` flushes and compacts them.
- Secrets in the config and on the command line are redacted before storage (best effort for
  free text). Set `CVPILOT_STORE_COMMAND_LINE=false` to store only the executable name and
  the number of arguments; `repro` then cannot re-run the command.
- Exit codes of `runs` and `repro`: `1` for a well-formed run id that is not in the store,
  `2` for a missing project, an invalid project or a malformed run id.
- `benchmarks/log_readback.py` logs 100 000 points and reads them back;
  `benchmarks/install_size.sh` checks that the core install carries no heavy framework;
  `benchmarks/ckpt_save.py` measures the checkpoint save overhead (the 5 % budget of ADR-0006 is
  not met: see the overview).

## Checkpoints that survive a kill

```python file=restart.py
import cvpilot
from cvpilot.ckpt import CheckpointManager, find_resume_point
from cvpilot.runs import recover_run

point = find_resume_point("fridge-sku")  # newest run of this project that can be resumed, or None
if point is not None and point.needs_recover:  # the earlier process was killed outright
    recover_run("fridge-sku", point.run_id)
run = cvpilot.init(
    "fridge-sku", task="detection", resume_from=None if point is None else point.run_id
)
ckpt = CheckpointManager(run, every_n_steps=1)  # directory: CVPILOT_CHECKPOINT_DIR
restored = ckpt.restore()  # None on a fresh start; this run's directory, then the runs it resumed
start = 0 if restored is None else restored.manifest.step + 1
print(f"starting at step {start}")
for step in range(start, start + 3):
    ckpt.maybe_save(step, model={"w": [float(step)]})
    if ckpt.preempted():
        break  # SIGTERM or SIGUSR1 set a flag; the run is already marked preempted
```

Run it twice. The first process stands in for a job that was killed, the second is its restart: a
new run that resumes the first and starts where it stopped. **The run store and the checkpoint
directory must both live on storage that survives the restart** (a persistent mount); the restart
finds its predecessor through them.

```bash
export CVPILOT_PROJECT=fridge-sku CVPILOT_STORE_DIR="$PWD/state/store" CVPILOT_CHECKPOINT_DIR="$PWD/state/ckpt"
python restart.py
python restart.py
```

```text expect
starting at step 0
starting at step 3
```

```text expect stderr
json: message=run_started project=fridge-sku status=running
json: message=run_started project=fridge-sku status=resumed
json: message=checkpoint_restored step=2 kind=last trust_level=anchored
```

On a node that has neither (a fresh machine that only reaches the bucket), configure `remote=` or
`CVPILOT_CHECKPOINT_REMOTE` and call `ckpt.restore(discover=True)`: it lists the newest runs on the
remote and restores the best step through the same checks as any other checkpoint
([docs/guide/checkpoints.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/checkpoints.md)).

A save is atomic and verified; a restore skips (and counts) anything truncated, corrupt, forged,
replaced or symlinked and never loads it; `cvpilot.resume_path("fridge-sku")` returns the newest
file that verifies. The files `CheckpointManager` writes are its own safetensors format: the
Lightning and Ultralytics `resume_arguments` helpers take only a verified framework `pt` file (see
the checkpoint guide for what a real Lightning or Ultralytics file allows). The default policy trusts the checkpoint directory's own digests, which is
right when only you can write it. **For a directory that is shared, restored from a bucket or
writable by others, set `CVPILOT_CKPT_KEY` and `CVPILOT_CKPT_REQUIRE_TRUST=authenticated`**
(a key stops forgery; detecting rollback or deletion also needs the run store, versioned storage
and `CVPILOT_CKPT_MIN_STEP`)
(see [docs/guide/trust.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/trust.md)). Walkthrough with a preemption, a damaged
file and a resume: [docs/tutorial.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/tutorial.md); design: ADR-0006.

## Evaluate predictions

Parquet files: predictions `image_id, cls, score, x1, y1, x2, y2`; ground truth `image_id, cls,
x1, y1, x2, y2` plus optional `iscrowd`, `width`, `height` and metadata columns (a row with a
null `cls` declares an image without objects); one class name per line (or a JSON list).

```bash notest
uv run cvpilot eval <RUN_ID> --preds p.parquet --gts g.parquet --classes names.txt \
  --slices camera=camera --step 10 --seed 0 --resamples 1000      # add --json for the snapshot
uv run cvpilot runs show <RUN_ID>                                  # lists the snapshot
uv run python benchmarks/eval_throughput.py                        # timing and peak memory
```

Exit codes of `eval`: `1` invalid inputs, a closed run (add `--allow-closed` to evaluate into a
finished run), an unknown run, a snapshot whose digest does not match, or an existing snapshot
(snapshots are immutable per set and step); `2` usage errors (bad `--slices`, a step or seed out
of range). `--max-memory-mb` caps the address space on Linux. Optional `--checkpoint-sha256` and
`--git-commit` are stored with the snapshot. Values are fractions in [0, 1].

## Baselines, compare and gate

Commands below were run against a store holding four runs of the same synthetic 40-image set
(`tests/support.py: write_eval_files`): a baseline, an equal run, a run with 10 broken
predictions and a run of a different evaluation set. Pins are append-only; `--replace` adds a
newer pin and keeps the old one.

```bash notest
cvpilot baseline set prod <BASELINE_RUN>        # pins the latest snapshot; --eval-set HASH, --step N
cvpilot baseline list                           # add --json, --history
cvpilot gate <RUN> --baseline tag:prod --max-drop mAP50_95=0.5 \
  --per-class-max-drop AP50_95=2.0 --slice-max-drop "camera=*:1.5" --min-images 20
cvpilot compare <RUN> --baseline tag:prod --report cmp.html --report-json cmp.json
```

```text
$ cvpilot gate <regressed run> --baseline tag:prod --max-drop mAP50_95=0.5 --per-class-max-drop AP50_95=2.0 --min-images 20
gate: FAIL (3 violations, worst: [class_drop] cat) run 01M4... step 0 vs baseline tag:prod
  - [class_drop] cat: AP50_95 of class dropped 43.56 points (limit 2)
  - [headline_drop] mAP50_95: mAP50_95 dropped 43.56 points (limit 0.5)
exit=1
$ cvpilot gate <equal run> ... --slice-max-drop "camera=*:1.5" --min-images 20
gate: PASS (5 checks, 0 slices skipped) run 01M4... step 0 vs baseline tag:prod
exit=0
$ cvpilot gate <run of another evaluation set> --baseline tag:prod ...
error: run 01M4... has no snapshot for eval set 65d37ab51abe        # event compare_refused
exit=1
```

Limits are in points on the 0..100 scale (`--max-drop mAP50_95=0.5` is 0.005 of the fraction in
the snapshot); a drop equal to the limit passes and improvements never fail. The gate fails
closed (exit 1) on an absent or non-finite metric, an absent or too small slice, fewer than
`--min-images` images (default 30), a threshold that gated nothing, a snapshot or baseline that
fails its integrity check, an unknown tag, and a different `eval_set_hash`, task, class names or
evaluation parameters; with no threshold at all it is a usage error (exit 2).
`--skip-small-slices` is the one explicit opt-in, capped by `--max-skipped` (default 0). In CI add `--recompute --gt GT --classes FILE`, `--require-commit SHA` and `--expect-baseline-chain SHA256` (the digest printed by `baseline show`); without them the gate trusts the store and protects against accidents only (see the trust model under "GitHub Action"). `--report` writes static HTML (no script, strict
CSP, every name escaped), `--report-json` the canonical JSON; paths must stay inside `--report-dir` (default: the current directory), outside the store and off runner files such as `$GITHUB_ENV`, and an existing file that cvpilot did not create needs `--force` (`--allow-outside` lifts the directory rule). Semantics: ADR-0007.
`python benchmarks/compare_throughput.py` times it on 5 000 images.

## Forward to a tracker and hook into a framework

```python notest
import cvpilot

run = cvpilot.init("fridge-sku", task="detection", forward_to=["mlflow"])  # or CVPILOT_FORWARD_TO
# Lightning: pip install "cvpilot[lightning]"; the Trainer gets a logger and a callback
run = cvpilot.autolog(trainer, project="fridge-sku", task="detection")
# Ultralytics: pip install "cvpilot[ultralytics]"; see the license note below
run = cvpilot.autolog(yolo_model, project="fridge-sku", task="detection")
```

Forwarders (`cvpilot[mlflow]`, `[wandb]`, `[trackio]`) run on their own threads behind bounded
queues: a tracker that is down, slow or hung is counted and logged once and never raises into or
blocks training (`finish` waits at most 5 s). Tokens stay in the tracker SDK's own environment
variables; CV-Pilot refuses credentials in URLs, plain http outside `CVPILOT_ENV=dev` and
disabled TLS verification. Hook callbacks never raise into the framework. The `ultralytics`
package is AGPL-3.0: using `cvpilot[ultralytics]` makes your combined work subject to AGPL-3.0;
CV-Pilot never imports it. These integrations are tested with fake trainers and injected fake
SDK modules; smoke tests against the real libraries exist behind `pytest -m real_libs` and were
not run in this environment.

## GitHub Action

`action.yml` is a composite action around `cvpilot gate`. **A gate is only a trust boundary if the
pull request it judges cannot change the code that runs, the baseline it compares with or the
settings it uses.** Run on its own it protects against accidents, not against a contributor who
controls the store, the workflow or the action. The full model is "Trust model for CI gates" in
[docs/architecture/06-cross-cutting.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/architecture/06-cross-cutting.md); in short:

- install cvpilot from a pinned release with hashes, never from the pull request;
- take the baseline store and the ground truth from the base branch or a protected artifact, and
  let a trusted script (from the base branch) evaluate the pull request's predictions into a copy
  of it with `cvpilot eval --git-commit`;
- gate with `recompute: "true"` (re-evaluates the stored predictions on the trusted ground truth),
  `require-commit` (the commit under test) and `expect-baseline-chain` (the digest printed by
  `cvpilot baseline show`, kept in a protected variable);
- accept only `tag:` baselines; protect `.github/` and `action.yml` with CODEOWNERS or required
  workflow rulesets; trigger on `pull_request` with read-only permissions, never
  `pull_request_target`.

```yaml
name: gate
on: pull_request # never pull_request_target
permissions:
  contents: read
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      # The base commit is trusted code: its action.yml, its CI scripts, its ground truth.
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          ref: ${{ github.event.pull_request.base.sha }}
          persist-credentials: false
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0
        with:
          python-version: "3.12"
      # A pinned release with hashes (pip-compile --generate-hashes), never the pull request.
      - run: python -m pip install --require-hashes --no-deps -r ci/cvpilot-requirements.txt
      # ci/make_candidate.py (from the base branch) copies the protected baseline store and runs
      # cvpilot eval on the pull request's predictions with the trusted ground truth and
      # --git-commit; it writes run-id=<ULID> to $GITHUB_OUTPUT.
      - id: candidate
        run: python ci/make_candidate.py
      - uses: ./
        with:
          run-id: ${{ steps.candidate.outputs.run-id }}
          baseline: tag:prod
          project: demo
          store-dir: ci-store
          recompute: "true"
          gt: ci/ground-truth.parquet
          classes: ci/classes.txt
          require-commit: ${{ github.event.pull_request.head.sha }}
          expect-baseline-chain: ${{ vars.CVPILOT_BASELINE_CHAIN }}
          max-drop: |
            mAP50_95=0.5
          slice-max-drop: |
            camera=*:1.5
          min-images: "30"
```

The Action has no third-party steps; inputs reach its script only through `env:` and are used
inside double quotes as single arguments (`--name=value`, the run id after `--`). The exit code
is available as the output `exit-code`; compare it as a string, for example
`if: steps.gate.outputs.exit-code == '0'`, because step outputs are strings. The step summary
holds the verdict (skipped slices included) in a fenced block. `tests/unit/test_action.py`
parses `action.yml`, every workflow and the example above and rejects any `${{ ... }}` expression
inside a `run:` script, an unpinned action, write permissions, unquoted variables, a checkout of
the pull request's head, an unhashed install, a gate without `recompute`, `expect-baseline-chain`
or `require-commit`, and `pull_request_target`; it also runs the real script against hostile input
values.

## More usage

### Evaluation snapshots in a plain training loop

`init`, `log`, `log_eval` and `finish` are built; passing `PredictionTable` and `AnnotationTable`
(from `cvpilot.eval`) instead of raw arrays; the hooks log scalars and leave evaluation to you.

```python notest
import cvpilot

run = cvpilot.init(project="fridge-sku", task="detection", config=cfg)
for step, batch in enumerate(loader):
    loss = train_step(batch)
    cvpilot.log({"train/loss": loss}, step=step)
snapshot = cvpilot.log_eval(preds, gts, step=step, slices={"camera": "camera_id"})
cvpilot.finish()
```

## Planned

The server, alert rules and channels, sync, and the Hugging Face, MMEngine and RF-DETR hooks are
not built; [docs/architecture/02-components.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/architecture/02-components.md) fixes their
interfaces and status.

## Architecture

![System context](https://raw.githubusercontent.com/B4teShai/CV-Pilot/main/docs/diagrams/png/context.png)

![Containers](https://raw.githubusercontent.com/B4teShai/CV-Pilot/main/docs/diagrams/png/containers.png)

- [docs/architecture/README.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/architecture/README.md): the full document set
  (overview, components, data model, flows, runtime, cross-cutting concerns).
- [docs/engineering/README.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/engineering/README.md): coding standards, testing,
  errors and logging, dependencies and licensing, workflow.
- [docs/adr/](https://github.com/B4teShai/CV-Pilot/blob/main/docs/adr/0001-scope-and-license.md): architecture decision records.
- [docs/engineering/release.md](https://github.com/B4teShai/CV-Pilot/blob/main/docs/engineering/release.md): the release runbook (publishing needs
  the maintainer's explicit approval).

## License

Apache-2.0, including the server. See [LICENSE](https://github.com/B4teShai/CV-Pilot/blob/main/LICENSE) and [NOTICE](https://github.com/B4teShai/CV-Pilot/blob/main/NOTICE).
