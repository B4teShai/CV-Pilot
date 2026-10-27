# Gating in CI

`cvpilot gate RUN --baseline tag:prod --max-drop mAP50_95=0.5` exits `0` when a run holds against a
pinned [baseline](concepts.md#glossary) and `1` when it regresses or when anything cannot be
checked. Limits are in points on the 0..100 scale. A drop equal to the limit passes, and
improvements never fail. Design: [ADR-0007](../adr/0007-compare-and-gate-semantics.md).

## 1. Evaluate predictions

Inputs are Parquet files: predictions with `image_id, cls, score, x1, y1, x2, y2`, and ground truth
with `image_id, cls, x1, y1, x2, y2` plus optional `iscrowd`, `width`, `height` and metadata
columns. A row with a null `cls` declares an image without objects. Class names are one per line.

```bash notest
cvpilot eval <RUN_ID> --preds p.parquet --gts g.parquet --classes names.txt \
  --slices camera=camera --step 10 --seed 0 --resamples 1000      # add --json for the snapshot
cvpilot runs show <RUN_ID>                                         # lists the snapshot
```

From a training loop, call `log_eval` with `PredictionTable` and `AnnotationTable` (from
`cvpilot.eval`). The framework hooks log scalars and leave evaluation to you.

```python notest
import cvpilot

run = cvpilot.init(project="fridge-sku", task="detection", config=cfg)
for step, batch in enumerate(loader):
    cvpilot.log({"train/loss": train_step(batch)}, step=step)
snapshot = cvpilot.log_eval(preds, gts, step=step, slices={"camera": "camera_id"})
cvpilot.finish()
```

A [snapshot](concepts.md#glossary) is never edited. Metric values are fractions in [0, 1]. Evaluating
into a finished run needs `--allow-closed`.

## 2. Pin a baseline, compare and gate

A baseline must come from a `finished` run, whether you pin it with `baseline set` or name it by run
id. There is no override: mark a dead run failed with `recover_run` and pin a run that finished.

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
$ cvpilot gate <run of another evaluation set> --baseline tag:prod ...
error: run 01M4... has no snapshot for eval set 65d37ab51abe        # event compare_refused
exit=1
```

The gate is [fail-closed](concepts.md#glossary). It exits `1` for a missing or non-finite metric, a
missing or too small slice, fewer than `--min-images` images (default 30), a threshold that gated
nothing, a failed integrity check, an unknown tag, and a different
[eval set hash](concepts.md#glossary), task, class list or evaluation setting. With no threshold at
all it is a usage error (exit `2`). `--skip-small-slices` is the one opt-in, capped by
`--max-skipped` (default 0).

`--report` writes static HTML with no script, and `--report-json` writes the canonical JSON. Reports
must stay inside `--report-dir` (default: the current directory), outside the store, and an existing
file that cvpilot did not create needs `--force`.

## 3. Make the gate a security control

Run on its own, the gate protects against accidents. It is a security control only if the pull
request it judges **cannot change** the code that runs, the baseline or the settings. A contributor
who controls the workflow, the store or the action can make any run pass. The attacks and controls
are in [Trust model for CI gates](../architecture/06-cross-cutting.md#trust-model-for-ci-gates). In
short:

- Install cvpilot from a pinned release with hashes, never from the pull request.
- Take the baseline store and the ground truth from the base branch or a protected artifact. A
  script from the base branch evaluates the pull request's predictions into a copy of the store
  (`cvpilot eval --git-commit`).
- Gate with `--recompute` (re-evaluates the stored predictions on trusted ground truth),
  `--require-commit SHA` and `--expect-baseline-chain SHA256`. The last is the digest printed by
  `baseline show`; keep it in a protected variable.
- Accept only `tag:` baselines, trigger on `pull_request` with read-only permissions (never
  `pull_request_target`) and protect `.github/` and `action.yml` with CODEOWNERS.

## The GitHub Action

`action.yml` wraps `cvpilot gate`. Its inputs reach the script only through `env:`, inside double
quotes. It has no third-party step. The exit code is the output `exit-code` (a string, so compare it
as `'0'`), and the step summary shows the verdict. `tests/unit/test_action.py` checks `action.yml`,
every workflow and the example below against these rules.

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

## Checkpoints in CI

A job that restores or verifies checkpoints from a shared folder or bucket should set
`CVPILOT_CKPT_KEY` from a protected secret and `CVPILOT_CKPT_REQUIRE_TRUST=authenticated`. Then
`cvpilot ckpt verify DIR --require-trust authenticated` fails the job (exit `1`) for anything not
provably yours. See [Trust model](trust.md).
