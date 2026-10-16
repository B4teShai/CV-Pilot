# Gating in CI

`cvpilot gate RUN --baseline tag:prod --max-drop mAP50_95=0.5` exits `0` when a run holds against
a pinned baseline and `1` when it regresses or when anything cannot be verified. Limits are in
points on the 0..100 scale; a drop equal to the limit passes and improvements never fail.

## Evaluate predictions

Inputs are Parquet files: predictions with `image_id, cls, score, x1, y1, x2, y2`; ground truth
with `image_id, cls, x1, y1, x2, y2` plus optional `iscrowd`, `width`, `height` and metadata
columns (a row with a null `cls` declares an image without objects); and one class name per line
(or a JSON list).

```bash notest
cvpilot eval <RUN_ID> --preds p.parquet --gts g.parquet --classes names.txt \
  --slices camera=camera --step 10 --seed 0 --resamples 1000      # add --json for the snapshot
cvpilot runs show <RUN_ID>                                         # lists the snapshot
python benchmarks/eval_throughput.py                               # timing and peak memory
```

From a training loop, pass `PredictionTable` and `AnnotationTable` (from `cvpilot.eval`) instead of
raw arrays. The framework hooks log scalars and leave evaluation to you.

```python notest
import cvpilot

run = cvpilot.init(project="fridge-sku", task="detection", config=cfg)
for step, batch in enumerate(loader):
    loss = train_step(batch)
    cvpilot.log({"train/loss": loss}, step=step)
snapshot = cvpilot.log_eval(preds, gts, step=step, slices={"camera": "camera_id"})
cvpilot.finish()
```

Exit codes of `eval`: `1` for invalid inputs, a closed run (add `--allow-closed` to evaluate into
a finished run), an unknown run, a snapshot whose digest does not match, or an existing snapshot
(snapshots are immutable per set and step); `2` for usage errors (bad `--slices`, a step or seed
out of range). `--max-memory-mb` caps the address space on Linux. Optional `--checkpoint-sha256`
and `--git-commit` are stored with the snapshot. Values are fractions in [0, 1].

## Baselines, compare and gate

A baseline is the model you compare against. The commands below were run against a store holding
four runs of the same synthetic 40-image set: a baseline, an equal run, a run with 10 broken
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
the snapshot); a drop equal to the limit passes and improvements never fail. The gate fails closed
(exit 1) on an absent or non-finite metric, an absent or too small slice, fewer than `--min-images`
images (default 30), a threshold that gated nothing, a snapshot or baseline that fails its
integrity check, an unknown tag, and a different `eval_set_hash`, task, class names or evaluation
parameters; with no threshold at all it is a usage error (exit 2). `--skip-small-slices` is the one
explicit opt-in, capped by `--max-skipped` (default 0).

In CI add `--recompute --gt GT --classes FILE`, `--require-commit SHA` and
`--expect-baseline-chain SHA256` (the digest printed by `baseline show`); without them the gate
trusts the store and protects against accidents only (see the trust conditions below).
`--report` writes static HTML (no script, strict CSP, every name escaped), `--report-json` the
canonical JSON; paths must stay inside `--report-dir` (default: the current directory), outside the
store and off runner files such as `$GITHUB_ENV`, and an existing file that cvpilot did not create
needs `--force` (`--allow-outside` lifts the directory rule). `python benchmarks/compare_throughput.py`
times a comparison on 5 000 images. Semantics: [ADR-0007](../adr/0007-compare-and-gate-semantics.md).

## A gate is a trust boundary only under conditions

Run on its own the gate protects against accidents. It is a security control only if the pull
request it judges **cannot change** the code that runs, the baseline it compares with or the
settings it uses. A contributor who controls the workflow, the store or the action can make any
run pass. The full model is in
[Trust model for CI gates](../architecture/06-cross-cutting.md); in short:

- install cvpilot from a pinned release with hashes, never from the pull request;
- take the baseline store and the ground truth from the base branch or a protected artifact, and
  let a script from the base branch evaluate the pull request's predictions into a copy of it
  (`cvpilot eval --git-commit`);
- gate with `recompute: "true"` (re-evaluates the stored predictions on the trusted ground truth),
  `require-commit` and `expect-baseline-chain` (the digest printed by `cvpilot baseline show`,
  kept in a protected variable);
- accept only `tag:` baselines; trigger on `pull_request` with read-only permissions, never
  `pull_request_target`; protect `.github/` and `action.yml` with CODEOWNERS.

## The Action

`action.yml` is a composite action around `cvpilot gate`. Its inputs reach the script only through
`env:` and are used inside double quotes as single arguments (`--name=value`, the run id after
`--`); it has no third-party step. The exit code is available as the output `exit-code`; compare it
as a string, for example `if: steps.gate.outputs.exit-code == '0'`, because step outputs are
strings. The step summary holds the verdict (skipped slices included) in a fenced block.

`tests/unit/test_action.py` parses `action.yml`, every workflow and the example below, and rejects
any `${{ ... }}` expression inside a `run:` script, an unpinned action, write permissions, unquoted
variables, a checkout of the pull request's head, an unhashed install, a gate without `recompute`,
`expect-baseline-chain` or `require-commit`, and `pull_request_target`; it also runs the real
script against hostile input values.

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

A job that restores or verifies checkpoints from a shared directory or a bucket should require
authentication: set `CVPILOT_CKPT_KEY` from a protected secret and `CVPILOT_CKPT_REQUIRE_TRUST`
to `authenticated`, then `cvpilot ckpt verify DIR --require-trust authenticated` fails the job
(exit `1`) for anything that is not provably yours. See [Trust model](trust.md).

## Semantics

[ADR-0007](../adr/0007-compare-and-gate-semantics.md) defines what is compared, when a comparison
is refused and what every exit code means.
