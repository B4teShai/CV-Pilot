# Gating in CI

`cvpilot gate RUN --baseline tag:prod --max-drop mAP50_95=0.5` exits `0` when a run holds against
a pinned baseline and `1` when it regresses or when anything cannot be verified. Limits are in
points on the 0..100 scale; a drop equal to the limit passes and improvements never fail.

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
`env:` and are used inside double quotes as single arguments; it has no third-party step. The
workflow example in the repository README is parsed
and linted by `tests/unit/test_action.py`.

```yaml
- uses: ./
  with:
    run-id: ${{ steps.candidate.outputs.run-id }}
    baseline: tag:prod
    recompute: "true"
    gt: ci/ground-truth.parquet
    classes: ci/classes.txt
    require-commit: ${{ github.event.pull_request.head.sha }}
    expect-baseline-chain: ${{ vars.CVPILOT_BASELINE_CHAIN }}
    max-drop: |
      mAP50_95=0.5
```

## Checkpoints in CI

A job that restores or verifies checkpoints from a shared directory or a bucket should require
authentication: set `CVPILOT_CKPT_KEY` from a protected secret and `CVPILOT_CKPT_REQUIRE_TRUST`
to `authenticated`, then `cvpilot ckpt verify DIR --require-trust authenticated` fails the job
(exit `1`) for anything that is not provably yours. See [Trust model](trust.md).

## Semantics

[ADR-0007](../adr/0007-compare-and-gate-semantics.md) defines what is compared, when a comparison
is refused and what every exit code means.
