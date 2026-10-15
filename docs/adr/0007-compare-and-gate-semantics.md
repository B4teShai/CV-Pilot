# ADR-0007: Compare and gate semantics

**Status:** accepted
**Date:** 2026-10-15

## Context

`cvpilot gate` decides whether a CI job passes. A gate that can pass by accident is worse than
no gate: the team trusts it. Three things can make it pass by accident. The inputs may be
untrustworthy (a snapshot edited after the fact, a baseline re-pointed, a stored "result" file
planted in the store). The inputs may be incomplete (a metric, a slice or a class that is
absent, `NaN`, or measured on too few images) and a check that cannot be made is easy to skip.
The thresholds may be ambiguous (fractions or points, what a drop of zero means, whether an
improvement can fail).

The evaluation engine stores metrics as fractions in `[0, 1]` and bootstrap intervals in the
snapshot. It does not store ground truth or per-image results, so a paired bootstrap of a
difference is not possible at compare time.

## Options considered

### Option A: pass unless a threshold is clearly exceeded

Missing data is ignored, small slices are skipped, `NaN` compares false.

Trade-offs: no friction on sparse data; but every gap is a silent pass, and `NaN > limit` is
false in every language, so a broken run passes.

### Option B: fail closed on everything that cannot be checked, with one explicit opt-in

The gate passes only if every requested check could be made and held. A missing or non-finite
value, an absent or mismatched slice, a too small evaluation set or slice, a threshold that
gated nothing and any verification failure are violations or refusals. Small slices may be
skipped only with `--skip-small-slices`, and a threshold whose slices were all skipped still
fails.

Trade-offs: noisier on sparse data (a wildcard slice threshold fails on a rare category until
`--min-images` is lowered or `--skip-small-slices` is given); in return the verdict never
depends on what was not looked at.

### Option C: fail closed, with a paired bootstrap of the delta at compare time

Needs the ground truth and per-image matches at compare time, so the store would have to keep
them or the CLI would have to be given them again.

Trade-offs: a statistically better significance statement; but a much larger store, a slower
gate and a second place where the inputs could differ. Not needed to be correct about the
threshold semantics.

## Decision

Option B.

1. **Units and sign.** Snapshots hold fractions; every delta, drop and limit is in points on
   the 0..100 scale. A drop is `(baseline - candidate) * 100`. `--max-drop mAP50_95=0.5`
   fails when the candidate is more than 0.5 points (0.005) below the baseline. A drop equal
   to the limit passes; drops are rounded to 1e-9 points before comparing so float noise does
   not fail an exact boundary. A negative drop (an improvement) never fails, not even against
   a limit of 0. Limits are finite numbers from 0 to 100 written without exponent or
   separators.
2. **Comparable only when equal.** `eval_set_hash`, `task`, `class_names` (order included) and
   `params` (IoU grid, confusion settings, `max_dets`) must match. Missing `params` count as
   different. Otherwise `CompareRefusedError` (`EvalSetMismatchError` for the hash): event
   `compare_refused` with the reason and both hashes, exit 1, no result. A candidate without a
   snapshot for the baseline's evaluation set is a refusal too.
3. **Fail closed.** Violations: `headline_missing`, `headline_not_finite`, `headline_drop`;
   `class_missing`, `class_not_finite`, `class_drop`, `no_class_checked`; `slice_missing`,
   `slice_no_match`, `slice_spec_mismatch`, `slice_too_small`, `slice_none_gated`,
   `slice_not_finite`, `slice_drop`; `too_few_images`. Classes without ground truth on both
   sides are not applicable (nothing to regress), counted and not failed. At least one
   threshold must be given (exit 2 otherwise): a gate with none would check nothing.
4. **`--min-images` (default 30, at least 1)** is the smallest evaluation set and the smallest
   slice the gate accepts. This follows the instruction that a gate must not pass on too
   little data; the first sketch in 04-flows.md only skipped small slices, which is now the
   opt-in `--skip-small-slices`.
5. **Trust.** `gate` never reads a stored comparison, verdict or report. It recomputes from
   snapshots that were verified on read (file digest, per-run chain), a tag from a baseline row
   that was verified on read (record digest, store-wide chain, pinned snapshot digest) and
   confusion matrices from SHA-256 verified artifacts. A verification failure is exit 1 with
   its own event.
6. **Exit codes.** 0 only after `gate_passed`; 1 for a violation, a refusal, an unknown tag
   or run, a verification failure or any unexpected exception (the single error boundary);
   2 for usage errors. There is no path that exits 0 on an exception.
7. **Intervals.** `compare` shows the bootstrap intervals stored in the snapshots and whether
   two intervals overlap. Non-overlap is a conservative sign of a real change; overlap is not
   proof of none. Thresholds are on the point estimate. `compare` does no resampling, so it is
   deterministic and its cost does not depend on the image count.
8. **Output safety.** Names from data are shown through `safe_text` (control, format and
   private-use characters escaped, length bounded) and every output line starts with `gate:`
   or two spaces, so a class named `::error::...` cannot become a CI workflow command.

## Consequences

- A team adopting the gate sees failures where data is thin. The message names the check and
  the fix (`--min-images`, `--skip-small-slices`, a threshold for a slice that exists).
- The gate is read-only: it opens the store, verifies and compares; it writes nothing.
- A stronger significance test (Option C) remains possible later without changing the
  threshold semantics: it would add a column to the report, not a way to pass.
- Anyone who can write both the database and the snapshot files can recompute every digest;
  nothing is signed. The gate is a control against mistakes and casual edits and against
  stale or planted results, not against a malicious writer of the store. Pinning the baseline
  digest in CI configuration is a possible follow-up.

## Links

- [../architecture/02-components.md](../architecture/02-components.md),
  [../architecture/03-data-model.md](../architecture/03-data-model.md),
  [../architecture/04-flows.md](../architecture/04-flows.md)
- [0005-run-store.md](0005-run-store.md)
