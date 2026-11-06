# ADR-0013: Oriented bounding boxes (task `obb`)

**Status:** accepted
**Date:** 2026-11-06

## Context

Detectors trained for shelves, fridges, aerial images and documents often predict oriented boxes:
a rectangle with a rotation. The YOLO framework trains them as task `obb` and its validator reports
mAP50 and mAP50-95 for them. CV-Pilot evaluates axis-aligned detection at parity with pycocotools
(ADR-0007 gates on those numbers); an oriented detector could not be evaluated or gated at all.

Facts that shape the decision:

- There are two overlap measures in use. DOTA-style evaluation computes the exact IoU of the two
  polygons. The YOLO-OBB validator matches with *probabilistic IoU* (`batch_probiou`): each box
  becomes a Gaussian and the IoU is one minus their Hellinger distance (arXiv:2106.06072). The two
  give different numbers for the same boxes (probiou is smooth and never exactly 1 for equal
  boxes: about 0.9996), so a snapshot is only comparable with one measured the same way.
- The YOLO validator also integrates AP differently from COCO: the trapezoidal rule over a
  linearly interpolated precision envelope at 101 points, where COCO averages the envelope sampled
  at 101 recall points. Its matching (greedy in confidence order, best unclaimed IoU per threshold)
  is the COCO matching without crowd regions. On the parity fixture its mAP50-95 differs from the
  COCO rule by 0.002 on average and 0.007 at most, with identical true positives.
- Tools exchange oriented boxes in two forms: four corners (DOTA labels, YOLO-OBB label files, the
  framework's `xyxyxyxy` output) and `xywhr` (centre, size, rotation; the framework's internal form).
  Corner order and winding differ between tools.
- The core stays numpy-only (ADR-0003); the framework is AGPL-3.0 and is never imported by
  CV-Pilot's code (ADR-0001). The run store's `run_meta.task` column has a CHECK constraint that
  only allows `detection` and `classification`.

## Options considered

### Option A: one IoU definition, polygon IoU

Trade-offs: one number per model, exact geometry, matches DOTA. But the numbers cannot be
reconciled with what the framework prints during training, which is the first thing a user checks.

### Option B: one IoU definition, probabilistic IoU

Trade-offs: matches the framework's matching decisions. But it is a training-loss surrogate, not
an overlap: it rates elongated boxes differently from their area overlap and never reaches 1, and
other tools (DOTA, MMRotate) report polygon IoU.

### Option C: both, recorded in the evaluation parameters (chosen)

`EvalParams.iou_type` is `polygon` (default for `obb`) or `probiou`; `box` for detection and
classification. Trade-offs: one more parameter, and two snapshots of the same model can differ. That
is exactly what the parameter makes visible: compare and gate refuse two snapshots with different
`iou_type`, like any other parameter difference.

### Option D: also reproduce the framework's AP integration

Trade-offs: CV-Pilot's mAP would equal the framework's printed mAP. It needs a second accumulation
path (trapezoid rule, no 101-point sampling) next to the COCO one, bootstrap and slices for both,
and a second parity reference; and the framework's rule is not the COCO definition every other
CV-Pilot number uses. Rejected; the difference is documented and measured by the parity tests.

## Decision

Option C, built as follows.

1. **Task** `obb` (the framework's name) joins `detection` and `classification`. Metric names are
   the detection names (`mAP50_95`, `mAP50`, `mAP75`, `mAP_small` ..., `AR1` ..., per class
   `AP50_95`, `AP50`, `AR100`, `support`), so gate thresholds written for detection apply.
2. **Box format**: four corners `x1, y1, x2, y2, x3, y3, x4, y4` in pixels, in order around the box,
   either winding, any starting corner. Parquet predictions are `image_id, cls, score, x1..y4`,
   ground truth `image_id, cls, x1..y4` plus the optional `width`, `height` and metadata columns.
   `xywhr` (radians; width axis `(cos r, sin r)` with `y` pointing down, the framework's
   convention) converts at the edges with `cvpilot.core.obb.xywhr_to_corners` and back with
   `corners_to_xywhr` (minimum-area rectangle, `w >= h`, `r` in `[0, pi)`).
3. **Validation**: corners must be finite, inside the coordinate limits and form a convex
   quadrilateral (degenerate boxes included, bow ties refused); ground-truth corners must lie
   inside the image when its size is known; `iscrowd` is refused (neither definition has a crowd
   rule). Boxes are stored in one canonical order (positive winding, starting at the smallest `x`),
   so the `eval_set_hash` does not depend on the order a tool wrote the corners in.
4. **IoU**: `polygon` is the exact intersection area of the two convex quadrilaterals (corners
   inside the other box and edge crossings, ordered by angle, shoelace area) over the union.
   `probiou` derives `xywhr` from the corners (exact for rectangles) and computes the framework's
   formula with the same epsilon and clamping in float64. Both are numpy, vectorised over box
   pairs in bounded chunks. Areas for the COCO size ranges are polygon areas.
5. **Matching and AP** are the COCO machinery of `cvpilot.core.metrics`, reused through a
   `BoxGeometry` (area and pairwise IoU) passed to `match_detections` and the confusion matrix;
   accumulation, 101-point AP, AR, slices, bootstrap intervals and recompute are unchanged.
6. **Comparability**: `iou_type` is part of `EvalParams`; a snapshot without it (written before
   this change) reads as `box`. A snapshot whose `iou_type` does not suit its task is invalid.
   `compare` and `gate` refuse different `iou_type` with the reason `iou_type`.
7. **Store**: migration 2 rebuilds `run_meta` with `task IN ('detection','classification','obb')`
   (SQLite cannot alter a CHECK). The migration runner switches foreign-key enforcement off around
   migrations and runs `foreign_key_check` before committing. `SCHEMA_VERSION` becomes 2, so an
   older library refuses a store that may hold `obb` runs; a resumed run is written with the
   current schema version.
8. **Hook**: the YOLO hook logs the same scalars for an `obb` model as for a detection model and
   records no snapshot. It refuses to attach when the model's `task` contradicts the run's
   (`obb` model in a `detection` run), so later snapshots use the right task.

Parity (`tests/parity/test_obb_parity.py`, tolerance 1e-9, requirement 1e-3): `polygon` against
pycocotools with a shapely IoU, `probiou` against pycocotools with the framework's `batch_probiou`
(when installed), and the framework's own `match_predictions` must flag the same true positives.

## Consequences

- An oriented detector is evaluated, sliced, recomputed, compared and gated like a detection model.
- CV-Pilot's `mAP50_95` for `obb` is a COCO-rule number, so it differs from the framework's printed
  mAP by up to about 0.01 even with `--iou-type probiou`. Gate against CV-Pilot baselines, never
  against numbers copied from a training log.
- Opening an existing store with this release migrates it (with the usual `runs.sqlite.bak-1`
  backup); older releases then refuse it. A read-only open of an unmigrated store asks for a
  writable command first.
- Polygon IoU costs about 2.5 microseconds per box pair; oriented matching uses a smaller block
  budget (`OBB_BUDGET`) than axis-aligned matching.
- Crowd regions and DOTA's "difficult" flag are not supported for `obb`; adding them needs a rule
  for `probiou` first.
- Follow-up: export of the framework's OBB predictions to the Parquet layout is the user's script
  (shown in the gating guide); the hook does not capture predictions.

## Links

- [ADR-0007](0007-compare-and-gate-semantics.md) (comparability), [ADR-0005](0005-run-store.md)
  (migrations), [ADR-0001](0001-scope-and-license.md) (AGPL boundary).
- [Data model](../architecture/03-data-model.md), [Gating in CI](../guide/gating-ci.md#oriented-boxes-task-obb).
