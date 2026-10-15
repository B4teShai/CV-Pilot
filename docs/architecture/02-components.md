# 02 Components

![Components by layer](../diagrams/png/components.png)

The package follows the four layers of
[ADR-0004](../adr/0004-layering-and-dependency-rule.md). Built today: `cli/`, `api.py`,
`runs/`, `repro`, `eval/`, `compare/`, `ckpt/`, `core/`, `store/`, `capture/`, `forward/`, `hooks/`,
`remote/` (checkpoint mirror and preemption notices), `registry.py`, `logging.py` and `settings.py`; each section below states what part is built.
Every other package is planned and its interface is the contract it will be built to.

| Layer | Packages |
|---|---|
| Interface | `cli/` (built), `api.py` (built: `init`, `log`, `log_artifact`, `log_eval`, `finish`) |
| Application | `runs/` (built, with the forwarder fan-out), `repro` (built), `eval/` (built), `compare/` (baselines, compare, gate built), `ckpt/`, `alerts/` |
| Domain | `core/schema.py`, `core/hashing.py` (built); `core/ulid.py`, `core/redact.py`, `core/jsonable.py`, `core/errors.py` (built); `core/metrics.py`, `core/confusion.py` (built) |
| Infrastructure | `store/`, `capture/`, `forward/`, `hooks/` (built); `sync/`, `remote/`, `notify/`, `server/` |
| Cross-cutting | `logging.py`, `settings.py`, `registry.py` (built) |

Signatures use `Literal` and Pydantic models from `core/schema.py`; `PredictionTable` and
`AnnotationTable` are typed wrappers over Arrow tables with the columns given in
[03-data-model.md](03-data-model.md).

## Interface layer

### `cli/`

Responsibility: Typer application, option parsing, the single error boundary, exit codes,
`stdout` for results and `stderr` for logs.

Public interface (built):

```python notest
def main() -> NoReturn            # entry point `cvpilot`; exit 0/1/2/130
# commands: --version, doctor, runs list|show, repro, eval, baseline set|list|show, compare, gate
```

`runs list|show` and `repro` accept `--project` (default `CVPILOT_PROJECT`; missing gives exit
2); `runs ls` is a hidden alias of `runs list`; `runs show --json` prints
`{meta, repro, metrics, artifacts}`. A malformed run id is a usage error (exit 2); a
well-formed id that is not in the store is a failure (exit 1, `command_failed`), following
the exit-code table in [06-cross-cutting.md](06-cross-cutting.md). A store written by a newer
library exits 1 with `store_version_unsupported`. `repro` prints a bash script (checkout of
`git_commit`, `git apply` of the stored diff, environment and dataset notes as comments, the
recorded command) on `stdout`.

Planned commands (each documents its exit codes in `--help` and in [04-flows.md](04-flows.md)):

```
cvpilot runs ls [--project P] [--status S] [--json]
cvpilot runs show <run_id> [--json]
cvpilot eval <run_id> --preds preds.parquet --gts gts.parquet --classes names.txt [--slices key=column]... [--step N] [--seed N] [--resamples N] [--json]
cvpilot compare <run_id> --baseline (tag:<tag>|<run_id>) [--eval-set HASH] [--step N] [--json]
cvpilot gate <run_id> --baseline ... [--max-drop K=V]... [--per-class-max-drop K=V]...
             [--slice-max-drop "key=*:V"]... [--min-images N] [--skip-small-slices] [--json]
cvpilot baseline set <tag> <run_id> [--eval-set HASH] [--step N] [--replace] | list [--history] | show <tag>
cvpilot ckpt list <dir> | verify <dir> | prune <dir> --run ID   # built; [--require-trust any|anchored|authenticated] [--project P] [--json]
cvpilot repro <run_id>
cvpilot sync [--project P] [--once]
cvpilot import (trackio|mlflow|tensorboard) <path> --project P
cvpilot server [--host H] [--port N]        # requires the [server] extra
```

Depends on: application packages (`runs/`, `repro`), `settings.py`, `logging.py`. Status:
built (`--version`, `doctor`, `runs list`, `runs show`, `repro`, `eval`, `baseline set|list|show`, `compare`, `gate`, `ckpt list|verify|prune`; `eval-file` is a hidden alias); planned (all other commands).

### `api.py`

Responsibility: the public Python API; holds the active run for the process; converts user
inputs into domain objects; delegates to `runs/`, `eval/`, `ckpt/`, `alerts/`.

```python notest
def init(
    project: str | None = None,          # default: CVPILOT_PROJECT
    *,
    task: Literal["detection", "classification"],
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    dataset_uri: str | PathLike[str] | None = None,
    seed: int | None = None,
    framework: str | None = None,        # "<name>==<version>", default the running Python
    resume_from: str | None = None,      # run id of a preempted or failed run
    store_dir: str | PathLike[str] | None = None,   # default: CVPILOT_STORE_DIR
    forward_to: Sequence[str] | None = None,        # default: CVPILOT_FORWARD_TO
) -> Run                                 # built (with `forward_to`); `resume: bool` is planned
def log(points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None
def log_eval(
    preds: PredictionTable,
    gts: AnnotationTable,
    *,
    step: int,
    class_names: Sequence[str] | None = None,   # default: ["0", ...] up to the largest GROUND-TRUTH id
    slices: Mapping[str, str] | None = None,
) -> EvalSnapshot
def log_artifact(path: str | PathLike[str], *, kind: ArtifactKind, meta: Mapping[str, Any] | None = None) -> Artifact
def finish(status: Literal["finished", "failed"] = "finished") -> None
def autolog(target: object, *, project: str | None = None, forward_to: Sequence[str] | None = None,
            **init_kwargs: Any) -> Run                  # init_kwargs needs task=
def resume_path(project: str, *, checkpoint_dir: str | PathLike[str] | None = None) -> Path | None

class Run:                        # handle returned by init(); same methods bound to one run
    run_id: str
    def log(self, points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None: ...
    def log_eval(self, preds: PredictionTable, gts: AnnotationTable, *, step: int, **kw: Any) -> EvalSnapshot: ...
    def finish(self, status: Literal["finished", "failed"] = "finished") -> None: ...
```

`autolog(target)` resolves the hook by inspecting `type(target)` against the `cvpilot.hooks`
registry and calls `hook.attach(target, run)`; it raises `HookNotFoundError` when no hook matches
and `MissingExtraError` naming the extra when the framework's package is missing (built).

One run is active per process (`RunAlreadyActiveError` otherwise); `Run` is a context manager
(`failed` on an exception). An `atexit` hook flushes buffered points but leaves the status
`running`, so a process that dies without `finish` is visibly unfinished.

Depends on: `runs/`, `repro`, `core/schema.py`, `settings.py`, `logging.py`; later `eval/`,
`ckpt/`. Status: built (`init`, `log`, `log_artifact`, `log_eval`, `finish`, `Run`, `autolog`, `forward_to`, `resume_path`).

## Application layer

### `runs/`

Responsibility: run lifecycle use cases; the `MetricSink` implementation handed to hooks;
status transitions; hand-off of every record to the store, the forwarders, the outbox and
the client-side alert rules.

```python notest
def start_run(meta: RunMeta, repro: Repro, *, store_dir=None, git_diff: bytes | None = None) -> RunHandle
def log_points(run: RunHandle, points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None
def finish_run(run: RunHandle, status: Literal["finished", "failed", "preempted"]) -> None
def resume_from(parent: RunMeta, *, run_id: str, host: str, name: str | None) -> RunMeta
def recover_run(project: str, run_id: str, *, store_dir=None) -> None   # running -> failed
def list_runs(project: str, *, status: RunStatus | None = None, store_dir=None) -> list[RunMeta]
def get_run_view(project: str, run_id: str, *, store_dir=None) -> RunView   # meta, repro, stats, artifacts

class RunHandle:                  # implements MetricSink.log and log_artifact
    def log(self, points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None
    def flush(self) -> None
    def finish(self, status: Literal["finished", "failed", "preempted"] = "finished") -> None
```

After a flush the batch is offered to the forwarders (`runs/forwarding.py`): a non-blocking
`put_nowait` on a bounded queue per forwarder, each served by its own daemon thread; `finish`
waits at most `CLOSE_TIMEOUT_S` (5 s) for them. `log` validates the whole call, buffers at most 10 000 points and writes one Parquet batch on
the caller's thread when that bound is reached or 2 s have passed since the last flush
(checked on each call; there is no timer thread). The longest a training step waits on I/O is
one batch write. A failed write keeps the points buffered and raises `StoreError`. A step must
be strictly greater than the last step logged for the same key (`DuplicateMetricError` for an
equal step, `MetricOrderError` for a lower one). `finish` flushes, compacts the metric files,
sets the final status and logs `run_finished`; a run whose process died stays `running` until
`recover_run` marks it `failed`. `resume_from` accepts only `preempted` or `failed` parents.

Depends on: `core/`, registry `stores`. Status: built (lifecycle, queries, recovery, resume
metadata, forwarder fan-out); planned: the outbox and client-side alerts.

### `eval/`

Responsibility: compute an `EvalSnapshot` from predictions and ground truth; per-class,
confusion with background, slices, size ranges (small, medium, large AP and AR); bootstrap confidence intervals over images;
record the result in a run.

```python notest
def evaluate(
    preds: PredictionTable, gts: AnnotationTable, *,
    task: Literal["detection", "classification"], class_names: Sequence[str],
    slices: Mapping[str, str] | None = None,      # slice key -> image metadata column
    iou_thresholds: Sequence[float] | None = None,  # COCO grid by default
    n_resamples: int = 1000, seed: int = 0, level: float = 0.95, workers: int = 4,
    confusion_iou: float = 0.5, confusion_score: float = 0.25,
) -> EvalResult                    # .to_snapshot(run_id=, step=, confusion_sha256=, predictions_sha256=)
def eval_set_hash(gts: AnnotationTable, class_names: Sequence[str], *, task=...) -> str
def bootstrap_ci(per_image: np.ndarray, *, n_resamples: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]
def record_eval(store, meta, preds, gts, *, step, class_names=None, slices=None, ...) -> EvalSnapshot
```

Inputs are `PredictionTable`, `AnnotationTable` and `ImageTable` (`eval/tables.py`): NumPy
columns built from lists, arrays or Arrow tables (`from_columns`, `from_table`). Image ids are
strings; boxes are `x1, y1, x2, y2`. The `AnnotationTable` carries the image list (including
images without boxes, sizes and metadata columns for slices). `eval/validate.py` rejects
non-finite scores and boxes, reversed boxes, class ids outside `[0, C)`, duplicated image ids,
predictions or labels for images outside the evaluation set, ground-truth boxes outside the
image, and counts above the named limits (`MAX_PREDICTIONS` and `MAX_ANNOTATIONS` 5 000 000,
`MAX_IMAGES` 1 000 000, `MAX_ANNOTATIONS_PER_IMAGE` 100 000, `MAX_CLASSES` 2 048,
`MAX_SLICE_VALUES` 256 per key, `MAX_SLICE_KEYS` 16, `MAX_RESAMPLES` 10 000). A slice key with
more values raises `SliceCardinalityError` before any metric is computed.

`record_eval` (arguments `allow_closed`, `provenance`; used by `RunHandle.log_eval`, `api.log_eval` and `cvpilot eval`) hashes the
ground truth, refuses an existing `(run, hash, step)`, evaluates, stores `predictions` and
`confusion` Parquet artifacts (confusion as sparse `(true_class, pred_class, count)` rows,
index `C` is background) and then the snapshot, and logs `eval_completed`. `eval/files.py`
reads the CLI's Parquet files through the `tables` registry (bounded: file size, rows, row
groups, decoded bytes, string widths).

The bootstrap resamples images with a seeded generator. Matching is per image, so one matching
pass serves every resample: weights are image multiplicities (`core.metrics.accumulate`
accepts them, slices use 0/1 weights) and a vectorised resampler works on true positives only.
Intervals cover `mAP50_95`, `mAP50`, `AR100` (detection) and `top1` (classification).

Deviations from the first sketch: `evaluate` returns an `EvalResult` (a snapshot needs a run id
and artifact hashes), takes the image list inside `AnnotationTable`, and slices map a key to a
metadata column. Metric values are fractions in [0, 1]; undefined ones are omitted. Parity with
pycocotools (tolerance 1e-4) is a test in the default suite, `tests/parity/`; the definitions are
in the `core/metrics.py` docstring. Depends on: `core/metrics.py`, `core/confusion.py`,
`core/hashing.py`, `core/schema.py`; store and Parquet I/O through the `stores` and `tables`
registries. Status: built.

### `compare/` and `gate`

Responsibility: pin baselines, load two verified snapshots, refuse when they are not
comparable, compute headline, per-class, slice and confusion deltas, evaluate thresholds and
return the exit code. Semantics and trade-offs: [ADR-0007](../adr/0007-compare-and-gate-semantics.md).

```python notest
@dataclass(frozen=True)
class Thresholds:
    max_drop: Mapping[str, float]              # {"mAP50_95": 0.5}, points on the 0..100 scale
    per_class_max_drop: Mapping[str, float]    # {"AP50_95": 2.0}
    slice_max_drop: Mapping[str, float]        # {"camera=*": 1.5} or {"camera=cam03": 1.5}
    min_images: int = 30
    skip_small_slices: bool = False            # explicit opt-in; the default fails small slices
    max_skipped: int = 0                       # most slices the opt-in may skip

def compare(candidate: EvalSnapshot, baseline: EvalSnapshot, *,
            confusion: ConfusionDelta | None = None) -> Comparison
def gate(cmp: Comparison, thresholds: Thresholds) -> GateResult   # .passed, .violations, .skipped
def resolve(project, run_id, baseline_ref, *, eval_set=None, step=None, store_dir=None,
            require_commit=None, expect_chain=None,
            recompute: RecomputeSpec | None = None) -> Resolved   # verified snapshots + confusion delta
def verify(store, snapshot, spec: RecomputeSpec) -> None          # compare/recompute.py
def set_baseline(project, tag, run_id, *, eval_set=None, step=None, replace=False, store_dir=None) -> Baseline
```

`compare()` raises `CompareRefusedError` (`EvalSetMismatchError` for a different
`eval_set_hash`) unless `eval_set_hash`, `task`, `class_names` (order included), the
evaluation parameters and `slice_hash` are equal, every slice holds the same number of images and every class has the same support (`class_support`); snapshots without recorded parameters are refused because
they cannot be checked. The CLI logs `compare_refused` with `reason` and both hashes and exits
1; no result or report exists. Deltas are `(candidate - baseline) * 100` points, negative
being a regression. Headline deltas carry the stored bootstrap intervals and whether they
overlap (`overlap`, `separated`, `unknown`); `compare` does no resampling, so it is
deterministic and its cost does not depend on the image count. Per-class deltas are sorted
worst first with the support of both sides; unusable rows (`missing`, `not_finite`) come before
the measured ones, classes without ground truth (`not_applicable`) last. Slice deltas flag
absent slices and slices built from another column (`slice_spec`). The confusion delta
(`compare/confusion.py`) reads both confusion artifacts through `open_artifact` (digest verified),
bounds size, rows and values, and reports the changed cells largest first plus the change in
missed ground truth (`[true, background]`) and spurious detections (`[background, pred]`).

`resolve()` additionally refuses a baseline that is the candidate or not `finished`
(`baseline_is_candidate`, `baseline_not_finished`), a candidate whose `provenance.git_commit`
differs from `--require-commit` or is absent (`commit_mismatch`, `provenance_missing`), a tag baseline
whose table head differs from `--expect-baseline-chain` (`BaselineIntegrityError`) and, with
`--recompute`, any stored result that differs from a re-evaluation of the stored predictions on
trusted ground truth (`RecomputeMismatchError`, event `gate_recompute_failed`); the confusion
artifacts are verified on every call. See the trust model in
[06-cross-cutting.md](06-cross-cutting.md).

`gate()` fails closed (module docstring of `compare/gate.py`): an absent or non-finite
metric, an absent, mismatched or too small slice, an evaluation set below `min_images`, a
per-class threshold that gated no class and a slice threshold that matched nothing are all
violations. At least one threshold is required. `resolve()` never reads a stored comparison or
report: snapshots come from `get_snapshot` (file digest and run chain verified), a tag from
`get_baseline` (record, baseline chain and pinned snapshot digest verified). Text output
(`compare/render_text.py`) passes every name through `safe_text` and starts each line with
`gate:` or two spaces so a class name cannot become a CI workflow command.

Reports (`compare/html_report.py`, `compare/report.py`): `--report PATH` writes a static HTML
page and `--report-json PATH` (both restricted by `--report-dir`, `--allow-outside` and `--force`, ADR-0007) the canonical JSON (`compare` and `gate` both take them; `--json`
prints the JSON). The page is built from element constructors that escape every string child
(names also pass `safe_text`), has no script, no external resource, no link and no data-carrying
attribute, and declares `default-src 'none'; style-src 'sha256-<its one inline style>';
base-uri 'none'; form-action 'none'` plus `no-referrer`. Content is severity first: violations
(at most 100, worst first), headline, per-class tables (50 rows each, worst first), slices (100)
and changed confusion cells (30); each table says how many rows it left out and the JSON has
them all. Neither format carries a timestamp, so equal inputs give equal bytes
(`tests/golden/test_report_golden.py`). Files are written atomically (temp file, fsync, rename,
mode 0644), refuse a symbolic link or a non-regular target and are limited to 8 MiB; a report
that cannot be written fails the command, also when the gate passed, and a refused comparison
writes none.

Deviations from the first sketch: `compare()` has no `seed` (no resampling, see above); there
is no prediction-diff gallery (out of scope for this phase, `max_gallery_images` does not
exist); `baseline rm` does not exist (pins are append-only); the report is written only to the
path given and is not registered as an artifact, so `gate` never writes to the store. Depends
on: `core/`, the `stores` and `tables` registries. Status: built (baselines, compare, gate,
JSON and HTML reports); planned: the prediction-diff gallery.

### `ckpt/`

Responsibility: checkpoint manager semantics from [ADR-0006](../adr/0006-checkpoint-semantics.md)
(including its amendment). Status: **built** and covered by unit, golden-style, chaos and
property-style tests.

```python notest
class CheckpointManager:
    def __init__(
        self,
        run: Run,
        *,
        dir: str | PathLike[str] | None = None,       # default: CVPILOT_CHECKPOINT_DIR
        remote: str | None = None,                    # default: CVPILOT_CHECKPOINT_REMOTE
        remote_store: RemoteStore | None = None,      # inject a store (tests, other backends)
        keep_last: int = 3,
        keep_best: int = 2,
        monitor: str | None = None,
        mode: Literal["max", "min"] = "max",
        every_n_minutes: float | None = None,         # none given: every 10 minutes
        every_n_steps: int | None = None,
        signals: Sequence[int] | None = None,         # default SIGTERM, SIGUSR1
        grace_seconds: float = 30.0,
        format: Literal["safetensors", "pt"] = "safetensors",
        allow_pt_format: bool | None = None,          # CVPILOT_CKPT_ALLOW_PT; pt needs it
        require_trust: Literal["any", "anchored", "authenticated"] | None = None,
        notice_source: NoticeSource | str | None = None,   # "aws", "gcp"; off by default
        ...                                           # limits, retries, clocks: see the API snapshot
    ) -> None
    def maybe_save(self, step, *, model, optimizer=None, scheduler=None, scaler=None,
                   rng=True, dataloader_state=None, metrics=None, ema=None) -> CheckpointManifest | None
    def save_now(self, step, *, kind="last", **state) -> CheckpointManifest
    def adopt(self, path, step) -> CheckpointManifest          # a framework-written file, as pt
    def restore(self, kind="last", *, map_location="cpu", run_ids=None) -> Restored | None
    def restore_latest_valid(self, *, map_location="cpu") -> dict[str, Any] | None
    def verify(self, manifest) -> bool
    def list(self) -> list[Entry]
    def preempted(self) -> bool
    def close(self) -> None
```

Modules: `directory.py` (`RunCheckpoints`: one run's directory, atomic save, the trust rules, the
walk back, retention), `fsops.py` (directory-descriptor file operations, no symlinks, bounded,
`O_EXCL`), `serializers.py` (safetensors with a strict one-pass header parser, opt-in `pt` with a
strict archive reader), `state.py` and `rng.py` (what a checkpoint holds and how it is validated
on restore), `auth.py` (HMAC tags), `preempt.py` (flag, signal guard, notice poller),
`remote.py` (`Mirror`: upload, verify, resume, fetch), `resume.py` (`resume_path`,
`find_resume_point`). `store/checkpoints.py` keeps the digest of every manifest as the external
anchor. Deviations from the first sketch: the `.pt` format is named `pt` and is opt-in, `last`
checkpoints are safetensors files with the non-tensor state as bounded JSON, the pointers are
`LATEST` and `LATEST_BEST`, `restore` returns a `Restored` (state, manifest, run, skipped,
trust level) and `restore_latest_valid` only its state, and `ckpt restore` does not exist (the
command is `ckpt verify`, `list` and `prune`).

Depends on: `core/`, the `remotes`, `spot_notices` and `stores` registries (through `runs/`),
`logging.py`, `settings.py`; the framework only for `pt`.

### `alerts/`

Responsibility: rule engine over the stream of metric points, snapshots and status changes;
dedupes and rate-limits; hands `Alert` objects to channels. Runs inside the training process
and inside the server.

```python notest
class Rule(Protocol):
    name: str
    def feed(self, event: RunEvent) -> Alert | None: ...

RULES: Mapping[str, type[Rule]]   # nan_loss, loss_spike, no_progress, gpu_util_below,
                                  # eval_regression_vs_baseline, new_best, failed, preempted, resumed
def evaluate(rules: Sequence[Rule], event: RunEvent) -> list[Alert]
def deliver(alerts: Sequence[Alert], channels: Sequence[str]) -> None   # via `channels` registry
```

Depends on: `core/schema.py`; `notify/` through the `channels` registry. Status: planned.

### `repro`

Responsibility: assemble the `Repro` record at `init` and print a reproduction script.

```python notest
def capture_repro(run_id, dataset_uri: str | None, config, seed, *, store_command_line: bool = True,
                  argv=None, cwd=None) -> Captured        # Repro + diff bytes + host
def script(view: RunView) -> str                          # bash: checkout, diff, env notes, command
```

`store_command_line` is passed from `Settings.store_command_line`; capture never reads the
environment itself. Depends on: `core/hashing.py`, `core/schema.py`, `core/jsonable.py`;
`capture/` through the `capturers` registry. Status: built.

## Domain layer

### `core/schema.py`

Responsibility: the Pydantic models `RunMeta`, `Repro`, `MetricPoint`, `EvalSnapshot`,
`Artifact`, `CheckpointManifest`, `Baseline`; the `MetricSink` protocol hooks talk to; the
`RunEvent` union consumed by alert rules; validators for the invariants in
[03-data-model.md](03-data-model.md).

```python notest
class MetricSink(Protocol):
    def log(self, points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None: ...
    def log_eval(
        self, preds: PredictionTable, gts: AnnotationTable, *, step: int
    ) -> EvalSnapshot: ...
    def log_artifact(
        self, path: Path, *, kind: ArtifactKind, meta: Mapping[str, Any] | None = None
    ) -> Artifact: ...
    def checkpoint_saved(self, manifest: CheckpointManifest) -> None: ...
```

Phase 1 built `RunMeta`, `Repro`, `MetricPoint`, `Artifact`, `MetricStats`, the `RunStore`
port that `runs/` uses and `MetricSink` with `log` and `log_artifact`; `EvalSnapshot`,
`CheckpointManifest` and `CheckpointPointer` (built with `ckpt/`), `Baseline` and the `RunStore`
methods `put_manifest`/`get_manifest_digest` are built; `RunEvent` and `MetricSink.log_eval/checkpoint_saved` are
planned (the manager records through the run handle instead). `Repro` redacts `command_line` and `config` on construction
(`core/redact.py`, [06-cross-cutting.md](06-cross-cutting.md)). `core/ulid.py` generates
monotonic ULIDs, `core/errors.py` holds the specific exceptions the CLI boundary reports.

Depends on: pydantic, stdlib. Status: built (subset above); planned (rest).

### `core/hashing.py`

```python notest
def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str
def dataset_hash(dataset_uri: str) -> str      # "dcp:<version>" or "merkle:<hex>"
def merkle_hash(root: Path) -> str
def eval_set_hash(class_names, image_ids, widths, heights, ann_image, ann_cls, ann_boxes, ann_crowd) -> str
def new_run_id() -> str                         # ULID
```

Depends on: stdlib. Status: built (`sha256_file`, `dataset_hash`, `merkle_hash`,
`new_run_id`, `eval_set_hash`).

### `core/metrics.py` and `core/confusion.py`

Pure functions on NumPy arrays; no I/O, no framework types.

```python notest
def match_detections(pred_image, pred_boxes, pred_scores, pred_cls, gt_image, gt_boxes, gt_cls, *,
                     n_images, n_classes, iou_thresholds, iscrowd=None, max_dets=100) -> MatchResult
def accumulate(match, *, image_weights=None, areas=None, max_dets=None) -> Curves  # [T, R, K, A, M]
def summarize(curves, iou_thresholds=None) -> dict[str, float]   # the 12 COCO numbers, -1 undefined
def per_class_ap(curves, *, iou=None, area="all") -> np.ndarray
def confusion_matrix(pred_image, pred_boxes, pred_scores, pred_cls, gt_image, gt_boxes, gt_cls, *,
                     n_images, n_classes, iscrowd=None, iou=0.5, score=0.25) -> np.ndarray  # (C+1)x(C+1)
def classification_metrics(true_cls, pred_cls, *, n_classes) -> ClassificationResult
```

Image indices are part of the signatures (matching is per image), which the first sketch left
out. Depends on: numpy (`core/schema.py` and `core/hashing.py` import it lazily so the CLI
start-up stays light). Status: built.

## Infrastructure layer

### `store/`

Responsibility: the local run store of [ADR-0005](../adr/0005-run-store.md): SQLite WAL for
rows, Parquet for metric points, content-addressed artifacts, DuckDB queries, migrations,
outbox table for `sync/`.

```python notest
class RunStore(Protocol):
    def open(project: str, root: Path | None = None) -> "RunStore": ...
    def put_run(self, meta: RunMeta, repro: Repro) -> None
    def set_status(self, run_id: str, status: RunStatus, ended_at: datetime | None) -> None
    def append_points(self, points: Sequence[MetricPoint]) -> None
    def put_snapshot(self, snapshot: EvalSnapshot) -> None
    def put_artifact(self, run_id: str, path: Path, kind: ArtifactKind, meta: Mapping[str, Any]) -> Artifact
    def put_manifest(self, manifest: CheckpointManifest) -> None
    def put_baseline(self, baseline: Baseline, *, replace: bool = False) -> None   # append-only, digested, chained
    def get_baseline(self, tag: str) -> Baseline                  # verified on every read
    def list_baselines(self, *, history: bool = False) -> list[Baseline]
    def snapshot_digest(self, run_id: str, eval_set_hash: str, step: int) -> str
    def get_snapshot(self, run_id: str, eval_set_hash: str, step: int | None = None) -> EvalSnapshot
    def query(self, sql: str, params: Mapping[str, Any] = {}) -> pa.Table      # DuckDB over runs.sqlite + metrics/*.parquet
    def outbox_pull(self, limit: int) -> list[OutboxItem]
    def outbox_ack(self, run_id: str, seq: int) -> None
```

Built (`store/local.py`, adapter `local` of the `stores` registry): `put_run`, `get_run`,
`get_repro`, `list_runs`, `set_status`, `append_points`, `read_points`, `metric_stats`,
`compact_metrics`, `put_artifact`, `put_artifact_bytes`, `list_artifacts`, `artifact_path`,
`put_snapshot`, `get_snapshot`, `list_snapshots`, `has_snapshot`, `snapshot_digest`, `put_baseline`, `get_baseline`, `list_baselines`, migrations (`store/migrations/0001_initial.py`), validated paths (`store/paths.py`). The
`RunStore` protocol lives in `core/schema.py` so `runs/` can type against it without
importing `store/`. The `eval_snapshot` table is additive (`store/additive.py`, created idempotently
at open, schema version unchanged; the database refuses UPDATE and DELETE on it). Parquet
reading for evaluation lives in `store/tables.py` (adapter `parquet` of the `tables` registry,
protocol `TableIO` in `core/schema.py`). Planned: `put_manifest`, `query`
(DuckDB), `outbox_*`.

Depends on: `core/schema.py`; sqlite3 and pyarrow (imported inside functions). Status: built
(run, metric, artifact and eval snapshot paths, Parquet tables); planned (the rest).

### `capture/`

Environment capture in `capture/env.py`: `git` (commit, dirty flag over tracked files, diff
of tracked changes with fsmonitor, external diff and textconv disabled), lockfile hash
(`uv.lock` or `poetry.lock` up to the repo root, else a hash of the installed set), Python,
CUDA and driver and GPU names (`pynvml` if importable, else `nvidia-smi`, else none; `torch`
is never imported), hostname, and the command line redacted per argument (`redact_argv`)
before quoting. Each capturer is a function registered under `capturers`, with optional
keyword parameters (`cwd`, `argv`, `store_command_line`); absent facilities give `None` or
empty values, never errors. Depends on: stdlib, `pynvml` (imported lazily). Status: built;
planned: the NVML sampler thread (every 10 s).

### `hooks/`

Framework adapters. Each adapter implements (`Hook` in `core/schema.py`):

```python notest
class Hook(Protocol):
    name: str  # "lightning", "ultralytics"; planned: "hf", "mmengine", "rfdetr", "torch"
    requires: tuple[str, ...]  # distributions that must be installed; "a|b" means either
    license_notice: str | None  # set for a copyleft framework

    def matches(self, target: object) -> bool: ...
    def attach(self, target: object, sink: MetricSink) -> None: ...
    def detach(self) -> None: ...
```

| Adapter | Mechanism | Captured | Status |
|---|---|---|---|
| `hooks/lightning.py` | `CVPilotLogger` (a Lightning `Logger`, scalars of `self.log`, rank zero only) plus `CVPilotCallback` (`on_fit_start` rank check, `on_fit_end` finishes, `on_exception` marks the run failed), both built when the hook is attached | scalars; hyperparameters are not forwarded | built |
| `hooks/hf_trainer.py` | `TrainerCallback`: `on_log`, `on_evaluate`, `on_save` | scalars, eval, checkpoints | planned |
| `hooks/mmengine.py` | MMEngine hook plus `VisBackend` | scalars, COCO metrics, checkpoints | planned |
| `hooks/rfdetr.py`, `hooks/torch_loop.py` | context manager plus explicit `log_eval` | explicit | planned |
| `hooks/ultralytics.py` (`cvpilot[ultralytics]`, AGPL-3.0 notice) | `add_callback("on_fit_epoch_end" / "on_train_end")` on the model; reads `trainer.metrics`, the loss items and `validator.metrics.box.maps` with `validator.names` | losses, mAP, per-class AP (`val/AP/<class>`), step = epoch | built; `resume_arguments(point)` gives the `resume=` target for a `pt` file `resume_path` verified here (`hooks/lightning.py` has the same for `ckpt_path`; safetensors files and unverified paths are refused); the confusion matrix and automatic adoption of framework checkpoints are planned |

`api.autolog(target, project=..., forward_to=..., task=...)` finds the hook whose `matches`
accepts the object (from the `hooks` registry, which imports no framework), checks its
`requires` against package metadata (`MissingExtraError` naming the extra; `hook_missing`),
starts a run (`init`) and attaches; `HookNotFoundError` when nothing matches, no run is started.
`hook_attached` carries the hook, the framework version and the license notice.

Lifecycle callbacks (finish, mark failed) are exempt from the disabled state of the metric guard (`always=True`), so ten failed logs cannot leave a run `running`. The Lightning hook finishes the run after the last stage (`test` or `predict`), not after `fit`, so `trainer.test()` metrics reach the store; a script that stops after `fit` or `validate` is finished by an `atexit` handler when training completed without an exception; every stage (`setup`) is rank-gated; the hook and `Run` refuse to be pickled with a clear message (`ddp_spawn` is not supported). The YOLO hook logs the epoch metrics and per-class AP only for epochs in which a validation ran (`on_val_end`), always logs the losses, and reports `hook_no_epochs` when `train` returns without any epoch callback (distributed training runs in child processes that carry no callbacks and is not supported).

Rules every hook follows (`hooks/__init__.py`, tests `test_hooks_*.py`; the YOLO hook also wraps the model's `train`, because the framework has no failure event: an exception or interrupt marks the run failed and propagates unchanged): callbacks never raise
into the host (a `Guard` catches `Exception`, counts it, logs the first as `hook_failed` with the
error type only, and disables the hook after 10 failures in a row with `hook_disabled`;
`KeyboardInterrupt` and `SystemExit` pass through); callbacks make no network call and block the
host for at most one buffered batch write (`sink.log`), because forwarders run on their own
threads; the framework's objects are read through narrow views that touch only the attributes
named in the module docstring, convert values defensively (`NaN` and `inf` pass, text, booleans
and failing tensors are skipped and counted), cap a call at 1 000 points, and a key is logged
only at a step greater than its last one, because the store requires strictly increasing steps.

The copyleft framework: it is declared only in the `[ultralytics]` extra, is never imported by
this package (the hook receives the framework's objects through `add_callback`), no module but
`hooks/ultralytics.py` may mention it (a layering test parses every module), and the adapter
carries the AGPL-3.0 notice in its docstring, in `license_notice` and in the `hook_attached`
event. The name of the framework is spelled in parts in source files because the repository QA
gate rejects the literal name in Python source. Depends on: `core/schema.py`, `logging.py`.

### `forward/`

```python notest
class Forwarder(Protocol):  # core/schema.py
    name: str  # "mlflow", "wandb", "trackio" (TensorBoard is planned)
    requires: tuple[str, ...]  # "a|b" means either distribution

    def open(self, meta: RunMeta, repro: Repro) -> None: ...
    def log_points(self, points: Sequence[MetricPoint]) -> None: ...
    def close(self, status: RunStatus) -> None: ...


# constructed as cls(allow_insecure=<CVPILOT_ENV == "dev">, allowed_hosts=<CVPILOT_FORWARD_HOSTS>);
# no token argument exists
```

Adapters: `forward/mlflow.py` (`cvpilot[mlflow]`, `mlflow-skinny`), `forward/wandb.py`
(`cvpilot[wandb]`), `forward/trackio.py` (`cvpilot[trackio]`), registered under
`cvpilot.forwarders`. Each imports its SDK inside `open()` only and raises
`MissingExtraError("... install cvpilot[<name>]")` when it is absent; the registry-level
check (`requires`) raises the same error at `init()` before any run exists. Endpoint and
credentials come from the SDK's own environment variables (`MLFLOW_TRACKING_URI`,
`WANDB_API_KEY`, ...): an adapter reads no credential, accepts none, stores none and logs
none. At construction (on the training thread, reading the environment once and passing the
checked values to the SDK explicitly) it accepts only an `https` endpoint (`http` in dev; no
other scheme), without credentials, on a host of `CVPILOT_FORWARD_HOSTS` when that is set, and
refuses the SDK switches that disable TLS verification (`MLFLOW_TRACKING_INSECURE_TLS`,
`WANDB_INSECURE_DISABLE_SSL`) and the certificate-authority overrides outside dev; the message
names the variable and never the value. Trackio passes no endpoint, so it cannot verify its
destination and refuses to run when an allow-list is set (ADR-0008). Adapters send the run id, project, task, parent run id and the metrics
(MLflow: tags; W&B and Trackio: a small config), never `Repro.config`, the command line, the
dataset path or artifacts. Keys: MLflow replaces characters it does not accept with `_` (steps
and float64 values are unchanged); W&B requires non-decreasing steps, so points are sent grouped
by ascending step and points older than the last forwarded step are counted in
`dropped_points`, reported to the fan-out (`adapter_dropped_points`, a one-time `forward_failed` with reason `step_regressed`, and the `forward_summary`), together with the points of any failed `log_points` call (`failed_points`). Artifact forwarding is not built (predictions can reveal the dataset; it will
be opt-in).

The fan-out lives in `runs/forwarding.py` (ADR-0008): a bounded queue (256 batches and 200 000 points), a daemon worker thread and
a stall timeout per forwarder, so a failing, slow or hung tracker is counted and logged once and
never raises into or blocks the training loop. Depends on: `core/`; the tracker SDK inside the
adapter. Status: built (`mlflow`, `wandb`, `trackio`); planned: TensorBoard.

### `sync/`

Background thread that pulls the outbox, posts batches to `CVPILOT_SERVER_URL` over HTTPS
with the token in a header, retries with exponential backoff (1 s to 60 s, jitter), and acks
`(run_id, seq)` on HTTP 200 or 409 (already stored). Uses stdlib `urllib.request` with a
10 s timeout; no extra needed. Depends on: `core/schema.py`, `settings.py`. Status: planned.

### `remote/`

Infrastructure adapters. `fsspec_store.py`: `RemoteStore` over fsspec for checkpoint upload and
download, registered under `remotes` for the schemes `s3`, `gs`, `az` and `memory`; TLS on, bounded
timeouts, the backend's retries turned off (the caller owns them), no credentials, validated keys,
backend exceptions replaced by an `OSError` that names only the type. `spot.py`: AWS and GCP
preemption notices, registered under `spot_notices` (fixed metadata hosts, no redirects, bounded
answers). Depends on: `fsspec` and the backend package (`s3fs`, `gcsfs`, `adlfs`) imported lazily.
Status: built for checkpoints (artifact export is planned).

### `notify/`

Channel adapters: Telegram Bot API, Slack incoming webhook, generic webhook, SMTP.

```python notest
class Channel(Protocol):
    name: str
    requires: tuple[str, ...]

    def send(self, alert: Alert) -> None: ...
```

Depends on: `core/schema.py`; `httpx` from the `[alerts]` extra for HTTP channels, stdlib
`smtplib` for SMTP. Status: planned.

### `server/`

Its own package with its own entry point (`cvpilot server`), shipped as a Docker image.
Imports `core/` only; never `runs/`, `store/` or `cli/`. Routes: `POST /api/v1/runs`,
`POST /api/v1/runs/{run_id}/events` (idempotent on `(run_id, seq)`), `GET /api/v1/runs`,
`GET /api/v1/runs/{run_id}/snapshots`, `WS /ws/runs/{run_id}`, `GET /share/{token}`, htmx
pages under `/ui`. Every route requires a project token except `/healthz` and valid share
links. Depends on: `core/schema.py`; FastAPI, Uvicorn, SQLAlchemy (SQLite or Postgres)
from the `[server]` extra. Status: planned.

## Cross-cutting

- `logging.py` (built): structlog processor chain per
  [errors-and-logging.md](../engineering/errors-and-logging.md).
- `settings.py` (built): `CVPILOT_ENV`, `CVPILOT_LOG_LEVEL`, `CVPILOT_PROJECT`,
  `CVPILOT_STORE_DIR`, `CVPILOT_STORE_COMMAND_LINE`; planned keys in
  [05-runtime.md](05-runtime.md).

## Registries and the plugin contract

`cvpilot/registry.py` (built; `stores` and `capturers` are populated, the other groups arrive
with their packages) exposes `Registry[T]` with `get(name) -> T`,
`names() -> list[str]` and `find(predicate) -> T | None`. Each registry is populated lazily
from entry points on first use; first-party adapters are declared in `pyproject.toml` under
the same groups, so first- and third-party adapters are indistinguishable to application
code.

| Registry | Entry-point group | Key | Plugin declares |
|---|---|---|---|
| `hooks` | `cvpilot.hooks` | hook name | a class implementing `Hook` with `name`, `requires`, `matches`, `attach`, `detach` |
| `forwarders` | `cvpilot.forwarders` | tracker name | a class implementing `Forwarder` |
| `channels` | `cvpilot.channels` | channel name | a class implementing `Channel` |
| `remotes` | `cvpilot.remotes` | URL scheme | a callable returning an fsspec filesystem |
| `capturers` | `cvpilot.capturers` | field group | a callable `() -> Mapping[str, Any]` |
| `stores` | `cvpilot.stores` | store name (`local` default) | a class implementing `RunStore` |

Rules for a plugin:

1. `requires` lists distribution names; the registry checks `importlib.metadata` before
   importing the module and raises `MissingExtraError("install cvpilot[lightning]")` or the
   plugin package name.
2. A plugin imports `cvpilot.core` only. Importing `cvpilot.runs`, `cvpilot.store` or
   `cvpilot.cli` from a plugin is rejected in review and by the import-linter contract.
3. A plugin that includes or derives from copyleft code is a separate distribution; a
   plugin that only imports a copyleft package lazily may live in this package as an extra.

## Dependency rule in this project

Forbidden imports, checked by review and by an import-linter contract once more than one
layer is populated:

- `core/*` importing anything from `cvpilot` outside `core/`, or importing `torch`,
  `pyarrow`, `duckdb`, `fsspec`, `fastapi`, any framework or tracker SDK.
- `store/`, `capture/`, `hooks/`, `forward/`, `sync/`, `remote/`, `notify/`, `server/`
  importing `runs/`, `eval/`, `compare/`, `ckpt/`, `alerts/`, `repro`, `api.py` or `cli/`.
- `runs/`, `eval/`, `compare/`, `ckpt/`, `alerts/`, `repro` importing `store/`, `capture/`,
  `hooks/`, `forward/`, `sync/`, `remote/`, `notify/` or `server/` by module path (they go
  through `registry.py`).
- `cli/` or `api.py` importing any infrastructure package directly.
- Any module importing `torch`, a framework SDK, a tracker SDK, `fastapi`, `httpx` or
  `pynvml` at module top level; these imports live inside the adapter function that needs
  them.
- `logging.py` and `settings.py` importing any other `cvpilot` module.
- Any module other than `hooks/ultralytics.py` importing `ultralytics` (or any other
  copyleft package); inside that adapter the import is local to the function, never at
  module top level, and never reachable from `core/` or an application package.
