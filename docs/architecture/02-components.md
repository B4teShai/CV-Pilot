# 02 Components

![The four layers and the one rule: imports point downward](../diagrams/png/components.png)

The package has four layers ([ADR-0004](../adr/0004-layering-and-dependency-rule.md)). A test
(`tests/unit/test_layering.py`) checks the rule on the syntax tree.

| Layer | Packages |
|---|---|
| Interface | `cli/`, `api.py` |
| Application | `runs/`, `eval/`, `compare/`, `ckpt/`, `alerts/`, `repro.py` |
| Domain | `core/`: schemas, hashing, metrics, confusion, errors, ULIDs, redaction |
| Infrastructure | `store/`, `capture/`, `hooks/`, `forward/`, `notify/`, `remote/` |
| Cross-cutting | `logging.py`, `settings.py`, `registry.py` |

## The dependency rule

- `core/` imports nothing else from `cvpilot`, and no framework, tracker SDK or `pyarrow` at module
  level.
- Infrastructure never imports application or interface code.
- Application code never imports an adapter by module path. It asks a registry for one by name.
- `cli/` and `api.py` never import infrastructure directly.
- Optional libraries are imported inside the function that needs them.
- Only `hooks/ultralytics.py` may mention the AGPL-licensed Ultralytics package, and it does not
  import it either: the hook receives the model through `add_callback`.

## Interface layer

**`cli/`** is the Typer application: option parsing, the single error boundary, exit codes, results
on `stdout` and logs on `stderr`. Commands: `doctor`, `runs list|show`, `repro`, `eval`,
`baseline set|list|show`, `compare`, `gate`, `alerts check`, `ckpt list|verify|prune|quarantine|trash`.
The generated [reference](../reference/cli.md) lists every option.

**`api.py`** is the public Python API. It holds the active run of the process, turns user input
into domain objects and delegates. `init`, `log`, `log_eval`, `log_artifact`, `finish`, `autolog` and
`resume_path` are documented in the [API reference](../reference/api.md). One run is active per
process, and an `atexit` hook flushes buffered points but leaves a crashed run `running`, so it is
visibly unfinished.

## Application layer

**`runs/`** runs the lifecycle: start, log, finish, recover and resume. `log` buffers up to 10,000
points and writes one Parquet batch when the buffer is full or 2 seconds have passed. A step must
be larger than the last for the same key. Flushed batches are offered to the forwarders through
bounded queues, one daemon thread each (`runs/forwarding.py`). A process killed without `finish`
stays `running` until `recover_run` marks it `failed`. `resume_from` accepts only `preempted` or
`failed` parents.

**`eval/`** turns predictions and ground truth into an `EvalSnapshot`: mAP, per-class AP, a
confusion matrix with a background class, slices and seeded bootstrap intervals. Inputs are
`PredictionTable` and `AnnotationTable`, validated with fixed limits (for example 5,000,000
predictions and 2,048 classes) before any metric is computed. `record_eval` refuses an existing
`(run, eval set hash, step)`, stores `predictions` and `confusion` artifacts, then the snapshot.
Matching is per image, so one pass serves every bootstrap resample.

**`compare/`** pins baselines, loads two verified snapshots, refuses when they are not comparable
and computes headline, per-class, slice and confusion deltas. `gate()` checks limits and fails
closed. Reports are static HTML and canonical JSON, with no timestamp, so equal inputs give equal
bytes. `compare` never reads a stored result: it recomputes from snapshots whose digests it
verifies on read. Rules: [ADR-0007](../adr/0007-compare-and-gate-semantics.md).

```python notest
@dataclass(frozen=True)
class Thresholds:
    max_drop: Mapping[str, float]            # {"mAP50_95": 0.5}, points on the 0..100 scale
    per_class_max_drop: Mapping[str, float]  # {"AP50_95": 2.0}
    slice_max_drop: Mapping[str, float]      # {"camera=*": 1.5}
    min_images: int = 30
    skip_small_slices: bool = False          # explicit opt-in
    max_skipped: int = 0

def gate(cmp: Comparison, thresholds: Thresholds) -> GateResult   # .passed, .violations, .skipped
```

**`ckpt/`** is the checkpoint manager ([ADR-0006](../adr/0006-checkpoint-semantics.md)). Modules:
`directory.py` (one run's folder, atomic save, trust rules, walk-back, retention), `fsops.py` (file
operations through directory descriptors, no symlinks), `serializers.py` (safetensors with a strict
header parser, opt-in `pt` with a strict archive reader), `state.py` and `rng.py`, `auth.py` (HMAC
tags), `preempt.py` (flag, signal guard, notice poller), `remote.py` (upload and fetch) and
`resume.py` (`resume_path`, `find_resume_point`). The run store keeps the digest of every manifest as
the outside anchor. Use is described in the [checkpoint guide](../guide/checkpoints.md).

**`alerts/`** is the rule engine that runs inside the training process: `config.py` validates the
JSON, `engine.py` holds the rules and rate limits, and `replay.py` is the dry run behind
`cvpilot alerts check`. `eval_regression` reuses the real gate. Channels are reached through the
`channels` registry, never imported here.

**`repro.py`** assembles the `Repro` record at `init` and prints the reproduction script.

## Domain layer

**`core/`** holds the Pydantic models (`RunMeta`, `Repro`, `MetricPoint`, `EvalSnapshot`,
`Artifact`, `CheckpointManifest`, `Baseline`), the protocols the adapters implement (`MetricSink`,
`RunStore`, `Hook`, `Forwarder`, `Channel`), hashing, ULIDs, redaction and the typed errors the CLI
boundary reports. `core/metrics.py` and `core/confusion.py` are pure NumPy functions with no I/O.
Parity with pycocotools is tested in `tests/parity/`.

## Infrastructure layer

**`store/`** is the local run store ([ADR-0005](../adr/0005-run-store.md)): SQLite in WAL mode for
rows, Parquet for metric points, content-addressed artifacts, forward-only migrations. The
`RunStore` protocol lives in `core/schema.py` so `runs/` can type against it without importing
`store/`:

```python notest
class RunStore(Protocol):
    def put_run(self, meta: RunMeta, repro: Repro) -> None
    def append_points(self, points: Sequence[MetricPoint]) -> None
    def put_snapshot(self, snapshot: EvalSnapshot) -> None
    def put_manifest(self, manifest: CheckpointManifest, manifest_sha256: str) -> None
    def put_baseline(self, baseline: Baseline, *, replace: bool = False) -> None   # append-only, chained
    def get_baseline(self, tag: str) -> Baseline                                    # verified on every read
```

**`capture/`** records the environment: git commit, dirty flag and diff, lockfile hash, Python,
CUDA and GPU names, host, and the redacted command line. Missing facilities give none, never
errors.

**`hooks/`** adapts frameworks. A hook matches an object, attaches to it and sends scalars to a
`MetricSink`.

| Hook | Records | Resume helper |
|---|---|---|
| Lightning | Scalars through a logger and a callback, rank zero only | Yes |
| Hugging Face | `train/` and `eval/` metrics, saves as events | No |
| Ultralytics | Losses, mAP, per-class AP per epoch | A helper exists, but a real `last.pt` is refused |
| MMEngine | `train/`, `val/` and `test/` metrics at `runner.iter` | No |
| RF-DETR | Every scalar the framework logs | No |

Every hook runs its callbacks under a guard that catches `Exception`, counts it, logs the first
and disables the hook after 10 failures in a row. Hooks make no network calls and never raise into
the training loop.

**`forward/`** has adapters for MLflow, W&B and Trackio. Each reads the SDK's own environment
variables, accepts only https endpoints (http in dev), refuses credentials in URLs and disabled TLS
verification, and sends run metadata and metrics, never the config or command line.

**`notify/`** has the `webhook`, `slack` and `telegram` channels, built like forwarders. One outbound
call (`_http.py`) accepts only public https addresses, checked once and connected to directly, with
no redirects.

**`remote/`** has the fsspec store for checkpoint upload (`s3`, `gs`, `az`, `memory`) and the AWS and
GCP preemption notice pollers.

## Registries

`registry.py` gives each adapter family a lazy, named lookup filled from entry points.

| Registry | Entry-point group | Key |
|---|---|---|
| `hooks` | `cvpilot.hooks` | Hook name |
| `forwarders` | `cvpilot.forwarders` | Tracker name |
| `channels` | `cvpilot.channels` | Channel name |
| `remotes` | `cvpilot.remotes` | URL scheme |
| `capturers` | `cvpilot.capturers` | Field group |
| `stores` | `cvpilot.stores` | Store name (`local`) |

A plugin lists the distributions it `requires`, and the registry checks them before importing, so a
missing library raises `MissingExtraError` naming the extra. A plugin imports `cvpilot.core` only.
Code that includes or derives from copyleft code must be a separate distribution.
