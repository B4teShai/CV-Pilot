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

## Layers

Interface (`cli/`, the public API) uses application packages (`runs/`, `eval/`, `compare/`,
`ckpt/`), which use the domain (`core/`: schemas, hashing, errors) and reach infrastructure
(`store/`, `hooks/`, `forward/`, `remote/`) only through registries. Optional dependencies are
imported inside the function that needs them. [ADR-0004](../adr/0004-layering-and-dependency-rule.md)
has the rule and a test enforces it.
