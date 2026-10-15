// Entities of the local run store with keys, foreign keys and cardinalities.
import { grid, entity, arrow, heading, note, legend, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const W = 330;
const g = grid({ x: 60, y: 110, colW: W, rowH: 340, gapX: 90, gapY: 80 });
const at = (c, r) => [g(c, r).x, g(c, r).y];

const repro = entity("repro", ...at(0, 0), W, "Repro", [
  "run_id: str            PK FK",
  "git_commit: str?",
  "git_dirty: bool",
  "git_diff_sha256: str?",
  "packages_lock_sha256: str",
  "python_version: str",
  "cuda: str?",
  "gpu_names: list[str]",
  "dataset_uri: str",
  "dataset_hash: str",
  "config: json",
  "seed: int?",
  "command_line: str (redacted)",
], COLORS.built);

const baseline = entity("baseline", ...at(1, 0), W, "Baseline", [
  "project: str",
  "tag: str               newest row wins",
  "run_id: str            FK",
  "eval_set_hash: str     FK",
  "step: int              FK",
  "snapshot_sha256: str",
  "pinned_at: datetime",
], COLORS.built);

const manifest = entity("manifest", ...at(2, 0), W, "CheckpointManifest", [
  "run_id: str            PK FK",
  "step: int              PK",
  "kind: last | best      PK",
  "path: str",
  "uri: str?  (intended)",
  "sha256: str",
  "size_bytes: int",
  "format: safetensors | pt",
  "metric_key/value/mode?",
  "versions: json",
  "git_commit: str?",
  "dataset_hash: str",
  "created_at: datetime",
  "prev_manifest_sha256",
  "prev_chain_sha256",
  "chain_sha256",
  "auth_tag: str? (HMAC)",
], COLORS.built);

const metric = entity("metric", ...at(0, 1), W, "MetricPoint", [
  "run_id: str            PK FK",
  "key: str               PK",
  "step: int              PK",
  "epoch: int?",
  "value: float64",
  "wall_time: float64",
], COLORS.built);

const run = entity("run", ...at(1, 1), W, "RunMeta", [
  "run_id: str (ULID)     PK",
  "project: str",
  "name: str",
  "status: enum",
  "parent_run_id: str?    FK",
  "task: detection | classification",
  "framework: str",
  "host: str",
  "started_at: datetime",
  "ended_at: datetime?",
  "schema_version: int",
], COLORS.built);

const artifact = entity("artifact", ...at(2, 1), W, "Artifact", [
  "run_id: str            PK FK",
  "sha256: str            PK",
  "kind: enum",
  "uri: str",
  "size_bytes: int",
  "meta: json",
  "created_at: datetime",
], COLORS.built);

const snapshot = entity("snapshot", ...at(1, 2), W, "EvalSnapshot", [
  "run_id: str            PK FK",
  "eval_set_hash: str     PK",
  "step: int              PK",
  "task, class_names, n_images",
  "metrics: json",
  "per_class: json",
  "slices: json",
  "ci: json, bootstrap: json?",
  "confusion_sha256: str  FK",
  "predictions_sha256: str FK",
], COLORS.built);

export default {
  elements: [
    heading("t", "CV-Pilot: run store data model", 40, 30),
    repro, baseline, manifest, metric, run, artifact, snapshot,

    arrow(repro.box, run.box, "0..1 → 1"),
    arrow(baseline.box, run.box, "0..* → 1"),
    arrow(manifest.box, run.box, "0..* → 1"),
    arrow(metric.box, run.box, "0..* → 1"),
    arrow(artifact.box, run.box, "0..* → 1"),
    arrow(snapshot.box, run.box, "0..* → 1"),

    legend(40, 1260, [[COLORS.built, "built (table in runs.sqlite; MetricPoint also in Parquet)"], [COLORS.planned, "planned"]]),
    note("n1", "PK = primary key, FK = foreign key, ? = nullable. Arrows point from the referencing entity to the referenced one, labelled with cardinality.", 40, 1300, 14),
    note("n2", "RunMeta.parent_run_id references RunMeta.run_id (resumed run → preempted parent). Baseline pins the EvalSnapshot (run_id, eval_set_hash, step) and its digest; rows are append-only, digested and chained. EvalSnapshot FKs reference Artifact.sha256.", 40, 1324, 14),
    note("n3", "CheckpointManifest rows (checkpoint_manifest) hold the digest of each manifest file, append-only; the files live in the checkpoint directory.", 40, 1348, 14),
  ],
};
