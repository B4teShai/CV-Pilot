// The records of the run store and how they relate. Fields are listed in 03-data-model.md.
import { box, arrow, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const W = 330, H = 170, GX = 120, GY = 110, FS = 24;
const x = (c) => 40 + c * (W + GX);
const y = (r) => 40 + r * (H + GY);
const o = { fontSize: FS, strokeWidth: 2 };
const a = { fontSize: 22, strokeWidth: 2 };

const repro = box("repro", x(0), y(0), W, H, "Repro\ngit, environment,\ndataset, seed", COLORS.store, o);
const baseline = box("baseline", x(1), y(0), W, H, "Baseline\ntag to pinned\nsnapshot", COLORS.store, o);
const manifest = box("manifest", x(2), y(0), W, H, "CheckpointManifest\nstep, SHA-256,\nchain, HMAC tag", COLORS.store, o);
const metric = box("metric", x(0), y(1), W, H, "MetricPoint\nkey, step, value", COLORS.store, o);
const run = box("run", x(1), y(1), W, H, "RunMeta\nrun_id, project,\nstatus, parent_run_id", COLORS.core, o);
const artifact = box("artifact", x(2), y(1), W, H, "Artifact\nSHA-256, kind, size", COLORS.store, o);
const snapshot = box("snapshot", x(1), y(2), W, H, "EvalSnapshot\neval set hash, step,\nmetrics, per class,\nslices", COLORS.store, o);

export default {
  elements: [
    repro, baseline, manifest, metric, run, artifact, snapshot,
    arrow(repro, run, "one", a),
    arrow(baseline, run, "many", a),
    arrow(manifest, run, "many", a),
    arrow(metric, run, "many", a),
    arrow(artifact, run, "many", a),
    arrow(snapshot, run, "many", a),
    note("cap", "Every record belongs to one run. Fields are listed on the data model page.", 40, y(2) + H + 40, 22),
  ],
};
