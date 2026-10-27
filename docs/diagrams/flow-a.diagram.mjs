// Flow A: log metrics, score predictions, gate in CI. Eight boxes, left to right.
import { box, arrow, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const W = 240, H = 150, GAP = 110, FS = 30;
const x = (i) => 40 + i * (W + GAP);
const R1 = 40, R2 = 330;
const o = { fontSize: FS, strokeWidth: 2 };
const a = { fontSize: 26, strokeWidth: 2 };

const code = box("code", x(0), R1, W, H, "Training code", COLORS.external, o);
const log = box("log", x(1), R1, W, H, "log() or\nautolog", COLORS.core, o);
const store = box("store", x(2), R1, W, H, "Run store", COLORS.store, o);
const ci = box("ci", x(3), R1, W, H, "CI: exit\n0 or 1", COLORS.external, o);
const data = box("data", x(0), R2, W, H, "Predictions\n+ ground\ntruth", COLORS.external, o);
const ev = box("ev", x(1), R2, W, H, "cvpilot eval\nor log_eval()", COLORS.core, o);
const snap = box("snap", x(2), R2, W, H, "Snapshot", COLORS.store, o);
const gate = box("gate", x(3), R2, W, H, "cvpilot gate\nvs baseline", COLORS.core, o);

export default {
  elements: [
    code, log, store, ci, data, ev, snap, gate,
    arrow(code, log, "", a),
    arrow(log, store, "", a),
    arrow(data, ev, "", a),
    arrow(ev, snap, "", a),
    arrow(snap, gate, "", a),
    arrow(gate, ci, "verdict", a),
    note("cap", "Hooks and log() send scalars. Scoring and the gate are separate steps that you call.", 40, 530, 26),
  ],
};
