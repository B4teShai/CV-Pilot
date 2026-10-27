// Flow B: save, preemption, restore. Eight boxes, left to right.
import { box, arrow, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const W = 240, H = 150, GAP = 110, FS = 30;
const x = (i) => 40 + i * (W + GAP);
const R1 = 40, R2 = 330;
const o = { fontSize: FS, strokeWidth: 2 };
const a = { fontSize: 26, strokeWidth: 2 };

const save = box("save", x(0), R1, W, H, "maybe_save", COLORS.core, o);
const temp = box("temp", x(1), R1, W, H, "Write and\nhash temp\nfiles", COLORS.store, o);
const link = box("link", x(2), R1, W, H, "Link manifest,\nthen data", COLORS.store, o);
const last = box("last", x(3), R1, W, H, "Record digest,\nmove LATEST", COLORS.store, o);
const term = box("term", x(0), R2, W, H, "SIGTERM\nsets a flag", COLORS.external, o);
const now = box("now", x(1), R2, W, H, "Save now,\nmark\npreempted", COLORS.core, o);
const up = box("up", x(2), R2, W, H, "Upload with\ntime left", COLORS.external, o);
const rest = box("rest", x(3), R2, W, H, "New run\nrestores\nnewest valid", COLORS.core, o);

export default {
  elements: [
    save, temp, link, last, term, now, up, rest,
    arrow(save, temp, "", a),
    arrow(temp, link, "", a),
    arrow(link, last, "", a),
    arrow(term, now, "", a),
    arrow(now, up, "", a),
    arrow(up, rest, "restart", a),
    note("cap", "Restore skips bad files, refuses a rollback and falls back to the bucket.", 40, 530, 26),
  ],
};
