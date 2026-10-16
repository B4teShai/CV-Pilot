// Overview: the whole idea in five boxes, left to right. No legend needed:
// orange = what you bring, blue = CV-Pilot, green = what you get.
import { box, arrow, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const W = 290, H = 130, GAP = 150, FS = 20;
const x = (i) => 40 + i * (W + GAP);
const Y = 40;
const opts = { fontSize: FS, strokeWidth: 2 };

const code = box("code", x(0), Y, W, H, "Your training code\nLightning · Ultralytics\nplain Python", COLORS.external, opts);
const tool = box("tool", x(1), Y, W, H, "CV-Pilot\nrecords · evaluates\nsaves checkpoints", COLORS.core, opts);
const compare = box("compare", x(2), Y, W, H, "Compare with\nyour best model", COLORS.core, opts);
const ci = box("ci", x(3), Y, W, H, "CI passes\nor fails", COLORS.built, opts);
const tracker = box("tracker", x(1), Y + H + 90, W, 100, "Your tracker\nMLflow · W&B · Trackio", COLORS.external, opts);

export default {
  elements: [
    code, tool, compare, ci, tracker,
    arrow(code, tool, "log", { fontSize: 18, strokeWidth: 2 }),
    arrow(tool, compare, "new model", { fontSize: 18, strokeWidth: 2 }),
    arrow(compare, ci, "verdict", { fontSize: 18, strokeWidth: 2 }),
    arrow(tool, tracker, "metrics", { fontSize: 18, strokeWidth: 2, strokeStyle: "dashed" }),
    note("caption", "Train as usual. CV-Pilot checks every new model and stops a worse one from shipping.", x(0), Y + H + 240, 20),
  ],
};
