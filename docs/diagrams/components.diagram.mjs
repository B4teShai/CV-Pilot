// Components: the layers and the dependency rule, eleven boxes.
import { box, lane, arrow, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const FS = 26, W = 300, H = 92, G = 30;
const col = (i) => 60 + i * (W + G);
const opts = { fontSize: FS, strokeWidth: 2 };
const LW = 3 * W + 2 * G + 40; // lane width

const [uiLane, uiT] = lane("ui_lane", 40, 40, LW, 150, "Interface");
const [appLane, appT] = lane("app_lane", 40, 270, LW, 270, "Application");
const [infLane, infT] = lane("inf_lane", 40, 620, LW, 150, "Infrastructure (adapters)");

const ui = box("ui", 60, 90, LW - 40, H, "cli/ and api.py\ncommands, Python API", COLORS.core, opts);
const app = [
  ["runs", "runs/\nlifecycle, logging"],
  ["eval", "eval/\nmAP, slices"],
  ["cmp", "compare/\ngate, reports"],
  ["ckpt", "ckpt/\nsave, restore"],
  ["alerts", "alerts/\nrules"],
  ["repro", "repro.py\nreproduce a run"],
].map(([id, text], i) =>
  box(id, col(i % 3), i < 3 ? 320 : 428, W, H, text, COLORS.core, opts));
const inf = [
  ["store", "store/ capture/\nfiles, environment"],
  ["hooks", "hooks/ forward/\ntrackers, hooks"],
  ["notify", "notify/ remote/\nalerts, buckets"],
].map(([id, text], i) => box(id, col(i), 670, W, H, text, COLORS.external, opts));
const domain = box("core", LW + 100, 40, 270, 730, "core/\nschemas\nhashing\nmetrics\nerrors\n\nimports nothing\nfrom the rest", COLORS.store, opts);

const a = { fontSize: 22, strokeWidth: 2 };
export default {
  elements: [
    uiLane, uiT, appLane, appT, infLane, infT,
    ui, ...app, ...inf, domain,
    arrow(uiLane, appLane, "calls", a),
    arrow(appLane, infLane, "by name, through registries", { ...a, strokeStyle: "dashed" }),
    arrow(appLane, domain, "uses", a),
    arrow(infLane, domain, "uses", a),
    note("cap", "Imports point downward: an application package never imports an adapter, it asks a registry for one.", 40, 810, 22),
  ],
};
