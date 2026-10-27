// Components: the layers and the dependency rule, ten boxes.
import { box, lane, arrow, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const FS = 26, W = 240, H = 92, G = 30;
const col = (i) => 60 + i * (W + G);
const opts = { fontSize: FS, strokeWidth: 2 };
const LW = 4 * W + 3 * G + 40; // lane width

const [uiLane, uiT] = lane("ui_lane", 40, 40, LW, 150, "Interface");
const [appLane, appT] = lane("app_lane", 40, 270, LW, 150, "Application");
const [infLane, infT] = lane("inf_lane", 40, 500, LW, 150, "Infrastructure (adapters)");

const ui = box("ui", 60, 90, LW - 40, H, "cli/ and api.py\ncommands, Python API", COLORS.core, opts);
const app = [
  ["runs", "runs/\nalerts, repro"],
  ["eval", "eval/\nmAP, slices"],
  ["cmp", "compare/\ngate, reports"],
  ["ckpt", "ckpt/\nsave, restore"],
].map(([id, text], i) => box(id, col(i), 320, W, H, text, COLORS.core, opts));
const inf = [
  ["store", "store/\nSQLite, files"],
  ["hooks", "hooks/\nframeworks"],
  ["fwd", "forward/\ntrackers"],
  ["notify", "notify/\nalert channels"],
].map(([id, text], i) => box(id, col(i), 550, W, H, text, COLORS.external, opts));
const domain = box("core", LW + 100, 40, 270, 610, "core/\nschemas\nhashing\nmetrics\nerrors\n\nimports nothing\nfrom the rest", COLORS.store, opts);

const a = { fontSize: 22, strokeWidth: 2 };
export default {
  elements: [
    uiLane, uiT, appLane, appT, infLane, infT,
    ui, ...app, ...inf, domain,
    arrow(uiLane, appLane, "calls", a),
    arrow(appLane, infLane, "by name, through registries", { ...a, strokeStyle: "dashed" }),
    arrow(appLane, domain, "uses", a),
    arrow(infLane, domain, "uses", a),
    note("cap", "Imports point downward: an application package never imports an adapter, it asks a registry for one.", 40, 690, 22),
  ],
};
