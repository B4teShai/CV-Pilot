// C4 level 3: packages of the cvpilot distribution by layer (ADR-0004).
import { grid, lane, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { component, statusLegend } from "../../../tools/diagrams/lib/c4.mjs";

const COLS = 8, COL_W = 170, GAP_X = 26, PITCH = COL_W + GAP_X, BOX_H = 72;
const X0 = 130, LANE_X = 40, LANE_W = X0 - LANE_X + COLS * PITCH - GAP_X + 20; // column 0 starts right of the lane titles
const LANE_H = 40 + BOX_H + 16;
// Lane tops: interface, application, domain, infrastructure.
const LANE_Y = [100, 100 + LANE_H + 100, 100 + 2 * LANE_H + 180, 100 + 3 * LANE_H + 260];
const laneGrid = LANE_Y.map((y) => grid({ x: X0, y: y + 40, colW: COL_W, rowH: BOX_H, gapX: GAP_X, gapY: 0 }));
// Column may be fractional (domain boxes sit between columns so vertical arrows pass beside them).
const at = (layer, col) => {
  const p = laneGrid[layer](0, 0);
  return [p.x + col * PITCH, p.y, COL_W, BOX_H];
};

const lanes = [
  lane("l0", LANE_X, LANE_Y[0], LANE_W, LANE_H, "Interface"),
  lane("l1", LANE_X, LANE_Y[1], LANE_W, LANE_H, "Application"),
  lane("l2", LANE_X, LANE_Y[2], LANE_W, LANE_H, "Domain"),
  lane("l3", LANE_X, LANE_Y[3], LANE_W, LANE_H, "Infrastructure"),
];

// Interface.
const api = component("api", ...at(0, 1), "api.py", "log, log_eval, autolog", true);
const cli = component("cli", ...at(0, 3), "cli/", "Typer app, error boundary", true);

// Application.
const repro = component("repro", ...at(1, 0), "repro", "capture at init, script", true);
const runs = component("runs", ...at(1, 1), "runs/", "lifecycle, bounded buffer", true);
const evalPkg = component("eval", ...at(1, 2), "eval/", "evaluate, CIs, slices", true);
const compare = component("compare", ...at(1, 3), "compare/ + gate", "baselines, deltas, gate", true);
const ckpt = component("ckpt", ...at(1, 5), "ckpt/", "atomic save, verified restore", true);
const alerts = component("alerts", ...at(1, 6), "alerts/", "rule engine");

// Domain (between columns).
const schema = component("schema", ...at(2, 0.5), "core/schema.py", "RunMeta, MetricPoint, ...", true);
const metrics = component("metrics", ...at(2, 3.5), "core/metrics.py", "AP, confusion on arrays", true);
const hashing = component("hashing", ...at(2, 5.5), "core/hashing.py", "dataset + eval-set hash", true);

// Infrastructure.
const capture = component("capture", ...at(3, 0), "capture/", "env, git, GPU", true);
const forward = component("forward", ...at(3, 1), "forward/", "MLflow, W&B, Trackio", true);
const sync = component("sync", ...at(3, 2), "sync/", "HTTPS, (run_id, seq)");
const storePkg = component("store", ...at(3, 3), "store/", "SQLite WAL, Parquet, CAS", true);
const hooks = component("hooks", ...at(3, 4), "hooks/", "Lightning, YOLO callbacks", true);
const remote = component("remote", ...at(3, 5), "remote/", "fsspec upload, spot notices", true);
const notify = component("notify", ...at(3, 6), "notify/", "Telegram, Slack, SMTP");
const server = component("server", ...at(3, 7), "server/", "FastAPI; imports domain only");

// Side lanes: cross-cutting modules and plugin entry-point groups.
const SIDE_X = LANE_X + LANE_W + 30, SIDE_W = 230, SIDE_BOX_W = 200;
const cross = lane("lx", SIDE_X, LANE_Y[0], SIDE_W, LANE_Y[1] + LANE_H - LANE_Y[0], "Cross-cutting");
const logging = component("logging", SIDE_X + 15, LANE_Y[0] + 40, SIDE_BOX_W, BOX_H, "logging.py", "structlog JSON, redaction", true);
const settings = component("settings", SIDE_X + 15, LANE_Y[0] + 40 + BOX_H + 16, SIDE_BOX_W, BOX_H, "settings.py", "CVPILOT_* from env", true);
const ENTRY_H = 40 + 6 * (BOX_H + 16) + 4;
const entry = lane("le", SIDE_X, LANE_Y[2], SIDE_W, ENTRY_H, "Entry-point groups");
const epHooks = component("ep_hooks", SIDE_X + 15, LANE_Y[2] + 40, SIDE_BOX_W, BOX_H, "cvpilot.hooks", "registry: hooks/", true);
const epForward = component("ep_forward", SIDE_X + 15, LANE_Y[2] + 40 + (BOX_H + 16), SIDE_BOX_W, BOX_H, "cvpilot.forwarders", "registry: forward/", true);
const epChannels = component("ep_channels", SIDE_X + 15, LANE_Y[2] + 40 + 2 * (BOX_H + 16), SIDE_BOX_W, BOX_H, "cvpilot.channels", "registry: notify/");
const epStores = component("ep_stores", SIDE_X + 15, LANE_Y[2] + 40 + 3 * (BOX_H + 16), SIDE_BOX_W, BOX_H, "cvpilot.stores", "registry: store/", true);
const epCapturers = component("ep_capturers", SIDE_X + 15, LANE_Y[2] + 40 + 4 * (BOX_H + 16), SIDE_BOX_W, BOX_H, "cvpilot.capturers", "registry: capture/", true);

const epTables = component("ep_tables", SIDE_X + 15, LANE_Y[2] + 40 + 5 * (BOX_H + 16), SIDE_BOX_W, BOX_H, "cvpilot.tables", "registry: Parquet I/O", true);

const LEGEND_Y = Math.max(LANE_Y[3] + LANE_H, LANE_Y[2] + ENTRY_H) + 40;

export default {
  elements: [
    heading("t", "CV-Pilot: components by layer (C4 L3)", 40, 30),
    lanes, cross, entry,
    api, cli,
    repro, runs, evalPkg, compare, ckpt, alerts,
    schema, metrics, hashing,
    capture, forward, sync, storePkg, hooks, remote, notify, server,
    logging, settings, epHooks, epForward, epChannels, epStores, epCapturers, epTables,

    // interface -> application
    arrow(api, runs, "log, finish"),
    arrow(api, ckpt),
    arrow(cli, compare),
    arrow(cli, evalPkg),
    arrow(cli, runs),
    arrow(cli, repro),
    arrow(cli, ckpt),
    // application -> domain (direct imports)
    arrow(repro, schema),
    arrow(runs, schema),
    arrow(evalPkg, metrics),
    arrow(compare, metrics),
    arrow(ckpt, hashing),
    // application -> infrastructure (through registries; dashed)
    arrow(repro, capture, null, { strokeStyle: "dashed" }),
    arrow(runs, forward, null, { strokeStyle: "dashed" }),
    arrow(runs, sync, null, { strokeStyle: "dashed" }),
    arrow(runs, storePkg, null, { strokeStyle: "dashed" }),
    arrow(compare, storePkg, null, { strokeStyle: "dashed" }),
    arrow(ckpt, remote, null, { strokeStyle: "dashed" }),
    arrow(alerts, notify, null, { strokeStyle: "dashed" }),

    statusLegend(40, LEGEND_Y),
    note("n1", "Solid arrow = direct import. Dashed arrow = adapter resolved through a registry by name at run time. Same-lane calls (runs/ -> alerts/, ckpt/ -> runs/) and infrastructure -> domain imports are not drawn.", 40, LEGEND_Y + 36, 14),
    note("n2", "hooks/ is entered by the training framework and talks to runs/ only through the MetricSink protocol in core/schema.py; runs/ hands flushed batches to forward/ through bounded queues, one worker thread per forwarder.", 40, LEGEND_Y + 60, 14),
  ],
};
