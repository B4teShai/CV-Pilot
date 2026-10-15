// C4 level 2: deployable and installable units of CV-Pilot.
import { grid, actor, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { container, store, external, boundary, c4Legend, statusLegend } from "../../../tools/diagrams/lib/c4.mjs";

const g = grid({ x: 60, y: 150, colW: 220, rowH: 84, gapX: 120, gapY: 60 });
const at = (c, r) => {
  const p = g(c, r);
  return [p.x, r === 0 ? p.y - 50 : p.y, p.w, p.h]; // row 0 sits higher so its arrow labels clear the boundary title
};
const pitchX = 340, pitchY = 144;

// Boundaries: the pip package (cols 0-2, rows 1-4) and the server image (col 3, rows 1-3).
const pkg = boundary("pkg", g(0, 1).x - 20, g(0, 1).y - 40, 3 * pitchX - 120 + 40, 4 * pitchY - 60 + 60, "cvpilot package  (pip install cvpilot)");
const srv = boundary("srv", g(3, 1).x - 20, g(3, 1).y - 40, 220 + 40, 3 * pitchY - 60 + 60, "cvpilot server  (Docker image, [server] extra)");

// Row 0: external callers.
const ci = external("ci", ...at(0, 0), "CI runner", "cvpilot gate");
const frameworks = external("frameworks", ...at(1, 0), "Training frameworks", "callbacks, loggers");
const objstore = external("objstore", ...at(2, 0), "Object storage", "S3 / GCS via fsspec");

// Package containers.
const cli = container("cli", ...at(0, 1), "cvpilot CLI", "Typer: eval, baseline, gate", true);
const api = container("api", ...at(1, 1), "Python API", "init, log, autolog", true);
const ckpt = container("ckpt", ...at(2, 1), "checkpoint manager", "atomic save, verified restore", true);
const compare = container("compare", ...at(0, 2), "compare + gate", "deltas, fail-closed gate, reports", true);
const evalEngine = container("eval", ...at(1, 2), "eval engine", "mAP, per-class, slices", true);
const sync = container("sync", ...at(2, 2), "sync client", "offline-first outbox");
const forward = container("forward", ...at(0, 3), "forwarders", "MLflow, W&B, Trackio", true);
const core = container("core", ...at(1, 3), "core library", "schema, hashing, store port", true);
const alertsClient = container("alertsc", ...at(2, 3), "client alert rules", "nan_loss, no_progress");
const runStore = store("runstore", ...at(1, 4), "local run store", "~/.cvpilot/<proj>/", true);

// Server containers.
const serverDb = store("serverdb", ...at(3, 1), "server DB", "SQLite or Postgres");
const serverApp = container("server", ...at(3, 2), "server app", "FastAPI, websocket, htmx");
const alertsServer = container("alertss", ...at(3, 3), "server alert rules", "regression, preempt");

// Right column and bottom row: externals.
const reviewer = actor("reviewer", g(4, 2).x, g(4, 2).y, "Reviewer\nbrowser", 200);
const channels = external("channels", ...at(3, 4), "Alert channels", "Telegram, Slack, SMTP");
const tracker = external("tracker", ...at(0, 5), "User's tracker", "MLflow, W&B, Trackio");

export default {
  elements: [
    heading("t", "CV-Pilot: containers (C4 L2)", 40, 30),
    pkg, srv,
    ci, frameworks, objstore,
    cli, api, ckpt, compare, evalEngine, sync, forward, core, alertsClient, runStore,
    serverDb, serverApp, alertsServer,
    reviewer, channels, tracker,

    arrow(ci, cli, "exit 0 / 1 / 2"),
    arrow(frameworks, api, "callbacks"),
    arrow(ckpt, objstore, "upload after verify"),
    arrow(cli, compare, "compare, gate"),
    arrow(api, ckpt, "save / restore"),
    arrow(api, evalEngine, "log_eval"),
    arrow(evalEngine, core, "EvalSnapshot"),
    arrow(compare, core, "snapshots, baselines"),
    arrow(core, runStore, "SQLite WAL, Parquet, CAS"),
    arrow(core, forward, "fan-out"),
    arrow(core, alertsClient, "rules"),
    arrow(core, sync, "outbox batches"),
    arrow(sync, serverApp, "HTTPS, seq"),
    arrow(serverApp, serverDb, "persist"),
    arrow(serverApp, alertsServer, "events"),
    arrow(alertsServer, channels, "deliver"),
    arrow(alertsClient, channels, "deliver (no server)"),
    arrow(reviewer[0], serverApp, "HTTPS / WSS"),
    arrow(forward, tracker, "scalars"),

    c4Legend(40, 1000),
    statusLegend(40, 1040),
    note("n1", "Blue containers are planned; green is built and tested today. Sync, alerts and the server are planned; hooks and the checkpoint manager live in the trainer process and are built.", 400, 1042, 14),
  ],
};
