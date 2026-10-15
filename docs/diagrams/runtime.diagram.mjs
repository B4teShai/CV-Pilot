// Deployment view: which process runs where and what crosses machine boundaries.
import { grid, actor, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { container, store, external, node, c4Legend, statusLegend } from "../../../tools/diagrams/lib/c4.mjs";

const g = grid({ x: 60, y: 160, colW: 220, rowH: 80, gapX: 110, gapY: 90 });
const at = (c, r, span = 1) => {
  const p = g(c, r, span);
  return [p.x, r === 0 ? p.y - 50 : p.y, p.w, p.h]; // row 0 sits higher so its arrow labels clear node titles
};
const PX = 330, PY = 170;
const nodeBox = (id, c, r, cols, rows, title) =>
  node(id, g(c, r).x - 20, at(c, r)[1] - 40, cols * PX - 110 + 40, rows * PY - 90 + 60, title);

// Deployment nodes.
const trainNode = nodeBox("n_train", 0, 1, 2, 2, "GPU training machine");
const serverNode = nodeBox("n_server", 3, 2, 2, 2, "Server host");
const devNode = nodeBox("n_dev", 5, 2, 1, 2, "Developer machine");
const ciNode = nodeBox("n_ci", 5, 0, 1, 1, "CI runner");

// Row 0.
const scheduler = external("scheduler", ...at(0, 0), "Scheduler / cloud", "SIGTERM, spot notice");
const nvml = external("nvml", ...at(1, 0), "GPU (NVML)", "util, memory, temp");
const reviewer = actor("reviewer", g(4, 0).x + 10, g(4, 0).y - 60, "Reviewer\nbrowser", 200);
const ci = container("ci", ...at(5, 0), "cvpilot gate", "exit 0 / 1 / 2", true);

// Training machine.
const trainer = container("trainer", ...at(0, 1, 2), "trainer process", "framework + hooks + api in-process; sync thread");
const ckptDir = store("ckptdir", ...at(0, 2), "/checkpoint", "last/, best/, LATEST");
const runStore = store("runstore", ...at(1, 2), "local run store", "~/.cvpilot/<proj>/", true);

// Server host.
const proxy = external("proxy", ...at(3, 2), "TLS reverse proxy", "nginx / Caddy");
const server = container("server", ...at(4, 2), "cvpilot server", "Docker: FastAPI, Uvicorn");
const serverDb = store("serverdb", ...at(3, 3), "server DB", "SQLite file or Postgres");

// Developer machine.
const cli = container("cli", ...at(5, 2), "cvpilot CLI", "runs, eval, gate", true);
const devStore = store("devstore", ...at(5, 3), "local run store", "~/.cvpilot/<proj>/", true);

// Externals below.  (Column 2 is left empty so the sync arrow label has room.)
const objstore = external("objstore", ...at(0, 3), "Object storage", "s3:// or gs:// via fsspec");
const channels = external("channels", ...at(4, 4), "Alert channels", "Telegram, Slack, SMTP");

export default {
  elements: [
    heading("t", "CV-Pilot: runtime and deployment", 40, 30),
    trainNode, serverNode, devNode, ciNode,
    scheduler, nvml, reviewer, ci,
    trainer, ckptDir, runStore,
    proxy, server, serverDb,
    cli, devStore,
    objstore, channels,

    arrow(scheduler, trainer, "signal"),
    arrow(trainer, nvml, "polls every 10 s"),
    arrow(trainer, ckptDir, "atomic save"),
    arrow(trainer, runStore, "SQLite WAL, Parquet"),
    arrow(ckptDir, objstore, "upload after verify"),
    arrow(runStore, proxy, "HTTPS batches, token header"),
    arrow(proxy, server, "HTTP"),
    arrow(server, serverDb, "persist"),
    arrow(server, channels, "alerts"),
    arrow(reviewer[0], proxy, "HTTPS, WSS"),
    arrow(ci, proxy, "HTTPS: pull snapshots"),
    arrow(cli, devStore, "read / write"),

    c4Legend(40, 1060),
    statusLegend(40, 1100),
    note("n1", "Built: the CLI, gate, the local run store, hooks and forwarder threads (in the trainer process). Sync, checkpoints and the server are planned. Client-side alert rules deliver straight to the channels when no server is configured (arrow omitted).", 400, 1102, 14),
  ],
};
