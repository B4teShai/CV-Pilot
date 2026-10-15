// C4 level 1: CV-Pilot in its environment.
import { grid, actor, arrow, heading, note } from "../../../tools/diagrams/lib/scene.mjs";
import { system, external, c4Legend } from "../../../tools/diagrams/lib/c4.mjs";

const g = grid({ x: 60, y: 110, colW: 240, rowH: 110, gapX: 100, gapY: 140 });
const at = (c, r, span = 1) => {
  const p = g(c, r, span);
  return [p.x, p.y, p.w, p.h];
};

// Row 0: inputs that drive a run.
const datacopilot = external("datacopilot", ...at(0, 0), "DataCopilot", "sibling: manifest.version");
const frameworks = external("frameworks", ...at(1, 0), "Training frameworks", "Ultralytics, Lightning, HF, MMEngine");
const nvml = external("nvml", ...at(2, 0), "GPU telemetry", "NVML via pynvml");
const scheduler = external("scheduler", ...at(3, 0), "Scheduler / cloud", "SkyPilot, SLURM, spot notices");

// Row 1: people left and right of the system.
const engineer = actor("engineer", g(0, 1).x + 20, g(0, 1).y, "CV engineer\ntrains and compares", 200);
const cvpilot = system("cvpilot", ...at(1, 1, 2), "CV-Pilot", "regression gates, remote monitoring, resume-safe checkpoints");
const reviewer = actor("reviewer", g(4, 1).x + 20, g(4, 1).y, "Reviewer\nreads gate reports", 200);

// Row 2: consumers of run data and alert delivery.
const ci = external("ci", ...at(0, 2), "CI runner", "runs cvpilot gate");
const tracker = external("tracker", ...at(1, 2), "User's tracker", "MLflow, W&B, Trackio");
const storage = external("storage", ...at(2, 2), "Object storage", "S3 / GCS via fsspec");
const channels = external("channels", ...at(3, 2), "Notification channels", "Telegram, Slack, webhook, SMTP");

export default {
  elements: [
    heading("t", "CV-Pilot: system context (C4 L1)", 40, 30),
    datacopilot, frameworks, nvml, scheduler,
    engineer, cvpilot, reviewer,
    ci, tracker, storage, channels,

    arrow(engineer[0], cvpilot, "CLI / API"),
    arrow(frameworks, cvpilot, "callbacks in-process"),
    arrow(cvpilot, nvml, "polls util, memory"),
    arrow(scheduler, cvpilot, "SIGTERM, spot notice"),
    arrow(cvpilot, datacopilot, "reads manifest.json"),
    arrow(cvpilot, channels, "alerts: HTTPS, SMTP"),
    arrow(channels, reviewer[0], "message"),
    arrow(reviewer[0], cvpilot, "live UI: HTTPS, WSS"),
    arrow(ci, cvpilot, "cvpilot gate, exit code"),
    arrow(cvpilot, tracker, "forwards scalars"),
    arrow(cvpilot, storage, "checkpoints, artifacts"),

    c4Legend(40, 800),
    note("n1", "Arrows point from caller to callee. All external integrations are planned; the CLI and the local run store are built.", 40, 840, 14),
  ],
};
