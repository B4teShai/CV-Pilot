// Context: who uses CV-Pilot and which outside tools it talks to, eight boxes.
import { box, arrow, note, COLORS } from "../../../tools/diagrams/lib/scene.mjs";

const FS = 26, W = 320, H = 110;
const L = 40, R = 40 + W + 150 + 340 + 150;
const rows = [60, 250, 440];
const o = { fontSize: FS, strokeWidth: 2 };
const a = { fontSize: 22, strokeWidth: 2 };

const you = box("you", L, rows[0], W, H, "You\nCV engineer", COLORS.agent, o);
const code = box("code", L, rows[1], W, H, "Your training code\nLightning, YOLO", COLORS.external, o);
const sched = box("sched", L, rows[2], W, H, "Scheduler or cloud\nspot instance", COLORS.external, o);
const tool = box("tool", L + W + 150, 60, 340, 490, "CV-Pilot\n\nrecords runs\nscores models\ngates CI\nsaves checkpoints", COLORS.core, o);
const tracker = box("tracker", R, rows[0], W, H, "Your tracker\nMLflow, W&B", COLORS.external, o);
const bucket = box("bucket", R, rows[1], W, H, "Object storage\nS3, GCS, Azure", COLORS.external, o);
const chat = box("chat", R, rows[2], W, H, "Slack, Telegram\nor a webhook", COLORS.external, o);
const ci = box("ci", L + W + 150, 650, 340, H, "CI runner\nGitHub Actions", COLORS.external, o);

export default {
  elements: [
    you, code, sched, tool, tracker, bucket, chat, ci,
    arrow(you, tool, "CLI, Python", a),
    arrow(code, tool, "hooks, log", a),
    arrow(sched, tool, "SIGTERM", a),
    arrow(tool, tracker, "metrics", a),
    arrow(tool, bucket, "checkpoints", a),
    arrow(tool, chat, "alerts", a),
    arrow(ci, tool, "cvpilot gate", a),
    note("cap", "Everything runs on your machine; CV-Pilot starts no server.", 40, 800, 22),
  ],
};
