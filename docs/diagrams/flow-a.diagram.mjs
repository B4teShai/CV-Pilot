// Flow A: framework hook -> log -> eval -> gate in CI.
import { sequence } from "../../../tools/diagrams/lib/sequence.mjs";
import { note } from "../../../tools/diagrams/lib/scene.mjs";

const steps = [
  { from: "fw", to: "hook", text: "on_fit_epoch_end(metrics)" },
  { from: "hook", to: "api", text: "sink.log(points, step)" },
  { from: "api", to: "store", text: "append MetricPoint batch" },
  { divider: "validation epoch" },
  { from: "fw", to: "hook", text: "on_val_end(preds, gts)" },
  { from: "hook", to: "api", text: "sink.log_eval(preds, gts, step)" },
  { from: "api", to: "eval", text: "evaluate(task, class_names, slices)" },
  { from: "eval", to: "store", text: "EvalSnapshot, predictions, confusion", kind: "return" },
  { divider: "CI:  cvpilot gate <run_id> --baseline tag:prod --max-drop mAP50_95=0.5" },
  { from: "ci", to: "gate", text: "gate(run, baseline, thresholds)" },
  { from: "gate", to: "store", text: "verify + load baseline row and snapshots" },
  { note: "mismatch → compare_refused", at: "gate" },
  { from: "gate", to: "gate", text: "deltas, stored CIs, slices; fail closed" },
  { from: "gate", to: "ci", text: "exit 0 passed / 1 failed", kind: "return" },
];
const totalH = 80 + 64 + 30 + steps.length * 64 + 40;

export default {
  elements: [
    sequence({
      gap: 100,
      title: "Flow A: hook → log → eval → gate",
      actors: [
        { id: "fw", label: "training framework", kind: "external" },
        { id: "hook", label: "hooks/ adapter" },
        { id: "api", label: "api.py + runs/" },
        { id: "store", label: "store/ (run store)", kind: "store" },
        { id: "eval", label: "eval/" },
        { id: "gate", label: "compare/ + gate" },
        { id: "ci", label: "CI runner", kind: "external" },
      ],
      steps,
    }),
    note("n1", "Logged events: run_started, hook_attached, metrics_flushed, eval_completed, compare_completed or compare_refused, gate_passed or gate_failed, run_finished.", 40, totalH + 20, 14),
  ],
};
