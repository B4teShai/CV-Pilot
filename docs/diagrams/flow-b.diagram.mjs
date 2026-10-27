// Flow B: atomic checkpoint save, preemption, restore_latest_valid on restart.
import { sequence } from "../../../tools/diagrams/lib/sequence.mjs";
import { note } from "../../../tools/diagrams/lib/scene.mjs";

const steps = [
  { from: "trainer", to: "ckpt", text: "maybe_save(step, state, metrics)" },
  { from: "ckpt", to: "disk", text: "temp file + fsync, then 0444" },
  { from: "ckpt", to: "disk", text: "write manifest, rename it, then the data" },
  { from: "ckpt", to: "ckpt", text: "re-hash; record digest in the store" },
  { from: "ckpt", to: "disk", text: "rewrite LATEST (after the data)" },
  { from: "ckpt", to: "s3", text: "upload .part, verify, rename; manifest last" },
  { note: "previous valid checkpoint kept until this one verifies", at: "ckpt" },
  { divider: "preemption" },
  { from: "sched", to: "trainer", text: "SIGTERM sets a flag (or spot notice)" },
  { from: "trainer", to: "ckpt", text: "next maybe_save: save, mark preempted, then upload" },
  { divider: "restart on a new node" },
  { from: "trainer", to: "ckpt", text: "restore_latest_valid()" },
  { from: "ckpt", to: "disk", text: "walk back; skip bad files; refuse a rollback" },
  { from: "ckpt", to: "s3", text: "none valid locally: fetch, verify" },
  { from: "ckpt", to: "trainer", text: "state (model, optimizer, rng…) or None", kind: "return" },
];
const totalH = 80 + 64 + 30 + steps.length * 64 + 40;

export default {
  elements: [
    sequence({
      gap: 170,
      title: "Flow B: atomic checkpoint → preemption → restore_latest_valid",
      actors: [
        { id: "sched", label: "scheduler / spot notice", kind: "external" },
        { id: "trainer", label: "trainer process" },
        { id: "ckpt", label: "ckpt/ manager" },
        { id: "disk", label: "local disk /checkpoint", kind: "store" },
        { id: "s3", label: "object storage", kind: "external" },
      ],
      steps,
    }),
    note("n1", "Logged events: checkpoint_saved, checkpoint_verified, checkpoint_uploaded, preemption_signal_received, run_preempted, checkpoint_skipped_corrupt or checkpoint_untrusted, checkpoint_restored, run_resumed.", 40, totalH + 20, 14),
  ],
};
