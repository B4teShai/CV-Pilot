// Flow B: atomic checkpoint save, preemption, restore_latest_valid on restart.
import { sequence } from "../../../tools/diagrams/lib/sequence.mjs";
import { note } from "../../../tools/diagrams/lib/scene.mjs";
import { c4Legend } from "../../../tools/diagrams/lib/c4.mjs";

const steps = [
  { from: "trainer", to: "ckpt", text: "maybe_save(step, state, metrics)" },
  { from: "ckpt", to: "disk", text: "write step_000420.pt.tmp + fsync" },
  { from: "ckpt", to: "ckpt", text: "sha256 over the file" },
  { from: "ckpt", to: "disk", text: "write manifest.json, os.replace .tmp → .pt" },
  { from: "ckpt", to: "disk", text: "update LATEST (only after manifest exists)" },
  { from: "ckpt", to: "s3", text: "upload .pt + manifest (fsspec)" },
  { note: "previous valid checkpoint kept until this one verifies", at: "ckpt" },
  { divider: "preemption" },
  { from: "sched", to: "trainer", text: "SIGTERM or spot notice (grace window)" },
  { from: "trainer", to: "ckpt", text: "save_now(); run status → preempted" },
  { divider: "restart on a new node" },
  { from: "trainer", to: "ckpt", text: "restore_latest_valid()" },
  { from: "ckpt", to: "disk", text: "read LATEST; verify sha256 + load; walk back if bad" },
  { from: "ckpt", to: "trainer", text: "state (step, optimizer, rng…) or None", kind: "return" },
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
    c4Legend(40, totalH + 20),
    note("n1", "Planned flow: only the cli/ skeleton exists today. Logged events: checkpoint_saved, checkpoint_uploaded, preemption_signal_received, run_preempted, checkpoint_verified or checkpoint_skipped_corrupt, checkpoint_restored, run_resumed.", 40, totalH + 60, 14),
  ],
};
