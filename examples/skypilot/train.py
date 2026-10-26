"""A training loop for a spot instance: the script the SkyPilot recipe runs on every node.

It is written for the case SkyPilot's managed jobs create: a preempted job is started again on a
NEW node that has neither the run store nor the checkpoint directory of the earlier node, only the
bucket. So the script asks the bucket for the newest checkpoint (``restore(discover=True)``), and
on a preemption notice it saves within the grace window and exits non-zero, so the job is not
reported as completed. Experimental: see docs/guide/skypilot.md.

Configuration comes from the environment (the recipe sets it):

* ``CVPILOT_PROJECT``, ``CVPILOT_CHECKPOINT_DIR``, ``CVPILOT_CHECKPOINT_REMOTE``: where things go.
* ``CVPILOT_CKPT_KEY`` (a secret) and ``CVPILOT_CKPT_REQUIRE_TRUST=authenticated``: only
  checkpoints written with your key are restored from the bucket.
* ``SPOT_NOTICE``: ``aws`` or ``gcp`` to poll the instance metadata service for a preemption notice
  (off when empty; ``SIGTERM`` and ``SIGUSR1`` always count).
* ``TOTAL_STEPS``, ``SAVE_EVERY``, ``STEP_SECONDS``: the length of this stand-in training.

The model is a stand-in (one weight vector); replace it with yours and pass ``model=``,
``optimizer=`` and ``scheduler=`` as in the checkpoint guide.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import cvpilot
from cvpilot.ckpt import CheckpointManager

PREEMPTED_EXIT = 143  # 128 + SIGTERM: "terminated", never 0, so the job is not marked done


def main() -> int:
    total = int(os.environ.get("TOTAL_STEPS", "1000"))
    every = int(os.environ.get("SAVE_EVERY", "50"))
    pause = float(os.environ.get("STEP_SECONDS", "0"))
    notice = os.environ.get("SPOT_NOTICE") or None
    run = cvpilot.init(task="detection", config={"total_steps": total, "save_every": every})
    ckpt = CheckpointManager(run, every_n_steps=every, notice_source=notice)
    restored = ckpt.restore(discover=True)  # this run's lineage first, then the bucket
    if restored is None:
        weights, start = np.zeros(8, dtype=np.float32), 0
        print("starting at step 0", flush=True)
    else:
        weights, start = restored.state["model"]["w"], restored.manifest.step + 1
        print(f"starting at step {start} from a checkpoint of run {restored.run_id}", flush=True)
    saved = start - 1
    for step in range(start, total):
        weights = weights + np.float32(0.001)  # stand-in for one optimizer step
        cvpilot.log({"train/loss": float(1.0 / (step + 1))}, step=step)
        if ckpt.maybe_save(step, model={"w": weights}) is not None:
            saved = step
        if ckpt.preempted():  # the save is done and the run is already marked preempted
            print(f"preempted after step {step}", flush=True)
            ckpt.close()
            return PREEMPTED_EXIT
        time.sleep(pause)
    if saved < total - 1:
        ckpt.save_now(total - 1, model={"w": weights})  # the final state, whatever the interval
    ckpt.close()
    cvpilot.finish()
    print(f"finished at step {total - 1}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
