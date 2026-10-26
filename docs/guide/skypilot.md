# Spot instances with SkyPilot (experimental)

**Experimental: this recipe has not been run on a real spot instance.** The training script and
the CV-Pilot preemption path it relies on are tested locally with real processes and real signals
(see [What is verified](#what-is-verified)); SkyPilot, a cloud's preemption notice and the way
SkyPilot treats the script's exit code are not.

[SkyPilot](https://github.com/skypilot-org/skypilot) managed jobs restart a preempted spot job on a
new node. That node has neither the run store nor the checkpoint directory of the old one, only
the bucket. The recipe is two files in `examples/skypilot/`: a task file and a training script that
saves on a preemption notice and resumes from the bucket.

## The task file

```yaml
# CV-Pilot on a spot instance with SkyPilot managed jobs. EXPERIMENTAL: this file has not been
# run on a real spot instance. The training script and the preemption path it relies on are
# tested locally (tests/docs/test_skypilot_recipe.py); SkyPilot itself is not. See
# docs/guide/skypilot.md for the walk-through and what is and is not verified.
#
# Launch (from a clone of the repository):
#   sky jobs launch -n fridge-sku examples/skypilot/cvpilot-spot.yaml --secret CVPILOT_CKPT_KEY
name: fridge-sku

resources:
  accelerators: T4:1  # any; the stand-in training runs on a CPU too
  use_spot: true
  job_recovery:
    strategy: EAGER_NEXT_REGION  # a preempted job starts again on a new node
    max_restarts_on_errors: 3  # the script exits 143 after a preemption save

workdir: .  # the repository clone, synced to every node

envs:
  CVPILOT_PROJECT: fridge-sku
  CVPILOT_CHECKPOINT_DIR: /tmp/cvpilot-ckpt  # node-local; the bucket is what survives
  CVPILOT_CHECKPOINT_REMOTE: s3://my-bucket/cvpilot/fridge-sku  # one prefix per job
  CVPILOT_CKPT_REQUIRE_TRUST: authenticated  # restore only checkpoints written with your key
  SPOT_NOTICE: aws  # poll the instance metadata for the two-minute notice (gcp on Google Cloud)
  TOTAL_STEPS: "20000"
  SAVE_EVERY: "500"

secrets:
  CVPILOT_CKPT_KEY: null  # given with --secret at launch; never written in this file

setup: |
  python3 -m venv ~/cvpilot-venv
  ~/cvpilot-venv/bin/pip install -e ".[safetensors,remote-s3]"

run: |
  ~/cvpilot-venv/bin/python examples/skypilot/train.py
```

The key is a SkyPilot secret, passed at launch and never written in the file. Use one bucket prefix
per job: on a fresh node CV-Pilot discovers checkpoints by listing the newest run prefixes under
`CVPILOT_CHECKPOINT_REMOTE`, and with `authenticated` it accepts only checkpoints written with your
key (see "A fresh node with only the bucket" in [Checkpoints and resume](checkpoints.md)).

## What happens on a preemption

1. The cloud announces the preemption (AWS: two minutes ahead, in the instance metadata). With
   `SPOT_NOTICE=aws` the `CheckpointManager`'s poller sees it; a `SIGTERM` or `SIGUSR1` counts too.
   The poller and the signal handler only set a flag.
2. The next `maybe_save` writes a checkpoint at once, outside its interval, inside the grace window
   (`grace_seconds`, default 30), marks the run `preempted` and uploads the checkpoint to the bucket
   with what is left of the window (events `preemption_signal_received`, `checkpoint_saved`,
   `run_preempted`, `checkpoint_uploaded`).
3. `ckpt.preempted()` is now true; `train.py` prints `preempted after step N` and exits with `143`,
   not `0`, so the job is not reported as completed.
4. SkyPilot starts the job again on a new node. `train.py` calls `ckpt.restore(discover=True)`: the
   new run has no lineage, so CV-Pilot lists the bucket, admits the newest checkpoint that passes the
   gate (the key, the size caps, the SHA-256 before anything is installed) and the script continues
   at step `N + 1` (`checkpoint_restored ... trust_level=authenticated`).

Launch it from a clone of the repository:

```bash notest
sky jobs launch -n fridge-sku examples/skypilot/cvpilot-spot.yaml --secret CVPILOT_CKPT_KEY
```

## What is verified

`tests/docs/test_skypilot_recipe.py` runs `train.py` as two "nodes" on one machine, with the bucket
served from a local directory under the `s3` scheme:

- node A trains with the key and `authenticated`; after two uploaded checkpoints it gets a real
  `SIGTERM`, saves, uploads, marks its run `preempted` and exits `143`;
- node B has its own empty run store and checkpoint directory, restores node A's checkpoint from the
  bucket at the trust level `authenticated`, starts at the step after the one node A saved last and
  finishes;
- a node with a different key restores nothing and fails instead of silently starting over;
- the task file asks for spot recovery, keeps the key out of `envs` and sets only variables the
  script or CV-Pilot reads.

This walk-through found two bugs before it passed, both fixed with their own unit tests:
`preempted()` was true before the preemption save had happened, and the preemption checkpoint was
never uploaded once the run had been marked `preempted`.

Not verified: SkyPilot itself, a real spot instance, a real AWS or Google Cloud notice (the notice
sources are unit-tested against fake metadata services), whether SkyPilot recovers a job whose script
exits `143` (`max_restarts_on_errors` covers it if SkyPilot counts it as an error), and whether a
real model's save and upload fit in the time between the notice and the end of the node.
