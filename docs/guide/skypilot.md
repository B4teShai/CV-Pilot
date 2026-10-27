# Spot instances with SkyPilot (experimental)

**Experimental: this recipe has not been run on a real spot instance.** The training script and the
preemption path are tested locally with real processes and signals. SkyPilot, a cloud's preemption
notice and the way SkyPilot treats the script's exit code are not.

A [SkyPilot](https://github.com/skypilot-org/skypilot) managed job restarts a preempted spot job on
a new node. That node has neither the run store nor the checkpoint folder of the old one, only the
bucket. The recipe is two files in `examples/skypilot/`: a task file, and a training script that
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
  CVPILOT_CHECKPOINT_DIR: ~/cvpilot-ckpt  # per-user and node-local (mode 0700, created by CV-Pilot); the bucket is what survives
  CVPILOT_CHECKPOINT_REMOTE: s3://my-bucket/cvpilot/fridge-sku  # one prefix per job
  CVPILOT_CKPT_REQUIRE_TRUST: authenticated  # restore only checkpoints written with your key
  SPOT_NOTICE: aws  # poll the instance metadata for the two-minute notice (gcp on Google Cloud)
  TOTAL_STEPS: "20000"
  SAVE_EVERY: "500"

secrets:
  CVPILOT_CKPT_KEY: null  # given with --secret at launch; never written in this file

setup: |
  # the environment of the committed lock (uv.lock), not whatever pip resolves today
  python3 -m pip install --user "uv==0.11.7"
  ~/.local/bin/uv sync --locked --extra safetensors --extra remote-s3

run: |
  ~/.local/bin/uv run --no-sync python examples/skypilot/train.py
```

The key is a SkyPilot secret passed at launch, never written in the file. Use one bucket prefix per job:
on a fresh node CV-Pilot lists the newest run prefixes under `CVPILOT_CHECKPOINT_REMOTE`, and with
`authenticated` it accepts only checkpoints written with your key (see
[a fresh node with only the bucket](checkpoints.md#a-fresh-node-with-only-the-bucket)).

```bash notest
sky jobs launch -n fridge-sku examples/skypilot/cvpilot-spot.yaml --secret CVPILOT_CKPT_KEY
```

## What happens on a preemption

1. The cloud announces it (AWS: two minutes ahead, in the instance metadata). With `SPOT_NOTICE=aws`
   the manager's poller sees it. A `SIGTERM` or `SIGUSR1` counts too. Both only set a flag.
2. The next `maybe_save` writes a checkpoint at once, marks the run `preempted` and uploads it with
   the time left in the grace window.
3. `train.py` prints `preempted after step N` and exits with `143`, not `0`, so the job is not
   reported as completed.
4. SkyPilot starts the job on a new node. `train.py` calls `ckpt.restore(discover=True)`, which
   lists the bucket and picks the newest checkpoint that passes every check, including your key.
   Training continues at step `N + 1`.

## What is verified

`tests/docs/test_skypilot_recipe.py` runs `train.py` as two "nodes" on one machine, with the bucket
served from a local folder under the `s3` scheme:

- Node A trains with the key. After two uploads it gets a real `SIGTERM`, saves, uploads, marks its
  run `preempted` and exits `143`.
- Node B starts empty, restores node A's checkpoint from the bucket as `authenticated`, continues
  at the next step and finishes.
- A node with a different key restores nothing and fails instead of silently starting over.
- The task file asks for spot recovery, keeps the key out of `envs` and sets only variables the
  script or CV-Pilot reads.

Not verified: SkyPilot itself, a real spot instance, a real AWS or Google Cloud notice, whether
SkyPilot recovers a job whose script exits `143` (`max_restarts_on_errors` covers it if SkyPilot
counts that as an error), and whether a real model's save and upload fit between the notice and the
end of the node.
