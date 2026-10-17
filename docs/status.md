# What is built and what is planned

CV-Pilot is in alpha. A part is "built" when it exists in the code, has tests, and its commands
appear in the documentation from a real run. Everything else is planned, however detailed its
description. The architecture pages mark each component the same way, and the
[changelog](https://github.com/B4teShai/CV-Pilot/blob/main/CHANGELOG.md) lists what changed.

| Part | State | Where to read about it |
|---|---|---|
| Record runs: environment, git state, dataset hash, metrics, config | Built | [Concepts](guide/concepts.md), [Tutorial](tutorial.md) |
| Local run store and `cvpilot runs list`, `runs show`, `repro` | Built | [Concepts](guide/concepts.md) |
| Evaluation of detection and classification, per class and per slice | Built | [Gating in CI](guide/gating-ci.md) |
| Baselines, `compare`, `gate`, HTML and JSON reports | Built | [Gating in CI](guide/gating-ci.md) |
| GitHub Action around `cvpilot gate` | Built | [Gating in CI](guide/gating-ci.md) |
| Send metrics to MLflow, Weights & Biases, Trackio | Built | [Concepts](guide/concepts.md) |
| Hooks for Lightning, Hugging Face `Trainer` and Ultralytics | Built | [Concepts](guide/concepts.md) |
| Crash-proof checkpoints, resume, retention, preemption handling | Built | [Checkpoints and resume](guide/checkpoints.md) |
| Copy checkpoints to S3, GCS or Azure storage | Built | [Checkpoints and resume](guide/checkpoints.md) |
| Signed checkpoints and the trust levels | Built | [Trust model](guide/trust.md) |
| Python API (frozen by a snapshot test), command line, structured JSON logs | Built | [Python API](reference/api.md), [Command line](reference/cli.md) |
| Documentation site and a tutorial whose every command is run by a test | Built | [Tutorial](tutorial.md) |
| Server, remote monitoring and sync | Planned | [Components](architecture/02-components.md) |
| Alert rules (NaN loss, spike, plateau, regression, failed, preempted) and channels (webhook, Slack-style webhook, Telegram) | Built | [Alerts](guide/alerts.md), [ADR-0009](adr/0009-alerts-and-hf-hook.md) |
| Email alerts, per-user routing, alert history | Not planned for this release | [ADR-0009](adr/0009-alerts-and-hf-hook.md) |
| Hooks for MMEngine and RF-DETR | Planned | [Components](architecture/02-components.md) |

The Lightning, Hugging Face, Ultralytics and tracker integrations are tested with fake trainers and fake tracker
modules (the Hugging Face hook also has a real-library smoke test and a real run in notebook 03). Alert delivery is tested against a local listener; no real Slack or Telegram account was used. Smoke tests against the real libraries exist behind `pytest -m real_libs`; they have not
been run in the development environment.

Benchmarks live in `benchmarks/`: `log_readback.py` logs 100 000 points and reads them back,
`install_size.sh` checks that the core install carries no heavy framework, `ckpt_save.py` measures
the checkpoint save overhead (the 5 % budget of [ADR-0006](adr/0006-checkpoint-semantics.md) is not
met: see the [overview](architecture/01-overview.md)), `eval_throughput.py` times an evaluation and
`compare_throughput.py` times a comparison of 5 000 images.
