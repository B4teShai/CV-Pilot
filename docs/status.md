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
| Hooks for Lightning, Hugging Face `Trainer`, Ultralytics and MMEngine | Built; each ran on a real training run in notebook 03 or 04 | [Concepts](guide/concepts.md), [ADR-0010](adr/0010-real-checkpoint-files-and-mmengine.md) |
| Crash-proof checkpoints, resume, retention, preemption handling | Built. Proven on real files: safetensors checkpoints of a real model and optimizer, and Lightning `last.ckpt`. Not resumable through CV-Pilot: Ultralytics `last.pt`, MMEngine `.pth` and a Hugging Face `Trainer` folder (refused, with the reason) | [Checkpoints and resume](guide/checkpoints.md) |
| Copy checkpoints to S3, GCS or Azure storage | Built, **experimental**: tested against an in-memory filesystem, never against a real bucket | [Checkpoints and resume](guide/checkpoints.md) |
| Signed checkpoints and the trust levels | Built. `anchored` and `authenticated` proven on a real Lightning file; `local_chain_only` and the cloud preemption notices **experimental** (not run on real files or services) | [Trust model](guide/trust.md) |
| Python API (frozen by a snapshot test), command line, structured JSON logs | Built | [Python API](reference/api.md), [Command line](reference/cli.md) |
| Documentation site and a tutorial whose every command is run by a test | Built | [Tutorial](tutorial.md) |
| Server, remote monitoring and sync | Planned | [Components](architecture/02-components.md) |
| Alert rules (NaN loss, spike, plateau, regression, failed, preempted) and channels (webhook, Slack-style webhook, Telegram) | Built | [Alerts](guide/alerts.md), [ADR-0009](adr/0009-alerts-and-hf-hook.md) |
| Email alerts, per-user routing, alert history | Not planned for this release | [ADR-0009](adr/0009-alerts-and-hf-hook.md) |
| Hook for RF-DETR | Planned | [Components](architecture/02-components.md) |
| Resume helpers for Ultralytics, MMEngine and Hugging Face files | Not built: their files pickle objects or form a folder, see the decision | [ADR-0010](adr/0010-real-checkpoint-files-and-mmengine.md) |

The Lightning, Hugging Face, Ultralytics, MMEngine and tracker integrations are tested with fake trainers and
fake tracker modules. The four hooks also ran on real training runs (the Hugging Face hook in notebook 03,
all four in notebook 04), and real-library smoke tests exist behind `pytest -m real_libs` (they ran for
Lightning, Hugging Face and MMEngine in the environment of notebook 04). The tracker smoke tests were not run for this release.
Alert delivery is tested against a local listener; no real Slack or Telegram account was used.

"Experimental" in the table means built and unit-tested but not yet shown to work on the real thing; read
its guarantees as design intent. What notebook 04 measured and refused is in
[ADR-0010](adr/0010-real-checkpoint-files-and-mmengine.md) and the
[checkpoint guide](guide/checkpoints.md#frameworks-that-write-their-own-files).

Benchmarks live in `benchmarks/`: `log_readback.py` logs 100 000 points and reads them back,
`install_size.sh` checks that the core install carries no heavy framework, `ckpt_save.py` measures
the checkpoint save overhead and where its time goes (the 5 % budget of
[ADR-0006](adr/0006-checkpoint-semantics.md) is withdrawn; the measured numbers are in
[ADR-0010](adr/0010-real-checkpoint-files-and-mmengine.md)), `eval_throughput.py` times an evaluation and
`compare_throughput.py` times a comparison of 5 000 images.
