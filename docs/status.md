# Project status

CV-Pilot is in alpha. This page is the only place that says what works. Every other page describes
behaviour that exists, unless it says otherwise.

- **Built**: in the code, tested, and its commands are shown from a real run.
- **Experimental**: built and unit-tested, but never run against the real thing.
- **Planned**: not in the code.

| Part | State | Note |
|---|---|---|
| Record runs, local store, `runs list`, `runs show`, `repro` | Built | [Concepts](guide/concepts.md) |
| Evaluation of detection and classification, per class and per slice | Built | [Gating in CI](guide/gating-ci.md) |
| Baselines, `compare`, `gate`, HTML and JSON reports, GitHub Action | Built | [Gating in CI](guide/gating-ci.md) |
| Hooks for Lightning, Hugging Face, Ultralytics, MMEngine, RF-DETR | Built | Each ran on a real training run. |
| Send metrics to MLflow, Weights & Biases, Trackio | Built | Tested with stand-in SDKs, not live services. |
| Alerts: NaN, spike, plateau, regression, failed, preempted | Built | Webhook, Slack-style and Telegram. Email is not planned. [Alerts](guide/alerts.md) |
| Crash-proof checkpoints, resume, retention, preemption handling | Built | Killed 1,000 times in tests; local disk only. [Checkpoints](guide/checkpoints.md) |
| Signed checkpoints and trust levels | Built | `local_chain_only` is Experimental. [Trust](guide/trust.md) |
| Copy checkpoints to S3, GCS or Azure | Experimental | Tested on an in-memory filesystem, never a real bucket. |
| Resume helpers for framework files | Built for Lightning | Ultralytics is Experimental. MMEngine, Hugging Face and RF-DETR files are not resumed. |
| Spot instances with SkyPilot | Experimental | Never run on SkyPilot or a real spot instance. [SkyPilot](guide/skypilot.md) |
| Server, remote monitoring and sync | Planned | A read-only server is designed, not built. |

CV-Pilot installs from a git clone on Python 3.10 to 3.14. Nothing is published on a package
index. What changed in each release is in the
[changelog](https://github.com/B4teShai/CV-Pilot/blob/main/CHANGELOG.md).
