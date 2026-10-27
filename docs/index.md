# CV-Pilot

CV-Pilot stops you from shipping a worse vision model and from losing training progress. It records
each training run, compares every new model with your current best, and fails your CI build if
something got worse. It also saves crash-proof checkpoints, so a killed job resumes where it
stopped. It runs on your machine from Python or the command line; no server is needed.

![CV-Pilot records your training, compares the new model with your best one and tells CI pass or fail](diagrams/png/overview.png)

- New here: [install it](guide/install.md), then follow the [tutorial](tutorial.md).
- Unsure of a word: the [glossary](guide/concepts.md#glossary).
- Look something up: [command line](reference/cli.md), [Python API](reference/api.md),
  [settings](reference/settings.md), [project status](status.md).
- How it is built: [architecture](architecture/README.md). To contribute, read the
  [engineering guide](engineering/README.md).

Apache-2.0.
