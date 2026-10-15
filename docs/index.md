# CV-Pilot

Regression gates, remote monitoring and resume-safe checkpoints for vision models, on top of
the experiment tracker you already use.

CV-Pilot records every run (environment, git state, dataset hash, metrics), evaluates detection
and classification predictions into immutable snapshots, pins baselines, and fails a pull request
when a model regresses. Its checkpoint manager writes atomically, verifies what it restores, and
survives a kill at any moment of a save.

## Start here

- [Install](guide/install.md), then the [Tutorial](tutorial.md): a synthetic run from logging to
  gating to a preemption and a damaged checkpoint. Every command in it is executed by a test.
- [Concepts](guide/concepts.md) for the vocabulary, [Checkpoints and resume](guide/checkpoints.md)
  for the checkpoint manager, [Gating in CI](guide/gating-ci.md) and the
  [Trust model](guide/trust.md) before you rely on either in a shared environment.
- [Command line](reference/cli.md) (generated from the code) and [Python API](reference/api.md).
- [Architecture](architecture/README.md), the [decision records](adr/0001-scope-and-license.md) and
  the [engineering guide](engineering/README.md) explain why it is built the way it is.

## Status

Version 0.1.0 is a release candidate: it has not been published to any index. The documents under
Architecture mark each component as built or planned; the server, alerts, sync and the other
frameworks' hooks are planned and are not part of this release.

## License

Apache-2.0. The optional Ultralytics hook talks to an AGPL-3.0 package that CV-Pilot never
imports; using that extra makes your combined work subject to that license.
