# ADR-0012: Crash safety under real kills, the RF-DETR hook and a spot recipe

**Status:** accepted
**Date:** 2026-10-26

## Context

The product's first promise is checkpoints that survive a crash. Until now that rested on a few
dozen kills (`tests/chaos/test_ckpt_sigkill.py`: six random delays and four chosen points). Three
more pieces of the roadmap were open: a hook for RF-DETR (Apache-2.0), the last framework of the
second set of hooks; a recipe for spot instances, where preemption is routine; and a decision about
the server (ADR-0011).

RF-DETR 1.11.2 trains through a trainer of the legacy Lightning package that `RFDETR.train()` builds
itself (`rfdetr.training.build_trainer`) and offers no callback argument; its old `callbacks`
dictionary is no longer read. Its `[train]` extra includes `roboflow`, an upload client that pins
`typer<0.26`. The published Nano weights are Apache-2.0 (the framework's model configuration says
so; XLarge and 2XLarge need a platform licence), about 366 MB, with an MD5 in the framework's
registry.

## Options considered

### Chaos: more random delays, or kills at exact places

Random delays (the existing test) land wherever the scheduler puts them: most land between saves.
Exact places need instrumentation inside the save, which tests the places named but nothing between
them. Chosen: both. A harness forks a writer per iteration; the writer stops itself (`SIGSTOP`) at a
seeded place inside the save and the harness kills it with `SIGKILL` there, 200 times at each of four
places; 200 more kills land 0 to 4 ms after a save started. Forking from a harness that has imported
everything keeps 1,000 iterations at about 2.5 minutes, so the suite stays in the default run.

### RF-DETR: wrap the trainer factory, or ask users to build the trainer

Users could call `build_trainer` themselves and pass our callbacks, but then they leave the
framework's documented `train()` path. Wrapping `build_trainer` process-wide would affect every model.
Chosen: wrap `train` on the model instance, and only for the duration of that call wrap the module's
`build_trainer`, adding the Lightning hook's logger and callback to the first trainer it builds; the
factory is put back when `train` returns or raises.

### RF-DETR extra: the framework's `[train]` extra, or its contents without `roboflow`

The lock resolves every extra together: `rfdetr[train]` held back the core `typer` (0.27.3 to 0.25.1)
for every install. Chosen: the extra lists `rfdetr` and the training dependencies of its `[train]`
extra without `roboflow`, which training does not import.

## Decision

1. **Chaos suite** (`tests/chaos/test_ckpt_kill_points.py`, `_ckpt_kill_points.py`): 200 seeded kills
   at each of `before_temp_write`, `mid_write` (a random number of bytes, at least one and less than
   the file, never fsynced), `after_write_before_rename` and `after_rename_before_chain` (after the
   manifest link, the data link, verification, the store record, or with the new pointer at its
   temporary name), and 200 at a random moment. Each iteration varies the number of earlier saves, the
   tensor size (4 bytes to 1 MiB), `keep_last`, a key and the `anchored` policy. After each kill a fresh
   process recovers and resumes the run, restores, compares the state byte for byte with what that step
   saved, checks every checkpoint the directory still admits, saves the next step and restores it. The
   expected step is exact for each place. `test_ckpt_truncated_upload.py` cuts 200 uploads to fsspec's
   in-memory filesystem (a dropped connection or a death; data or manifest; any byte count) and
   requires that a node with nothing restores exactly the newest completed upload and that every remote
   manifest describes a data object of its size and digest.
2. **What the suites found, fixed with their own tests**: a `memory://` remote could not save (its URL
   had a third slash); `preempted()` reported the raw flag, so the documented loop could leave without
   the preemption save; the preemption checkpoint was not uploaded once the run was marked (its store
   was closed). The local save path itself needed no change: all 1,000 kills restored the expected
   checkpoint.
3. **RF-DETR hook** (`hooks/rfdetr.py`, extra `rfdetr`), as described above: every scalar the
   framework logs at the trainer's global step, rank zero only (`trainer.is_global_zero` and `RANK`),
   callbacks under the `Guard`, `train` returning finishes the run, an exception fails it, an interrupt
   preempts it, the exception propagates unchanged. Fake-trainer unit tests; two `real_libs` smoke tests
   train a weightless RF-DETR Nano offline; notebook 05 fine-tunes the published Nano weights on 64
   COCO128 photos with the hook and finds the recorded values equal to the framework's `metrics.csv`.
4. **Licences before downloads.** Notebook 05 checks the installed package's licence, the Nano
   weights' licence in the framework's model configuration and the registry's URL and MD5 before the
   weights are downloaded; the download is SHA-256 pinned (the pin was computed once from the file
   whose MD5 equals the registry's, and the notebook checks the MD5 again).
5. **Spot recipe** (`examples/skypilot/`, guide page): a SkyPilot task file (spot, job recovery, the key
   as a secret) and a training script that restores with `discover=True`, saves on a notice or a signal
   and exits 143 after a preemption so the job is not reported done. Labelled experimental: no real
   spot instance or SkyPilot run was available. A docs test runs the script as two nodes against a
   directory-backed bucket with a real `SIGTERM`.

## What was deliberately not built

- **A resume helper for RF-DETR.** The framework's `train(resume=...)` reads the file with its own
  loader in the mode that falls back to full unpickling when the restricted load fails, so a file
  swapped after CV-Pilot verified it could be unpickled. The files RF-DETR wrote in notebook 05 pass the
  strict scan; that is not enough when the framework's next read can fall back. `CheckpointManager` is
  the verified path.
- **Accepting the published RF-DETR weights in CV-Pilot's reader.** The strict scan refuses them for
  one `argparse.Namespace`; the allow-list stays as it is, because CV-Pilot never loads those weights
  (the framework does, with PyTorch's restricted loader).
- **Kill tests on other file systems, power loss, or a real bucket.** The suite runs on the local disk
  of the machine that runs the tests; a kill is not a power cut (the page cache survives a killed
  process), and the cut uploads are against an in-memory filesystem.
- **Running SkyPilot or a real spot instance**, and a test of the real AWS or Google Cloud notice; the
  notice sources stay tested against fake metadata services.
- **The RF-DETR `[train]` extra as published**, with `roboflow` (see above), and a `torch_loop` hook for
  plain PyTorch.
- **Any server code** (ADR-0011 is a design).

## Consequences

- The crash-safety claim is backed by 1,000 kills at named places and 200 cut uploads, run on every
  `pytest` (about 2.5 minutes) and reproducible from the seed.
- `preempted()` changed meaning (handled, not requested); code that polled it for the request itself
  reads `flag.is_set`.
- The `rfdetr` extra must be kept in step with the framework's `[train]` extra by hand; a framework
  release that adds a training dependency breaks `train` until the extra is updated.
- RF-DETR's `build_trainer` is wrapped while `train` runs; another thread building an RF-DETR trainer
  at the same moment in the same process would also get the hook's logger and callback. One training
  per process is the supported case.

## Links

- [0006-checkpoint-semantics.md](0006-checkpoint-semantics.md),
  [0010-real-checkpoint-files-and-mmengine.md](0010-real-checkpoint-files-and-mmengine.md),
  [0011-minimal-read-only-server.md](0011-minimal-read-only-server.md)
- [../guide/checkpoints.md](../guide/checkpoints.md), [../guide/skypilot.md](../guide/skypilot.md),
  [../status.md](../status.md)
