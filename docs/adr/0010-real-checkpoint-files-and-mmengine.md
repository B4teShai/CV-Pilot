# ADR-0010: Checkpoint claims proven on real framework files, and the MMEngine hook

**Status:** accepted
**Date:** 2026-10-18

## Context

The checkpoint verifier, the strict `.pt` archive reader and the resume helpers were built and tested
against files that tests wrote themselves (Python's `zipfile`, fake frameworks). Three claims had never
met a real file: that a framework's own resume file can be adopted, verified and resumed; that the
strict reader reads what the framework writes; and that a save stays within 5 % of the framework's
own. Notebook 04 ran all of it on real files from real short training runs (Lightning, Ultralytics
YOLO11n on COCO128, a Hugging Face `Trainer` fine-tune of ResNet-18, MMEngine) and found:

1. **The reader refused every file the framework writes**, with `unsafe_archive` ("bytes lie between
   the members and the central directory"). The framework's writer (checked with the minimum supported
   release and the newest installed one) flags every member with a zip data descriptor, writes zero
   checksum and sizes in the local header and puts a zip64 end record and locator in front of the
   classic end record, even for a small file. The reader had been written for the layout Python's
   `zipfile` produces. This also meant that CheckpointManager's own opt-in `pt` format could not be
   restored with the real framework, and that the real-library smoke tests (which had never been run)
   had two bugs of their own.
2. **`adopt` stored files that resume would refuse**, logged them as saved and verified, and a restart
   then got `None`, which looks like a fresh start. An Ultralytics `last.pt` is this case: it pickles
   the model object.
3. **`cvpilot ckpt verify` and `list` ignored the `pt` opt-in** (every `pt` checkpoint showed as
   `pt_not_allowed`), and verify did not scan the archive, so it could call valid a file that resume
   refuses.
4. A pickle written with protocol 2 names the builtin `set` as `__builtin__.set`; the allow-list only
   knew `builtins.set`.
5. The message for a file that is not a zip archive (a safetensors file, a JSON file) said "raw
   pickles are refused", which is wrong for those.
6. The guide's restore snippet passed NumPy arrays to `load_state_dict`.

The save-overhead budget (5 % over `torch.save`) had been recorded as not met without a measurement
against `torch.save` itself.

## Options considered

### Option A: accept any zip archive that the framework's loader accepts

Trade-offs: no more refused files. But the reader's whole purpose is to decide before any framework
code runs; loosening it to "whatever opens" gives up the point. Rejected.

### Option B: accept exactly the layout the framework writes, and nothing looser

A member may carry the descriptor flag only with zero local values and one signed 16-byte descriptor
repeating the central directory; a zip64 end record is accepted only as the plain 56-byte record whose
every field equals the classic record's, with a locator that points at it. Trade-offs: a small amount
of strict code, each deviation a test. Chosen.

### Option C: allow-list the framework's own classes in the pickle scan (Ultralytics, MMEngine)

Trade-offs: those files would verify and resume. But the framework then unpickles them with
`weights_only=False`, which executes whatever the pickle says; an allow-list of class names does not
bound what their constructors do. Rejected.

### Option D: digest-only adoption of pickled model objects

Record size and SHA-256 of an Ultralytics `last.pt` without scanning, and hand the path back. Trade-offs:
tamper evidence for the most used detection framework. But it hands an unscanned pickle to a loader
that runs code, so a writer of the directory who also forges the run store gets code execution; the
trust levels would then mean "this is the file I saved", not "this is safe to open". Not built (see
below); the honest result is that CV-Pilot cannot vouch for these files.

## Decision

Option B, and:

1. **Layout, exactly** (`ckpt/serializers.py`): tests build archives byte for byte like the framework
   (`tests/zip_support.py`, compared with a file the installed framework wrote by a marked test) and
   every deviation (wrong descriptor values, descriptor without signature, sizes in the local header
   next to the flag, a zip64 record that disagrees, extensible data, a locator elsewhere, trailing bytes)
   is refused.
2. **Scan at adopt time.** `adopt` runs the same check resume runs (loader version, archive scan) before
   it stores anything. The scan at resume time stays: it reads the bytes that were hashed.
3. **Verify judges like resume.** `RunCheckpoints.inspect` (behind `ckpt verify` and retention) scans a
   `pt` archive, but does not need the framework installed (the loader version belongs to the machine
   that resumes). The command line honours `CVPILOT_CKPT_ALLOW_PT`.
4. The protocol-2 `set` global is allowed; the non-zip message names what is not adopted.
5. **MMEngine hook** (`hooks/mmengine.py`, extra `mmengine`): a `Hook` built at attach time records
   `train/` scalars every `interval` iterations (default 10, like the framework's own logging), `val/` and
   `test/` metrics, at `step = runner.iter`, rank zero only; `train`, `val` and `test` are wrapped so an
   exception marks the run failed and an interrupt preempted; the run finishes when `test` returns (or at
   exit after `train`). Same guard, defensive conversion, no network and no pickling as the Lightning
   and Hugging Face hooks. No resume helper, because the framework's checkpoints pickle
   `HistoryBuffer` by class and the scan refuses them.
6. **Save overhead: the 5 % rule is withdrawn** and replaced by the measured numbers below.

Measured (notebook 04, macOS arm64, local SSD, Python 3.13, median of 9 steady-state saves with
`keep_last=3`, against `torch.save` of the same state written and `fsync`ed to the same disk; one run,
and a repeat of the same notebook changed the YOLO11n comparison between 47 % and 103 % because the
framework's own save takes a few milliseconds):

| State | Bytes | `save_now` | `torch.save` | Ratio | Two SHA-256 passes |
|---|---|---|---|---|---|
| tiny Lightning classifier | 15 692 | 4.36 ms | 0.31 ms | 14.1 x | under 1 % |
| YOLO11n weights (float32) | 10 653 476 | 17.52 ms | 11.89 ms | 1.5 x | 37 % |
| ResNet-18 weights | 44 790 276 | 53.30 ms | 15.36 ms | 3.5 x | 51 % |

Where the time goes for ResNet-18: writing the file while hashing it 39.8 %, reading it back and hashing it
again (the verification that makes a save trustworthy) 33.0 %, serialising 11.8 %, retention 5.9 %, the
pointer 1.3 %. The fsync is not the cost on this machine (macOS `fsync` flushes to the drive cache, so a
Linux server will differ). For small states the fixed cost (retention, pointer, manifest, locking) dominates
and the ratio is meaningless. At the default save interval of 10 minutes the ResNet-18 save is 0.009 % of the
wall time. Two SHA-256 passes alone are 27 ms for 44.8 MB, more than `torch.save` takes in total (15.4 ms), so no
change that keeps the digest and the read-back can reach 5 %.

## What was deliberately not built

- **Digest-only or class-allow-listed adoption of pickled model objects** (Ultralytics `last.pt`,
  MMEngine `.pth`): see options C and D. Ultralytics and MMEngine resume stays the framework's own and is
  not verified by CV-Pilot.
- **Verification of a Hugging Face `Trainer` checkpoint folder** (a manifest over a directory). It would
  be a new verification mechanism; the supported verified path for that model is `CheckpointManager`
  (model and optimizer state as safetensors), which notebook 04 shows restoring bit-identical weights and
  optimizer state.
- **Resume helpers for Hugging Face and MMEngine.**
- **An MMEngine `VisBackend`, a checkpoint hook or per-class metrics**; the hook records scalars and
  metrics only. The interval is a constructor argument, not a setting or an environment variable.
- **Save-path optimisations**: hashing in a thread while writing, comparing the read-back with the bytes in
  memory instead of hashing again, hashing with a different function. The first two would save a fraction of
  a save and touch the verification code; none can reach the old budget.
- **zip64 archives** (over 4 GiB), **legacy non-zip `.pt` files**, a safetensors export of Ultralytics
  state.
- **A real bucket test of the remote mirror** and of the cloud preemption notices: they remain tested
  against an in-memory filesystem and local listeners and are labelled experimental in the status page.

## Consequences

- Lightning `last.ckpt` files are adopted, verified under `anchored` and `authenticated`, tamper-checked
  and resumed with identical weights; Ultralytics, MMEngine and Hugging Face resume files are refused,
  with the reader's exact message recorded in notebook 04 and the guide.
- A file that cannot be resumed is refused when it is adopted, not at the restart that needs it.
- The framework-layout test builder is a second description of the layout; the marked smoke test fails if
  the framework changes it.
- `ckpt verify` of a large `pt` file reads it into memory once (the same bound as resume).
- The numbers above are for one machine and one run; they are a record, not a promise.

## Links

- [0006-checkpoint-semantics.md](0006-checkpoint-semantics.md), [0009-alerts-and-hf-hook.md](0009-alerts-and-hf-hook.md)
- [../guide/checkpoints.md](../guide/checkpoints.md), [../guide/trust.md](../guide/trust.md), [../status.md](../status.md)
