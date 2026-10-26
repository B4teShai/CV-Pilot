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
   the members and the central directory"). The framework's writer (files from 2.6.0, the minimum
   supported release, and from 2.14.1 are kept in `tests/data`) flags every member with a zip data descriptor, writes zero
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
7. The member-name rule refused the folder the framework derives from the file stem: a Lightning file
   saved as `epoch=1-step=8.ckpt` (its default name) has members under `epoch=1-step=8/`, with `=`;
   `my model.pt` has a space. Both were refused.
8. A state with two names for one tensor (a shared embedding) was refused as `unloadable`, because the
   size bound counted the storage once per name.
9. The allow-listed codec call (`_codecs.encode`) was reachable by the framework's loader with any
   arguments, so a nested hex or base64 encode could multiply memory before the loader returned.
10. `adopt` read the source twice (once to scan and hash, once to store), so a swapped or still-written
    file could be stored as verified, and held about twice the file in memory.

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

1. **Layout, exactly** (`ckpt/serializers.py`): the scanner accepts the four real files kept in
   `tests/data` (written by 2.6.0 and 2.14.1, one plain and one saved to `epoch=1-step=8.ckpt`), and the
   test builder `tests/zip_support.py` reproduces each of them byte for byte; a marked test repeats the
   comparison with the installed framework. That is the evidence: two releases and two file names, not
   every release. Every deviation (wrong descriptor values, descriptor without signature, sizes in the
   local header next to the flag, a zip64 record that disagrees, extensible data, a locator elsewhere,
   bytes before the end record, trailing bytes, hidden gaps, encryption flags, a newer zip version,
   stored members with unequal sizes or a wrong checksum) is refused by its own check, and each test
   asserts which check fired. A mutation run over the checks (removing each in turn) was killed by the
   tests in all but one case, which cannot matter (the bytes after a descriptor always exist).
2. **Scan at adopt time, one read.** `adopt` reads the source once into one buffer sized from `fstat`
   (refusing a file over the zip64 limit before reading a byte), runs the same check resume runs (loader
   version, archive scan) on those bytes, and stores, hashes and tags those same bytes: a swap or a rewrite
   between two reads cannot happen. The scan at resume time stays: it reads the bytes that were hashed.
   `fsops.read_regular_file` fails if the file changes size while it is read.
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
6. **Pickle scan and archive hardening.** `_codecs.encode` is allowed only as `encode(<string>, 'latin1')`
   (the shape protocol 2 uses for a byte string; a different codec or an argument that is itself a call is
   refused); stored members need equal sizes and a matching checksum; a development, alpha, beta or
   candidate build of the first fixed release is not a safe loader; a loader whose own `__version__`
   disagrees with its package metadata is refused; text taken from a pickle is cut to 40 plain ASCII
   characters in messages; tensors that share a storage are counted once; folder names may use
   `= + , @ ( ) [ ]` and a space (not shell or path characters, not non-ASCII: a non-ASCII stem makes the
   writer fall back to `archive`).
7. **Save overhead: the 5 % rule is withdrawn** and replaced by the measured numbers below.

Measured (notebook 04, macOS arm64, local SSD, Python 3.13, median of 9 steady-state saves with
`keep_last=3`, after a warm-up, with the save, the framework's own save and the plain write interleaved in
a seeded random order inside each round, against `torch.save` of the same state written and `fsync`ed to
the same disk). Over three runs of the notebook the YOLO11n ratio ranged from 1.5 to 2.0 (47 % to 103 %
over `torch.save`) and the ResNet-18 ratio from 3.2 to 3.5, because the framework's own save takes only
about 12 ms; the table is the run committed with this record (2026-10-18):

| State | Bytes | `save_now` | `torch.save` | Ratio | Two SHA-256 passes |
|---|---|---|---|---|---|
| tiny Lightning classifier | 15 692 | 4.48 ms | 0.44 ms | 10.2 x | under 1 % |
| YOLO11n weights (float32) | 10 653 476 | 19.34 ms | 12.05 ms | 1.6 x | 36 % |
| ResNet-18 weights | 44 790 276 | 52.71 ms | 16.67 ms | 3.2 x | 56 % |

Notebook 04 was re-executed twice on 2026-10-26 (its download cache moved to a private folder, ADR-0012).
The committed run gave, on the same machine: the tiny classifier 4.58 ms against 0.47 ms, YOLO11n weights
21.20 ms against 11.60 ms (1.8 x, two SHA-256 passes 33 %), ResNet-18 weights 54.87 ms against 18.69 ms
(2.9 x, 54 %), 0.0091 % of the wall time at the default interval; the earlier re-run gave 54.12 ms against
16.44 ms (3.3 x). All runs are inside the ranges above; the ratio moves with the framework's own save, which
is the small number.

Where the time goes for ResNet-18 (exclusive times, nothing counted twice): writing the file while hashing it
40.3 %, reading it back and hashing it again (the verification that makes a save trustworthy) 33.2 %,
serialising 12.0 %, retention 5.7 %, the pointer 1.4 %, and 6.5 % that the phases do not explain (state
build, locking, scans, the timers). The fsync is not the cost on this machine (macOS `fsync` flushes to the
drive cache, so a Linux server will differ). For small states the fixed cost (retention, pointer, manifest,
locking) dominates and the ratio is meaningless. At the default save interval of 10 minutes the ResNet-18
save is 0.009 % of the wall time. Two SHA-256 passes alone are 29.5 ms for 44.8 MB (2 x 14.77 ms), more
than `torch.save` takes in total (16.67 ms), so no change that keeps the digest and the read-back can reach
5 %.

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

## Recorded limits and review items not taken

- **Local padding bytes and header attributes are not checked.** The alignment padding bytes of a local
  header are never read by anyone, and the central directory's version, time and attribute fields are not
  used (nothing is extracted), so requiring particular values would only create false refusals for
  archives made by other writers.
- **Verify-to-load gap.** `resume_arguments` hands a framework a path in the checkpoint directory, and a
  writer of that directory can still replace the file between the check and the framework's open. The
  helpers recheck inode, size and SHA-256 immediately before returning the path (now opening without
  following links), which narrows the window and does not close it. `cvpilot.ckpt.private_copy` closes it
  (a private 0700 directory, digest checked while copying) and is the answer for a directory others can
  write: a `pt` checkpoint in a directory that is not yours alone is outside what the default helpers
  protect. The helpers do not copy by default, because a copy of a multi-gigabyte file on every resume is a
  cost most single-user setups do not need.
- **Memory.** A `pt` file is read into memory once for adopt, verify and resume (the same bound as
  resume), up to the zip64 limit (4 GiB) or `max_checkpoint_bytes`, whichever is smaller.
- **A non-integer `RANK` counts as rank zero** (documented), so a garbled launcher variable duplicates
  logging instead of silencing it; only plain ASCII digits are read, so a non-ASCII digit string no longer
  raises.
- **Detaching two hooks from one runner in the wrong order** can drop the other's wrapper (the stage
  wrappers nest); one hook attached twice undoes cleanly.

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
