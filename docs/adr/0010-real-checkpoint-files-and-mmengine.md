# ADR-0010: Checkpoint claims proven on real framework files, and the MMEngine hook

**Status:** accepted
**Date:** 2026-10-18

## Context

The checkpoint verifier, the strict `.pt` archive reader and the resume helpers had been tested only
against files that tests wrote (Python's `zipfile`, fake frameworks). Three claims had never met a
real file: that a framework's own resume file can be adopted, verified and resumed; that the strict
reader reads what the framework writes; and that a save stays within 5 % of the framework's own.
Notebook 04 ran all of it on real files from short real training runs (Lightning, Ultralytics
YOLO11n on COCO128, a Hugging Face `Trainer` fine-tune of ResNet-18, MMEngine) and found:

1. **The reader refused every file the framework writes** (`unsafe_archive`, "bytes lie between the
   members and the central directory"). The framework's writer (files from 2.6.0, the minimum
   supported release, and from 2.14.1 are kept in `tests/data`) flags every member with a zip data
   descriptor, writes zero checksum and sizes in the local header and puts a zip64 end record and
   locator before the classic end record, even for a small file. The reader had been written for
   the layout Python's `zipfile` produces. So CheckpointManager's own opt-in `pt` format could not
   be restored with the real framework either.
2. **`adopt` stored files that resume would refuse**, logged them as verified, and a restart then got
   `None`, which looks like a fresh start. An Ultralytics `last.pt` is this case: it pickles the
   model object.
3. **`ckpt verify` and `list` ignored the `pt` opt-in** (every `pt` checkpoint showed as
   `pt_not_allowed`), and verify did not scan the archive, so it could call valid a file that
   resume refuses.
4. A protocol-2 pickle names the builtin `set` as `__builtin__.set`; the allow-list knew only
   `builtins.set`.
5. The message for a non-zip file (a safetensors or JSON file) said "raw pickles are refused".
6. The guide's restore snippet passed NumPy arrays to `load_state_dict`.
7. The member-name rule refused the folder the framework derives from the file stem: a Lightning
   file `epoch=1-step=8.ckpt` has members under `epoch=1-step=8/`, and `my model.pt` has a space.
8. A state with two names for one tensor (a shared embedding) was refused as `unloadable`, because
   the size bound counted the storage once per name.
9. The allow-listed `_codecs.encode` could be called by the loader with any arguments, so a nested
   hex or base64 encode could multiply memory.
10. `adopt` read the source twice (scan and hash, then store), so a swapped or still-written file
    could be stored as verified, and it held about twice the file in memory.

The 5 % save-overhead budget had been recorded as not met without measuring against `torch.save`.

## Options considered

### Option A: accept any zip archive the framework's loader accepts

No more refused files, but the reader exists to decide before any framework code runs, and "whatever
opens" gives that up. Rejected.

### Option B: accept exactly the layout the framework writes, and nothing looser

A member may carry the descriptor flag only with zero local values and one signed 16-byte descriptor
repeating the central directory. A zip64 end record is accepted only as the plain 56-byte record
whose every field equals the classic record's, with a locator that points at it. A little strict
code, each deviation a test. Chosen.

### Option C: allow-list the framework's own classes in the pickle scan (Ultralytics, MMEngine)

Those files would verify and resume, but the framework then unpickles them with `weights_only=False`,
which runs whatever the pickle says, and a class-name allow-list does not bound what constructors
do. Rejected.

### Option D: digest-only adoption of pickled model objects

Record size and SHA-256 of an Ultralytics `last.pt` without scanning and hand the path back. That is
tamper evidence for the most used detection framework, but it hands an unscanned pickle to a loader
that runs code, so a writer of the directory who also forges the run store gets code execution, and
trust levels would mean "the file I saved", not "safe to open". Not built: CV-Pilot cannot vouch for
these files.

## Decision

Option B, and:

1. **Layout, exactly** (`ckpt/serializers.py`). The scanner accepts the four real files in
   `tests/data` (2.6.0 and 2.14.1, one plain and one saved to `epoch=1-step=8.ckpt`), and the test
   builder `tests/zip_support.py` reproduces each byte for byte. A marked test repeats the
   comparison with the installed framework. That is the evidence: two releases and two file names,
   not every release. Every deviation (wrong descriptor values, no signature, sizes beside the flag,
   a disagreeing zip64 record, extensible data, a locator elsewhere, bytes before the end record or
   trailing, hidden gaps, encryption flags, a newer zip version, stored members with unequal sizes
   or a wrong checksum) is refused by its own check and each test asserts which check fired. A
   mutation run removing each check in turn was caught by the tests in all but one case, which
   cannot matter.
2. **Scan at adopt time, one read.** `adopt` reads the source once into one buffer sized from `fstat`
   (refusing a file over the zip64 limit before reading), runs the same check resume runs on those
   bytes, and stores, hashes and tags the same bytes. The scan at resume stays. `fsops.read_regular_file`
   fails if the file changes size while read.
3. **Verify judges like resume.** `RunCheckpoints.inspect` (behind `ckpt verify` and retention)
   scans a `pt` archive but does not need the framework installed. The command line honours
   `CVPILOT_CKPT_ALLOW_PT`.
4. The protocol-2 `set` global is allowed, and the non-zip message names what is not adopted.
5. **MMEngine hook** (`hooks/mmengine.py`, extra `mmengine`). A `Hook` built at attach time records
   `train/` scalars every `interval` iterations (default 10), `val/` and `test/` metrics, at
   `step = runner.iter`, rank zero only. `train`, `val` and `test` are wrapped so an exception marks
   the run failed and an interrupt preempted. The run finishes when `test` returns, or at exit after
   `train`. Same guard, defensive conversion, no network and no pickling as the other hooks. No
   resume helper, because MMEngine checkpoints pickle `HistoryBuffer` by class and the scan refuses
   it.
6. **Pickle scan and archive hardening.** `_codecs.encode` is allowed only as
   `encode(<string>, 'latin1')`. Stored members need equal sizes and a matching checksum. A
   development, alpha, beta or candidate build of the first fixed release is not a safe loader, and
   neither is one whose `__version__` disagrees with its package metadata. Text from a pickle is cut
   to 40 plain ASCII characters in messages. Tensors sharing a storage are counted once. Folder
   names may use `= + , @ ( ) [ ]` and a space (not shell or path characters, not non-ASCII: the
   writer falls back to `archive` for a non-ASCII stem).
7. **Save overhead: the 5 % rule is withdrawn** and replaced by measurements.

Measured (notebook 04, macOS arm64, local SSD, Python 3.13, median of 9 steady-state saves with
`keep_last=3` after a warm-up, interleaved in random order, against `torch.save` of the same state
written and `fsync`ed to the same disk). Over three runs the YOLO11n ratio ranged from 1.5 to 2.0
and ResNet-18 from 2.9 to 3.5, because the framework's own save takes only about 12 ms. The committed
run of 2026-10-18:

| State | Bytes | `save_now` | `torch.save` | Ratio | Two SHA-256 passes |
|---|---|---|---|---|---|
| tiny Lightning classifier | 15 692 | 4.48 ms | 0.44 ms | 10.2 x | under 1 % |
| YOLO11n weights (float32) | 10 653 476 | 19.34 ms | 12.05 ms | 1.6 x | 36 % |
| ResNet-18 weights | 44 790 276 | 52.71 ms | 16.67 ms | 3.2 x | 56 % |

A later re-execution on 2026-10-26, same machine, gave 4.58 ms against 0.47 ms, YOLO11n 21.20 ms
against 11.60 ms (1.8 x, hashing 33 %) and ResNet-18 54.87 ms against 18.69 ms (2.9 x, 54 %), or
0.0091 % of the wall time at the default interval. All runs are inside the ranges above.

Where the ResNet-18 time goes (exclusive): writing while hashing 40.3 %, reading back and hashing
again 33.2 %, serialising 12.0 %, retention 5.7 %, the pointer 1.4 %, unexplained 6.5 %. `fsync` is
not the cost on this machine (macOS flushes to the drive cache, so a Linux server will differ). For
small states fixed costs dominate and the ratio is meaningless. At the default 10-minute interval a
ResNet-18 save is 0.009 % of the wall time. Two SHA-256 passes alone take 29.5 ms for 44.8 MB, more
than `torch.save` takes in total, so no change that keeps the digest and the read-back can reach 5 %.

## What was deliberately not built

- **Digest-only or class-allow-listed adoption of pickled model objects** (Ultralytics `last.pt`,
  MMEngine `.pth`): options C and D. Their resume stays the framework's own and is not verified.
- **Verification of a Hugging Face `Trainer` checkpoint folder** (a manifest over a directory). It
  would be a new mechanism. The verified path for that model is `CheckpointManager` (safetensors),
  which notebook 04 shows restoring bit-identical weights and optimizer state.
- **Resume helpers for Hugging Face and MMEngine.**
- **An MMEngine `VisBackend`, a checkpoint hook or per-class metrics.** The interval is a
  constructor argument, not a setting.
- **Save-path optimisations** (hashing in a thread while writing, comparing the read-back with
  memory, another hash function). The first two save a fraction of a save and touch verification
  code, and none can reach the old budget.
- **zip64 archives** (over 4 GiB), **legacy non-zip `.pt` files**, a safetensors export of
  Ultralytics state.
- **A real-bucket test of the remote mirror and of the cloud notices.** They stay tested against an
  in-memory filesystem and local listeners and are labelled experimental in the status page.

## Recorded limits and review items not taken

- **Local padding bytes and header attributes are not checked.** Nobody reads the alignment padding,
  and nothing is extracted, so requiring values would only refuse archives from other writers.
- **Verify-to-load gap.** `resume_arguments` hands a framework a path, and a writer of the directory
  can swap the file between the check and the framework's open. The helpers recheck inode, size and
  SHA-256 just before returning the path (opening without following links), which narrows the window
  without closing it. `cvpilot.ckpt.private_copy` closes it (a private 0700 directory, digest
  checked while copying) and is the answer for a directory others can write. The helpers do not copy
  by default, because copying a multi-gigabyte file on every resume costs most single-user setups
  more than it gives.
- **Memory.** A `pt` file is read into memory once for adopt, verify and resume, up to the zip64
  limit (4 GiB) or `max_checkpoint_bytes`, whichever is smaller.
- **A non-integer `RANK` counts as rank zero**, so a garbled launcher variable duplicates logging
  instead of silencing it. Only plain ASCII digits are read.
- **Detaching two hooks from one runner in the wrong order** can drop the other's wrapper; one hook
  attached twice undoes cleanly.

## Consequences

- Lightning `last.ckpt` files are adopted, verified under `anchored` and `authenticated`,
  tamper-checked and resumed with identical weights. Ultralytics, MMEngine and Hugging Face resume
  files are refused, with the reader's exact message recorded in notebook 04 and the guide.
- A file that cannot be resumed is refused when adopted, not at the restart that needs it.
- The test builder is a second description of the layout, and the marked smoke test fails if the
  framework changes it.
- `ckpt verify` of a large `pt` file reads it into memory once.
- The numbers are for one machine and one run: a record, not a promise.

## Links

- [0006-checkpoint-semantics.md](0006-checkpoint-semantics.md), [0009-alerts-and-hf-hook.md](0009-alerts-and-hf-hook.md)
- [../guide/checkpoints.md](../guide/checkpoints.md), [../guide/trust.md](../guide/trust.md), [../status.md](../status.md)
