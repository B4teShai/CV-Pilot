# ADR-0006: Checkpoint semantics

**Status:** accepted
**Date:** 2026-10-14

## Context

Training jobs on spot instances and shared clusters are killed without warning or with a
short grace window (SIGTERM, cloud preemption notices). A checkpoint written at that moment
can be truncated on local disk, half-uploaded to object storage, or complete on disk but
missing from the bucket. Frameworks resolve "last" by file name (Lightning `ckpt_path="last"`,
Ultralytics `resume=True` on `last.pt`) and do not verify what they load; a truncated file
turns into a crash at startup or, worse, a silent restart from scratch. SkyPilot managed jobs
relaunch on another node and lose local disk unless `/checkpoint` is a mounted bucket, and
recommend loading a checkpoint on every start.

A resumable checkpoint needs more than weights: optimizer, scheduler, AMP scaler, RNG state
and dataloader/sampler state. A deployable checkpoint needs weights and EMA only. Keeping
every full-state checkpoint is too expensive; keeping only one is unsafe.

Checkpoints must be loadable without arbitrary code execution
([../engineering/standards.md](../engineering/standards.md): `weights_only=True`,
`safetensors` preferred).

## Options considered

### Option A: framework-native "last" checkpoint

Rely on `ModelCheckpoint(save_last=True)` and `last.pt`.

Trade-offs: zero code; but no checksum, no atomic rename on some paths, no manifest of
what the file contains, no distinction between "exists" and "valid", and no handling of a
kill during the write or the upload.

### Option B: cloud-native checkpointing (bucket-only, framework-agnostic streaming)

Write directly to object storage with multipart upload and treat the bucket as the only
location.

Trade-offs: survives node loss by construction; but every save pays network latency, a
preemption mid-upload leaves an invisible partial object or a visible stale one, and
on-prem clusters without a bucket get nothing.

### Option C: atomic local write, sidecar manifest, LATEST pointer, verified restore

Write to a temporary file, `fsync`, hash, write a manifest, rename, then update a `LATEST`
pointer; upload to the bucket after the local file is verified; on restart walk back from
`LATEST` to the newest checkpoint that verifies and loads.

Trade-offs: more code and a chaos test tier; in return every state is recoverable and the
semantics are the same on a laptop, a SLURM node and a spot instance.

## Decision

Option C, implemented by `ckpt/` with the following rules.

1. **Atomic write.** Save to `<dir>/<run_id>/<kind>/step_<step:09d>.<ext>.tmp`, call
   `fsync` on the file and its directory, compute SHA-256 over the file, write
   `step_<step:09d>.manifest.json` (a `CheckpointManifest`), then `os.replace` the `.tmp`
   to its final name. Only after the manifest exists and the rename succeeded is `LATEST`
   (or `BEST`) rewritten atomically with the manifest's relative path.
2. **Integrity on restore.** `restore_latest_valid()` reads `LATEST`, verifies the file size
   and SHA-256 against the manifest, loads it (`safetensors`, or `torch.load(...,
   weights_only=True)`), and returns the state. If any step fails it logs
   `checkpoint_skipped_corrupt` and tries the previous manifest in step order. It returns
   `None` when no valid checkpoint exists **and nothing is known to have existed**: when the run
   store, a tag-checked pointer or `min_step` proves that a step was saved, finding nothing at or
   above it raises `CheckpointUntrustedError` (`rolled_back` or `checkpoint_missing`, with the
   step in the message) under `anchored` and `authenticated`; under `any` it returns `None` and
   logs `checkpoint_progress_lost` with the floor. Starting over is `accept_rollback_to=0`.
   Files without a manifest are never loaded.
3. **Last versus best.** `kind="last"` holds the full training state (model, optimizer,
   scheduler, scaler, RNG, dataloader state) in a `.pt` file loaded with `weights_only=True`.
   `kind="best"` holds model weights and EMA as `safetensors`. Retention is per kind:
   `keep_last` and `keep_best`, defaults 3 and 2.
4. **Never delete before verify.** A checkpoint is eligible for deletion only when a newer
   checkpoint of the same kind has been written, verified and (if a remote is configured)
   uploaded with a matching remote checksum. The newest valid checkpoint of each kind is
   never deleted.
5. **Remote mirror.** When `remote=` is set, the `.pt`/`.safetensors` file and its manifest
   are uploaded through `fsspec` after local verification; the remote manifest is written
   last. On restore, if local `LATEST` is missing (new node), the manager lists remote
   manifests and applies rule 2 to them, downloading to `<dir>` first. A restart has a new run
   id, so the remote is searched by the run's lineage (the run store must have survived) or, with
   `restore(discover=True)`, by listing the newest runs on the remote and ordering their
   candidates by step; every candidate passes the same gate before a byte is installed.
6. **Preemption.** A handler for SIGTERM and SIGUSR1 (configurable) and a poller for cloud
   preemption notices (AWS and GCP metadata endpoints; Azure has no notice source and relies on
   the signal) set a flag; `maybe_save` saves
   at the next call inside the grace window regardless of `every_n_minutes`. The run is
   marked `preempted`. A run that calls `restore_latest_valid()` and finds a state starts
   as a new run with `parent_run_id` set to the preempted run and continues its step
   timeline.
7. **Framework adapters** resolve resume paths through the manager: `resume_path()` returns
   the latest *verified* `last` file or `None`. The adapters pass it on (Lightning `ckpt_path`,
   Ultralytics `resume=` target) only for a `pt` checkpoint that this process verified: never
   the string `"last"`, an arbitrary path or a safetensors file, which is the manager's own
   format. A `pt` file skipped by the strict scan is reported (`resume_point_skipped`,
   `resume_point_unavailable`), not silently dropped. Whether a particular framework file
   scans clean is a property of that file: Ultralytics `last.pt` pickles the model object and
   does not.
8. **Save overhead budget.** Hashing and manifest writing add at most 5 % to the wall time of
   `torch.save` for the same state on local SSD; measured by `benchmarks/ckpt_save.py`.

## Amendment (2026-10-15): trust and formats as built

- **Trust.** The directory's digests and chain detect damage, not an adversary who can write
  the directory. Restore therefore reports a trust level (`authenticated`: HMAC tag verified
  with `CVPILOT_CKPT_KEY`; `anchored`: manifest digest matches the run store; `local_chain_only`)
  and enforces a policy (`CVPILOT_CKPT_REQUIRE_TRUST`, default `any`). Under `anchored` and
  `authenticated` each candidate needs its own proof, nothing is vouched for by a neighbour, and
  a failure to reach the level raises `checkpoint_untrusted` instead of loading a weaker file.
  See "Who can write the checkpoint directory" in
  [../architecture/06-cross-cutting.md](../architecture/06-cross-cutting.md).
- **Formats.** safetensors is the default for both kinds (rule 3's `.pt` for `last` is replaced:
  non-tensor state is bounded JSON inside the safetensors file). The pickle-based format is
  named `pt` in the manifest, is opt-in (`allow_pt_format` / `CVPILOT_CKPT_ALLOW_PT=1`), needs
  the framework at 2.6 or newer because `weights_only=True` was bypassed before that, is
  pre-scanned without being executed and is loaded only with `weights_only=True` from the
  verified bytes.
- **Names.** The pointers are `LATEST` and `LATEST_BEST` (`BEST` would collide with the `best/`
  directory on case-insensitive filesystems). Manifests also carry `prev_manifest_sha256`,
  `prev_chain_sha256`, `chain_sha256`, `metric_mode` and `auth_tag`; `uri` is the intended
  remote location, and a `step_N.remote.json` marker records a verified upload.

- **One parser.** What is checked is what is loaded. The safetensors header is parsed once
  by a strict reader (duplicate keys including a repeated `__metadata__`, NaN and Infinity,
  fractions, negative or oversized integers, non-UTF-8 or BOM text, trailing bytes, unknown
  dtype spellings, wrong byte counts, overlapping, gapped or unsorted spans, bounded rank,
  element and tensor counts), and after the library loads the same bytes the loaded names,
  dtypes, shapes and byte total must equal what the strict parse declared. A `.pt` archive is
  read by our own bounded zip reader, not by the loader: central and local headers must agree,
  nothing may precede, separate or follow the members, names are unique ignoring case, there
  are no encrypted, descriptor, zip64, extra-field or comment constructs (a local header may
  carry the framework's alignment padding only), `data.pkl` is decompressed and its checksum
  verified here, and the loaded state is bounded against the archive's tensor data. Residual
  risk for `.pt`: the reader has been exercised on archives built by the standard library and
  by hand, not on files written by a real framework release, archives over 4 GiB are refused
  (zip64), and tensor names and dtypes cannot be derived from the pickle without executing it,
  so they are bounded, not compared.

- **A restored checkpoint is data.** Even at trust level `local_chain_only`, a forged checkpoint
  can supply weights and training state and cannot execute code through cvpilot: the JSON
  skeleton is parsed strictly (no duplicate keys, NaN, Infinity or fractions where integers
  belong, integers within 19 digits, at most 1 MiB, depth and node limits) and rebuilt only into
  dicts, lists, strings, numbers, booleans, `None`, bytes and arrays, which `validate_plain`
  re-checks; nothing restored is used to import a module, look up an attribute or pick a class
  (the `state_dict` and tensor-conversion lookups run on objects the caller passed when saving,
  never on restored values). RNG structures are validated completely (Python: version 3 and 625
  integers in range; NumPy: `MT19937`, 624 `uint32` values, position, flag and finite cached
  value; framework: a `uint8` array of the exact size) before any generator is touched, applied
  together and rolled back if any step fails; a checkpoint with a malformed RNG state is skipped
  as `unloadable`. What a forged checkpoint can still do is what weights and optimizer state can
  do: degrade or bias a model, or exhaust memory up to the bounds above; that is why a shared
  directory needs `authenticated`.

## Consequences

- Easier: any kill, at any point of save or upload, leaves a loadable checkpoint; resume is
  one call; the same recipe works for SkyPilot (`/checkpoint` mounted in MOUNT mode), SLURM
  and local runs.
- Harder: a chaos test tier (`tests/chaos/`) that kills the process mid-write and truncates
  uploads is mandatory before `ckpt/` is marked built; retention needs care when the
  remote is slower than the local save interval.
- Forbidden: loading a checkpoint without a manifest; `torch.load` without
  `weights_only=True`; deleting the only valid checkpoint of a kind.
- Built: `cvpilot ckpt list|verify|prune` (instead of `ls|verify|restore`); follow-up: SkyPilot and SLURM recipes in the
  documentation, `preempted`/`resumed` alert rules.

## Links

- [../architecture/04-flows.md](../architecture/04-flows.md) (flow B)
- [../architecture/03-data-model.md](../architecture/03-data-model.md) (`CheckpointManifest`)
- [0005-run-store.md](0005-run-store.md)
