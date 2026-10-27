# ADR-0006: Checkpoint semantics

**Status:** accepted
**Date:** 2026-10-14

## Context

Training jobs on spot instances and shared clusters are killed without warning, or with a short
grace window (SIGTERM, cloud preemption notices). A checkpoint written at that moment can be
truncated on disk, half-uploaded, or complete locally but missing from the bucket. Frameworks resolve
"last" by file name (Lightning `ckpt_path="last"`, Ultralytics `resume=True`) and do not verify what
they load, so a truncated file becomes a crash at startup or a silent restart from scratch. SkyPilot
managed jobs relaunch on another node and lose local disk unless `/checkpoint` is a mounted bucket.

A resumable checkpoint needs optimizer, scheduler, AMP scaler, RNG and dataloader state as well as
weights. A deployable one needs weights and EMA only. Keeping every full checkpoint is too
expensive; keeping one is unsafe. Checkpoints must load without arbitrary code execution
([standards](../engineering/standards.md): `weights_only=True`, `safetensors` preferred).

## Options considered

### Option A: framework-native "last" checkpoint

Rely on `ModelCheckpoint(save_last=True)` and `last.pt`. Zero code, but no checksum, no atomic rename
on some paths, no manifest, no difference between "exists" and "valid", and nothing for a kill during
the write or upload.

### Option B: cloud-native checkpointing

Write straight to object storage and treat the bucket as the only location. It survives node loss,
but every save pays network latency, a preemption mid-upload leaves an invisible or stale object, and
on-prem clusters without a bucket get nothing.

### Option C: atomic local write, sidecar manifest, LATEST pointer, verified restore

Write a temp file, `fsync`, hash, write a manifest, rename, then update `LATEST`. Upload after the
local file verifies. On restart walk back from `LATEST` to the newest checkpoint that verifies. More
code and a chaos test tier, but every state is recoverable and the semantics are the same on a
laptop, a SLURM node and a spot instance.

## Decision

Option C, implemented by `ckpt/`, with these rules.

1. **Atomic write.** Save to `<dir>/<run_id>/<kind>/step_<step:09d>.<ext>.tmp`, `fsync` the file and
   its directory, hash it, write `step_<step:09d>.manifest.json` (a `CheckpointManifest`), then
   `os.replace` the temp file to its final name. Only after that is `LATEST` rewritten atomically.
2. **Integrity on restore.** `restore_latest_valid()` reads `LATEST`, checks size and SHA-256
   against the manifest, loads (`safetensors`, or `torch.load(..., weights_only=True)`) and returns
   the state. On any failure it logs `checkpoint_skipped_corrupt` and tries the previous manifest by
   step. It returns `None` only when no valid checkpoint exists **and nothing is known to have
   existed**. When the run store, a tag-checked pointer or `min_step` proves a step was saved,
   finding nothing at or above it raises `CheckpointUntrustedError` (`rolled_back` or
   `checkpoint_missing`, with the step) under `anchored` and `authenticated`. Under `any` it returns
   `None` and logs `checkpoint_progress_lost`. Starting over is `accept_rollback_to=0`. Files
   without a manifest are never loaded.
3. **Last versus best.** `kind="last"` holds the full training state. `kind="best"` holds weights
   and EMA as `safetensors`. Retention is per kind: `keep_last` and `keep_best`, defaults 3 and 2.
4. **Never delete before verify.** A checkpoint may be deleted only when a newer one of the same
   kind has been written, verified and, with a remote, uploaded with a matching checksum. The newest
   valid checkpoint of each kind is never deleted.
5. **Remote mirror.** With `remote=`, the data file and manifest are uploaded through `fsspec` after
   local verification, the manifest last. If local `LATEST` is missing (a new node), the manager
   lists remote manifests, applies rule 2 and downloads to `<dir>` first. A restart has a new run
   id, so the remote is searched by the run's lineage (the run store must have survived) or, with
   `restore(discover=True)`, by listing the newest runs and ordering candidates by step. Every
   candidate passes the same gate before a byte is installed.
6. **Preemption.** Handlers for SIGTERM and SIGUSR1 (configurable) and a poller for AWS and GCP
   preemption notices (Azure has none and relies on the signal) set a flag. `maybe_save` saves at
   the next call inside the grace window, whatever `every_n_minutes` says, and the run is marked
   `preempted`. A run that restores a state starts as a new run with `parent_run_id` set to the
   preempted run and continues its step timeline.
7. **Framework adapters** resolve resume paths through the manager: `resume_path()` returns the
   latest *verified* `last` file or `None`. Adapters pass it on (Lightning `ckpt_path`, Ultralytics
   `resume=`) only for a `pt` checkpoint this process verified, never the string `"last"`, an
   arbitrary path or a safetensors file (the manager's own format). A `pt` file the strict scan
   skips is reported (`resume_point_skipped`, `resume_point_unavailable`). Whether a framework file
   scans clean is a property of that file: Ultralytics `last.pt` pickles the model object and does
   not.
8. **Save overhead budget.** Hashing and manifest writing add at most 5 % to the wall time of
   `torch.save` for the same state on local SSD, measured by `benchmarks/ckpt_save.py`.
   **Amendment (2026-10-16): the budget is not met.** SHA-256 (about 1.3 ms per MB) is paid once per
   save, so against a page-cache-speed write the overhead is several times the write (0.19 s for a
   128 MB state, `keep_last=3`, macOS arm64). The verification pass (`verify_after_save=True`, the
   default) is the price of the integrity guarantee and stays on. **Amendment (2026-10-18): the 5 %
   rule is withdrawn.** Against `torch.save` of the same state (ResNet-18 weights 52.71 ms against
   16.67 ms, 3.2 times; YOLO11n weights 1.6 times, 1.5 to 2.0 over repeated runs) two SHA-256 passes
   alone exceed the framework's whole save. The measured numbers replace the budget
   ([ADR-0010](0010-real-checkpoint-files-and-mmengine.md)).

## Amendment (2026-10-15): trust and formats as built

- **Trust.** Digests and the chain detect damage, not a writer of the directory. Restore reports a
  trust level (`authenticated`: HMAC tag with `CVPILOT_CKPT_KEY`; `anchored`: manifest digest in the
  run store; `local_chain_only`) and enforces a policy (`CVPILOT_CKPT_REQUIRE_TRUST`, default
  `any`). Under `anchored` and `authenticated` each candidate needs its own proof, and failing to
  reach the level raises `checkpoint_untrusted` instead of loading a weaker file. See
  [06 Cross-cutting](../architecture/06-cross-cutting.md) and the [trust guide](../guide/trust.md).
- **Formats.** safetensors is the default for both kinds, replacing rule 3's `.pt` for `last`:
  non-tensor state is bounded JSON inside the file. The pickle-based format is named `pt`, is opt-in
  (`allow_pt_format` or `CVPILOT_CKPT_ALLOW_PT=1`), needs the framework at 2.6 or newer because
  `weights_only=True` was bypassed before that, is scanned without being executed and is loaded only
  with `weights_only=True` from the verified bytes.
- **Names.** The pointers are `LATEST` and `LATEST_BEST` (`BEST` would collide with the `best/`
  directory on case-insensitive filesystems). Manifests also carry `prev_manifest_sha256`,
  `prev_chain_sha256`, `chain_sha256`, `metric_mode` and `auth_tag`. `uri` is the intended remote
  location, and a `step_N.remote.json` marker records a verified upload.
- **One parser.** What is checked is what is loaded. The safetensors header is parsed once by a
  strict reader (duplicate keys, NaN, trailing bytes, unknown dtypes, wrong byte counts,
  overlapping or gapped spans, bounded counts), and after the library loads the same bytes, names,
  dtypes, shapes and byte total must equal that parse. A `pt` archive is read by our own bounded zip
  reader: headers must agree, nothing may precede, separate or follow the members, names are unique
  ignoring case, and encrypted, descriptor, zip64, extra-field and comment constructs are refused
  (the framework's alignment padding in a local header is the one exception). `data.pkl` is
  decompressed and checksummed here. Residual risk for `pt`: archives over 4 GiB are refused, and
  tensor names and dtypes cannot be derived from the pickle without executing it, so they are
  bounded, not compared. ADR-0010 later tested real framework files.
- **A restored checkpoint is data.** Even at `local_chain_only`, a forged checkpoint can supply
  weights and state but cannot execute code through cvpilot. The JSON skeleton is parsed strictly
  (no duplicate keys, NaN or Infinity; integers within 19 digits; at most 1 MiB; depth and node
  limits) and rebuilt only into plain types. Nothing restored selects a module, attribute or class.
  RNG structures are fully validated before any generator is touched, applied together and rolled
  back on failure; a malformed one skips the checkpoint as `unloadable`. A forged checkpoint can
  still degrade a model or exhaust memory up to the bounds, which is why a shared directory needs
  `authenticated`.

## Amendment (2026-10-16): the rules of a save

A save must exceed a floor made of every self-consistent manifest (trusted or not, quarantined
included), the run store's highest step, a tag-valid pointer, `min_step` and the remote's newest
admitted step. Untrusted manifests above every trusted one go on a quarantine list (never ignored,
never moved) until an operator releases them, and deleting files is a separate confirmed command. A
new manifest chains only to the newest trusted checkpoint. With none and a floor known, it fails
closed (`chain_unavailable`) unless the manager was built with `restart_chain=True`, which is
recorded as `chain_reset` inside the tagged manifest. Files are created exclusively and linked into
place without replacing, the pointer only moves forward, and the pointer's target is never pruned.

## Amendment (2026-10-16): failure is not permission, and a key is not freshness

- A rejection never moves or deletes a file. `ckpt trash --yes` is the only command that moves
  rejected files.
- The rule "a merely damaged newer checkpoint lets an older one be served" is removed. Under
  `anchored` and `authenticated`, a step below a known floor needs a good checkpoint at or above the
  floor in the same inspection, or an explicit, logged `accept_rollback_to`. `min_step` is
  unconditional.
- Remote listings are canonical, de-duplicated and probed until enough steps are admitted. Entries
  that exist but none passes are an error (`remote_untrusted`). Under `any`, a remote manifest that
  is neither anchored nor authenticated never raises the floor.
- Authentication evidence comes from tag-verified manifests only. Strict policies require a complete
  anchor, and a failed anchor write holds the pointer.
- A key proves authorship, not recency. Freshness needs the run store, versioned storage or a step
  the operator supplies.
- One validated `AdmitContext` is used by restore, list, verify, the mirror and lineage.

## Consequences

- Easier: a kill at any point of a save or upload leaves a loadable checkpoint; resume is one call;
  the same recipe works for SkyPilot, SLURM and local runs.
- Harder: a chaos test tier (`tests/chaos/`) that kills the process mid-write and truncates uploads
  was mandatory before `ckpt/` counted as built; retention needs care when the remote is slower than
  the save interval.
- Forbidden: loading a checkpoint without a manifest; `torch.load` without `weights_only=True`;
  deleting the only valid checkpoint of a kind.
- The CLI is `cvpilot ckpt list|verify|prune` (instead of `ls|verify|restore`).

## Links

- [04 Flows](../architecture/04-flows.md) (Flow B)
- [03 Data model](../architecture/03-data-model.md) (`CheckpointManifest`)
- [0005-run-store.md](0005-run-store.md)
