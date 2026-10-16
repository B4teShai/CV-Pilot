# Trust model

CV-Pilot reads files that other processes (and, in a shared setup, other people) can write:
checkpoint directories, buckets, Parquet inputs, the run store. This page says what each control
protects against, and what it does not.

## Who can write the checkpoint directory?

If only you can, the defaults are enough: digests, a hash chain between manifests and a `LATEST`
pointer detect everything that goes wrong without an adversary (a kill during a write, a truncated
upload, a flipped bit). If anyone else can, they can write a new manifest, a data file with the
matching SHA-256, a new pointer and a consistent chain, and a restore that trusts the directory
will load it. Three levels of trust exist; every restore reports one (`trust_level` in the
`checkpoint_restored` event, `Restored.trust`):

| Level | Proof | Works on a fresh machine | Stops a writer of the directory |
|---|---|---|---|
| `authenticated` | HMAC-SHA256 tag with your key (`CVPILOT_CKPT_KEY`) | yes | yes, unless the key leaks |
| `anchored` | the manifest digest is in the run store | no (needs the store) | yes, unless the store is writable too |
| `local_chain_only` | the directory's own chain | yes | no |

The policy `CVPILOT_CKPT_REQUIRE_TRUST` (`any` by default, `anchored`, `authenticated`; the CLI
takes `--require-trust`) says how much proof a restore accepts. Under `anchored` and
`authenticated` every checkpoint needs its **own** proof: a forged newer manifest cannot vouch for
an older one. If checkpoints exist but none reaches the level, restore raises
`checkpoint_untrusted` and loads nothing; it never falls back to a weaker file.

**Settings only tighten.** The level in force is the strictest of the manager's constructor
argument, `CVPILOT_CKPT_REQUIRE_TRUST` and the implicit default (`authenticated` as soon as
`CVPILOT_CKPT_KEY` is set). A per-call `require_trust=`, a `--require-trust` flag or an argument to
`resume_path` can raise the level but never lower it; a weaker request is ignored and logged as
`checkpoint_trust_downgrade_ignored`. The same holds for the rollback floor: `CVPILOT_CKPT_MIN_STEP`
and the constructor's `min_step=` both bind (the larger wins) and `accept_rollback_to=` cannot
accept a restore below them. The one way to weaken a manager is the constructor argument
`allow_weaker_trust=True` (not a call, a flag or an environment variable); it is logged once as
`checkpoint_weaker_trust_allowed`, and `Restored.policy` / `Restored.weaker_allowed` and the
`policy` field of `ckpt verify --json` show what was in force. Nothing in the checkpoint directory
(manifests, markers, extra files) is ever read as a setting.

**Nothing left is not a fresh start.** When the run store, a tag-checked pointer or `min_step`
proves that a step was saved, a restore that finds no valid checkpoint at or above it raises
`CheckpointUntrustedError` (`rolled_back`, or `checkpoint_missing` when manifests are present but
nothing verifies; the step is in the message) under `anchored` and `authenticated`. Under `any`
rollback is not enforced, but the empty result is logged as `checkpoint_progress_lost` with the
floor. `find_resume_point` and `resume_path` behave the same. Starting over is an explicit
`accept_rollback_to=0` (logged); after it only verified saves move the floor forward, and
re-saving an old step number is refused by the run store.

**Damage is not a licence to fall back.** Under `anchored` and `authenticated`, when something is
known to have existed at or above a step (the run store, a tag-checked pointer, `min_step`), an
older checkpoint is served only if a good checkpoint at or above that step was seen. If the newest
one is truncated, deleted or unverifiable, the restore fails and names the damaged step; going back
to an older one is `accept_rollback_to=N` (logged as `checkpoint_rollback_accepted`). `min_step` is
applied unconditionally. Under `any` nothing is enforced and the walk back to the newest valid one
is the documented behaviour.

**A key alone gives no freshness.** A key proves who wrote a checkpoint, not when. With only the
key, an attacker who can write the directory can put back an older, genuine, tagged checkpoint and
it verifies. Freshness comes from something the attacker cannot rewrite: the run store (an anchor),
versioned or write-once bucket storage, or a step you supply (`min_step`, `--expect-min-step`).
A key shorter than 128 bits of entropy is accepted with a warning (`checkpoint_key_weak`), not
refused; use 32 random bytes.

**Evidence comes from tag-verified manifests only.** When a key exists, a manifest whose tag does
not verify never counts as proof that the run was authenticated or as a floor
(`checkpoint_unverified_tags_ignored`), so planting tagged-looking files cannot move either.
Strict policies require a complete anchor (`max_step` and `auth_evidence`); a save whose record
cannot be written to the run store fails (`checkpoint_unanchored`) and the pointer is not moved.

**Recommendation.** Keep the default `any` only for a directory that nobody else can write (a
laptop, a private node). For a directory that is shared, restored from a bucket or writable by
others, set both:

```bash notest
export CVPILOT_CKPT_KEY="$(openssl rand -hex 32)"   # keep it in your secret manager
export CVPILOT_CKPT_REQUIRE_TRUST=authenticated
```

The key is read from the environment only; it is never logged, stored, put in a manifest or
repeated in an error message. It must be 32 to 1024 bytes; a value made only of hex digits must be
at least 64 of them (32 bytes of key material) and a padded base64 value must decode to at least 32
bytes. A key that looks guessable (a repeated pattern, few distinct characters, low entropy) is
accepted but logged once as `checkpoint_key_weak` with the reason, never the value; generate it with
`openssl rand -hex 32`. Tags cover a domain label, the run id
and the canonical bytes of the record, so a manifest cannot be moved to another run or replayed as
a pointer. A manifest also names its run's `project` and `parent_run_id` inside the tagged record: a
restart of another project refuses it (`binding_mismatch`), also from a shared bucket, and a
lineage restore refuses a manifest that misstates its parent. Under a key the `step_N.remote.json`
upload marker is tagged too, so a forged marker cannot make retention believe a checkpoint is
safely uploaded. **Rotation:** tags made under the old key fail under the new one (`auth_invalid`);
re-save to re-tag. Files written before a key existed have no tag and are accepted only under
`any`.

## Downgrades

Authentication only ratchets up. Once a run held a tagged checkpoint (a valid tag in the directory,
a tagged record in the run store, or a tag-covered `LATEST` pointer, which carries `auth_since`),
anything untagged that is newer than the first tagged step is refused as `auth_downgrade` under
every policy: stripping tags or deleting the pointer does not reset it, because the run store keeps
the record. The only way through is the explicit constructor argument `allow_weaker_trust=True`.

* **No key.** Restoring or saving a run that was authenticated without `CVPILOT_CKPT_KEY` fails
  closed with `key_required` and says to set the key; it never degrades to `local_chain_only`.
* **A different key.** A key other than the one the run was saved with is refused as `key_changed`,
  for restore and for save. Rotation path: restore with the **old** key; switch to the new key;
  build the manager with `accept_key_change=True` and `restart_chain=True`. The next save starts a
  visibly reset chain under the new key (`checkpoint_key_change_accepted` and
  `checkpoint_chain_reset` are logged); the old checkpoints stay unreadable under the new key.
* **Versions and formats.** Manifests carry `schema_version`, `chain_version`, `tag_alg` and
  `key_id`, all covered by the tag, and the serializer format is part of the chain digest, so a
  safetensors checkpoint cannot be presented as a `pt` one and stripping the version fields breaks
  the chain. A manifest from before versions existed is refused under `anchored` and
  `authenticated` (`legacy_manifest`) and reported (`checkpoint_legacy_manifest`) under `any`; an
  unknown version is refused under every policy. The `pt` opt-in applies at every path.
* **Trust regression.** Under `any`, a manifest the run store never recorded (newer than its highest
  step) is served at `local_chain_only` although the run reached `anchored`: that is logged as
  `trust_regressed`.
* **The remote only adds evidence.** A name in a bucket listing is not evidence that a checkpoint
  exists. Only canonical, de-duplicated names are considered, and `Mirror.admitted` probes
  newest-first (at most 64) until enough steps pass the gate with their data object at the recorded
  size. Junk names above the real ones are passed over; entries that exist but none passes raise
  `CheckpointUntrustedError` (`remote_untrusted`), never a fresh start. Under `any` a remote
  manifest that is neither anchored nor authenticated does not raise the floor (a bucket writer
  could otherwise plant a huge step and stop every save: `checkpoint_remote_floor_ignored`). A failed lookup
  is not cached. Retention deletes a local copy on an upload marker only after the remote is read
  back and holds the same manifest bytes and a data object of the recorded size.
* **Quarantine is a list, not a move.** A rejected checkpoint is never moved or deleted because of
  the rejection. The step is recorded in a `QUARANTINE` list that counts toward the save floor;
  `cvpilot ckpt quarantine --clear` only releases list entries. Deleting the debris is a separate
  command, `cvpilot ckpt trash --run ID --yes`, which moves the listed files into a trash
  directory.
* **A pickle scan is defence in depth.** The `pt` scan refuses unknown globals including those
  built with memoised names; it narrows what a hostile file can do and does not make loading a
  `pt` file safe. Prefer safetensors.
* **Liveness.** A run holds a lease on its checkpoint directory; `recover_run` refuses a run whose
  lease is held. The lease is an advisory file lock: it is not reliable on NFS or FUSE mounts, and
  a run that cannot take one logs `checkpoint_lease_unavailable`.

## Verified, then opened: the gap

`resume_path` verifies a file and hands you its path; your framework opens it later. Anyone who can
write the checkpoint directory in that interval can swap the file after the check (a
time-of-check-to-time-of-use race that no check inside this library can close, because the open
happens in the framework). Restrict write access to the directory, which the first section already
asks for, or close the gap yourself: `cvpilot.ckpt.private_copy(point)` copies the verified bytes
into a new directory only you can use (mode 0700, the file 0600), checking the SHA-256 and size
while it copies, and returns that path for `resume_arguments`; the hooks' `resume_arguments` take the verified path as it is and recheck the file's identity
(inode, size, modification time) and SHA-256 immediately before returning it, which narrows the gap
to the time between that recheck and your framework's open; they do not copy. Call `private_copy`
first to remove it. The restore path of
`CheckpointManager.restore` does not have the gap at all, because the bytes that were hashed are the
bytes that are parsed. A `pt` file is read once on the way to a resume point: the archive scan runs
over the bytes whose digest was just checked.

## Forgery is not rollback

A key stops **forgery**: a writer without it cannot produce a checkpoint, a pointer or an upload
marker that verifies. It does not, by itself, stop **rollback or deletion**: everything in the
directory or bucket may be genuine and still be old, or gone. Rollback is detected from a floor
the writer cannot lower: the run store (a digest and the highest step of every checkpoint), a
tag-checked `LATEST` pointer, and the numbers you supply (`CVPILOT_CKPT_MIN_STEP`,
`--expect-min-step`). **A fresh machine that has only the key, with no run store and no remembered
step, cannot tell that newer checkpoints once existed**: if someone deleted every newer object and
left an older genuine one, the older one verifies as `authenticated`. Close that gap outside the
tool: turn on bucket versioning and object lock (or a retention policy) so nothing newer can be
deleted, keep the run store on storage the bucket's writers cannot reach, and pass the step you
last saw as `CVPILOT_CKPT_MIN_STEP`. `cvpilot ckpt verify --expect-min-step N` checks a directory
against such a number, and `--accept-rollback-to N` accepts a rollback from the store or pointer on
purpose (logged); neither lowers the operator's own floor.

## What a forged checkpoint can and cannot do

Even at `local_chain_only` a forged checkpoint can supply weights and training state, so it can
poison or degrade a model, and it can use memory up to the documented bounds. It cannot execute
code through cvpilot: safetensors has no executing loader, the JSON state is parsed strictly
(no duplicate keys, NaN, Infinity or oversized integers; at most 1 MiB; depth and node limits) and
comes back as plain data, RNG structures are validated completely before any generator is set and
rolled back on failure, and nothing restored selects a module, attribute or class.

## The pickle-based `pt` format

safetensors is the default and the recommended format. `pt` (a framework's own archive) needs
`allow_pt_format=True` or `CVPILOT_CKPT_ALLOW_PT=1`, a framework at version 2.6 or newer (an
earlier `weights_only=True` could be bypassed), an archive that passes a strict in-house reader
(no hidden bytes, duplicate names, descriptors, zip64 or extra fields; `data.pkl` limited to an
allow-list of tensor-storage globals and never executed), and is loaded only with
`weights_only=True`. Residual risk: a flaw in the framework's own restricted unpickler.

## The gate and the run store

A gate that reads a store a pull request controls is only as strong as that store: see
[Gating in CI](gating-ci.md). Snapshot and baseline files carry digests and chains verified on
every read; nothing is signed, so someone who can write both the database and the files can
recompute every digest. The external anchor is `--expect-baseline-chain`.

## Reporting

Report suspected vulnerabilities privately to the maintainers rather than in a public issue.
