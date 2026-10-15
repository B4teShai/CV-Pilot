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

**Recommendation.** Keep the default `any` only for a directory that nobody else can write (a
laptop, a private node). For a directory that is shared, restored from a bucket or writable by
others, set both:

```bash notest
export CVPILOT_CKPT_KEY="$(openssl rand -hex 32)"   # keep it in your secret manager
export CVPILOT_CKPT_REQUIRE_TRUST=authenticated
```

The key is read from the environment only; it is never logged, stored, put in a manifest or
repeated in an error message. It must be 32 to 1024 bytes. Tags cover a domain label, the run id
and the canonical bytes of the record, so a manifest cannot be moved to another run or replayed as
a pointer. **Rotation:** tags made under the old key fail under the new one (`auth_invalid`);
re-save to re-tag. Files written before a key existed have no tag and are accepted only under
`any`.

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
