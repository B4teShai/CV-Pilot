# Trust model

CV-Pilot reads files that other processes, and in a shared setup other people, can write:
checkpoint folders, buckets, Parquet inputs and the run store. This page says what each control
protects against and what it does not. Terms are in the [glossary](concepts.md#glossary).

## Who can write the checkpoint folder?

If only you can, the defaults are enough. Hashes and the hash chain catch everything that goes
wrong without an adversary: a kill during a write, a truncated upload, a flipped bit.

If anyone else can, they can forge a manifest, a data file with a matching SHA-256 and a consistent
chain, and a restore that trusts the folder will load it. Every restore reports one of three trust
levels (`trust_level` in the `checkpoint_restored` event):

| Level | Proof | Works on a fresh machine | Stops a writer of the folder |
|---|---|---|---|
| `authenticated` | HMAC tag made with your key | Yes | Yes, unless the key leaks |
| `anchored` | The manifest hash is in the run store | No, needs the store | Yes, unless the store is writable too |
| `local_chain_only` | The folder's own hash chain | Yes | No |

**Recommendation.** Keep the default only for a folder nobody else can write (a laptop, a private
node). For a folder that is shared, restored from a bucket or writable by others, set both:

```bash notest
export CVPILOT_CKPT_KEY="$(openssl rand -hex 32)"   # keep it in your secret manager
export CVPILOT_CKPT_REQUIRE_TRUST=authenticated
```

## How a restore decides

`CVPILOT_CKPT_REQUIRE_TRUST` (`any` by default; the CLI takes `--require-trust`) sets the weakest
level a restore accepts. Under `anchored` and `authenticated` every checkpoint needs its own proof,
so a forged newer manifest cannot vouch for an older one. If checkpoints exist but none reaches the
level, restore raises `checkpoint_untrusted` and loads nothing. It never falls back to a weaker file.

**Settings only tighten.** The level in force is the strictest of the constructor argument, the
environment variable and the default (`authenticated` as soon as a key is set). A per-call argument
or flag can raise it, never lower it. The one way to weaken a manager is the constructor argument
`allow_weaker_trust=True`, which is logged. Nothing inside the checkpoint folder is ever read as a
setting.

## Nothing left is not a fresh start

When the run store, a tag-checked pointer or `CVPILOT_CKPT_MIN_STEP` proves that a step was saved, a
restore that finds nothing at or above that [floor](concepts.md#glossary) raises
`CheckpointUntrustedError` under `anchored` and `authenticated`. Under `any` it logs
`checkpoint_progress_lost`. When the newest checkpoint is damaged the strict levels also fail
instead of serving an older one. Going back on purpose is `accept_rollback_to=N`, which is logged.
`find_resume_point` and `resume_path` behave the same.

## A key proves who, not when

A key stops forgery. It does not stop rollback: an attacker who can write the folder can put back
an older, genuine, tagged checkpoint. Freshness needs something the attacker cannot rewrite: the run
store, versioned or write-once bucket storage, or a step you supply (`CVPILOT_CKPT_MIN_STEP`,
`--expect-min-step`). A fresh machine that has only the key cannot know that newer checkpoints
existed. Turn on bucket versioning and object lock, keep the run store out of the bucket writers'
reach, and pass the last step you saw as `CVPILOT_CKPT_MIN_STEP`.

## Keys

The key is read from the environment only. It is never logged, stored, put in a manifest or shown in
an error. It must be 32 to 1024 bytes; use 32 random bytes. A weak-looking key is accepted but
logged once as `checkpoint_key_weak`. The tag also covers the run id, project and parent run, so a
manifest cannot be moved to another run. Once a run has held a tagged checkpoint, an untagged newer
one is refused (`auth_downgrade`), and a different key is refused (`key_changed`). To rotate: restore
with the old key, then build the manager with `accept_key_change=True` and `restart_chain=True`.

## Verified, then opened

`resume_path` verifies a file and hands you its path, and your framework opens it later. Someone who
can write the folder in between can swap the file. Restrict write access, or call
`cvpilot.ckpt.private_copy(point)`: it copies the verified bytes into a private folder and returns
that path. `CheckpointManager.restore` has no such gap, because the bytes that were hashed are the
bytes that are parsed.

## The pickle-based `pt` format

safetensors is the default and the recommended format. `pt` needs `allow_pt_format=True` or
`CVPILOT_CKPT_ALLOW_PT=1`, a framework at version 2.6 or newer, and an archive that passes a strict
in-house reader. The reader refuses unknown pickle globals, and the file is loaded only with
`weights_only=True`. This narrows what a hostile file can do. It does not make loading one safe.
Files that pickle classes (Ultralytics `last.pt`, MMEngine `.pth`) are refused. See
[Checkpoints](checkpoints.md#frameworks-that-write-their-own-files).

## What a forged checkpoint can do

At any level it can supply weights and training state, so it can degrade a model. It cannot run code
through CV-Pilot: safetensors has no executing loader, the JSON state is parsed strictly and comes
back as plain data, and nothing restored picks a module or class.

## The gate and the run store

A gate that reads a store a pull request controls is only as strong as that store. Snapshots and
baselines carry hashes checked on every read, but nothing is signed. The outside anchor is
`--expect-baseline-chain`. See [Gating in CI](gating-ci.md).

## What is proven

Checked on files from real training runs: size and SHA-256 checks with refusal of a flipped byte, a
truncation and a foreign file; the `anchored` and `authenticated` levels; and the strict reader.
Experimental, never run against the real thing: the S3, GCS and Azure mirror, the AWS and GCP
preemption pollers, `local_chain_only` on real framework files and network file systems.

## Appendix: error codes

| Code | Meaning |
|---|---|
| `checkpoint_untrusted` | Checkpoints exist but none reaches the required level. Nothing was loaded. |
| `rolled_back`, `checkpoint_missing` | Something proves a later step existed, but no valid checkpoint reaches it. |
| `auth_missing`, `auth_invalid` | The tag is absent, or made with another key. |
| `auth_downgrade`, `key_required` | The run was authenticated before; an untagged file or a missing key is refused. |
| `key_changed` | A different key from the one the run was saved with. |
| `binding_mismatch` | The manifest belongs to another project or parent run. |
| `legacy_manifest` | A manifest from before versions existed; refused under the strict levels. |
| `not_anchored`, `anchor_error` | The run store has no record of the manifest, or could not be read. |
| `remote_untrusted` | The bucket lists checkpoints but none passes the checks. |
| `checkpoint_trust_downgrade_ignored` | A call asked for weaker trust than the configuration. Ignored. |

Report suspected vulnerabilities privately to the maintainers, not in a public issue.
