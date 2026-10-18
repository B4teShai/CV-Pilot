# Changelog

All notable changes are recorded here. Format: [Keep a Changelog](https://keepachangelog.com/);
versions follow semantic versioning. Before 1.0, minor versions may break the public API.

## [Unreleased]

### Fixed (checkpoints, found by running the verifier on real framework files, ADR-0010)

- The strict `.pt` reader refused every file the framework writes (zip data descriptors and a zip64 end
  record, even for a small file), so adopting, verifying or resuming a real Lightning checkpoint, and
  restoring CheckpointManager's own `pt` format with the real framework, could not work. It now accepts
  exactly that layout and still refuses every deviation (`tests/zip_support.py` builds it byte for byte;
  a marked test compares it with a file the installed framework wrote).
- `CheckpointManager.adopt` scans the archive before it stores anything. It used to store, log as saved
  and verified, and only refuse at the restart that needed the file (an Ultralytics `last.pt` pickles the
  model object): a file that cannot be resumed is now refused with the reason.
- `cvpilot ckpt verify` and `list` honour `CVPILOT_CKPT_ALLOW_PT` (every `pt` checkpoint showed as
  `pt_not_allowed`), and verify scans `pt` archives so it no longer calls valid a file that resume refuses.
- A pickle written with protocol 2 that holds a `set` (`__builtin__.set`) passes the scan; the message
  for a file that is not a zip archive (safetensors, JSON) no longer says "raw pickles are refused".
- The guide's restore snippet converts NumPy arrays before `load_state_dict`.
- The real-library smoke tests for Lightning files had never run and had two bugs; they are fixed and run.

### Changed (checkpoints)

- The save-overhead budget (5 % over `torch.save`) is withdrawn and replaced by measured numbers: on the
  measured machine a steady-state save of ResNet-18 weights took 53.30 ms against 15.36 ms for
  `torch.save` of the same state (3.5 times), YOLO11n weights 17.52 ms against 11.89 ms (1.5 times),
  with two SHA-256 passes taking 51 % and 37 % of a save. `benchmarks/ckpt_save.py` gained a per-phase
  profile (`profile_state`).
- The status page labels the least-proven checkpoint features **experimental**: the remote mirror
  (never run against a real bucket), the `local_chain_only` level on real files and the cloud preemption
  notices. `anchored` and `authenticated` are proven on a real Lightning file.

### Added (checkpoints)

- Notebook 04 (`notebooks/04_real_checkpoints.ipynb`): real Lightning, Ultralytics YOLO11n, Hugging Face
  `Trainer` and MMEngine runs, the exact refusal messages for each file layout, tamper runs (flip,
  truncate, swap), resume comparisons with uninterrupted runs, and the save cost by phase on three real
  model sizes. Ultralytics, MMEngine and Hugging Face resume files are refused by the reader and have no
  verified resume; see the guide and ADR-0010.
- An MMEngine hook (`pip install "cvpilot[mmengine]"`, `cvpilot.autolog(runner, ...)`): `train/`
  scalars every `interval` iterations (default 10), `val/` and `test/` metrics, the end of the run, with
  the guard, isolation and rank rules of the other hooks. No resume helper.

### Fixed

- `cvpilot baseline set` (and `cvpilot.compare.baselines.set_baseline`) now refuses a run that is not
  `finished`, like `--baseline RUN_ID` already did; there is no override.
- A resumed `CheckpointManager` seeds its save schedule from the restored `last` checkpoint
  (step and time), so the first save after a resume happens one interval later instead of at once.
- `cvpilot ckpt verify`: the help text now says what each trust policy reports for a missing newest
  checkpoint. Under the default policy `any` the exit code is unchanged and one `note:` line on stderr
  tells when the run store or a pointer records a newer step than the newest valid checkpoint.
- The events of `find_resume_point` and `recover_run` are JSON on stderr when called before `init`.

### Changed

- The README is now a short front page (what it is for, one picture, a runnable example, links).
  The settings, exit codes, evaluation, baseline and gate walkthrough, GitHub Action example and
  status moved into the docs (`docs/reference/settings.md`, `docs/status.md`, the guide pages). A new
  overview diagram replaces the architecture diagrams in the README.

### Fixed (alerts, before their first release)

- A name-resolution stall can no longer hold an alert worker: resolution runs inside the 5 second
  attempt deadline. A stuck delivery only drops the alerts that arrive while it lasts and the channel
  recovers; the end-of-run alert is no longer lost to an earlier stall.
- `finish` evaluates the `run_failed` alert before the final flush, so a store that cannot be written
  no longer silences it; an uncaught exception outside a `with` block now sends it from the exit handler.
- `finish` waits up to 12 s for alert channels after up to 5 s for forwarders (one full delivery
  budget), and the docs now give that worst case.
- Metric names in messages cannot be links (`://`, dot before a letter); IPv6 addresses are allowed
  only inside `2000::/3` (after unwrapping mapped, 6to4 and NAT64 forms); development mode accepts
  loopback addresses only and logs `insecure_dev_mode`; a socket cannot leak when a connection object
  fails to build; a channel that fails to construct stops `init`.
- A failed rule uses its own rate key and cannot use up the real alert's window; a broken logger or a
  value too large for a float cannot stop the rules; a non-string channel entry is a configuration error.
- Every suppression kind is reported once (including the per-run cap) and `finish` writes `alert_summary`.
- `eval_regression` is not compared again while its alert is muted. The dry run interleaves snapshots
  with scalars by step and says that snapshots carry no time.
- Hugging Face hook: only rank zero from the first callback; interrupts and exits mark the run
  preempted; custom evaluation prefixes (`val_`, `test_`) are recorded once under `eval/`.

### Added

- Alerts, in the training process and without a server (ADR-0009): `cvpilot.init(alerts=...)` or
  `CVPILOT_ALERTS` point at a small JSON file with the rules `nan_loss`, `loss_spike`, `no_progress`,
  `eval_regression` (the real baseline comparison, run when `log_eval` stores a snapshot),
  `run_failed` and `run_preempted`. Channels `webhook`, `slack` (a Slack-style incoming webhook) and
  `telegram` (registry `cvpilot.channels`) take their address and credential from environment
  variables only and deliver over https to public addresses, with no redirects, a 5 second deadline,
  one retry and the existing bounded queue and isolation of the forwarders; a dead channel never
  slows or fails training. One alert per rule and run per window, 20 per run. Messages are constant
  templates with plain-character values. `cvpilot alerts check CONFIG [--project P --run RUN]`
  validates a configuration and replays a recorded run, sending nothing.
- A Hugging Face `Trainer` hook (`pip install "cvpilot[hf]"`, `cvpilot.autolog(trainer, ...)`): the
  training log under `train/`, evaluation metrics under `eval/`, saves as events, the end of training;
  an exception in `train()` marks the run failed. No resume helper (the trainer resumes from a
  directory bundle; see ADR-0009). The extra now installs `transformers[torch]>=4.46` (the real
  smoke test passes on 4.46.0 and 5.20.1).
- Notebook `03_alerts_and_hf_trainer.ipynb`: a real ResNet-18 from the Hugging Face Hub (pinned commit,
  SHA-256 checked) fine-tuned on COCO128 crops with the `Trainer`; injected NaN loss and plateau, a real
  regression, and step times with dead channels, with the received messages.
- `CheckpointManager.restore(discover=True)` and `restore_latest_valid(discover=True, run_ids=...)`:
  a restarted script on a node without the run store or checkpoint directory finds the earlier
  run's checkpoints on the remote (newest 20 runs, best step first), through the same gate as any
  other checkpoint. A fresh-node test runs across real processes. The restart recipe of the
  checkpoint guide is executed by the test suite; `recover_run`, `list_runs` and `get_run_view` now honour
  `CVPILOT_STORE_DIR` like `init` does. Remote listings are bounded.
- `cvpilot ckpt verify --expect-min-step N` and `--accept-rollback-to N`; `CVPILOT_CKPT_MIN_STEP`
  applies to `verify` as well. The trust guide now separates forgery (stopped by a key) from
  rollback and deletion (needs a floor from the run store, versioned storage or a step you
  supply) and says what a fresh machine with only a key cannot detect.

### Security

- A rejected checkpoint is never moved or deleted: the quarantine is a list in the run directory
  that counts toward the save floor, `ckpt quarantine --clear` only releases entries, and the new
  `ckpt trash --run ID --yes` is the only command that moves rejected files. Key and evidence
  checks now run before anything is recorded, so a wrong key cannot fill the list.
- A damaged newest checkpoint no longer lets an older one be served below a known floor under
  `anchored` and `authenticated`: the restore fails naming the damaged step, and going back needs
  `accept_rollback_to`. `min_step` is unconditional. A remote only counts when its data object is
  present at the recorded size.
- The `pt` archive scan treated memoised names as unknown values; a global built from a memo fetch
  is now refused.
- Remote listings: canonical names only, de-duplicated, probed newest-first until enough steps are
  admitted (at most 64 probes); entries that exist but none passes raise `remote_untrusted`. Under
  `any`, a remote manifest that is neither anchored nor authenticated no longer raises the floor,
  and failed lookups are not cached. Retention checks the remote before it trusts an upload marker.
- Authentication evidence comes from tag-verified manifests only. Strict policies require a
  complete anchor, and a save whose record cannot be anchored holds the pointer and fails.
- `find_resume_point` passes the lineage and refuses when a newer member's candidates were
  rejected as untrusted. `CheckpointManager.list`, `verify`, the mirror and lineage use the same
  gate context as restore; `RunCheckpoints.existing` judges; `min_step`, `accept_rollback_to` and
  the byte cap share one validator; `CVPILOT_CKPT_MIN_STEP` has an upper bound.
- Verified-file paths remember inode, size and SHA-256 and are rechecked before use;
  `recover_run` refuses a run whose lease is held; `$intdict` keys are strict and unique.
- A key alone gives no freshness; the guide, README and ADR-0006 say so.

### Fixed

- The README is the package index page, so it no longer carries a frozen "release candidate,
  nothing published, install from source" status or relative links and images that break on the
  index. `scripts/check_readme.py` refuses relative links, links outside the project and
  release-status wording; the release workflow runs it before the build and a test covers it.

- Four published extras (`server`, `hf`, `mmdet`, `alerts`) installed nothing and the docs told
  you to install them; they are gone from the metadata and described as planned. New extras
  `remote-s3`, `remote-gcs` and `remote-azure` bring `fsspec` together with `s3fs`, `gcsfs` or
  `adlfs`, and the missing-backend error names the extra for the scheme in use. Tests fail on an
  empty extra or a documented extra that does not exist.

- Signal handlers and the notice poller are installed all or nothing: an unknown `notice_source`, an
  invalid interval or a poller that cannot start no longer leaves handlers installed and the
  liveness lease held. A handler that was already installed for the same signal is chained after
  the flag is set, reported once (`signal_handler_displaced`) and restored by `close()`.
- The notice poller gave up silently after five errors and its thread could die unseen. It now
  retries with exponential backoff, logs `notice_poller_degraded` and `notice_poller_recovered`, and
  `preemption_poller_stopped` with the reason. AWS counts a 200 only when its JSON `action` is
  terminate, stop or hibernate, treats 401 and 403 as an unavailable service, and GCP is polled at
  the link-local address instead of a DNS name. The ADR no longer lists an Azure notice source.

- Retention re-hashed every kept checkpoint after every save (and `verify_after_save` hashed the new
  one twice), which made a steady-state save at `keep_last=3` about three times as slow as
  necessary (0.52 s against 0.19 s for a 128 MB state on macOS arm64, `benchmarks/ckpt_save.py`).
  Retention now hashes a file only if this process has not verified it with the same inode, size and
  times; only a checkpoint that verifies is ever deleted. The budget of ADR-0006 (5 % over a plain
  write) is still not met: SHA-256 is paid once per save.

- The preemption path caught only a short list of exception types, so a model whose `state_dict`
  raised `RuntimeError` escaped into the training loop and the run was never marked `preempted`.
  It now catches everything, marks the run right after the local checkpoint verifies, uploads with
  only the remaining grace budget (`Mirror.budget`) and prunes last; `preemption_upload_deferred`
  reports a cut-short upload.

- The Lightning and Ultralytics `resume_arguments` helpers accepted any absolute path. They now
  accept only a `pt` checkpoint that `resume_path` returned in this process (as a path or a
  `ResumePoint`) and refuse safetensors files with an explanation, since no framework can resume
  from CheckpointManager's own format. `find_resume_point` logs `resume_point_skipped` and
  `resume_point_unavailable` (with the reason, for example `unsafe_archive`) instead of returning
  a bare `None`. Real-library smoke tests (`pytest -m real_libs`) cover the adoption of plain and
  pickled-model files and of a Lightning `last.ckpt`; the README, guide, ADR-0006 and component
  docs no longer claim that every framework file can be resumed from.

- Saving and restoring share one set of bounds: an integer of 20 digits below 2**64, a state of
  more than 100 000 nodes, an oversized key or a nesting the restore refuses is now refused at save
  with a clear error instead of being saved and then failing to load. A randomised save-then-load
  test covers it.
- `cvpilot ckpt prune` ranked `best` checkpoints with `--mode max` by default and deleted the
  lowest loss of a `min` run. It now uses the mode the checkpoints were saved with; a contradicting
  `--mode` is a usage error that deletes nothing. Retention that keeps non-adjacent best
  checkpoints no longer turns the older ones into `link_mismatch` corruption: manifests record the
  step of their predecessor (`prev_step`), so a gap is told apart from a replaced predecessor.

- A resumed run seeds its "best so far" from this run and every run it resumed (checkpoints that
  pass the gate, same monitored metric and mode), so it no longer saves a worse `best` just because
  its own directory is empty.

### Security

- The build backend and its dependencies are pinned exactly in `build-system.requires` (equal to the
  lock; pins, not hashes), so a newer backend cannot change what a build produces. The packaging
  tests also walk every archive member (only regular files and directories), compare the wheel with
  a committed file list and a suffix allow-list, run broader secret patterns (GitHub, Slack, API
  keys, JWTs, private keys) over artifacts and the source tree, and refuse symlinks in the source.

- Authentication, format and trust only ratchet up. Once a run held a tagged checkpoint (directory,
  run store or the tag-covered pointer's `auth_since`), an untagged newer manifest is refused as
  `auth_downgrade` under every policy; restoring or saving without the key fails closed
  (`key_required`), a different key needs `accept_key_change=True` (`key_changed`), manifests carry
  tag-covered `schema_version`, `chain_version`, `tag_alg` and `key_id` and the format is bound into
  the chain digest (legacy manifests are refused under the strict policies and reported under `any`),
  a manifest the store never recorded is flagged `trust_regressed` under `any`, and a remote only
  adds evidence: a listed name no longer loosens the rollback check, only manifests that pass the
  gate count. Reproduced against the previous code: a keyless restore of an authenticated run, and
  rollback laundering through a junk name in the bucket listing.

- The release workflow refuses a commit that is not an ancestor of `origin/main` (full history
  fetched), runs one release at a time with timeouts, builds with the uv cache off and an exactly
  pinned uv, and audits the locked runtime dependencies before building; CI audits for every
  supported Python and nightly, with timeouts and cancelling of superseded runs. The runbook gains a
  mandatory evidence step (read-only `gh api` calls) for the repository settings a workflow cannot
  enforce, and says that none exist yet.

- A resume point is verified, then opened by your framework later: that gap is now documented and
  closable (`cvpilot.ckpt.private_copy(point)` copies the verified bytes into a private 0700
  directory, checking them as it copies). A `pt` file is read once on the way to a resume point
  instead of twice, and the default restore size limit is 8 GiB.

- Manifests name their run's `project` and `parent_run_id` inside the tagged record: another
  project's checkpoint (also from a shared bucket, also through `discover`) and a lineage that is
  misstated are refused as `binding_mismatch`. Under a key the upload marker is tagged, so a forged
  marker cannot make retention delete on a false "uploaded". A hex key shorter than 32 bytes or a
  padded base64 key that decodes to fewer is refused, and a guessable key is logged once as
  `checkpoint_key_weak` (the reason, never the value).

- Saves follow strict rules. The step must exceed a floor made of every self-consistent manifest
  (trusted or not, quarantined included), the run store, a tag-valid pointer, `min_step` and the
  remote; an untrusted manifest above every trusted one is quarantined and reported
  (`cvpilot ckpt quarantine DIR --run ID [--clear]`) instead of silently blocking or being chained
  to. A manifest chains only to a trusted predecessor; with none and a floor known the save raises
  `CheckpointChainError` unless `restart_chain=True`, which is recorded as `chain_reset` in the
  tagged manifest. Data, manifests and markers are never replaced (hard link, never rename), a
  same-bytes save is a logged no-op, `adopt` and the remote install obey the same rule, the pointer
  only moves forward and the pointer's target is never pruned. The earlier change that let a
  planted high step be ignored is superseded: it now counts, visibly.

- A save took "the newest checkpoint" and "the previous one" from every manifest in the directory,
  admitted or not. A planted manifest at a huge step therefore made every later save fail, and a
  forged one could become the predecessor a genuine checkpoint chained to. Both now come only from
  manifests the gate admits.

- No valid checkpoint with a known floor is never a fresh start: restore, `resume_path` and
  `find_resume_point` raise `rolled_back` or `checkpoint_missing` (with the step) under the strict
  policies and log `checkpoint_progress_lost` under `any`; `accept_rollback_to=0` starts over on
  purpose.

- Trust and rollback settings are monotone: the effective policy is the strictest of the
  constructor argument, `CVPILOT_CKPT_REQUIRE_TRUST` and the implicit `authenticated` default under
  a key; per-call, `--require-trust` and `resume_path` arguments can only raise it
  (`checkpoint_trust_downgrade_ignored`). `CVPILOT_CKPT_MIN_STEP` is a floor no call can lower and
  `accept_rollback_to=0` no longer waives it. Weakening needs the constructor argument
  `allow_weaker_trust=True` (logged as `checkpoint_weaker_trust_allowed`).

## [0.1.0] - release candidate (not published)

Nothing has been uploaded to any package index; publishing needs the maintainer's explicit
approval (docs/engineering/release.md).

### Added

- Checkpoint manager (`cvpilot.ckpt.CheckpointManager`, ADR-0006): atomic saves (private `O_EXCL`
  temp file, `fsync`, manifest, rename, re-read and re-hash, digest recorded in the run store, then
  the `LATEST` pointer), verified restore that walks back and skips and counts anything truncated,
  corrupt, forged, replaced or symlinked, retention that counts only valid checkpoints and never
  deletes the newest one, `best` checkpoints by a monitored metric, resume across runs
  (`parent_run_id`), a directory lock for concurrent savers, and RNG state capture (restoring the generators is explicit: `apply_rng(state["rng"])`).
  Tensors go to safetensors files with the non-tensor state as bounded JSON; the pickle-based `pt`
  format is opt-in. Events `checkpoint_saved`, `checkpoint_verified`, `checkpoint_restored`,
  `checkpoint_skipped_corrupt`, `checkpoint_untrusted`, `checkpoint_uploaded`,
  `checkpoint_upload_failed`, `checkpoint_save_failed`, `checkpoint_restore_empty`,
  `preemption_signal_received`, `preemption_save_incomplete`, `run_preempted`.
- Preemption: SIGTERM and SIGUSR1 handlers that only set a flag, a save at the next `maybe_save`
  inside a grace window, the run marked `preempted`, optional AWS and GCP notice polling (off by
  default, bounded). Tested with real signals in a subprocess.
- Remote mirror over `fsspec` (extra `cvpilot[remote]`): upload after local verification, resumed
  `.part` objects, size and SHA-256 read back, manifest last, retries inside one deadline, TLS on,
  no credentials in locations, logs or the store, restore from the remote on a fresh node.
- `cvpilot.resume_path(project)`, `cvpilot.ckpt.find_resume_point`, `CheckpointManager.adopt` for
  framework-written files, `resume_arguments` helpers for the Lightning and Ultralytics hooks, and
  `cvpilot ckpt list|verify|prune` (`--require-trust`, `--project`, `--json`).
- Run store: an append-only `checkpoint_manifest` table (the digest of every manifest, the
  directory's external anchor); `Run.handle`.
- Public API frozen: `__all__` in `cvpilot` and `cvpilot.ckpt`, a snapshot test of names and
  signatures, and `cvpilot.deprecation` (decorator, warning, registry, a test that fails when a
  deprecation is due).
- Documentation site (MkDocs Material, strict, pinned, no remote assets): user guide, a tutorial
  whose every command is executed by a test, a CLI reference generated from the Typer app with a
  drift test, a Python API reference, and the release runbook.
- Release pipeline: `release.yml` (tag-triggered; a secret-free build job and a separate
  trusted-publishing job in the protected `pypi` environment that publishes the exact built
  artifact after re-checking its digests; tag must equal `__version__`), CI jobs `lock`, `docs`,
  `package` and `audit`, a workflow lint with 23 mutation tests, wheel and sdist content tests,
  `twine check --strict`, and `scripts/verify_wheel.py` (clean-environment install on 3.10 and
  3.13). **Nothing has been published.**
- Chaos tests with real processes: SIGKILL at random moments and at four chosen points inside a
  save, a full disk, three concurrent savers, a file replaced after it was read, symlink swaps,
  orphan temp files, clock skew, a forged coherent directory, real signals.
- `benchmarks/ckpt_save.py`: the save overhead (the 5 % budget of ADR-0006 is **not met**: hashing
  dominates; see the overview for numbers).

### Security

- Trust model for checkpoints: HMAC-SHA256 tags (`CVPILOT_CKPT_KEY`, environment only, never
  logged or stored) over a domain label, the run id and the canonical bytes of each manifest and
  pointer; a trust level on every restore (`authenticated`, `anchored`, `local_chain_only`); a
  policy `CVPILOT_CKPT_REQUIRE_TRUST` (default `any`: use `authenticated` for any directory that is
  shared, restored from a bucket or writable by others) under which every checkpoint needs its own
  proof, nothing is vouched for by a neighbour, a failing store lookup fails closed, and a restore
  that cannot reach the level raises `checkpoint_untrusted` instead of loading a weaker file.
- Parsers: the safetensors header is parsed once, strictly (duplicate keys, NaN, fractions,
  oversized integers, overlapping or gapped spans, wrong byte counts), and the loaded tensors must
  equal what that parse declared; the `pt` archive is read by an in-house strict zip reader (no
  hidden bytes, duplicate names, descriptors, zip64 or extra fields) with an opcode allow-list for
  `data.pkl`, a framework version floor of 2.6, an explicit opt-in and `weights_only=True` only.
- A restored checkpoint is data: strict bounded JSON state, RNG structures validated before any
  generator is touched and rolled back on failure, nothing restored selects a module or a class.
- All paths are opened relative to directory descriptors without following symlinks; files are
  regular, bounded and read once; new files are `O_EXCL` 0600 then 0444.

### Fixed

- Found by the new tests: 0-d arrays came back one-dimensional through safetensors (now carried
  by a `$scalar` marker), and a race in creating the lock file returned `ENOENT` on macOS.

### Changed

- No core dependency changed: `safetensors` and `fsspec` are optional extras. `pytest` gains the
  `packaging` marker; `docs`, `audit` dependency groups and `build`, `twine`, `hatchling` in `dev`.

### Added (earlier in this release)

- Correctness fixes: lifecycle callbacks (finish, mark failed) still run after ten failed logs; the
  Lightning hook finishes after the last stage so `trainer.test()` metrics reach the store, is
  rank-gated in every stage and refuses to be pickled; the YOLO hook logs metrics only for epochs
  with a validation; points a forwarder dropped or lost in failed calls are counted and summarised;
  a missing snapshot or artifact file is an integrity error; the JSON report holds every violation;
  IoU grids are compared with a tolerance and interval overlap needs equal bootstrap settings;
  `none_gated` is decided per selector.
- Forwarder endpoint checks close the allow-list bypasses: W&B always receives an explicit pinned
  `base_url`, URLs are validated structurally (no backslash, percent-encoding, `@`, query, fragment,
  non-ASCII or IPv6 hosts) and the normalised URL is what the SDK receives, ports other than 443
  need a `host:port` entry in `CVPILOT_FORWARD_HOSTS`, and `MLFLOW_REGISTRY_URI` is checked and pinned.
- Smaller hardening: the error boundary maps `SystemExit` and `KeyboardInterrupt` (an `exit(0)` inside
  a command is a failure, never success); a limit of 100 points is refused because it can never
  fail; snapshots whose per-class support differs are refused (`class_support`); `open_artifact`
  reads the file once from one descriptor, hashes those bytes and returns them from memory; a
  forwarder worker survives a `BaseException` from an adapter (disable, log once, close); error
  lines pass through `safe_text`; the repository workflow checks out with
  `persist-credentials: false`.
- Hardening: report files are written only inside `--report-dir` (default the current directory),
  never in the store or onto `$GITHUB_ENV`/`$GITHUB_OUTPUT`/`$GITHUB_PATH`/`$GITHUB_STEP_SUMMARY`,
  and only over files cvpilot created unless `--force` (`--allow-outside` lifts the root rule);
  metric steps are bounded to the store range and a batch that fails with a data error is dropped
  (`metrics_dropped`) instead of growing the buffer; forwarder queues are bounded in points
  (200 000), only rank zero forwards (`RANK`, `LOCAL_RANK`), forked children drop the inherited run
  and fan-out, and a crashed YOLO training marks the run failed; forwarders accept only https
  endpoints (http in dev), an optional `CVPILOT_FORWARD_HOSTS` allow-list, refuse CA overrides
  outside dev and read and freeze their environment once on the training thread.
- Trusted CI gates: `gate --recompute --gt GT --classes FILE` re-evaluates the stored predictions on
  trusted ground truth and fails unless every stored metric matches; `slice_hash` (slice membership)
  is stored in snapshots and must be equal, with equal per-slice image counts; `--require-commit`
  binds the candidate to the commit under test; a baseline run must be finished and is never the
  candidate; `--max-skipped` caps `--skip-small-slices` (default 0); `baseline show` prints the chain
  head and `--expect-baseline-chain` anchors it outside the store. The Action gains the matching
  inputs, accepts only `tag:` baselines unless `allow-run-baseline` is set and writes the verdict to
  the step summary. README and docs/architecture/06-cross-cutting.md state the trust model.
- `benchmarks/compare_throughput.py`: `cvpilot gate` with both reports on a 5 000-image snapshot
  takes 0.27 s end to end in a child process (81 MB peak), against a budget of 3 s and 300 MB.
- GitHub Action: `action.yml`, a composite action around `cvpilot gate`. Inputs reach the script only
  through `env:` and are quoted, options are passed as `--name=value` and the run id after `--`; no
  third-party action. `tests/unit/test_action.py` parses the YAML, rejects any `${{ ... }}` in a
  `run:` block, unpinned actions, write permissions and unquoted variables, and runs the real script
  with hostile inputs. The repository workflow now pins its actions to full SHAs. `pyyaml` joins
  the dev dependencies.
- Hooks: `cvpilot.autolog(target, project=..., task=...)` attaches a Lightning logger and callback
  (`cvpilot[lightning]`) or Ultralytics callbacks (`cvpilot[ultralytics]`, AGPL-3.0: the notice is
  in the adapter, in the `hook_attached` event and in the docs; the package is never imported by
  CV-Pilot). Callbacks never raise into the host (a guard counts failures, logs the first as
  `hook_failed` with the error type only and disables the hook after 10 in a row), make no network
  call, read framework objects through narrow views and cap what they take. Registry
  `cvpilot.hooks`.
- Forwarders: `init(forward_to=[...])` or `CVPILOT_FORWARD_TO` sends every logged scalar to MLflow
  (`cvpilot[mlflow]`), W&B (`cvpilot[wandb]`) or Trackio (`cvpilot[trackio]`) through the
  `cvpilot.forwarders` registry. Each forwarder has a daemon worker thread behind a bounded queue:
  a failing, slow or hung tracker is counted and logged once (`forward_failed`,
  `forward_summary`), never raised into or blocking the training loop (`finish` waits at most 5 s).
  Tokens stay in the SDK's own environment; adapters refuse credentials in URLs, plain http outside
  dev and disabled TLS verification. ADR-0008.
- Reports: `--report PATH` (static HTML: every name escaped, no script or link, CSP limited to its
  hashed inline style, severity-first with row caps) and `--report-json PATH` (canonical JSON, no
  timestamps) on `compare` and `gate`; atomic writes that refuse symbolic links; golden tests.
- `cvpilot compare RUN --baseline tag:NAME|RUN_ID` and `cvpilot gate RUN --baseline ... --max-drop
  NAME=POINTS --per-class-max-drop NAME=POINTS --slice-max-drop "KEY=*:POINTS" --min-images N`.
  Deltas and limits are points on the 0..100 scale (a drop is `(baseline - candidate) * 100`;
  improvements never fail). The gate fails closed: absent or non-finite metrics, absent,
  mismatched or too small slices, too few images, thresholds that gated nothing, snapshots or
  baselines that fail verification, and a different `eval_set_hash`, task, class names or
  evaluation parameters (`compare_refused`, exit 1) all fail. Events `compare_completed`,
  `compare_refused`, `gate_passed`, `gate_failed`, `slice_skipped`. ADR-0007.
- Baselines: `cvpilot baseline set|list|show` pin a verified snapshot under a tag. The `baseline`
  table is append-only (`--replace` adds a newer pin and keeps the history), each row carries a
  digest and a hash chain, and every read re-verifies the record, the chain and the pinned
  snapshot digest. Events `baseline_set`, `baseline_refused`, `baseline_integrity_failed`.
- Eval engine: `cvpilot.log_eval` / `Run.log_eval` and `cvpilot eval RUN_ID --preds P --gts G
  --classes C` (alias `eval-file`). COCO-protocol detection metrics (101-point AP, small/medium/large ranges,
  per-class AP/AR, confusion with background), top-1 classification, slices on image metadata
  (at most 256 values per key), seeded bootstrap intervals over images, canonical `eval_set_hash`,
  immutable `EvalSnapshot` keyed `(run_id, eval_set_hash, step)` with predictions and confusion as
  content-addressed Parquet artifacts; `runs show` lists snapshots; event `eval_completed`.
  Parity with pycocotools 2.0.11 is exact on 300 randomized sets and the edge cases
  (tolerance 1e-4). `benchmarks/eval_throughput.py`: 5 000 images and 100 000 detections in 0.7 s,
  1 000 bootstrap resamples +2.1 s, 140 MB peak.
- Bounded Parquet input (`store/tables.py`, registry `tables`): file size, rows, row groups,
  decoded bytes and string widths are checked from metadata before reading.
- Run capture: `cvpilot.init`, `log`, `log_artifact`, `finish` and the `Run` handle. Metric
  points are validated per call, buffered (at most 10 000 points or 2 s) and written as
  batched Parquet files; a failed write keeps the points. Steps must be strictly increasing
  per metric key. Runs left `running` by a dead process can be marked `failed`
  (`runs.recover_run`); `init(resume_from=...)` continues a `preempted` or `failed` run.
- Local run store: SQLite in WAL mode with `schema_version` and a `migrations` table, forward
  migrations with a backup, Parquet metric files indexed after an atomic rename, compaction at
  `finish`, a content-addressed artifact directory, validated store paths and run ids.
- Domain: `RunMeta`, `Repro`, `MetricPoint`, `Artifact`, the `MetricSink` and `RunStore`
  protocols, monotonic ULID run ids, dataset Merkle hash and plain-file hash.
- Reproducibility capture: git commit, dirty flag and diff of tracked changes, lockfile or
  installed-set hash, Python, host, CUDA and GPU names (optional `pynvml`, else `nvidia-smi`,
  else none) and the command line, redacted per argument. Works outside git and without a GPU.
- CLI: `runs list` (alias `ls`), `runs show`, `repro`; events `run_started`, `run_resumed`,
  `metrics_flushed`, `run_finished`; one `error:` line plus one event for expected failures.
- Settings: `CVPILOT_PROJECT`, `CVPILOT_STORE_DIR`, `CVPILOT_STORE_COMMAND_LINE`.
- Entry-point registries (`cvpilot.stores`, `cvpilot.capturers`) and layering tests for ADR-0004.
- Tests: unit, golden (100 000-point round trip, `repro` script round trip) and chaos
  (SIGKILL of a writer mid-batch); `benchmarks/log_readback.py`, `benchmarks/install_size.sh`.

### Changed

- New core dependency: `numpy` (eval math, imported lazily so the CLI start-up is unchanged);
  `pycocotools` is a dev-only dependency for the parity tests. ADR-0003, ADR-0005 and the
  overview are amended; `faster-coco-eval` is not added.
- New core dependency: `pyarrow` (imported lazily, store only). ADR-0003, ADR-0005 and the
  overview constraints are amended; DuckDB stays out until cross-run queries exist.

### Security

- The gate is documented as a trust boundary only under the conditions of the new trust model;
  without `--recompute`, `--require-commit` and `--expect-baseline-chain` it protects against
  accidents, not against a contributor who controls the store or the workflow.
- Snapshot integrity: per-snapshot SHA-256 and a per-run hash chain verified on every read,
  exclusive 0444 snapshot files written in the insert transaction, artifacts re-hashed before
  deduplication and read-only, triggers on `artifact` and `eval_snapshot`, `recursive_triggers`.
- Eval inputs: Parquet decoded size and string checks run on the decoded batches, leaf-column size
  mapping, bounded confusion matrix, slice work and memory budgets, `--max-memory-mb`, class file
  read without following symlinks, control characters and lone surrogates rejected, closed runs
  refused without `--allow-closed`, scrubbed tracebacks and error lines.
- Eval semantics: score ties across images follow the numeric image id, default class names come
  from the ground truth only, snapshots record parameters, provenance, slice spec and `ci_status`.

- Secret redaction for stored command lines and configs: by key (camelCase and run-together
  names included), by argument position, bearer credentials, URL user-info and signed query
  parameters. Best effort for free text; `CVPILOT_STORE_COMMAND_LINE=false` stores no arguments.
