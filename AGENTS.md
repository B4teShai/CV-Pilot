# AGENTS.md

Guide for coding agents and new contributors. Binding rules live in
[docs/engineering/](docs/engineering/README.md); this page is the map, not a copy.

## What it is

CV-Pilot is a CV-aware layer over training runs: it records runs, evaluates detections (COCO mAP at
parity with pycocotools), gates a candidate against a pinned baseline, and keeps crash-safe checkpoints.
Python 3.10+, Apache-2.0, alpha; [docs/status.md](docs/status.md) is the only page that says what works.

## Commands

Run from the repository root. `uv sync` creates `.venv` with the `dev` and `docs` groups (CI: `--locked`).

```bash
uv sync                                    # install from a clone
uv run cvpilot doctor                      # prints "ok"; exit 0
uv run cvpilot --help                      # eval compare gate baseline runs ckpt alerts repro doctor
uv run ruff check . && uv run ruff format --check .
uv run mypy                                # strict, src/ only
uv run pytest                              # default run: everything except real_libs and clone
uv run pytest --cov=cvpilot --cov-fail-under=85   # the CI form
uv lock --check                            # lockfile matches pyproject.toml
```

Test tiers ([testing.md](docs/engineering/testing.md) says what each proves):

| Tier | Command | Notes |
|---|---|---|
| Unit + contract | `uv run pytest tests/unit -q` | ~3200 tests, ~3 min; includes `test_*_contract.py`, `test_layering.py`, `test_api_surface.py` |
| Golden | `uv run pytest tests/golden -q` | byte/field-identical outputs |
| Parity | `uv run pytest -m parity -q` | 465 cases against pycocotools 2.0.11, ~30 s |
| Chaos | `uv run pytest -m chaos -q` | SIGKILL inside saves; minutes; part of the default run too |
| Docs | `uv run pytest tests/docs -q` | runs every tutorial/README command, checks the CLI reference |
| Notebooks | `uv run pytest tests/notebooks -q` | |
| Release | `uv run pytest tests/release -q` | builds wheel + sdist, checks contents and clean installs (`-m packaging` subset) |
| Smoke | `uv run pytest -m real_libs tests/smoke` | real framework/tracker libs; skips when the extra is absent |
| Clone | `uv run pytest -m clone` | fresh clone + `pip install .`; needs the network |

Docs and generated files (never edit `docs/reference/cli.md` by hand):

```bash
NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict      # writes site/ (ignored)
uv run python scripts/gen_cli_reference.py             # after any CLI option/help change
CVPILOT_UPDATE_API_SNAPSHOT=1 uv run pytest tests/unit/test_api_surface.py   # intended API change only
python3 scripts/check_readme.py                         # README is the package index page
```

Moved the checkout? Stale bytecode and venv shebangs keep the old path: delete `__pycache__/`, `uv sync --reinstall`.

## Module map (`src/cvpilot/`)

Layers per [ADR-0004](docs/adr/0004-layering-and-dependency-rule.md) and
[02-components](docs/architecture/02-components.md); imports point downward only.

- `cli/` (interface): Typer app in `cli/app.py`, the single error boundary and exit codes; one `*_cmds.py` per group.
- `api.py` (interface): public facade: `init`, `log`, `log_eval`, `log_artifact`, `finish`, `autolog`, `resume_path`.
- `runs/` (application): run lifecycle, queries, forwarder/channel fan-out.
- `eval/` (application): evaluation engine, slices, bootstrap intervals, input validation.
- `compare/` (application): baselines, compare, gate, text/HTML/JSON reports, recompute.
- `ckpt/` (application): checkpoint manager: atomic save, verify, restore, retention, preemption, remote mirror, HMAC trust.
- `alerts/` (application): JSON rule config, rule engine, replay.
- `repro.py` (application): the reproducibility record, filled from the `capturers` registry.
- `core/` (domain): schemas and plugin protocols, metrics, confusion, hashing, ULIDs, redaction, errors.
- `store/` (infra): local run store: SQLite WAL, Parquet metrics, content-addressed artifacts, migrations.
- `capture/` (infra): git/packages/python/host/gpu/command capture.
- `hooks/` (infra): Lightning, Hugging Face Trainer, Ultralytics, MMEngine, RF-DETR.
- `forward/` (infra): MLflow, W&B, Trackio forwarders.
- `notify/` (infra): webhook, Slack-style, Telegram channels (send-only) over one outbound client, `notify/_http.py`.
- `remote/` (infra): fsspec checkpoint remote (`s3`, `gs`, `az`, `memory`), AWS/GCP preemption notices.
- Cross-cutting: `logging.py`, `settings.py` (`CVPILOT_*`), `registry.py` (entry-point registries), `deprecation.py`.
- Outside `src/`: `action.yml` (GitHub Action running `cvpilot gate`), `scripts/`, `benchmarks/`, `examples/`, `notebooks/`.

## Invariants

- **Core dependencies are exactly** `click`, `numpy`, `pyarrow`, `pydantic`, `structlog`, `typer`.
  Adding one is an ADR-0003 amendment; the workspace QA gate and `tests/release` fail otherwise.
- **Frameworks and trackers are extras only**, imported lazily inside the adapter function; the
  import error names the extra. `tests/unit/test_layering.py` rejects heavy top-level imports.
- **Apache-only core.** Copyleft code is never vendored or a core dependency; the AGPL Ultralytics
  extra is referenced only from `hooks/ultralytics.py`, which receives the model and never imports
  it. See [dependencies-and-licensing.md](docs/engineering/dependencies-and-licensing.md).
- **Store** ([ADR-0005](docs/adr/0005-run-store.md)): steps strictly increase per metric key,
  eval snapshots are immutable and verified on read. **Checkpoints**
  ([ADR-0006](docs/adr/0006-checkpoint-semantics.md)): tmp + fsync + manifest + `os.replace`, then
  `LATEST`; restore verifies size/SHA-256/chain before loading; a file without a manifest is never
  loaded; nothing is deleted before a newer one is verified (and uploaded); `pt` is opt-in and loads
  with `weights_only=True`.
- **Gate** ([ADR-0007](docs/adr/0007-compare-and-gate-semantics.md)): limits in points on 0..100,
  fail closed on anything missing, non-finite or unverifiable; refuse unless eval set, task, classes
  and params are equal; exit 0 only after `gate_passed`. Recompute from verified snapshots, never a
  stored verdict. In CI pin the baseline as `tag:NAME` (the Action refuses a run id by default).
- **Forwarders/channels never block or raise into training** ([ADR-0008](docs/adr/0008-forwarders.md),
  [ADR-0009](docs/adr/0009-alerts-and-hf-hook.md)): bounded queues, daemon workers, failures logged once
  by exception type; https only (http only with `CVPILOT_ENV=dev`), `CVPILOT_FORWARD_HOSTS` allow-list.
- **Exit codes** 0 ok, 1 failure, 2 usage/config, 130 interrupted; results on stdout, logs on stderr
  ([errors-and-logging.md](docs/engineering/errors-and-logging.md)).
- **Logging contract** ([ADR-0002](docs/adr/0002-logging.md), `src/cvpilot/logging.py`,
  `tests/unit/test_logging_contract.py`): JSON lines on stderr with `@timestamp`, `level`, `message`,
  `service`, `env`, `request_id`, `logger`. `message` is a constant `snake_case` event name; variable
  data goes in separate `snake_case` fields. `request_id` is bound per CLI invocation and cleared at
  exit. Errors are logged once, at the boundary in `cli/app.py`. `DEBUG` only with `CVPILOT_ENV=dev`.
- **Never log credentials.** Redaction is key-based (`token`, `password`, `api_key`, `jwt`,
  `authorization`, ...), so a credential must never appear in free text or the `message`. Channel
  credentials come from the environment only and live in one private attribute (not in `repr`,
  errors, logs or pickles). The Telegram bot credential sits in the Bot API URL path, so that URL is
  built inside `notify/telegram.py` from the fixed host and never leaves the object; `_http.py` errors
  never contain a URL. Forwarders take no credential argument: the tracker SDK reads its own variables.

## Adding a hook, forwarder, channel or remote

A plugin is a module plus an entry point in `pyproject.toml`: `cvpilot.hooks`, `cvpilot.forwarders`,
`cvpilot.channels`, `cvpilot.remotes` (also `capturers`, `stores`, `tables`, `spot_notices`).

1. Implement the protocol in `src/cvpilot/core/schema.py` (`Hook`, `Forwarder`, `Channel`,
   `RemoteStore`); declare `requires` (distribution names, `a|b` for alternatives) so the registry
   raises `MissingExtraError` before importing. Import the library inside the method that needs it.
2. Add the optional extra in `pyproject.toml`, run `uv lock`, check the license table.
3. Test with the fakes (`tests/fake_frameworks.py`, `tests/fake_forwarders.py`,
   `tests/fake_channels.py`) in `tests/unit/`, plus a `real_libs` test in `tests/smoke/`.
4. Update `docs/architecture/02-components.md` (registries), `05-runtime.md` (extras), the guide
   page, `docs/status.md`, and an ADR for any new decision.

Defining rules: [ADR-0004](docs/adr/0004-layering-and-dependency-rule.md) (registries), [ADR-0008](docs/adr/0008-forwarders.md)
(forwarders), [ADR-0009](docs/adr/0009-alerts-and-hf-hook.md) (channels, HF hook),
[ADR-0010](docs/adr/0010-real-checkpoint-files-and-mmengine.md) (MMEngine), [ADR-0012](docs/adr/0012-chaos-rfdetr-and-spot-recipe.md) (RF-DETR).

## Definition of done ([full list](docs/engineering/workflow.md#definition-of-done))

- `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` pass;
  coverage stays at or above 85 %. A test that failed before the change covers it.
- `uv lock --check`, the strict mkdocs build and `uv run pytest tests/docs tests/release -q` pass.
- CLI reference regenerated; API snapshot changed only on purpose (removals need a deprecation).
- Docs, diagrams and `docs/status.md` updated in the same change.
- `python3 tools/qa_check.py cv-pilot`, run from the workspace root above this repo, prints `ok   cv-pilot`.

## Don'ts

- Don't add a core dependency, an AGPL/GPL import, vendored weights or `torch` in core.
- Don't import an adapter by module path from application code; go through `registry.py`.
- Don't catch broad exceptions to keep going, print tracebacks, or log and re-raise.
- Don't put credentials, or URLs carrying them, in code, fixtures, logs or errors; fake values in
  tests must look fake. Configuration comes from `CVPILOT_*` variables.
- Don't weaken a gate, trust or checkpoint check to make a test pass; tolerance changes are reviewed.
- Don't `pickle` untrusted data. Don't publish, tag, upload or release: that needs the maintainer
  ([release.md](docs/engineering/release.md)).
- Don't add tool, assistant or generator attribution to commits or source; tool-specific agent
  configuration is gitignored and stays local.
