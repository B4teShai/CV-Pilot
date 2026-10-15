# Release runbook

This page is the procedure for publishing CV-Pilot to PyPI. **Publishing is an irreversible
external action: it needs the maintainer's explicit approval, and the PyPI trusted-publisher setup
below must be done by the maintainer. Nothing in the repository, and no agent or script that works
on it, uploads, registers a name or creates a release.** The pipeline can only publish after a
maintainer pushes a version tag *and* approves the protected `pypi` environment.

## Names

`cvpilot` and `cv-pilot` were free on PyPI when they were checked read-only on 2026-10-15 (and
again on 2026-10-16: HTTP 404 for `https://pypi.org/pypi/cvpilot/json` and `/cv-pilot/`, and for
the same name on TestPyPI). **Re-check before every first publication**: a name can be taken at
any moment. PyPI normalises a name by lower-casing it and collapsing runs of `-`, `_` and `.` into
one `-`, so `cv-pilot`, `cv_pilot` and `CV.Pilot` are the same project, while `cvpilot` (no
separator) is a different one: check both spellings. Nothing is registered until the first upload.

## One-time setup (the maintainer)

1. Create the PyPI project by adding a **pending trusted publisher** at
   `https://pypi.org/manage/account/publishing/`: project name `cvpilot`, owner `B4teShai`,
   repository `CV-Pilot`, workflow `release.yml`, environment `pypi`. No API token is created and
   none may be stored anywhere: the workflow has no secret.
2. In the repository settings create the environment **`pypi`** with *required reviewers* (at least
   one person other than the one who pushes the tag) and restrict it to version tags.
3. Make the jobs of `ci.yml` (`test`, `lock`, `docs`, `package`, `audit`) required status checks on
   `main`, and protect `.github/` with CODEOWNERS: the workflows are part of the trust boundary.

4. Add a **tag ruleset** for `refs/tags/v*` (restrict creation to maintainers, block deletion and
   updates), a **deployment tag policy** on the `pypi` environment (only `v*` tags), **branch
   protection** on `main` and a `CODEOWNERS` entry for `.github/`.

### Evidence (mandatory, once, before the first tag)

The workflow cannot enforce repository settings, so the maintainer records proof that they are in
place. All of these commands are read-only; keep the output with the release notes.

```bash notest
gh api repos/B4teShai/CV-Pilot/environments/pypi --jq '{reviewers: .protection_rules, branch_policy: .deployment_branch_policy}'
gh api repos/B4teShai/CV-Pilot/rulesets --jq '.[] | {name, target, enforcement}'    # a tag ruleset on refs/tags/v*
gh api repos/B4teShai/CV-Pilot/branches/main/protection --jq '{checks: .required_status_checks.contexts, reviews: .required_pull_request_reviews}'
gh api repos/B4teShai/CV-Pilot/contents/.github/CODEOWNERS --jq .path
```

On PyPI, open the project's *Publishing* page and confirm the trusted publisher names this
repository, `release.yml` and the environment `pypi`. As of 2026-10-16 GitHub reports no rulesets
and no branch protection for this repository, so none of this evidence exists yet: do not push a
version tag until it does. The build job also refuses a commit that is not an ancestor of
`origin/main`, so a tag on a side branch cannot be published even if the tag ruleset is missing.

## Before tagging

All of this runs locally and uploads nothing.

```bash notest
uv sync --locked
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest --cov=cvpilot --cov-fail-under=85          # also on a real 3.13 environment
uv lock --check
python3 scripts/check_readme.py                          # the README is the package index page
python3 /path/to/tools/qa_check.py .
NO_MKDOCS_2_WARNING=1 uv run mkdocs build --strict
out="$(mktemp -d)"                                     # a fresh directory: no stale artifact can match
version="$(python3 scripts/print_version.py)"
uv build --out-dir "$out"
uv run twine check --strict "$out/cvpilot-$version.tar.gz" "$out/cvpilot-$version-py3-none-any.whl"
python scripts/verify_wheel.py "$out/cvpilot-$version-py3-none-any.whl" --python 3.10 --python 3.13
(cd "$out" && sha256sum -- *)                          # on macOS: shasum -a 256 -- *
```

0. The README is the long description on the package index: every link is an absolute `https://`
   URL into this repository, and it carries no release-status wording ("release candidate",
   "nothing has been published", "install from source"). `scripts/check_readme.py` checks that
   locally and the release workflow runs it before the build, so a tag cannot be built while the
   README is wrong. The status of a release lives in this runbook and the CHANGELOG, not in the
   README.
1. The version is `__version__` in `src/cvpilot/__init__.py`; it changes only in the final
   release-preparation commit, together with the `CHANGELOG` heading. `tests/unit/test_deprecation.py`
   fails if a deprecation is due for removal in that version.
2. Re-check the names (above) and the pins of the third-party actions in `release.yml` and `ci.yml`
   (`gh api repos/<owner>/<repo>/git/ref/tags/<tag>`, and for an annotated tag
   `git/tags/<sha>`; `tests/unit/test_release_workflow.py` lists the verified pins).
3. `tests/release/test_packaging.py` builds both artifacts and checks their contents (license and
   NOTICE present, `py.typed` present, no tests, benchmarks, plans, editor or agent configuration directories, `.env`, caches or
   fixtures, size within budget, only the documented core dependencies), runs `twine check
   --strict`, builds a wheel from the sdist and installs the wheel into clean environments for 3.10
   and 3.13.
4. Record the digests (`sha256sum`, or `shasum -a 256` on macOS, run inside the fresh build
   directory) with the release notes.

## TestPyPI

A rehearsal under the same name may be impossible: a project of that name can already belong to
someone else on TestPyPI (this happened to the sibling project DataCopilot). Rehearse with the
local build and `scripts/verify_wheel.py`, which installs the wheel into clean environments, and
with a local file index (`uv pip install --find-links dist cvpilot`) rather than by uploading.

## Publishing (only with the maintainer's explicit approval)

1. Merge the release-preparation commit to `main`; wait for CI.
2. `git tag v0.1.0 && git push origin v0.1.0` (the tag must equal `v` plus `__version__`; the build
   job fails otherwise).
3. `release.yml` runs: the **build** job (read-only, no secrets) checks the tag against the
   version, runs lint, types and tests, builds the sdist and wheel, runs `twine check --strict`,
   logs the SHA-256 of each file and uploads exactly `dist/`.
4. The **publish** job waits in the `pypi` environment. A reviewer compares the logged digests with
   the ones recorded before tagging and approves. It then downloads the artifact, re-computes and
   compares the digests, checks that every file name carries the tag's version and publishes with
   trusted publishing (`id-token: write`, no token).
5. Verify from a clean machine: `pip install cvpilot==0.1.0`, `cvpilot --version`, `cvpilot doctor`.

## If something goes wrong

A published version cannot be replaced: yank it on PyPI, fix, bump and tag again. If the
`pypi` environment's reviewers did not approve, nothing was published. A leaked key for
authenticated checkpoints (`CVPILOT_CKPT_KEY`) is a deployment matter, not a release one: rotate
it and re-save (see the [trust model](../guide/trust.md)).
