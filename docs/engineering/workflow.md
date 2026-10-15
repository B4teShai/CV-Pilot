# Development workflow

## Branches and commits

- `main` is always releasable. Work happens on short-lived branches named
  `<type>/<topic>` (`feat/coco-reader`, `fix/redact-nested-keys`).
- Commits follow Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `chore:`. One logical change per commit. The body states why, not what.
- No tool, assistant or generator attribution in commit messages or source.

## Pull requests

A pull request contains:

1. The failing test(s), then the change that makes them pass.
2. Documentation updates in the same PR: README for user-visible behaviour, the matching
   `docs/architecture/` document and diagram for structural change, an ADR for a decision.
3. A description with: problem, approach, how it was verified (commands and output),
   anything not verified.

## Review checklist

Reviewers check, in this order:

1. **Correctness:** tests exist, failed first, and cover the edge cases in
   [testing.md](testing.md). Behaviour matches the documented contract.
2. **Boundaries:** imports respect [ADR-0004](../adr/0004-layering-and-dependency-rule.md);
   infrastructure is reached through registries, not direct imports from application code.
3. **Contracts:** exit codes and logging follow
   [errors-and-logging.md](errors-and-logging.md); settings are validated; no secrets.
4. **Dependencies and licenses:** [dependencies-and-licensing.md](dependencies-and-licensing.md).
5. **Docs:** architecture documents and diagrams updated; status markers (built/planned)
   correct; every documented command was actually run.
6. **Scope:** nothing unrelated slipped in.

## Definition of done

A change is done when all of the following are true:

- `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`
  pass locally and in CI on every supported Python version.
- New behaviour has a test that failed before the change.
- Logging and exit codes follow the contracts; no secrets or personal data in code, logs
  or fixtures.
- Untrusted input is validated; failures produce one clear message and the documented exit
  code.
- Dependencies and licenses comply with the allow-list.
- README, architecture documents and diagrams reflect what is implemented.
- Review findings are resolved or explicitly deferred with an issue link.

## Continuous integration

`.github/workflows/ci.yml` runs lint, format check, type check and tests with coverage on
Python 3.10 to 3.13 on every push to `main` and every pull request. A red CI blocks merge.

## Releases

- Semantic versioning. Before 1.0, minor versions may break the public API; the CHANGELOG
  says so explicitly.
- Deprecations: the old name keeps working for one minor version and emits a
  `DeprecationWarning` that names the replacement.
- Release steps: update `CHANGELOG.md`, bump `__version__`, tag `vX.Y.Z`, build with
  `uv build`, publish to TestPyPI, install the wheel in a clean environment and run the
  `doctor` command, then publish to PyPI.
- A release includes: wheel and sdist, CHANGELOG entry, updated docs, the benchmark numbers
  used for any performance claim.
