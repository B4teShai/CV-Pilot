# Engineering guide

How code gets written, tested, reviewed and released in CV-Pilot. These documents are
binding for every contribution; the review checklist refers to them by section.

| Document | Covers |
|---|---|
| [standards.md](standards.md) | Python version, project layout, typing, naming, lint and format, docstrings |
| [testing.md](testing.md) | Test tiers, what each proves, fixtures, coverage, what never runs in a unit test |
| [errors-and-logging.md](errors-and-logging.md) | Exit-code contract, the single error boundary, the structured-logging contract |
| [dependencies-and-licensing.md](dependencies-and-licensing.md) | Core versus extras, license allow-list, model weights, how to add a dependency |
| [workflow.md](workflow.md) | Branching, pull requests, review checklist, definition of done, CI, releases |

Quick start for a contributor:

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

All four commands must pass locally before a pull request is opened. CI runs the same
commands on every supported Python version.
