# Contributing to CV-Pilot

Thank you for considering a contribution. This page is the short version; the binding
details are in [docs/engineering/](docs/engineering/README.md).

## Setup

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

All four commands must pass before you open a pull request.

## Before you start

- Read [docs/architecture/](docs/architecture/README.md) to see where your change belongs
  and which layer owns it.
- Open an issue for anything larger than a bug fix so the approach can be agreed first.

## The rules in one screen

- Apache-2.0. No AGPL code in this package; no vendored model weights. See
  [dependencies and licensing](docs/engineering/dependencies-and-licensing.md).
- Test first. Every change ships with the test that failed before it. See
  [testing](docs/engineering/testing.md).
- Structured logging and the exit-code contract are not optional. See
  [errors and logging](docs/engineering/errors-and-logging.md).
- Imports point downward through the layers. See
  [ADR-0004](docs/adr/0004-layering-and-dependency-rule.md).
- Docs and diagrams change in the same pull request as the code.

## Pull request flow

Branch from `main`, keep the PR focused, fill in the description (problem, approach,
verification), and work through the [review checklist](docs/engineering/workflow.md).
