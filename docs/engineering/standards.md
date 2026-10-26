# Coding standards

## Language and layout

- Python 3.10 is the floor for the published package; CI runs 3.10 through 3.14. New syntax
  is used only when every supported version has it. (The floor is a deliberate choice for
  adoption; raising it is an ADR.)
- `src/` layout. The import package is `cvpilot`; the distribution is built with hatchling
  and managed with `uv`. The lockfile is committed.
- One responsibility per module. A module that needs a second paragraph to describe is two
  modules. Files over ~400 lines are a review flag.
- Packages follow the layering in [ADR-0004](../adr/0004-layering-and-dependency-rule.md):
  interface → application → domain → infrastructure. Imports only point downward.

## Typing

- `mypy --strict` on `src/` is part of CI. No `Any` in public signatures; no `type: ignore`
  without a reason in the same line.
- Pydantic v2 models at every boundary where data enters the process: CLI options after
  parsing, files read from disk, network payloads, plugin inputs. Inside the domain, plain
  dataclasses or Pydantic models are both acceptable; choose by whether validation is needed.
- Public functions declare return types. Optional values are `X | None`, never implicit.
- `Literal` and `Enum` for closed sets (task kinds, split names, statuses). No bare strings.

## Naming

- Modules, functions and variables: `snake_case`. Classes: `PascalCase`. Constants:
  `UPPER_SNAKE`.
- Names say what a thing is, not what type it has: `images`, not `image_list`.
- Boolean names read as predicates: `is_crowd`, `has_mask`, `strict`.
- Public API names are stable once released (semver). Renames go through a deprecation
  cycle described in [workflow.md](workflow.md).

## Lint and format

- `ruff check` with rule sets `E, F, I, UP, B, S`; `ruff format` for layout. Line length 100.
- Security rules (`S`) are not silenced in `src/`. Tests may silence `S101` (assert) and, for
  fixtures that deliberately contain fake secrets, `S105`/`S106`.

## Docstrings and comments

- Every public module, class and function has a docstring that states purpose, inputs,
  outputs and raised exceptions. Google style.
- Comments explain why, not what. A comment that restates the code is removed in review.
- No commented-out code. No TODOs without an issue link.

## Performance habits

- Hot paths operate on arrays and tables (NumPy, Arrow/Parquet, DuckDB), never per-item
  Python loops over images or annotations.
- I/O that can be batched is batched. Files are streamed when they can exceed memory.
- Any claim about speed or memory is backed by a script under `benchmarks/` and a number in
  [../architecture/01-overview.md](../architecture/01-overview.md).

## Security habits

- No secrets in code, tests, fixtures or logs. Configuration comes from environment
  variables with the `CVPILOT_` prefix.
- Untrusted input (files, network, plugin output) is validated before use. Paths are
  resolved and checked against the intended root before reading or writing.
- No dynamic code execution, no `pickle` of untrusted data, `weights_only=True` for torch
  checkpoints, `safetensors` preferred.
