# ADR-0004: Layering and the dependency rule

**Status:** accepted
**Date:** 2026-10-14

## Context

CV-Pilot will grow adapters for many external formats, frameworks, model backends and
services. Without a stated rule, application code starts importing adapters directly,
adapters start depending on each other, and the core becomes impossible to test without
the heavy optional dependencies installed.

## Options considered

### Option A: flat package, convention only

Trade-offs: no ceremony; but nothing stops a `core` module from importing `torch` through
an adapter, which breaks the CPU-only promise silently.

### Option B: four layers with a downward-only import rule and registries

Trade-offs: slightly more structure to learn; in return every layer is testable in
isolation and optional dependencies stay optional.

### Option C: full hexagonal architecture with ports as abstract classes for everything

Trade-offs: maximal decoupling; too much ceremony for a library whose adapters are already
small and whose domain is data tables.

## Decision

Option B. Packages are assigned to one of four layers:

| Layer | Contains | May import |
|---|---|---|
| Interface | `cli/`, the public Python API module | application, domain |
| Application | use cases: the commands' logic, orchestration, reports | domain, infrastructure **through registries only** |
| Domain | schemas, invariants, versioning and hashing, pure computations | standard library, `pydantic`, `numpy` |
| Infrastructure | adapters: file formats, stores, model backends, framework hooks, forwarders, network clients | domain, their own optional dependency |

Rules that reviewers check:

1. Imports point downward only. Infrastructure never imports application or interface;
   domain imports no project package outside itself.
2. Application code selects infrastructure through a registry (`formats`, `embedders`,
   `hooks`, …) keyed by name, never by importing an adapter module by path. Registries
   resolve lazily so unused adapters are never imported.
3. Optional dependencies are imported inside the adapter function that needs them, never at
   module top level in a package that core imports.
4. Cross-cutting modules (`logging.py`, `settings.py`) sit beside the layers and may be
   imported by any layer; they import nothing from the layers.

## Consequences

- The core test suite runs with only the core dependencies installed.
- A new adapter is a new module plus a registry entry; nothing else changes.
- Plugins from other packages register through the same registries (entry points), so
  first-party and third-party adapters are indistinguishable to application code.
- A linter rule (import-linter contract) enforces the layer graph once the package has more
  than one layer populated.

## Links

- `docs/architecture/02-components.md`
