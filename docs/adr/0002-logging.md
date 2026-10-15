# ADR-0002: Structured JSON logging

**Status:** accepted
**Date:** 2026-10-14

## Context

CV-Pilot runs as a CLI in developers' shells and in CI, and later inside long-running
processes. Logs are collected by Loki, ELK or Datadog style pipelines that index JSON
fields. The organisation standard requires structured JSON, fixed required fields,
`snake_case` keys, constant messages, redaction of secrets and errors logged once. The
process also prints results to `stdout` that scripts consume.

## Options considered

### Option A: standard library `logging` with a JSON formatter

Trade-offs: no dependency; but context binding (per-invocation `request_id`), processors
for redaction and key normalisation must be hand-built, and every call site must remember
to pass `extra={}`.

### Option B: `structlog` with a processor chain

Trade-offs: one small pure-Python dependency; bound context, processors and JSON rendering
are first-class, which makes the contract testable in one place.

### Option C: `loguru`

Trade-offs: convenient API, but opinionated global state and weaker support for strict
field contracts and processors.

## Decision

Option B. `structlog` renders JSON lines to `stderr`; `stdout` is reserved for command
output. The processor chain enforces:

- Required fields on every record: `@timestamp` (UTC ISO 8601), `level` (upper case),
  `message`, `service`, `env`, `request_id`, `logger`.
- `message` is a constant event name; dynamic values are separate `snake_case` fields.
- Redaction of `token`, `password`, `api_key`, `jwt`, `authorization` keys: case-, hyphen-
  and underscore-insensitive, exact or `*_<key>` suffix, nested in dicts, lists, tuples and
  sets. Key-based only: free text and the `exception` string are not scrubbed, so secrets
  must never be logged as text.
- Errors are logged once, at the CLI boundary, with `exc_info`.
- `request_id` is a UUID bound per invocation and cleared afterwards. Bound context
  (`service`, `env`, `request_id`) cannot be overridden by caller kwargs.
- `INFO` by default; `DEBUG` only when `CVPILOT_ENV=dev`, otherwise capped.
- Logging never raises: non-serialisable values fall back to `str()`; on a serialisation
  failure a minimal valid record with `message=log_serialization_failed` is emitted;
  non-finite floats become strings; structures deeper than 8 levels or self-referencing are
  truncated to `[TRUNCATED]`.
- An invalid `CVPILOT_*` setting is reported as one plain line on `stderr` with exit
  code 2 before logging is configured.
- Any future Java component uses Log4j 2.x ≥ 2.17.1, never 1.x.

## Consequences

- Every log call is a contract: `tests/unit/test_logging_contract.py` fails on a missing
  field, a non-constant message or an unredacted key.
- Call sites cannot log free-form strings; they name an event and attach fields.
- Pydantic models and dataclasses are not walked by redaction; callers pass plain data.

## Links

- Implementation: `src/cvpilot/logging.py`
- Contract and usage: `docs/engineering/errors-and-logging.md`
