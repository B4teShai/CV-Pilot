# Errors and logging

## Exit-code contract

| Code | Meaning | Who sets it |
|---|---|---|
| 0 | Success | command returned normally |
| 1 | The command ran and failed: an unexpected exception, or a check that was asked to fail (for example a gate or `--fail-on`) | error boundary in `cli/app.py`, or the command explicitly |
| 2 | Usage or configuration error: unknown command, bad option, invalid `CVPILOT_*` setting | Typer usage errors; settings validation before logging is configured |
| 130 | Interrupted by the user (Ctrl-C) | error boundary |

Rules:

- `stdout` carries command output only (results, JSON, paths). `stderr` carries logs and
  usage errors. Scripts can therefore pipe `stdout` safely.
- A failing command prints one actionable line to `stderr` for a human and one structured
  `command_failed` log event for machines. Never a traceback by default.
- New commands document their exit codes in their `--help` text and in
  [../architecture/04-flows.md](../architecture/04-flows.md).

## The single error boundary

Exceptions are caught once, in `main()` in `cli/app.py`. Inside the application and domain
layers:

- Raise specific exceptions (`SettingsError`, `FormatError`, `ValidationError`, …) with
  messages that name the offending input.
- Never catch broad exceptions to "keep going" silently. If a step can be skipped, the skip
  is logged at `WARN` with the reason and counted in the command's report.
- Never log and re-raise; the boundary logs. Logging twice is a bug.

## Structured logging contract

Logging follows [ADR-0002](../adr/0002-logging.md) and the organisation standard. The
implementation is `src/cvpilot/logging.py`; the contract test is
`tests/unit/test_logging_contract.py`.

- JSON lines on `stderr`. Required fields on every record: `@timestamp` (UTC ISO 8601),
  `level`, `message`, `service`, `env`, `request_id`, `logger`.
- `message` is a constant event name in `snake_case` (`doctor_completed`,
  `command_failed`). Variable data goes in separate `snake_case` fields.
- `request_id` is bound per CLI invocation and cleared at exit.
- Keys `token`, `password`, `api_key`, `jwt`, `authorization` (case-, hyphen- and
  underscore-insensitive, exact or suffix, nested in dicts and lists) are redacted. Redaction
  is key-based: never put secrets in free text or in the `message`.
- Levels: `INFO` in production, `DEBUG` only when `CVPILOT_ENV=dev`, `WARN` for
  recoverable problems, `ERROR` for failures that need action.
- Logging never raises. Non-serialisable values fall back to `str()`; non-finite floats are
  emitted as strings; cyclic or deeper-than-8 structures are truncated.

Usage:

```python
from cvpilot.logging import get_logger

log = get_logger(__name__)
log.info("dataset_loaded", image_count=n_images, annotation_count=n_annotations)
```
