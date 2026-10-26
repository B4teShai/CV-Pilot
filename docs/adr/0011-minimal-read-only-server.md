# ADR-0011: A minimal read-only server (design only)

**Status:** proposed
**Date:** 2026-10-26

## Context

The architecture has carried a full server since the start (02-components, `server/`): ingest from a
sync client, its own database (SQLite or Postgres), websockets, project tokens, htmx pages, share
links, server-side alerts, a Docker image. None of it is built. What people ask for first is smaller:
to look at the runs of a training machine from a laptop, and to watch a run's metrics while it
trains, without copying the store around.

Facts that shape the decision:

- Everything a viewer needs is already in the local run store (ADR-0005): run rows and the captured
  environment in SQLite (WAL mode), metric points in Parquet, snapshots and baselines with digests
  that every read verifies. The store is written by one training process per run and read by the
  CLI concurrently today.
- Every phase so far found real defects in each new mechanism, and a network listener is the largest
  attack surface this project would have. Aim's 2026 reports (an unauthenticated tracking server
  exposing run data) are the failure to avoid.
- The organisation's logging standard applies (structured JSON, constant messages, snake_case
  fields, no tokens in logs), as does the dependency allow-list: FastAPI (MIT), Uvicorn (BSD) and
  `websockets` (BSD) are allowed in an optional extra, never in core.
- No server code is written in this phase. This record fixes the scope, the threat model and what
  is left out, so the build in the next phase is reviewed against it.

## Options considered

### Option A: the full server of the architecture (ingest, database, UI, share links, alerts)

Trade-offs: one design for everything, remote training machines included. It needs a sync client
with an outbox, a second schema and migrations, a write API that accepts data from the network, a
web UI with its escaping and CSRF surface, and token lifecycle management. Every one of them is a
review round. Not now.

### Option B: a read-only server over the existing project stores

A process on the machine that holds the store (or has it mounted) serves the runs, read-only, to
token holders: JSON over HTTPS behind a reverse proxy, and one websocket per watched run that pushes
new metric points as the store receives them. Trade-offs: no new storage, no write path, no schema;
the store's own verification stays in charge. It cannot see a run whose store it cannot read, so a
remote training machine still needs a shared volume (or the later sync). Chosen.

### Option C: no server; document `sshfs` or a mounted bucket plus the CLI

Trade-offs: nothing to secure. But there is no live view, and people then expose the store
directory through ad hoc file servers, which is worse than a small authenticated one.

## Decision

Option B, built in Phase 8 as `cvpilot server` behind a `[server]` extra, to this scope and these
rules:

**Scope.**

1. Read-only routes, JSON only, versioned under `/api/v1`: `GET /api/v1/projects/{project}/runs`
   (the `runs list` columns, paged, at most 200 per page), `GET /api/v1/projects/{project}/runs/{run_id}`
   (the `runs show` fields: status, task, timestamps, config as stored (already redacted), metric
   keys with point counts and step range, snapshot ids), and
   `GET /api/v1/projects/{project}/runs/{run_id}/metrics?key=...&after_step=...` (points of up to 20
   keys, at most 10 000 points per response). `GET /healthz` is the only unauthenticated route and
   returns `ok` and nothing else.
2. `WS /ws/projects/{project}/runs/{run_id}`: after authentication the server sends the points
   logged since `after_step`, then new points as they appear (polling the store at most once per
   second per run, shared by all viewers of that run), then a final `status` message when the run
   ends. Messages are JSON with a schema version; the server never accepts data on the socket after
   the authentication message.
3. Data comes only through the existing read paths (`runs.queries`, the store's verified reads),
   with the SQLite file opened read-only (`mode=ro`). No route takes a file path, a URL or a query
   language from the client; `project` must match the project name pattern and `run_id` must be a
   ULID before anything is opened.

**Authentication.**

4. Bearer tokens per project, 32 random bytes, created by `cvpilot server token add PROJECT` (shown
   once). The server stores only `sha256(token)` in a file named by `CVPILOT_SERVER_TOKEN_HASHES`
   (mode 0600, refused if group or world readable), compares with `hmac.compare_digest`, and a
   token grants read access to its own project only. HTTP routes take the token only in the
   `Authorization` header (a token in a query string is refused, because URLs end up in logs and
   browser history). The websocket takes it in the first message, which must arrive within 5 s.
5. Every route except `/healthz` requires a token; a test enumerates the app's routes and fails if one
   answers without a token. Failed authentication is rate-limited per client address (10 per minute)
   and logged as `auth_failed` without the presented value.

**Transport and limits.**

6. The server binds to `127.0.0.1` by default. Binding another address requires
   `--behind-proxy` (TLS is terminated by a reverse proxy; the server never handles certificates) or
   `CVPILOT_ENV=dev`. No CORS headers are sent; websocket connections with an `Origin` header that is
   not in `--allow-origin` are refused.
7. Bounds, as constants with boundary tests: request body 0 bytes (no route reads a body), 64
   concurrent websockets in total and 4 per token, a send queue of 100 messages per socket (a client
   that does not read is disconnected, never buffered without bound), 600 requests per minute per
   token, response sizes as in rule 1, a 30 s idle timeout.
8. Logging per the organisation standard: one `request_completed` or `request_failed` event per
   request with method, route template (not the raw path), status code, duration, project and request
   id; tokens, `Authorization` headers and query strings are never logged.

## Threat model

Assets: the run data (configs, metrics, the captured environment), the tokens, and the training
machine that runs the server. Trust boundaries: browser or script ↔ reverse proxy ↔ server; server
↔ the store files (written by training processes of the same user).

| Threat | Where | Mitigation in the design |
|---|---|---|
| Reading runs without a token | every route | token on every route but `/healthz`; route-enumeration test; read access per project only |
| Token leak through logs, URLs or errors | server, proxy logs | header or first websocket message only; query-string tokens refused; nothing token-like logged; hashes on disk, never tokens |
| Guessing a token | auth | 256-bit tokens; constant-time comparison; failed attempts rate-limited and logged |
| Reading another project's data with a valid token | routes | the token's project must equal the path's project; checked before the store is opened |
| Path traversal through `project` or `run_id` | routes | pattern and ULID checks before any path is built; the store resolves and contains every path (ADR-0005) |
| Data the store must not reveal (diffs, command lines with secrets, checkpoint files) | run detail | the git diff, the stored command line and every file (artifacts, checkpoints) are not served at all; the config is served as stored, which is already redacted by key |
| Denial of service: many sockets, slow readers, huge pages | server | the limits of rule 7; one store poll per run per second however many viewers |
| The server disturbing training (locks, writes) | store | read-only connection, no writes, WAL readers do not block the writer; polling at most once per second |
| Cross-site use from a browser (CSRF, other origins) | browser | no cookies (bearer tokens), no CORS, origin check on websockets, no state-changing route exists |
| Interception | network | loopback by default; other addresses only behind a TLS proxy (`--behind-proxy`) |
| A tampered store served as truth | store | the store's own digest checks on snapshots and baselines apply; the server does not vouch for more than the CLI does |
| Vulnerable dependencies | `[server]` extra | pinned lower bounds, the lock, the audit job; never in core |

Out of the model: a host where the attacker can already read the store files (they can read the
data directly), and a proxy misconfigured to strip TLS.

## What is deliberately not built

- Any write API: no ingest, no run creation, no status changes, no baseline pins, no deletes.
- The sync client and its outbox; remote training machines are served only through a shared volume.
- A database of the server's own, migrations, Postgres.
- Web pages (htmx or other); the API is JSON and a separate viewer can be built on it later.
- Share links, users, roles, SSO, token expiry and rotation beyond adding and removing a hash line.
- Server-side alerts and Telegram or Slack commands; alerts stay in the training process (ADR-0009).
- Serving files: artifacts, checkpoints, reports, the git diff, the reproduction script.
- `gate --from-server`; CI keeps reading a store or a checked-out copy.
- A Docker image and TLS handling inside the server.

## Consequences

- The server adds a network surface without a write path; a bug in it can leak data, but not change
  a run, a baseline or a checkpoint.
- 02-components' full `server/` design remains the long-term picture; this ADR is the first, smaller
  step, and the parts above stay planned.
- The build will need: route-enumeration and auth tests, limit boundary tests, a websocket
  backpressure test, a read-only test that runs the server against a store a training process is
  writing, and the logging contract test for the server's events.
- Remote training machines need a shared volume until a sync client exists.

## Links

- [0005-run-store.md](0005-run-store.md), [0009-alerts-and-hf-hook.md](0009-alerts-and-hf-hook.md)
- [../architecture/02-components.md](../architecture/02-components.md),
  [../architecture/06-cross-cutting.md](../architecture/06-cross-cutting.md)
