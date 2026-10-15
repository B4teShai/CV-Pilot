# ADR-0008: Tracker forwarders that cannot hurt training

**Status:** accepted
**Date:** 2026-10-15

## Context

Teams keep their experiment tracker (MLflow, W&B, Trackio) authoritative for scalars, so CV-Pilot
forwards every logged scalar there. A tracker is a network service: it is down, slow, rate
limiting or misconfigured often enough that a training job (hours, a GPU, sometimes a spot
instance) must never depend on it. The adapters also sit next to credentials (API keys, tracking
tokens) and talk over the network, so they are a place where secrets leak into logs or URLs and
where TLS verification gets switched off to "make it work".

`log()` already buffers points and writes one Parquet batch on the caller's thread; the
forwarders receive the same batches after that write.

## Options considered

### Option A: call the SDK on the training thread after each flush

Trade-offs: simplest; but every tracker stall is a training stall, and an SDK exception needs a
try/except at every call site.

### Option B: one daemon worker thread per forwarder behind a bounded queue

The training thread only does `put_nowait`; the worker calls the adapter. A full queue drops
the batch (counted). An adapter call that runs longer than a stall timeout disables the
forwarder. Closing waits for a bounded time.

Trade-offs: a thread per tracker and dropped batches when a tracker is far behind; in return the
training thread never waits and the failure modes are explicit and countable.

### Option C: a separate process that tails the store and forwards

Trade-offs: strongest isolation and survives a training crash; but a second process to run,
supervise and secure, and it needs the store to be readable while written. Considered for the
`sync/` client, not needed for scalars.

## Decision

Option B, with these rules (each has a test):

1. **Never blocks, never raises.** `Fanout.submit` is non-blocking and wraps everything; the only
   wait on the training side is `Fanout.close` at `finish`, bounded at `CLOSE_TIMEOUT_S` (5 s)
   in total. Workers are daemon threads, so a hung SDK cannot keep the process alive.
2. **Bounded.** `QUEUE_BATCHES` (256) batches and `MAX_QUEUED_POINTS` (200 000) points pending per
   forwarder (queued or being sent; a batch is at most 10 000 points), so a stalled tracker pins a
   bounded amount of memory in points as well as in batches;
   `CALL_TIMEOUT_S` (30 s): a call running longer disables the forwarder the next time a batch is
   offered. A hung call occupies only its own worker; the SDK's own timeouts end it eventually.
3. **Failures are counted and logged once.** A `BaseException` from an adapter (an SDK calling
   `sys.exit`) is handled the same way, disables the forwarder and calls `adapter.close` once. The first failure per forwarder is `forward_failed`
   (WARN) with the forwarder, run id, reason and the exception *type*; the exception text is never
   logged because it can contain a URL or a credential. `forward_summary` at close carries the
   counters. A failed call is retried with the next batch; a failed `open` or a stall disables
   the forwarder for the run.
4. **Configuration errors are loud and early.** An unknown forwarder name, a missing extra
   (checked by distribution name before any import, `a|b` for alternatives) and an unsafe
   endpoint raise from `init()` before a run exists. A plugin that fails with any other
   exception while being constructed is skipped and logged.
5. **Secrets and transport.** Adapters take no token argument and read no credential variable:
   the SDK reads its own environment. Every configured endpoint must be `https` (`http` only when
   `CVPILOT_ENV=dev`; `file:`, `sqlite:`, other schemes and bare paths are refused in every
   mode), carry no credentials and, when `CVPILOT_FORWARD_HOSTS` is set, be on exactly one of the
   listed hosts (whole-host equality, no suffix or substring matching; entries are `host` or
   `host:port`, an entry without a port allows only 443 (80 for http), a non-default port needs its
   own `host:port` entry even without an allow-list, wildcards and empty entries are refused at
   start; an empty variable means no restriction; an unset endpoint means the SDK default, which must itself
   be listed, and MLflow's local default has no host so it is refused). The SDK switches that
   disable TLS verification and the certificate-authority overrides
   (`MLFLOW_TRACKING_SERVER_CERT_PATH`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `SSL_CERT_FILE`,
   `SSL_CERT_DIR`) are refused outside development. The few variables that decide where an
   adapter connects are read **once, on the training thread, at construction**, checked, and
   passed to the SDK explicitly (`MlflowClient(tracking_uri=...)`, `Settings(base_url=...)`), so
   a later change of the environment cannot swap a checked value. URLs are validated structurally
   before the host is compared, because the allow-list and the HTTP client must read the same
   host: backslashes, percent-encoding, whitespace, control characters, `@` and `?` or `#`
   anywhere, non-ASCII hosts (punycode must be written as `xn--`, and is compared as such), IPv6
   literals, empty labels and userinfo in any form are refused; the host is lower-cased and one
   trailing dot removed, and the **rebuilt, normalised URL** (not the original text) is what the
   SDK receives. W&B always receives an explicit `base_url` (the validated one, or
   `https://api.wandb.ai` after allow-list approval of that default) so an unset variable cannot
   leave the endpoint to a settings file in the working directory, `~/.config/wandb/settings` or
   netrc, which a checkout can control; explicit settings are the SDK's highest-precedence source.
   I did not find an SDK option that disables reading those files and do not claim one: pinning
   the endpoint is the control. MLflow's `MLFLOW_REGISTRY_URI` gets the same checks and the client
   receives explicit tracking and registry URIs (the registry follows the tracking URI when
   unset). Tests record what the fake SDKs receive and spy on `os.environ` to show that
   construction reads only the endpoint and TLS variables, never a credential, and that `open`,
   `log_points` and `close` read nothing. Messages name the variable and
   never the value. Nothing from `Repro` (config, command line, dataset path) is sent.
   **Trackio:** the adapter passes no endpoint and no Space, so it makes no request of its own;
   Trackio's optional Space sync is configured outside this adapter, which cannot see or verify
   it. The adapter therefore applies the CA-override check and refuses to run when
   `CVPILOT_FORWARD_HOSTS` is set (an allow-list cannot be honoured for a destination the
   adapter cannot see).
6. **One copy per job.** Only rank zero starts forwarders: when `RANK` or `LOCAL_RANK` is set to
   a non-zero integer `init` logs `forward_skipped_rank` and starts none. A forked child (a
   dataloader worker, a spawned rank) inherits the `Fanout` object but not its threads;
   `os.register_at_fork` marks it closed in the child, drops the inherited run handle (its SQLite
   connection is never used or closed in the child) and forgets the active run, so the child
   neither forwards the parent's points again nor waits for threads that do not exist.
7. **Extras and licences.** SDKs are optional extras imported inside `open()`; core never
   imports them (layering test). MLflow uses `mlflow-skinny` (Apache-2.0).
8. **Scope.** Scalars only. Artifact forwarding is deferred because predictions can reveal the
   dataset; it will be opt-in.

## Consequences

- A tracker outage costs forwarded points (counted in `forward_summary`), never training time;
  the local store always has every point.
- A tracker that is merely slow loses points once its 256-batch queue fills; raise the limits
  only with a measured reason.
- Each tracker has its own step rule (W&B needs non-decreasing steps); adapters group by step and
  count what they cannot send.
- Real-library behaviour is covered by smoke tests behind the `real_libs` marker (off by
  default); the default suite uses injected fake modules and needs neither the SDKs nor a network.

## Links

- [../architecture/02-components.md](../architecture/02-components.md),
  [../architecture/05-runtime.md](../architecture/05-runtime.md),
  [../architecture/06-cross-cutting.md](../architecture/06-cross-cutting.md)
- [0004-layering-and-dependency-rule.md](0004-layering-and-dependency-rule.md),
  [0005-run-store.md](0005-run-store.md)
