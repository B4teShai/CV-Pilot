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
2. **Bounded.** `QUEUE_BATCHES` (256) batches per forwarder (a batch is at most 10 000 points);
   `CALL_TIMEOUT_S` (30 s): a call running longer disables the forwarder the next time a batch is
   offered. A hung call occupies only its own worker; the SDK's own timeouts end it eventually.
3. **Failures are counted and logged once.** The first failure per forwarder is `forward_failed`
   (WARN) with the forwarder, run id, reason and the exception *type*; the exception text is never
   logged because it can contain a URL or a credential. `forward_summary` at close carries the
   counters. A failed call is retried with the next batch; a failed `open` or a stall disables
   the forwarder for the run.
4. **Configuration errors are loud and early.** An unknown forwarder name, a missing extra
   (checked by distribution name before any import, `a|b` for alternatives) and an unsafe
   endpoint raise from `init()` before a run exists. A plugin that fails with any other
   exception while being constructed is skipped and logged.
5. **Secrets.** Adapters take no token argument and read no credential variable: the SDK reads
   its own environment. They refuse credentials inside a URL, plain `http` (unless
   `CVPILOT_ENV=dev`) and the SDK switches that disable TLS verification, with messages that
   name the variable and never the value. Nothing from `Repro` (config, command line, dataset
   path) is sent.
6. **Extras and licences.** SDKs are optional extras imported inside `open()`; core never
   imports them (layering test). MLflow uses `mlflow-skinny` (Apache-2.0).
7. **Scope.** Scalars only. Artifact forwarding is deferred because predictions can reveal the
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
