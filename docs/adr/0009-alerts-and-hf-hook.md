# ADR-0009: Client-side alerts and the Hugging Face Trainer hook

**Status:** accepted
**Date:** 2026-10-17

## Context

A training job that has gone wrong (a NaN loss, a loss that stopped moving, a model that regressed
against the one in production, a crashed or preempted run) costs hours and GPU money for as long as
nobody notices. People want a message when that happens. The message has to be sent from inside the
training process, because the server that would send it is not built (and is out of scope for this
release), and because a person on call needs it even when no server exists.

An alert path is also a new outbound network path next to a training loop and next to credentials.
It can leak a webhook URL (a Slack-style URL is a credential), be pointed at an internal service
(server-side request forgery), be flooded by a loop that fails on every step, put attacker-chosen
text (a metric name, a class name) into a chat message, or block or crash the job it is watching.

The most common training stack is the Hugging Face `Trainer`, which has callbacks but no failure
event, and which resumes from a directory bundle.

## Options considered

### Option A: a new delivery path written for alerts

A small thread and queue in `alerts/`. Trade-offs: simple to write; but it is a second copy of the
bounded queue, daemon worker, stall detection and counters that the tracker forwarders already
have and that have been reviewed three times.

### Option B: reuse the forwarder fan-out with a small description of what differs

`Fanout` takes a `Delivery` (the adapter method, the log events, the unit in the log fields).
Channels are adapters with `open`, `send` and `close`, resolved through their own registry
(`cvpilot.channels`) and checked with the same URL and host rules. Trade-offs: the fan-out grows by
one dataclass and three call sites; in return there is one implementation of the isolation rules.

### Option C: only an HTTP callback in the config (any URL)

Trade-offs: the least code; but a URL in a file is exactly where credentials and SSRF end up. Rejected.

## Decision

Option B, with these rules (each has a test):

1. **Rules fail safe.** `nan_loss`, `loss_spike`, `no_progress`, `eval_regression`, `run_failed`,
   `run_preempted`. A rule that raises is reported (`alert_rule_failed` once, an alert saying the rule
   could not be evaluated) and cannot suppress another rule. A baseline comparison that cannot be made
   (no pinned snapshot, a different evaluation set, a failed integrity check) is an alert with a
   constant reason, never silence. `eval_regression` is `run_gate` itself, so the verdict equals the
   one `cvpilot gate` gives.
2. **A tiny configuration.** One JSON document (JSON, not TOML, because Python 3.10 has no TOML reader
   and a new core dependency was not worth one file): `channels`, `rate_limit_s`, and at most one
   instance of each of six rules with a handful of numbers. Bounded read from a descriptor without
   following links, duplicate keys, `NaN`, unknown keys, repeated rules and out-of-range numbers are
   refused, errors name fields and never values. No code, no templating.
3. **Constant messages, plain values.** A message is a constant template; a value from data (a metric
   key, a class name) is cut and reduced to `[A-Za-z0-9 _./:()+%=-]`, anything else becomes `?`. The
   Slack channel also escapes `&`, `<`, `>` and turns `mrkdwn` off; Telegram sends no `parse_mode`.
4. **No storms.** One alert per rule and run per `rate_limit_s` (default 300 s), at most 20 alerts
   per run (the end-of-run alerts are exempt), at most 16 watched keys per rule, at most 1000 points
   looked at per call. Suppressions are counted and the first one per rule is logged.
5. **Channels.** `webhook`, `slack` (a Slack-style incoming webhook) and `telegram`. The address and
   credential come from environment variables only (`CVPILOT_ALERT_WEBHOOK_URL`, `..._WEBHOOK_TOKEN`,
   `CVPILOT_ALERT_SLACK_URL`, `CVPILOT_ALERT_TELEGRAM_TOKEN`, `CVPILOT_ALERT_TELEGRAM_CHAT`), read
   once at construction, checked with `check_endpoint` (https only, no userinfo, query or odd
   characters, port 443 unless listed, `CVPILOT_FORWARD_HOSTS` allow-list), a host that is not an IP
   number in any notation or `localhost`, and certificate-authority overrides refused. The secret
   lives in one private attribute: not in `repr`, not in an error, not in a log event, not picklable.
6. **One outbound call.** `notify/_http.py`: https only; the name is resolved once and every address
   must be public (an IPv4 address that is globally routable and not multicast, or an IPv6 address
   inside `2000::/3`; mapped, 6to4 and NAT64 forms are judged by the IPv4 address they carry; one bad
   address refuses the name; resolution runs in a helper thread inside the attempt deadline, so a
   resolver that never answers cannot stall the worker) and the connection goes to the checked
   address, so a rebinding name cannot reach an internal service; no redirect is followed; only the
   status line and headers are read (a huge answer costs nothing); a total deadline of 5 s per
   attempt enforced by closing the socket (a slowly dripped answer cannot outlast it, and a cut-off
   answer is a timeout, not a success); at most two attempts (connection errors, 429 and 5xx).
   Messages never contain the URL. Plain http is allowed only with `CVPILOT_ENV=dev`, which
   replaces the address policy by its opposite (every resolved address must be loopback) and logs one
   `insecure_dev_mode` warning per channel.
7. **Never blocks, never raises.** The engine runs on the calling thread (cheap, bounded, wrapped);
   delivery runs on the channels' daemon workers behind a queue of 16 alerts; a call longer than 20 s
   marks the channel stalled, which only drops the alerts that arrive while that call lasts (the
   channel recovers when it returns, so a later alert such as `run_failed` is not lost); a channel
   that fails to construct stops `init` (fail closed); failures are counted and logged once by exception type
   (`alert_delivery_failed`), with `alert_delivery_summary` at the end. `finish` waits up to 5 s for the
   forwarders and then up to 12 s for the channels (one full delivery budget, so the end-of-run
   alert is not abandoned in the middle of its retry); worst case about 17 s, only with dead
   endpoints. The alert for a failed run is evaluated before the final flush, so a store that cannot
   be written does not silence it, and an uncaught exception that ends the process sends it from the
   exit handler. Only rank zero alerts. Every suppression is counted and the first of each rule and
   reason is logged; `finish` writes one `alert_summary` event (queued, suppressed, delivered,
   delivery_failed, dropped).
8. **Dry run.** `cvpilot alerts check CONFIG [--project P --run RUN]` validates the file, builds the
   channels from the environment without sending, and replays a recorded run (stored scalars, the real
   baseline comparison for every snapshot, the final status) with a clock taken from the recording,
   printing what would fire. It exits 2 for a bad configuration or channel, never for an alert.
9. **The Hugging Face hook.** `hooks/hf_trainer.py`, extra `hf` (`transformers[torch]>=4.46`; the real
   smoke test passes on 4.46.0 and 5.20.1). A `TrainerCallback` built at attach time records `on_log`
   scalars (`train/...`), `on_evaluate` metrics (`eval/...`), `on_save` as an event with the step only,
   and finishes the run at `on_train_end`; `train` is wrapped so an exception marks the run failed and
   propagates unchanged. Same guard, defensive conversion and rank rule as the Lightning hook.

## What was deliberately not built

- **A server, sync, a UI** and server-side alert rules: out of scope for this release.
- **An email/SMTP channel**: it needs credentials in a URL or a second secret scheme and a mail
  protocol client; webhook, Slack-style and Telegram cover the chat tools asked for.
- **Per-user or per-rule routing**, escalation and acknowledgement: every rule goes to every channel.
- **Alert history storage**: alerts are logged events, not stored rows; nothing can be tampered with.
- **A templating language** or user-written rules (no code in the configuration).
- **An alert for `sys.exit`, a kill or a lost machine**: nothing runs to send it.
- **A hung-process alert** (no metric logged for N minutes): it cannot be seen from inside a hung
  process and a watchdog thread would be a new mechanism; the plateau rule is the in-process signal.
- **GPU utilisation, `new_best` and `resumed` rules** of the first sketch.
- **A resume helper for the Hugging Face hook.** The trainer resumes from a directory of files that
  includes pickled optimizer and random state; the checkpoint verification covers single files (size,
  SHA-256, format) and extending it to a bundle would be a new verification mechanism with its own
  attack surface. Restore a `CheckpointManager` checkpoint into the model and pass the weights.
- **MMEngine and RF-DETR hooks.**

## Consequences

- A dead, slow or hostile channel costs alerts (counted), never training time beyond the bounded wait at
  `finish`. The notebook `03_alerts_and_hf_trainer.ipynb` measures the step time with and without
  dead channels on a real `Trainer` run.
- The loopback listener of tests and notebooks needs the development setting; production refuses http.
- Slack and Telegram delivery was checked against a local listener and by unit tests; no real account
  was used.
- The Hugging Face `Trainer` replaces NaN training losses by the average of the earlier losses since the last log (0.0 when it logs every step) before callbacks see them (`logging_nan_inf_filter`), so `nan_loss` sees them only with that setting off; the evaluation loss is not filtered. Found while running notebook 03 and documented in the guide.
- The dry run has no recording time for evaluation snapshots: each is replayed one rate-limit window after the previous event, in step order, and the command says so.
- Metrics logged after `on_train_end` (a manual `trainer.evaluate()` after `train()`) are not recorded.

## Links

- [0008-forwarders.md](0008-forwarders.md), [0007-compare-and-gate-semantics.md](0007-compare-and-gate-semantics.md)
- [../guide/alerts.md](../guide/alerts.md), [../architecture/06-cross-cutting.md](../architecture/06-cross-cutting.md)
