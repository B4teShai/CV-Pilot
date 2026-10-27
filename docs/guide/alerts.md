# Alerts

CV-Pilot can message you while a job is still running: the loss became NaN, it stopped improving,
the new model is worse than the one you trust, or the run failed or was preempted. The rules run
inside your training process. No server is needed, and a broken channel never slows or stops
training.

## A configuration in ten lines

```json
{
  "channels": ["slack"],
  "rules": [
    {"rule": "nan_loss"},
    {"rule": "loss_spike", "factor": 3, "window": 20},
    {"rule": "no_progress", "patience": 50, "min_delta": 0.001},
    {"rule": "eval_regression", "baseline": "tag:prod", "max_drop": {"mAP50_95": 0.5}},
    {"rule": "run_failed"}, {"rule": "run_preempted"}
  ]
}
```

`rate_limit_s` (default 300) is the shortest time between two alerts of the same rule in one run.
Save the file as `alerts.json` and start your run with `cvpilot.init(..., alerts="alerts.json")`
(a dictionary works too), or set `CVPILOT_ALERTS=alerts.json`. `cvpilot.autolog(trainer, ...,
alerts="alerts.json")` takes it as well.

| Rule | Fires when |
|---|---|
| `nan_loss` | A loss is NaN or infinite. |
| `loss_spike` | A loss is more than `factor` times the mean of the previous `window` values. |
| `no_progress` | `patience` logged values pass without a new lowest loss (gains under `min_delta` do not count). |
| `eval_regression` | A `log_eval` snapshot fails the gate against `baseline` by more than `max_drop` points, or cannot be compared. |
| `run_failed`, `run_preempted` | The run ends that way. |

A loss is any metric whose name contains `loss`, or the `metric` named in the rule. Each rule
appears at most once. The configuration is checked when the run starts: an unknown key, a repeated
rule or an out-of-range number stops `init` before a run exists, and the message names the field,
never the value.

## Where the messages go

You name channels in the configuration. The address and the credential come from environment
variables only, never from the file.

| Channel | Variables | Notes |
|---|---|---|
| `webhook` | `CVPILOT_ALERT_WEBHOOK_URL`, optional `CVPILOT_ALERT_WEBHOOK_TOKEN` | JSON body with `rule`, `project`, `run_id`, `step`, `text`; the token is a bearer header. |
| `slack` | `CVPILOT_ALERT_SLACK_URL` | A Slack-style incoming webhook. The URL is a secret. |
| `telegram` | `CVPILOT_ALERT_TELEGRAM_TOKEN`, `CVPILOT_ALERT_TELEGRAM_CHAT` | Plain text through the Bot API. |

Every address must be `https` on port 443, without credentials in the URL, and must name a public
DNS host. An IP address, `localhost` or anything that resolves to a private address is refused, and
redirects are never followed. `CVPILOT_FORWARD_HOSTS` limits the hosts. A message is a fixed sentence
with the run id, the step and a cleaned-up metric or class name. With no channel an alert is only
logged (`alert_fired`).

## Never in the way of training

Metric rules run on the calling thread in microseconds. `eval_regression` runs one verified baseline
comparison per `log_eval` snapshot. Delivery runs on background threads, so a closed port, a silent
endpoint or a redirect costs only the alerts, which are counted and logged once
(`alert_delivery_failed`). Each rule sends at most one alert per `rate_limit_s` and 20 per run, and
only rank zero of a distributed job alerts (`init` checks `RANK` and `LOCAL_RANK`). At the end of a run `finish` waits up to 5 seconds for
tracker forwarders and up to 12 for alert channels, so the worst case with dead endpoints is about
17 seconds. `finish` logs one `alert_summary` with the counts.

`run_failed` is sent when the run ends as failed: `finish("failed")`, an exception inside a `with`
block or a hook, or an uncaught exception that ends the process. A process that ends with
`sys.exit`, is killed or loses its machine sends nothing, because no code runs.

## Try a configuration without sending anything

```bash notest
cvpilot alerts check alerts.json                       # valid? can the channels be built?
cvpilot alerts check alerts.json --project demo --run RUN_ID   # what would have fired on this run
```

The second form replays a recorded run's scalars and snapshots against the real baseline
comparison and prints `would fire: ...` lines. It sends nothing. Exit code `2` means a bad
configuration or a channel that cannot be built.

To receive alerts on your own machine set `CVPILOT_ENV=dev`. It allows `http://` endpoints that
resolve to loopback addresses, and nothing else.

## Hugging Face Trainer

```python notest
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val)
cvpilot.autolog(trainer, project="crops", task="classification", alerts="alerts.json")
trainer.train()
```

Install `cvpilot[hf]`. **One setting matters for `nan_loss`.** By default the `Trainer` replaces a NaN
training loss with an average of earlier losses before any callback sees it
(`logging_nan_inf_filter=True`), so the alert then fires only from the evaluation loss. Set
`logging_nan_inf_filter=False` in `TrainingArguments` to watch the training loss at every step.

The hook records the training log under `train/`, evaluation metrics under `eval/`, each save as an
event, and the end of training. An exception in `train()` marks the run failed. `KeyboardInterrupt`
and `SystemExit` mark it preempted. There is no resume helper: the `Trainer` resumes from a folder
that includes pickled state, which CV-Pilot's single-file verification does not cover
([ADR-0009](../adr/0009-alerts-and-hf-hook.md)).
