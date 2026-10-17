# Alerts

CV-Pilot can send you a message while a job is still running: the loss became NaN, it stopped
improving, the new model is worse than the one you trust, the run failed or was preempted. The
rules run inside your training process. No server is needed, and a broken channel never slows or
stops training.

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

`rate_limit_s` (default 300) is the shortest time between two alerts of the same rule in one run. Save it as `alerts.json` and start your run with `cvpilot.init(..., alerts="alerts.json")` (a
dictionary with the same content works too), or set `CVPILOT_ALERTS=alerts.json`. It also works with
`cvpilot.autolog(trainer, ..., alerts="alerts.json")`.

| Rule | Fires when |
|---|---|
| `nan_loss` | a loss is NaN or infinite |
| `loss_spike` | a loss is more than `factor` times the mean of the previous `window` values |
| `no_progress` | `patience` logged values pass without a new lowest loss (improvements under `min_delta` do not count) |
| `eval_regression` | `log_eval` produced a snapshot that fails the gate against `baseline` by more than `max_drop` points, or cannot be compared with it |
| `run_failed`, `run_preempted` | the run ends that way |

A loss is any metric whose name contains `loss`, or exactly the `metric` named in the rule. Each rule
appears at most once. The configuration is checked when the run starts: an unknown key, a repeated
rule or an out-of-range number stops `init` before a run exists, and the message names the field,
never the value. A rule that cannot be evaluated is reported as an alert, not skipped.

## Where the messages go

Channels are named in the configuration; **where** they send, and the credential, come from
environment variables only, never from the file.

| Channel | Variables | Notes |
|---|---|---|
| `webhook` | `CVPILOT_ALERT_WEBHOOK_URL`, optional `CVPILOT_ALERT_WEBHOOK_TOKEN` | JSON body with `rule`, `project`, `run_id`, `step`, `text`; the token is sent as a bearer header |
| `slack` | `CVPILOT_ALERT_SLACK_URL` | a Slack-style incoming webhook; the URL is a secret |
| `telegram` | `CVPILOT_ALERT_TELEGRAM_TOKEN`, `CVPILOT_ALERT_TELEGRAM_CHAT` | plain text through the Bot API |

Every address must be `https`, on port 443 (or listed), without credentials in the URL, and must
name a public DNS host: an address written as an IP number, `localhost` and anything that resolves to
a private or loopback address is refused. `CVPILOT_FORWARD_HOSTS` restricts the hosts, as for the
trackers. Redirects are never followed. A message is a fixed sentence with the run id, the step and
the metric or class name cut down to plain characters. Without any channel an alert is only logged
(the event `alert_fired`).

## Never in the way of training

Alerts are checked on the calling thread in microseconds and delivered by background threads. A
closed port, an endpoint that never answers and a redirect each cost only the alerts (counted, and
logged once as `alert_delivery_failed`). At most one alert per rule and run goes out per
`rate_limit_s`, and at most 20 per run. When a run ends, `finish` waits at most five seconds for the
channels. Only rank zero of a distributed job alerts.

## Try a configuration without sending anything

```bash notest
cvpilot alerts check alerts.json                       # valid? can the channels be built?
cvpilot alerts check alerts.json --project demo --run RUN_ID   # what would have fired on this run
```

The second form replays the stored scalars and snapshots of a recorded run, with the real baseline
comparison, and prints `would fire: ...` lines. It sends nothing. Exit code 2 means a bad
configuration or a channel that cannot be built.

## Development and tests

Production refuses plain http. To receive alerts on your own machine set `CVPILOT_ENV=dev`, which
allows `http://127.0.0.1:PORT/...` endpoints. The notebook `notebooks/03_alerts_and_hf_trainer.ipynb`
does exactly that with a local listener and shows the messages it received.

## Hugging Face Trainer

```python notest
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val)
cvpilot.autolog(trainer, project="crops", task="classification", alerts="alerts.json")
trainer.train()
```

`pip install "cvpilot[hf]"`. **One setting matters for `nan_loss`:** by default the `Trainer` replaces a NaN training loss with an earlier value before any callback sees it (`logging_nan_inf_filter=True`), so the alert then comes only from the evaluation loss at the end of the epoch. Set `logging_nan_inf_filter=False` in `TrainingArguments` to have the training loss watched at every step. The hook records the training log under `train/`, the evaluation
metrics under `eval/`, a save as an event with the step, and the end of training. An exception in
`train()` marks the run failed. There is no resume helper for this framework: it resumes from a
directory that includes pickled state, which CV-Pilot's single-file verification does not cover. See
[ADR-0009](../adr/0009-alerts-and-hf-hook.md) for what is deliberately not built.
