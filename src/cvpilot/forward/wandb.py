"""Weights & Biases forwarder (``cvpilot[wandb]``).

The API key is read by the SDK from its own environment. This adapter reads, once at
construction, ``WANDB_BASE_URL`` (https only, no credentials, on an allowed host when
``CVPILOT_FORWARD_HOSTS`` is set; unset means ``api.wandb.ai``), refuses the TLS switches
(``WANDB_INSECURE_DISABLE_SSL`` and the certificate-authority overrides) outside development, and
passes the checked URL to the SDK explicitly. It sends the
project, run name, a run id derived from the CV-Pilot run id, and the metrics; never the config
or the command line. W&B requires non-decreasing steps per run: points are sent grouped by step
in ascending order and points older than the last forwarded step are counted in
``dropped_points``. A W&B run that already existed (the user's own ``wandb.init``) is logged to
but not finished by this adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from cvpilot.core.schema import MetricPoint, Repro, RunMeta, RunStatus
from cvpilot.forward._common import (
    CA_OVERRIDE_VARIABLES,
    check_endpoint,
    freeze_env,
    group_by_step,
    load_sdk,
    refuse_ca_overrides,
    refuse_insecure_flag,
)

DEFAULT_HOST = "api.wandb.ai"
DEFAULT_URL = f"https://{DEFAULT_HOST}"
"""The host the SDK talks to when ``WANDB_BASE_URL`` is unset."""
INIT_TIMEOUT_S = 30
"""Seconds the SDK may take to start its run before ``init`` fails (and the failure is counted)."""


class WandbForwarder:
    name = "wandb"
    requires = ("wandb",)

    def __init__(self, *, allow_insecure: bool = False, allowed_hosts: Sequence[str] = ()) -> None:
        env = freeze_env(("WANDB_BASE_URL", "WANDB_INSECURE_DISABLE_SSL", *CA_OVERRIDE_VARIABLES))
        checked = check_endpoint(
            env["WANDB_BASE_URL"],
            allow_insecure=allow_insecure,
            what="WANDB_BASE_URL",
            allowed_hosts=allowed_hosts,
            default_host=DEFAULT_HOST,
        )
        refuse_insecure_flag(env, "WANDB_INSECURE_DISABLE_SSL", allow_insecure=allow_insecure)
        refuse_ca_overrides(env, allow_insecure=allow_insecure)
        # always pinned: an unset variable must not leave the endpoint to settings files or
        # netrc, which a checkout can control
        self._base_url = checked or DEFAULT_URL
        self._run: Any = None
        self._owns = False
        self._last_step = -1
        self.dropped_points = 0

    def open(self, meta: RunMeta, repro: Repro) -> None:
        sdk = load_sdk("wandb", "wandb")
        self._owns = getattr(sdk, "run", None) is None
        settings: dict[str, Any] = {"init_timeout": INIT_TIMEOUT_S, "base_url": self._base_url}
        self._run = sdk.init(
            project=meta.project,
            name=meta.name,
            id=meta.run_id.lower(),
            config={"cvpilot_run_id": meta.run_id, "task": meta.task},
            settings=sdk.Settings(**settings),
        )

    def log_points(self, points: Sequence[MetricPoint]) -> None:
        for step, values in group_by_step(points):
            if step < self._last_step:
                self.dropped_points += len(values)
                continue
            self._run.log(values, step=step)
            self._last_step = step

    def close(self, status: RunStatus) -> None:
        if self._owns:
            self._run.finish(exit_code=0 if status == "finished" else 1)
