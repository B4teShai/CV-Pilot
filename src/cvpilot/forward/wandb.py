"""Weights & Biases forwarder (``cvpilot[wandb]``).

The API key is read by the SDK from its own environment; this adapter reads only
``WANDB_BASE_URL`` and ``WANDB_INSECURE_DISABLE_SSL`` to refuse unsafe settings. It sends the
project, run name, a run id derived from the CV-Pilot run id, and the metrics; never the config
or the command line. W&B requires non-decreasing steps per run: points are sent grouped by step
in ascending order and points older than the last forwarded step are counted in
``dropped_points``. A W&B run that already existed (the user's own ``wandb.init``) is logged to
but not finished by this adapter.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from cvpilot.core.schema import MetricPoint, Repro, RunMeta, RunStatus
from cvpilot.forward._common import (
    check_endpoint,
    group_by_step,
    load_sdk,
    refuse_insecure_flag,
)

INIT_TIMEOUT_S = 30
"""Seconds the SDK may take to start its run before ``init`` fails (and the failure is counted)."""


class WandbForwarder:
    name = "wandb"
    requires = ("wandb",)

    def __init__(self, *, allow_insecure: bool = False) -> None:
        check_endpoint(
            os.environ.get("WANDB_BASE_URL"), allow_insecure=allow_insecure, what="WANDB_BASE_URL"
        )
        refuse_insecure_flag("WANDB_INSECURE_DISABLE_SSL", allow_insecure=allow_insecure)
        self._run: Any = None
        self._owns = False
        self._last_step = -1
        self.dropped_points = 0

    def open(self, meta: RunMeta, repro: Repro) -> None:
        sdk = load_sdk("wandb", "wandb")
        self._owns = getattr(sdk, "run", None) is None
        self._run = sdk.init(
            project=meta.project,
            name=meta.name,
            id=meta.run_id.lower(),
            config={"cvpilot_run_id": meta.run_id, "task": meta.task},
            settings=sdk.Settings(init_timeout=INIT_TIMEOUT_S),
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
