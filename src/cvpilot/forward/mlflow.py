"""MLflow forwarder (``cvpilot[mlflow]``): one MLflow run per CV-Pilot run.

Endpoint and credentials come from MLflow's own environment variables; this adapter reads only
``MLFLOW_TRACKING_URI`` and ``MLFLOW_TRACKING_INSECURE_TLS`` to refuse unsafe settings. It
sends the run id, project, task and parent run id as tags and the metrics; never the config,
the command line, the dataset path or an artifact. MLflow limits metric names to
alphanumerics and ``_ - . space /`` (250 characters), so other characters in a key become ``_``;
steps and float64 values are passed unchanged.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

from cvpilot.core.schema import MetricPoint, Repro, RunMeta, RunStatus
from cvpilot.forward._common import check_endpoint, load_sdk, refuse_insecure_flag

MAX_BATCH_METRICS = 1000
"""MLflow accepts at most 1000 metrics per ``log_batch`` call."""
MAX_KEY_CHARS = 250
_BAD_KEY = re.compile(r"[^A-Za-z0-9_\-. /]")
_STATUS = {"finished": "FINISHED", "failed": "FAILED"}


def _key(key: str) -> str:
    return _BAD_KEY.sub("_", key)[:MAX_KEY_CHARS]


class MlflowForwarder:
    name = "mlflow"
    requires = ("mlflow|mlflow-skinny",)

    def __init__(self, *, allow_insecure: bool = False) -> None:
        check_endpoint(
            os.environ.get("MLFLOW_TRACKING_URI"),
            allow_insecure=allow_insecure,
            what="MLFLOW_TRACKING_URI",
        )
        refuse_insecure_flag("MLFLOW_TRACKING_INSECURE_TLS", allow_insecure=allow_insecure)
        self._client: Any = None
        self._run_id = ""
        self._metric: Any = None

    def open(self, meta: RunMeta, repro: Repro) -> None:
        sdk = load_sdk("mlflow", "mlflow")
        client = sdk.MlflowClient()
        experiment = client.get_experiment_by_name(meta.project)
        experiment_id = (
            experiment.experiment_id if experiment else client.create_experiment(meta.project)
        )
        tags = {
            "cvpilot.run_id": meta.run_id,
            "cvpilot.project": meta.project,
            "cvpilot.task": meta.task,
        }
        if meta.parent_run_id:
            tags["cvpilot.parent_run_id"] = meta.parent_run_id
        run = client.create_run(experiment_id, run_name=meta.name, tags=tags)
        self._client, self._run_id, self._metric = client, run.info.run_id, sdk.entities.Metric

    def log_points(self, points: Sequence[MetricPoint]) -> None:
        for lo in range(0, len(points), MAX_BATCH_METRICS):
            chunk = points[lo : lo + MAX_BATCH_METRICS]
            metrics = [
                self._metric(_key(p.key), p.value, int(p.wall_time * 1000), p.step) for p in chunk
            ]
            self._client.log_batch(self._run_id, metrics=metrics)

    def close(self, status: RunStatus) -> None:
        self._client.set_terminated(self._run_id, _STATUS.get(status, "KILLED"))
