"""Trackio forwarder (``cvpilot[trackio]``): scalars grouped by step, ascending.

Trackio keeps its data in a local SQLite file by default and has no endpoint or credential of
its own here. This adapter sends the project, run name and the metrics; never the config or
the command line.
"""

from __future__ import annotations

from collections.abc import Sequence

from cvpilot.core.schema import MetricPoint, Repro, RunMeta, RunStatus
from cvpilot.forward._common import group_by_step, load_sdk


class TrackioForwarder:
    name = "trackio"
    requires = ("trackio",)

    def __init__(self, *, allow_insecure: bool = False) -> None:
        self._sdk: object = None

    def open(self, meta: RunMeta, repro: Repro) -> None:
        sdk = load_sdk("trackio", "trackio")
        sdk.init(project=meta.project, name=meta.name, config={"cvpilot_run_id": meta.run_id})
        self._sdk = sdk

    def log_points(self, points: Sequence[MetricPoint]) -> None:
        for step, values in group_by_step(points):
            self._sdk.log(values, step=step)  # type: ignore[attr-defined]

    def close(self, status: RunStatus) -> None:
        self._sdk.finish()  # type: ignore[attr-defined]
