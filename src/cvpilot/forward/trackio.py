"""Trackio forwarder (``cvpilot[trackio]``): scalars grouped by step, ascending.

Trackio keeps its data in a local SQLite file by default; the adapter passes no endpoint and no
Space, so it makes no network request itself. It still refuses the certificate-authority
overrides outside development, and it refuses to run when ``CVPILOT_FORWARD_HOSTS`` is set,
because an allow-list can only be honoured for a destination the adapter can see. It sends the
project, run name and the metrics; never the config or the command line.
"""

from __future__ import annotations

from collections.abc import Sequence

from cvpilot.core.schema import MetricPoint, Repro, RunMeta, RunStatus
from cvpilot.forward._common import (
    CA_OVERRIDE_VARIABLES,
    ForwardConfigError,
    freeze_env,
    group_by_step,
    load_sdk,
    refuse_ca_overrides,
)


class TrackioForwarder:
    name = "trackio"
    requires = ("trackio",)

    def __init__(self, *, allow_insecure: bool = False, allowed_hosts: Sequence[str] = ()) -> None:
        if allowed_hosts:
            # The adapter passes no endpoint and the SDK's own optional Space sync is configured
            # outside this adapter, so the destination cannot be verified: fail closed.
            raise ForwardConfigError(
                "CVPILOT_FORWARD_HOSTS is set but the trackio forwarder cannot verify where "
                "Trackio sends data; remove trackio from forward_to or unset the allow-list"
            )
        refuse_ca_overrides(freeze_env(CA_OVERRIDE_VARIABLES), allow_insecure=allow_insecure)
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
