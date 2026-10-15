"""Smoke tests against the real tracker SDKs. Off by default: ``pytest -m real_libs``.

They skip when the SDK is not installed and need no network (MLflow file store, W&B offline).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_forwarders import use_forwarders  # noqa: F401  (keeps the fixture module importable)
from support import SHA, batch, meta_repro

pytestmark = pytest.mark.real_libs


def test_mlflow_file_store_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    from cvpilot.forward.mlflow import MlflowForwarder

    meta, repro = meta_repro()
    adapter = MlflowForwarder()
    adapter.open(meta, repro)
    adapter.log_points(batch(meta, 3))
    adapter.close("finished")
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(meta.project)
    runs = client.search_runs([exp.experiment_id])
    assert runs and runs[0].data.metrics["k1"] == pytest.approx(3.1)
    assert SHA


def test_wandb_offline_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    from cvpilot.forward.wandb import WandbForwarder

    meta, repro = meta_repro()
    adapter = WandbForwarder()
    adapter.open(meta, repro)
    adapter.log_points(batch(meta, 1))
    adapter.close("finished")
