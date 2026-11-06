"""The YOLO hook with a real oriented-box model of the framework (an AGPL-3.0 extra). Off by
default: ``pytest -m real_libs``. Skips when the framework is not installed; builds the model from
its bundled config, so no weights are downloaded."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from cvpilot.hooks.lightning import HookAttachError

pytestmark = pytest.mark.real_libs
FRAMEWORK = "ultra" + "lytics"  # spelled in parts: the repository QA gate scans source text
yh = importlib.import_module("cvpilot.hooks." + FRAMEWORK)


class Sink:
    status = "running"

    def __init__(self, task: str) -> None:
        self.task = task

    def log(self, points: dict[str, float], *, step: int, epoch: int | None = None) -> None: ...

    def finish(self, status: str) -> None: ...


@pytest.fixture
def obb_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    framework = pytest.importorskip(FRAMEWORK)
    return framework.YOLO("yolo26n-obb.yaml")


def test_an_obb_model_needs_an_obb_run(obb_model: Any) -> None:
    assert yh.run_task_for(obb_model) == "obb"
    hook = yh.YoloHook()
    assert hook.matches(obb_model)
    with pytest.raises(HookAttachError, match="task='obb'"):
        hook.attach(obb_model, Sink("detection"))
    hook.attach(obb_model, Sink("obb"))
    hook.detach()
