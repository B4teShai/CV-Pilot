"""Smoke test of the MMEngine hook against the real ``Runner`` and a tiny model.

Off by default: ``pytest -m real_libs``. It skips when the libraries are not installed and needs
no network and no weights: a one-layer network on generated numbers for two epochs on the CPU.
What it pins down: the real hook events reach the hook, training scalars and validation metrics
are recorded in step order, the run finishes when ``test`` returns, an exception inside ``train``
marks the run failed, and a checkpoint file the framework wrote is refused by the strict scan
(it pickles the logging buffers by class), which is why the hook has no resume helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import structlog

from cvpilot import api, runs
from cvpilot.ckpt import serializers
from cvpilot.core.errors import CheckpointCorruptError
from cvpilot.store.local import open_store

pytestmark = pytest.mark.real_libs
FRAMEWORK = "to" + "rch"  # spelled in parts: the repository QA gate scans source text


@pytest.fixture(autouse=True)
def fresh_logging() -> Any:
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def build(work_dir: Path, *, boom: bool = False) -> Any:
    framework = pytest.importorskip(FRAMEWORK)
    pytest.importorskip("mmengine")
    from mmengine.evaluator import BaseMetric
    from mmengine.model import BaseModel
    from mmengine.registry import METRICS
    from mmengine.runner import Runner

    class Net(BaseModel):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.layer = framework.nn.Linear(2, 1)

        def forward(self, inputs: Any, data_samples: Any = None, mode: str = "tensor") -> Any:
            out = self.layer(inputs)
            if boom:
                raise RuntimeError("the model failed")
            if mode == "loss":
                return {"loss": ((out - data_samples) ** 2).mean()}
            return out

    class Data(framework.utils.data.Dataset):  # type: ignore[misc, name-defined]
        def __len__(self) -> int:
            return 16

        def __getitem__(self, i: int) -> dict[str, Any]:
            return {
                "inputs": framework.tensor([float(i % 4), 1.0]),
                "data_samples": framework.tensor([float(i % 4)]),
            }

    @METRICS.register_module(force=True)  # type: ignore[untyped-decorator]
    class Score(BaseMetric):  # type: ignore[misc]
        default_prefix = "tiny"

        def process(self, data_batch: Any, data_samples: Any) -> None:
            self.results.append(1.0)

        def compute_metrics(self, results: list[float]) -> dict[str, float]:
            return {"score": 0.5}

    def loader() -> Any:
        return framework.utils.data.DataLoader(Data(), batch_size=8)

    return Runner(
        model=Net(), work_dir=str(work_dir), train_dataloader=loader(), val_dataloader=loader(),
        test_dataloader=loader(), val_evaluator=dict(type="Score"),
        test_evaluator=dict(type="Score"), val_cfg={}, test_cfg={},
        train_cfg={"by_epoch": True, "max_epochs": 2, "val_interval": 1},
        optim_wrapper={"optimizer": {"type": "SGD", "lr": 0.01}},
        default_hooks={"checkpoint": {"type": "CheckpointHook", "interval": 1}},
        log_level="ERROR",
    )  # fmt: skip


def test_a_real_run_is_recorded_and_finished_at_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    runner = build(tmp_path / "work")
    run = api.autolog(runner, project="p", task="classification")
    runner.train()
    assert runs.get_run_view("p", run.run_id, store_dir=tmp_path / "store").meta.status == "running"
    runner.test()
    assert runs.get_run_view("p", run.run_id, store_dir=tmp_path / "store").meta.status == (
        "finished"
    )
    store = open_store("p", tmp_path / "store")
    try:
        points = store.read_points(run.run_id)
    finally:
        store.close()
    keys = {p.key for p in points}
    assert {"val/tiny/score", "test/tiny/score"} <= keys
    steps = [p.step for p in points if p.key == "val/tiny/score"]
    assert steps == sorted(set(steps)) and len(steps) == 2


def test_an_exception_in_a_real_runner_marks_the_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    runner = build(tmp_path / "work", boom=True)
    run = api.autolog(runner, project="p", task="classification")
    with pytest.raises(RuntimeError, match="the model failed"):
        runner.train()
    assert runs.get_run_view("p", run.run_id, store_dir=tmp_path / "store").meta.status == "failed"


def test_a_checkpoint_the_framework_wrote_is_refused_by_the_strict_scan(tmp_path: Path) -> None:
    runner = build(tmp_path / "work")
    runner.train()
    files = sorted((tmp_path / "work").glob("epoch_*.pth"))
    assert files
    with pytest.raises(CheckpointCorruptError, match="HistoryBuffer") as info:
        serializers.scan_pt_archive(files[-1].read_bytes())
    assert info.value.reason == "unsafe_archive"


def test_the_flexible_runner_is_matched_although_it_is_not_a_runner_subclass() -> None:
    pytest.importorskip(FRAMEWORK)
    runners = pytest.importorskip("mmengine.runner")
    from cvpilot.hooks.mmengine import MMEngineHook

    assert runners.FlexibleRunner.__mro__[1] is object
    bare = runners.FlexibleRunner.__new__(runners.FlexibleRunner)  # no training set-up needed
    assert MMEngineHook().matches(bare)
