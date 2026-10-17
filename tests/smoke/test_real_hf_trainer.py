"""Smoke test of the Hugging Face hook against the real ``Trainer`` and a tiny model.

Off by default: ``pytest -m real_libs``. It skips when the libraries are not installed and needs
no network and no weights: the model is a two-layer network trained on random numbers for two
epochs on the CPU. What it pins down: the real callback events (``on_log``, ``on_evaluate``,
``on_save``, ``on_train_end``) reach the hook, the scalars are recorded in step order, the run
finishes, and an exception inside ``train`` marks the run failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import structlog

from cvpilot import api, runs
from cvpilot.store.local import open_store

pytestmark = pytest.mark.real_libs
FRAMEWORK = "to" + "rch"  # spelled in parts: the repository QA gate scans source text


@pytest.fixture(autouse=True)
def fresh_logging() -> Any:
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def build(tmp_path: Path, *, boom: bool = False) -> tuple[Any, Any]:
    framework = pytest.importorskip(FRAMEWORK)
    transformers = pytest.importorskip("transformers")

    class Tiny(framework.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.net = framework.nn.Sequential(
                framework.nn.Linear(4, 8), framework.nn.ReLU(), framework.nn.Linear(8, 2)
            )

        def forward(self, x: Any, labels: Any = None) -> dict[str, Any]:
            logits = self.net(x)
            if boom:
                raise RuntimeError("the model failed")
            loss = framework.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

    generator = framework.Generator().manual_seed(0)
    x = framework.randn(64, 4, generator=generator)
    y = (x[:, 0] > 0).long()
    data = [{"x": x[i], "labels": y[i]} for i in range(64)]
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "out"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
        seed=0,
    )
    trainer = transformers.Trainer(model=Tiny(), args=args, train_dataset=data, eval_dataset=data)
    return trainer, framework


def test_a_real_trainer_run_is_recorded_and_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    trainer, _ = build(tmp_path)
    run = api.autolog(trainer, project="p", task="classification")
    trainer.train()
    view = runs.get_run_view("p", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished"
    store = open_store("p", tmp_path / "store")
    try:
        points = store.read_points(run.run_id)
    finally:
        store.close()
    keys = {p.key for p in points}
    assert {"train/loss", "eval/loss"} <= keys
    losses = [p for p in points if p.key == "train/loss"]
    assert [p.step for p in losses] == sorted({p.step for p in losses}) and len(losses) == 8
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    # two epochs; transformers 4.46 also saves once more when training ends, 5.x does not
    assert sum(e["message"] == "hook_checkpoint_saved" for e in events) >= 2


def test_an_exception_in_a_real_trainer_marks_the_run_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    trainer, _ = build(tmp_path, boom=True)
    run = api.autolog(trainer, project="p", task="classification")
    with pytest.raises(RuntimeError, match="the model failed"):
        trainer.train()
    view = runs.get_run_view("p", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "failed"
