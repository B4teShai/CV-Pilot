"""Smoke test of the RF-DETR hook against the real framework and a tiny model.

Off by default: ``pytest -m real_libs``. It skips when the framework is not installed and needs
no network and no weights: an RF-DETR Nano built without pretrained weights (``pretrain_weights=
None``, so nothing is downloaded; the Hugging Face Hub and the framework's weight cache are
pointed at nothing) trains one epoch on eight generated 128-pixel images on the CPU. What it pins
down: the hook really reaches the trainer that ``train`` builds, the framework's training losses and
validation metrics are recorded, the run finishes when ``train`` returns, the framework's
``build_trainer`` is put back afterwards, and an exception inside ``train`` marks the run failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog

from cvpilot import api, runs
from cvpilot.store.local import open_store

pytestmark = pytest.mark.real_libs


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("RF_HOME", str(tmp_path / "no-weights"))
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def dataset(root: Path) -> Path:
    image_module = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(0)
    for split, count in (("train", 8), ("valid", 4), ("test", 4)):
        folder = root / split
        folder.mkdir(parents=True)
        images, annotations = [], []
        for i in range(count):
            pixels = np.full((128, 128, 3), 30, dtype=np.uint8)
            x, y = int(rng.integers(10, 60)), int(rng.integers(10, 60))
            pixels[y : y + 40, x : x + 40] = (220, 40, 40)
            image_module.fromarray(pixels).save(folder / f"{i}.jpg")
            images.append({"id": i, "file_name": f"{i}.jpg", "width": 128, "height": 128})
            annotations.append(
                {"id": i, "image_id": i, "category_id": 1, "bbox": [x, y, 40, 40],
                 "area": 1600, "iscrowd": 0}
            )  # fmt: skip
        categories = [
            {"id": 0, "name": "shapes", "supercategory": "none"},
            {"id": 1, "name": "square", "supercategory": "shapes"},
        ]
        document = {"images": images, "annotations": annotations, "categories": categories}
        (folder / "_annotations.coco.json").write_text(json.dumps(document))
    return root


def test_a_real_tiny_training_is_recorded_and_finished(tmp_path: Path) -> None:
    rfdetr = pytest.importorskip("rfdetr")
    training = pytest.importorskip("rfdetr.training")
    factory = training.build_trainer
    model = rfdetr.RFDETRNano(pretrain_weights=None)
    run = api.autolog(model, project="rf", task="detection")
    model.train(
        dataset_dir=str(dataset(tmp_path / "data")), epochs=1, batch_size=2, resolution=128,
        device="cpu", output_dir=str(tmp_path / "out"), num_workers=0, progress_bar=None,
    )  # fmt: skip
    assert training.build_trainer is factory
    view = runs.get_run_view("rf", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.meta.framework.startswith("rfdetr==")
    store = open_store("rf", tmp_path / "store")
    keys = {p.key for p in store.read_points(run.run_id)}
    store.close()
    assert {"train/loss", "val/mAP_50_95", "val/mAP_50"} <= keys


def test_a_failing_training_marks_the_run_failed(tmp_path: Path) -> None:
    rfdetr = pytest.importorskip("rfdetr")
    pytest.importorskip("rfdetr.training")
    model = rfdetr.RFDETRNano(pretrain_weights=None)
    run = api.autolog(model, project="rf", task="detection")
    with pytest.raises(Exception):  # noqa: B017, PT011  # whatever the framework raises
        model.train(
            dataset_dir=str(tmp_path / "missing"), epochs=1, batch_size=2, resolution=128,
            device="cpu", output_dir=str(tmp_path / "out"), num_workers=0, progress_bar=None,
        )  # fmt: skip
    view = runs.get_run_view("rf", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "failed"
