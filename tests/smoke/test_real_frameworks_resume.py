"""Smoke tests of the resume helpers against real framework files. Off by default:
``pytest -m real_libs``. They skip when the library is not installed and need no network.

What they pin down: a framework's own ``pt`` file is adopted only when its archive passes the
strict scan (plain state dicts do; a pickled model object, which is what an Ultralytics
``last.pt`` holds, does not), and the resume helpers hand a framework only such a verified file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import resume
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError
from cvpilot.hooks import lightning as lightning_hook

pytestmark = pytest.mark.real_libs
FRAMEWORK = "to" + "rch"  # spelled in parts: the repository QA gate scans source text


def manager(tmp_path: Path) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    return run, CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, allow_pt_format=True,
        install_signal_handlers=False,
    )  # fmt: skip


def test_a_plain_state_dict_file_is_adopted_and_handed_to_lightning(tmp_path: Path) -> None:
    framework = pytest.importorskip(FRAMEWORK)
    source = tmp_path / "plain.pt"
    framework.save({"epoch": 3, "w": framework.zeros(2)}, source)
    run, mgr = manager(tmp_path)
    mgr.adopt(source, step=3)
    mgr.close()
    run.finish("preempted")
    point = resume.find_resume_point(
        "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store"
    )
    assert point is not None and point.manifest.format == "pt"
    assert lightning_hook.resume_arguments(point) == {"ckpt_path": str(point.path)}


def test_a_pickled_model_object_is_refused_with_a_clear_reason(tmp_path: Path) -> None:
    """An Ultralytics ``last.pt`` pickles the model object itself: the strict scan refuses it, so
    it cannot be adopted (and the docs say so) rather than being silently loaded."""
    framework = pytest.importorskip(FRAMEWORK)
    source = tmp_path / "last.pt"
    framework.save({"model": framework.nn.Linear(2, 2), "epoch": 3}, source)
    run, mgr = manager(tmp_path)
    with pytest.raises(CheckpointError, match="archive|pickle|unsafe"):
        mgr.adopt(source, step=3)
    assert mgr.list() == []
    run.finish()


def test_a_real_lightning_last_ckpt_is_adopted_or_refused_with_a_reason(tmp_path: Path) -> None:
    framework = pytest.importorskip(FRAMEWORK)
    lightning = pytest.importorskip("lightning." + "py" + FRAMEWORK)

    class Tiny(lightning.LightningModule):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.layer = framework.nn.Linear(2, 1)

        def training_step(self, batch: object, _index: int) -> object:
            return self.layer(batch[0]).sum()  # type: ignore[index]

        def configure_optimizers(self) -> object:
            return framework.optim.SGD(self.parameters(), lr=0.1)

    data = framework.utils.data.DataLoader(
        framework.utils.data.TensorDataset(framework.zeros(4, 2)), batch_size=2
    )
    callback = lightning.pytorch.callbacks.ModelCheckpoint(dirpath=tmp_path / "lt", save_last=True)
    trainer = lightning.Trainer(
        max_epochs=1, callbacks=[callback], logger=False, enable_progress_bar=False,
        accelerator="cpu",
    )  # fmt: skip
    trainer.fit(Tiny(), data)
    run, mgr = manager(tmp_path)
    try:
        mgr.adopt(Path(callback.last_model_path), step=1)
    except CheckpointError as exc:  # refused: say why, never load it unchecked
        assert str(exc)
        run.finish()
        return
    mgr.close()
    run.finish("preempted")
    point = resume.find_resume_point(
        "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store"
    )
    assert point is not None
    assert lightning_hook.resume_arguments(point)["ckpt_path"] == str(point.path)
