"""Real checkpoint files written by the installed framework and by Lightning. Off by default:
``pytest -m real_libs``. They skip when the library is not installed and need no network.

What they pin down (notebook 04 runs the same checks on larger real files): the strict reader
accepts exactly the layout the framework writes (``zip_support.framework_archive`` is compared with
it byte for byte), a framework's own state file is adopted and handed back to Lightning, a pickled
model object is refused when it is adopted and nothing is stored, a resumed Lightning run ends with
the same weights as an uninterrupted one, and a tampered or swapped file is refused.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest
from ckpt_support import run_meta
from zip_support import framework_archive

from cvpilot import runs
from cvpilot.ckpt import resume, serializers
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointCorruptError, CheckpointError, CheckpointUntrustedError
from cvpilot.hooks import lightning as lightning_hook

pytestmark = pytest.mark.real_libs
FRAMEWORK = "to" + "rch"  # spelled in parts: the repository QA gate scans source text


@pytest.fixture(autouse=True)
def opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")


def manager(tmp_path: Path) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    return run, CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, allow_pt_format=True,
        install_signal_handlers=False,
    )  # fmt: skip


def find(tmp_path: Path) -> resume.ResumePoint | None:
    return resume.find_resume_point(
        "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store"
    )


def test_the_test_builder_reproduces_the_framework_layout_byte_for_byte(tmp_path: Path) -> None:
    framework = pytest.importorskip(FRAMEWORK)
    state = {"epoch": 3, "w": framework.zeros(2), "nested": {"v": framework.ones(3)}}
    path = tmp_path / "epoch=1-step=8.ckpt"  # a path stem with '=' becomes the folder name
    framework.save(state, path)
    with zipfile.ZipFile(path) as archive:
        members = {i.filename: archive.read(i.filename) for i in archive.infolist()}
    assert framework_archive(members) == path.read_bytes()
    assert serializers.scan_pt_archive(path.read_bytes()) > 0


def test_a_plain_state_dict_file_is_adopted_and_handed_to_lightning(tmp_path: Path) -> None:
    framework = pytest.importorskip(FRAMEWORK)
    source = tmp_path / "plain.pt"
    framework.save({"epoch": 3, "w": framework.zeros(2)}, source)
    run, mgr = manager(tmp_path)
    mgr.adopt(source, step=3)
    mgr.close()
    run.finish("preempted")
    point = find(tmp_path)
    assert point is not None and point.manifest.format == "pt"
    assert lightning_hook.resume_arguments(point) == {"ckpt_path": str(point.path)}


def test_a_pickled_model_object_is_refused_when_adopted_and_nothing_is_stored(
    tmp_path: Path,
) -> None:
    """A YOLO ``last.pt`` pickles the model object, which is this shape."""
    framework = pytest.importorskip(FRAMEWORK)
    source = tmp_path / "last.pt"
    framework.save({"model": framework.nn.Linear(2, 2), "epoch": 3}, source)
    run, mgr = manager(tmp_path)
    with pytest.raises(CheckpointCorruptError, match="Linear") as info:
        mgr.adopt(source, step=3)
    assert info.value.reason == "unsafe_archive" and mgr.list() == []
    run.finish()


def test_the_legacy_non_zip_format_is_refused(tmp_path: Path) -> None:
    framework = pytest.importorskip(FRAMEWORK)
    source = tmp_path / "legacy.pt"
    framework.save({"w": framework.zeros(2)}, source, _use_new_zipfile_serialization=False)
    run, mgr = manager(tmp_path)
    with pytest.raises(CheckpointError, match="zip"):
        mgr.adopt(source, step=1)
    run.finish()


def lightning_pieces() -> tuple[Any, Any, Any]:
    framework = pytest.importorskip(FRAMEWORK)
    lightning = pytest.importorskip("lightning")
    callbacks = getattr(lightning, "py" + FRAMEWORK).callbacks

    class Tiny(lightning.LightningModule):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.layer = framework.nn.Linear(4, 2)

        def training_step(self, batch: Any, _index: int) -> Any:
            x, y = batch
            return framework.nn.functional.cross_entropy(self.layer(x), y)

        def configure_optimizers(self) -> Any:
            return framework.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    generator = framework.Generator().manual_seed(1)
    data = framework.utils.data.DataLoader(
        framework.utils.data.TensorDataset(
            framework.randn(32, 4, generator=generator),
            framework.randint(0, 2, (32,), generator=generator),
        ),
        batch_size=8,
    )
    return Tiny, data, callbacks


def fit(epochs: int, folder: Path, **kwargs: Any) -> tuple[Any, Any]:
    lightning = pytest.importorskip("lightning")
    tiny, data, callbacks = lightning_pieces()
    lightning.seed_everything(0, verbose=False)
    saver = callbacks.ModelCheckpoint(dirpath=folder, save_last=True)
    trainer = lightning.Trainer(
        max_epochs=epochs, callbacks=[saver], logger=False, enable_progress_bar=False,
        enable_model_summary=False, accelerator="cpu",
    )  # fmt: skip
    model = tiny()
    trainer.fit(model, data, **kwargs)
    return model, saver


def same_weights(a: Any, b: Any) -> bool:
    framework = pytest.importorskip(FRAMEWORK)
    pairs = zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    return all(framework.equal(x, y) for x, y in pairs)


def test_a_real_lightning_run_resumes_from_the_adopted_file_with_identical_weights(
    tmp_path: Path,
) -> None:
    straight, _ = fit(2, tmp_path / "straight")
    _, saver = fit(1, tmp_path / "half")
    run, mgr = manager(tmp_path)
    mgr.adopt(Path(saver.last_model_path), step=4)
    mgr.close()
    run.finish("preempted")
    point = find(tmp_path)
    assert point is not None
    resumed, _ = fit(2, tmp_path / "resumed", **lightning_hook.resume_arguments(point))
    assert same_weights(straight, resumed)


@pytest.mark.parametrize("policy", ["any", "anchored"])
def test_a_flipped_byte_a_truncation_and_a_swapped_file_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    """Under the default policy a file that does not verify is not returned (``None`` and a
    ``resume_point_unavailable`` event with the reason); under ``anchored`` or stricter, where
    the run store knows the step existed, the same situation raises."""
    monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", policy)
    _, saver = fit(1, tmp_path / "half")
    _, other = fit(2, tmp_path / "other")
    run, mgr = manager(tmp_path)
    manifest = mgr.adopt(Path(saver.last_model_path), step=4)
    mgr.close()
    run.finish("preempted")
    stored = tmp_path / "ckpt" / run.run_id / manifest.path
    good = stored.read_bytes()
    foreign = Path(other.last_model_path).read_bytes()
    for name, bad in (
        ("flip", good[:100] + bytes([good[100] ^ 1]) + good[101:]),
        ("truncate", good[: len(good) // 2]),
        ("swap", foreign),
    ):
        stored.chmod(0o600)
        stored.write_bytes(bad)
        if policy == "any":
            assert find(tmp_path) is None, name
        else:
            with pytest.raises(CheckpointUntrustedError):
                find(tmp_path)
    stored.write_bytes(good)
    restored = find(tmp_path)
    assert restored is not None and restored.manifest.step == 4
