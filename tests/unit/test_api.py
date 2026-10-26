from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import structlog

import cvpilot
from cvpilot import api, runs
from cvpilot.core.errors import (
    InvalidResumeError,
    NoActiveRunError,
    RunAlreadyActiveError,
    RunNotFoundError,
)
from cvpilot.core.hashing import new_run_id
from cvpilot.core.redact import REDACTED
from cvpilot.core.ulid import is_valid_ulid
from cvpilot.logging import clear_request_context
from cvpilot.settings import SettingsError


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_LOG_LEVEL", "CVPILOT_STORE_COMMAND_LINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", "--epochs", "2"])
    structlog.reset_defaults()
    yield
    if api._active is not None:  # noqa: SLF001
        api._active.abandon()  # noqa: SLF001
        api._active = None  # noqa: SLF001
    clear_request_context()
    structlog.reset_defaults()


def init(tmp_path: Path, **kw: Any) -> api.Run:
    kw.setdefault("task", "detection")
    return api.init("proj", store_dir=tmp_path / "store", **kw)


def stored(tmp_path: Path, run_id: str) -> runs.RunView:
    return runs.get_run_view("proj", run_id, store_dir=tmp_path / "store")


def test_init_log_finish_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = init(tmp_path, config={"lr": 0.1}, seed=3)
    assert is_valid_ulid(run.run_id)
    api.log({"train/loss": 1.0, "val/mAP": 0.5}, step=0)
    api.log({"train/loss": 0.5}, step=1, epoch=0)
    api.finish()
    view = stored(tmp_path, run.run_id)
    assert view.meta.status == "finished" and view.meta.task == "detection"
    assert view.repro.config == {"lr": 0.1} and view.repro.seed == 3
    assert view.repro.command_line == "train.py --epochs 2"
    assert view.stats.n_points == 3
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    names = [line["message"] for line in lines]
    assert names == ["run_started", "run_finished"]  # metrics_flushed is DEBUG, dev only
    for line in lines:
        assert {"@timestamp", "level", "message", "service", "env", "request_id", "logger"} <= set(
            line
        )
        assert line["service"] == "cvpilot-client" and line["run_id"] == run.run_id


def test_run_handle_methods_and_context_manager(tmp_path: Path) -> None:
    with init(tmp_path) as run:
        run.log({"a": 1.0}, step=0)
    assert stored(tmp_path, run.run_id).meta.status == "finished"
    assert api._active is None  # noqa: SLF001

    with pytest.raises(RuntimeError, match="boom"):
        with init(tmp_path) as run2:
            run2.log({"a": 1.0}, step=0)
            raise RuntimeError("boom")
    assert stored(tmp_path, run2.run_id).meta.status == "failed"
    assert stored(tmp_path, run2.run_id).stats.n_points == 1


def test_explicit_finish_inside_with_is_not_repeated(tmp_path: Path) -> None:
    with init(tmp_path) as run:
        run.finish("preempted")
    assert stored(tmp_path, run.run_id).meta.status == "preempted"


def test_module_level_misuse(tmp_path: Path) -> None:
    with pytest.raises(NoActiveRunError):
        api.log({"a": 1.0}, step=0)
    with pytest.raises(NoActiveRunError):
        api.finish()
    init(tmp_path)
    with pytest.raises(RunAlreadyActiveError):
        init(tmp_path)
    api.finish()
    with pytest.raises(NoActiveRunError):
        api.finish()
    init(tmp_path)  # a new run is fine after finish
    api.finish("failed")


def test_every_init_gets_a_new_run_id(tmp_path: Path) -> None:
    ids = []
    for _ in range(3):
        ids.append(init(tmp_path).run_id)
        api.finish()
    assert len(set(ids)) == 3 and ids == sorted(ids)


def test_log_artifact(tmp_path: Path) -> None:
    run = init(tmp_path)
    f = tmp_path / "cfg.json"
    f.write_text("{}")
    art = api.log_artifact(f, kind="config", meta={"n": 1})
    api.finish()
    assert [a.sha256 for a in stored(tmp_path, run.run_id).artifacts] == [art.sha256]


def test_invalid_environment_setting_raises_from_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_ENV", "banana")
    with pytest.raises(SettingsError, match="CVPILOT_ENV"):
        init(tmp_path)
    assert not (tmp_path / "store").exists()


def test_dataset_hash_and_missing_dataset(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.jpg").write_bytes(b"1")
    run = init(tmp_path, dataset_uri=str(ds))
    api.finish()
    assert stored(tmp_path, run.run_id).repro.dataset_hash.startswith("merkle:")
    with pytest.raises(FileNotFoundError):
        init(tmp_path, dataset_uri=str(tmp_path / "missing"))
    assert api._active is None  # noqa: SLF001
    assert len(runs.list_runs("proj", store_dir=tmp_path / "store")) == 1


def test_config_secrets_are_redacted_before_storage(tmp_path: Path) -> None:
    run = init(tmp_path, config={"api_key": "k", "lr": 1})
    api.finish()
    assert stored(tmp_path, run.run_id).repro.config == {"api_key": REDACTED, "lr": 1}


def test_command_line_setting_default_stores_the_redacted_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--token=abc123", "--lr", "0.1"])
    run = init(tmp_path)
    api.finish()
    line = stored(tmp_path, run.run_id).repro.command_line
    assert "abc123" not in line and "--lr 0.1" in line


@pytest.mark.parametrize("value", ["false", "0", "no"])
def test_command_line_setting_false_stores_no_argument_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CVPILOT_STORE_COMMAND_LINE", value)
    monkeypatch.setattr(sys, "argv", ["/home/bob/train.py", "--token=abc123", "--data", "/priv/x"])
    run = init(tmp_path)
    api.finish()
    view = stored(tmp_path, run.run_id)
    assert view.repro.command_line == "train.py [3 arguments omitted]"
    dumped = view.repro.model_dump_json() + view.meta.model_dump_json()
    for text in ("abc123", "/priv/x", "--data", "/home/bob"):
        assert text not in dumped
    raw = (view.store_path / "runs.sqlite").read_bytes()
    assert b"abc123" not in raw and b"/priv/x" not in raw


def test_resume_from_failed_run(tmp_path: Path) -> None:
    first = init(tmp_path)
    api.finish("failed")
    child = init(tmp_path, resume_from=first.run_id)
    api.finish()
    view = stored(tmp_path, child.run_id)
    assert (view.meta.status, view.meta.parent_run_id) == ("finished", first.run_id)


def test_resume_from_errors(tmp_path: Path) -> None:
    running = init(tmp_path)
    api.finish()  # finished runs cannot be resumed
    with pytest.raises(InvalidResumeError):
        init(tmp_path, resume_from=running.run_id)
    with pytest.raises(RunNotFoundError):
        init(tmp_path, resume_from=new_run_id())
    failed = init(tmp_path)
    api.finish("failed")
    with pytest.raises(ValueError, match="task"):
        init(tmp_path, task="classification", resume_from=failed.run_id)
    assert api._active is None  # noqa: SLF001


def test_exit_hook_flushes_but_keeps_the_run_running(tmp_path: Path) -> None:
    run = init(tmp_path)
    api.log({"a": 1.0}, step=0)
    api._flush_active_at_exit()  # noqa: SLF001
    view = stored(tmp_path, run.run_id)
    assert view.meta.status == "running" and view.stats.n_points == 1


def test_metrics_errors_reach_the_caller(tmp_path: Path) -> None:
    init(tmp_path)
    api.log({"a": 1.0}, step=1)
    with pytest.raises(Exception, match="duplicate"):
        api.log({"a": 1.0}, step=1)
    api.finish()


def test_public_names_are_lazily_exported() -> None:
    assert cvpilot.init is api.init and cvpilot.log is api.log and cvpilot.finish is api.finish
    code = "import sys, cvpilot; assert 'cvpilot.api' not in sys.modules; cvpilot.init;"
    code += "assert 'cvpilot.api' in sys.modules and 'pyarrow' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
    with pytest.raises(AttributeError):
        cvpilot.nope  # noqa: B018


def test_project_and_store_dir_fall_back_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_PROJECT", "from-env")
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "envstore"))
    run = api.init(task="detection")
    api.finish()
    assert runs.get_run_view("from-env", run.run_id, store_dir=tmp_path / "envstore")


def test_explicit_arguments_beat_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_PROJECT", "from-env")
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "envstore"))
    run = init(tmp_path)  # project="proj", store_dir=tmp_path/"store"
    api.finish()
    assert stored(tmp_path, run.run_id).meta.project == "proj"
    assert not (tmp_path / "envstore").exists()


def test_missing_project_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CVPILOT_PROJECT"):
        api.init(task="detection", store_dir=tmp_path)


# A real manifest written by `datacopilot ingest <tiny COCO set> --format coco -o x.dcp`
# (DataCopilot 0.1.0, commit 335ffb3), copied byte for byte: not written by hand.
DATACOPILOT_MANIFEST = Path(__file__).parents[1] / "data" / "datacopilot_0_1_0_manifest.json"


def test_a_dataset_prepared_with_datacopilot_is_recorded_by_its_version(tmp_path: Path) -> None:
    dcp = tmp_path / "dataset.dcp"
    dcp.mkdir()
    raw = DATACOPILOT_MANIFEST.read_bytes()
    (dcp / "manifest.json").write_bytes(raw)
    (dcp / "images.parquet").write_bytes(b"not read: the version comes from the manifest")
    run = init(tmp_path, dataset_uri=str(dcp))
    api.finish()
    repro = stored(tmp_path, run.run_id).repro
    assert repro.dataset_hash == "dcp:" + json.loads(raw)["version"]
    assert repro.dataset_hash == "dcp:3b87778c1af0e31665bdabf06112f3eb"
    assert repro.dataset_uri == str(dcp)
