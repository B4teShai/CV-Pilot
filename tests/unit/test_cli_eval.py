"""``cvpilot eval`` (offline files into a run) and the snapshot listing of ``runs show``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import structlog

from cvpilot import api, runs
from cvpilot.cli import app as cli
from cvpilot.logging import clear_request_context

A, B = (10.0, 10.0, 50.0, 50.0), (60.0, 60.0, 120.0, 140.0)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in (
        "CVPILOT_ENV",
        "CVPILOT_LOG_LEVEL",
        "CVPILOT_PROJECT",
        "CVPILOT_STORE_COMMAND_LINE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


def main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["cvpilot", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def events(err: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in err.splitlines() if x.startswith("{")]


def human(err: str) -> str:
    """The lines meant for people: everything that is not a JSON log record."""
    return "\n".join(line for line in err.splitlines() if not line.startswith("{"))


def new_run(task: str = "detection", status: str = "finished") -> str:
    run = api.init("proj", task=task, name="r")  # type: ignore[arg-type]
    run.finish(status)  # type: ignore[arg-type]
    return run.run_id


def write_gts(path: Path, **overrides: Any) -> Path:
    cols: dict[str, Any] = {
        "image_id": ["i1", "i1", "i2", "i3"],
        "cls": pa.array([0, 1, 0, None], type=pa.int64()),
        "x1": pa.array([A[0], B[0], A[0], None]),
        "y1": pa.array([A[1], B[1], A[1], None]),
        "x2": pa.array([A[2], B[2], A[2], None]),
        "y2": pa.array([A[3], B[3], A[3], None]),
        "iscrowd": [False, False, False, False],
        "width": [200, 200, 200, 200],
        "height": [200, 200, 200, 200],
        "camera": ["c1", "c1", "c1", "c2"],
    }
    cols.update(overrides)
    pq.write_table(pa.table(cols), path)
    return path


def write_preds(path: Path, **overrides: Any) -> Path:
    cols: dict[str, Any] = {
        "image_id": ["i1", "i1", "i2", "i3"],
        "cls": [0, 1, 0, 1],
        "score": [0.9, 0.8, 0.7, 0.6],
        "x1": [A[0], B[0], A[0], B[0]],
        "y1": [A[1], B[1], A[1], B[1]],
        "x2": [A[2], B[2], A[2], B[2]],
        "y2": [A[3], B[3], A[3], B[3]],
    }
    cols.update(overrides)
    pq.write_table(pa.table(cols), path)
    return path


def classes(path: Path, names: list[str] | None = None, *, as_json: bool = False) -> Path:
    names = names or ["cat", "dog"]
    path.write_text(json.dumps(names) if as_json else "\n".join(names) + "\n")
    return path


def eval_args(tmp_path: Path, run_id: str, *extra: str) -> list[str]:
    return [
        "eval",
        run_id,
        "--project",
        "proj",
        "--preds",
        str(tmp_path / "p.parquet"),
        "--gts",
        str(tmp_path / "g.parquet"),
        "--classes",
        str(tmp_path / "c.txt"),
        "--allow-closed",
        *extra,
    ]


@pytest.fixture
def files(tmp_path: Path) -> Path:
    write_preds(tmp_path / "p.parquet")
    write_gts(tmp_path / "g.parquet")
    classes(tmp_path / "c.txt")
    return tmp_path


def test_eval_stores_a_snapshot_in_a_finished_run_and_prints_the_headline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    capsys.readouterr()
    code = main(monkeypatch, *eval_args(files, rid, "--step", "5", "--resamples", "50"))
    out, err = capsys.readouterr()
    assert code == 0, err
    assert "mAP50_95: 1.0000" in out and "step: 5" in out and "n_images: 3" in out
    done = [e for e in events(err) if e["message"] == "eval_completed"]
    assert len(done) == 1 and done[0]["run_id"] == rid and done[0]["step"] == 5
    view = runs.get_run_view("proj", rid, store_dir=files / "store")
    assert len(view.snapshots) == 1 and view.snapshots[0].n_images == 3  # i3 has no objects
    assert view.snapshots[0].bootstrap is not None


def test_eval_json_prints_the_snapshot_and_eval_file_is_an_alias(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    capsys.readouterr()
    args = eval_args(files, rid, "--json", "--resamples", "0")
    args[0] = "eval-file"
    assert main(monkeypatch, *args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == rid and data["metrics"]["mAP50"] == pytest.approx(1.0)
    assert data["ci"] == {} and data["class_names"] == ["cat", "dog"]


def test_long_option_names_and_json_class_files_work(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    classes(files / "c.json", ["cat", "dog"], as_json=True)
    argv = [
        "eval", rid, "--project", "proj",
        "--predictions", str(files / "p.parquet"),
        "--ground-truth", str(files / "g.parquet"),
        "--classes", str(files / "c.json"), "--resamples", "0", "--allow-closed",
    ]  # fmt: skip
    assert main(monkeypatch, *argv) == 0


def test_slices_are_read_from_ground_truth_columns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    capsys.readouterr()
    assert (
        main(monkeypatch, *eval_args(files, rid, "--slices", "camera=camera", "--resamples", "0"))
        == 0
    )
    out = capsys.readouterr().out
    assert "camera=c1" in out and "camera=c2" in out
    snap = runs.get_run_view("proj", rid, store_dir=files / "store").snapshots[0]
    assert snap.slices["camera=c1"]["n_images"] == 2.0


def test_classification_eval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rid = new_run("classification")
    pq.write_table(pa.table({"image_id": ["a", "b"], "cls": [0, 1]}), tmp_path / "g.parquet")
    pq.write_table(
        pa.table({"image_id": ["a", "b"], "cls": [0, 0], "score": [0.9, 0.8]}),
        tmp_path / "p.parquet",
    )
    classes(tmp_path / "c.txt")
    capsys.readouterr()
    assert main(monkeypatch, *eval_args(tmp_path, rid, "--resamples", "0")) == 0
    assert "top1: 0.5000" in capsys.readouterr().out


def test_runs_show_lists_snapshots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    assert main(monkeypatch, *eval_args(files, rid, "--resamples", "0")) == 0
    capsys.readouterr()
    assert main(monkeypatch, "runs", "show", rid, "--project", "proj") == 0
    text = capsys.readouterr().out
    assert "snapshot: " in text and "step=0" in text and "mAP50_95=1.0000" in text
    assert "artifact: predictions" in text and "artifact: confusion" in text
    assert main(monkeypatch, "runs", "show", rid, "--project", "proj", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["snapshots"]) == 1 and data["snapshots"][0]["step"] == 0


def test_the_same_files_give_the_same_snapshot_in_two_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    out = []
    for _ in range(2):
        rid = new_run()
        capsys.readouterr()
        assert (
            main(monkeypatch, *eval_args(files, rid, "--json", "--seed", "3", "--resamples", "30"))
            == 0
        )
        data = json.loads(capsys.readouterr().out)
        data.pop("run_id")
        out.append(data)
    assert out[0] == out[1]


# -- failures: exit 1 with one line and one event, nothing stored --------------------------


def failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> tuple[int, str, list[dict[str, Any]]]:
    capsys.readouterr()
    code = main(monkeypatch, *argv)
    out, err = capsys.readouterr()
    assert out == ""
    return code, err, events(err)


def test_second_eval_of_the_same_set_and_step_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    argv = eval_args(files, rid, "--resamples", "0")
    assert main(monkeypatch, *argv) == 0
    code, err, ev = failing(monkeypatch, capsys, argv)
    assert code == 1 and "immutable" in err and "Traceback" not in human(err)
    assert len(human(err).splitlines()) == 1
    assert ev[-1]["message"] == "command_failed"
    assert main(monkeypatch, *eval_args(files, rid, "--step", "1", "--resamples", "0")) == 0


@pytest.mark.parametrize(
    ("which", "overrides", "needle"),
    [
        ("p", {"score": [float("nan"), 0.8, 0.7, 0.6]}, "non-finite scores"),
        ("p", {"cls": [0, 7, 0, 1]}, "class id 7"),
        ("p", {"image_id": ["i1", "i1", "i2", "ghost"]}, "not in the evaluation set"),
        ("g", {"x2": pa.array([5.0, B[2], A[2], None])}, "x2 < x1"),
        ("g", {"width": [200, 150, 200, 200]}, "conflicting"),
        ("g", {"width": [200, 200, 200, 200], "height": [10, 10, 10, 10]}, "outside the image"),
    ],
)
def test_invalid_data_exits_1_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    files: Path,
    which: str,
    overrides: dict[str, Any],
    needle: str,
) -> None:
    rid = new_run()
    (write_preds if which == "p" else write_gts)(files / f"{which}.parquet", **overrides)
    code, err, ev = failing(monkeypatch, capsys, eval_args(files, rid, "--resamples", "0"))
    assert code == 1 and needle in err and "Traceback" not in human(err)
    assert ev and ev[-1]["message"] == "command_failed"
    view = runs.get_run_view("proj", rid, store_dir=files / "store")
    assert view.snapshots == []


def test_malformed_and_hostile_parquet_files_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    (files / "p.parquet").write_bytes(b"PAR1" + b"\x00" * 100)
    code, err, _ = failing(monkeypatch, capsys, eval_args(files, rid, "--resamples", "0"))
    assert code == 1 and "not a readable Parquet" in err
    write_preds(files / "p.parquet")
    pq.write_table(pa.table({"image_id": ["a"]}), files / "g.parquet")  # no cls column
    code, err, _ = failing(monkeypatch, capsys, eval_args(files, rid, "--resamples", "0"))
    assert code == 1 and "missing column 'cls'" in err


@pytest.mark.parametrize(
    "content", ["", "cat\ncat\n", "[1, 2]", "x" * 200, "\n".join(f"c{i}" for i in range(2049))]
)
def test_bad_class_files_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path, content: str
) -> None:
    rid = new_run()
    (files / "c.txt").write_text(content)
    code, err, _ = failing(monkeypatch, capsys, eval_args(files, rid, "--resamples", "0"))
    assert code == 1 and "class" in err


def test_oversized_class_file_is_refused_without_reading_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    (files / "c.txt").write_bytes(b"a\n" * (600 * 1024))
    code, err, _ = failing(monkeypatch, capsys, eval_args(files, rid, "--resamples", "0"))
    assert code == 1 and "too large" in err


def test_unknown_run_exits_1_and_malformed_run_id_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    new_run()
    from cvpilot.core.hashing import new_run_id

    code, err, _ = failing(monkeypatch, capsys, eval_args(files, new_run_id()))
    assert code == 1 and "not found" in err
    assert main(monkeypatch, *eval_args(files, "not-an-id")) == 2


@pytest.mark.parametrize("slices", ["camera", "=col", "bad key=col", "a=b=c", ""])
def test_bad_slices_syntax_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path, slices: str
) -> None:
    rid = new_run()
    capsys.readouterr()
    assert main(monkeypatch, *eval_args(files, rid, "--slices", slices)) == 2
    assert capsys.readouterr().out == ""


def test_missing_files_and_bad_numbers_are_usage_errors(
    monkeypatch: pytest.MonkeyPatch, files: Path
) -> None:
    rid = new_run()
    (files / "p.parquet").unlink()
    assert main(monkeypatch, *eval_args(files, rid)) == 2
    write_preds(files / "p.parquet")
    assert main(monkeypatch, *eval_args(files, rid, "--resamples", "-3")) == 2
    assert main(monkeypatch, *eval_args(files, rid, "--step", "-1")) == 2
    assert main(monkeypatch, *eval_args(files, rid, "--resamples", "999999")) == 2


def test_a_slice_column_with_unbounded_cardinality_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    n = 300
    ids = [f"img{i:04d}" for i in range(n)]
    write_gts(
        files / "g.parquet",
        image_id=ids,
        cls=pa.array(np.zeros(n, dtype=np.int64)),
        x1=[A[0]] * n, y1=[A[1]] * n, x2=[A[2]] * n, y2=[A[3]] * n,
        iscrowd=[False] * n, width=[200] * n, height=[200] * n, camera=ids,
    )  # fmt: skip
    write_preds(
        files / "p.parquet", image_id=ids[:1], cls=[0], score=[0.9],
        x1=[A[0]], y1=[A[1]], x2=[A[2]], y2=[A[3]],
    )  # fmt: skip
    code, err, _ = failing(
        monkeypatch, capsys, eval_args(files, rid, "--slices", "camera=camera", "--resamples", "0")
    )
    assert code == 1 and "300 distinct values" in err


def test_class_file_reader_refuses_symlinks_fifos_and_deep_json(tmp_path: Path) -> None:
    import os

    from cvpilot.core.errors import EvalInputError
    from cvpilot.eval.files import read_class_names

    real = classes(tmp_path / "real.txt")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    with pytest.raises(EvalInputError, match="symbolic|regular file"):
        read_class_names(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(EvalInputError, match="regular file"):
        read_class_names(fifo)
    with pytest.raises(EvalInputError, match="regular file"):
        read_class_names(tmp_path)
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 200_000)
    with pytest.raises(EvalInputError, match="JSON"):
        read_class_names(deep)
    assert read_class_names(real) == ["cat", "dog"]


# -- memory ceiling --------------------------------------------------------------------------


def test_memory_limit_helper_lowers_the_address_space_limit_on_linux_only() -> None:
    import resource

    from cvpilot.cli import eval_cmds

    calls: list[tuple[int, tuple[int, int]]] = []

    def fake_set(which: int, limits: tuple[int, int]) -> None:
        calls.append((which, limits))

    soft = resource.RLIM_INFINITY
    assert eval_cmds.apply_memory_limit(
        512, platform="linux", setrlimit=fake_set, getrlimit=lambda w: (soft, soft)
    )
    assert calls == [(resource.RLIMIT_AS, (512 * 1024 * 1024, soft))]
    assert not eval_cmds.apply_memory_limit(512, platform="darwin", setrlimit=fake_set)
    assert not eval_cmds.apply_memory_limit(0, platform="linux", setrlimit=fake_set)
    assert len(calls) == 1


def test_the_memory_option_reaches_the_helper(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    from cvpilot.cli import eval_cmds

    seen: list[int] = []
    monkeypatch.setattr(eval_cmds, "apply_memory_limit", lambda mb, **kw: seen.append(mb) or True)
    rid = new_run()
    assert (
        main(monkeypatch, *eval_args(files, rid, "--resamples", "0", "--max-memory-mb", "777")) == 0
    )
    assert main(monkeypatch, *eval_args(files, rid, "--max-memory-mb", "-1")) == 2
    assert seen == [777]


# -- closed runs, parameters, provenance, argument validation ---------------------------------


@pytest.mark.parametrize("status", ["finished", "failed", "preempted"])
def test_closed_runs_are_refused_without_the_explicit_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path, status: str
) -> None:
    rid = new_run(status=status)
    argv = [a for a in eval_args(files, rid, "--resamples", "0") if a != "--allow-closed"]
    code, err, _ = failing(monkeypatch, capsys, argv)
    assert code == 1 and "allow-closed" in human(err)
    assert runs.get_run_view("proj", rid, store_dir=files / "store").snapshots == []
    assert main(monkeypatch, *eval_args(files, rid, "--resamples", "0")) == 0


def test_snapshots_record_parameters_provenance_and_slice_spec(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    capsys.readouterr()
    argv = eval_args(
        files, rid, "--resamples", "5", "--json", "--slices", "camera=camera",
        "--checkpoint-sha256", "c" * 64, "--git-commit", "abc1234",
    )  # fmt: skip
    assert main(monkeypatch, *argv) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["params"]["confusion_iou"] == 0.5 and len(data["params"]["iou_thresholds"]) == 10
    assert data["provenance"] == {"checkpoint_sha256": "c" * 64, "git_commit": "abc1234"}
    assert data["slice_spec"] == {"camera": "camera"} and data["ci_status"] == "computed"
    assert data["bootstrap"]["numpy_version"] == np.__version__


def test_a_single_image_records_that_no_interval_was_computed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rid = new_run()
    write_gts(
        tmp_path / "g.parquet", image_id=["i1"], cls=pa.array([0], type=pa.int64()),
        x1=pa.array([A[0]]), y1=pa.array([A[1]]), x2=pa.array([A[2]]), y2=pa.array([A[3]]),
        iscrowd=[False], width=[200], height=[200], camera=["c"],
    )  # fmt: skip
    write_preds(
        tmp_path / "p.parquet", image_id=["i1"], cls=[0], score=[0.9],
        x1=[A[0]], y1=[A[1]], x2=[A[2]], y2=[A[3]],
    )  # fmt: skip
    classes(tmp_path / "c.txt")
    capsys.readouterr()
    assert main(monkeypatch, *eval_args(tmp_path, rid, "--resamples", "50", "--json")) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ci"] == {} and data["ci_status"] == "not_computed_too_few_images"


@pytest.mark.parametrize(
    "extra",
    [
        ["--step", str(2**63)],
        ["--seed", str(2**64)],
        ["--slices", "caméra=camera"],
        ["--slices", "camera=a", "--slices", "camera=b"],
        ["--checkpoint-sha256", "xyz"],
        ["--git-commit", "NOT-HEX"],
    ],
)
def test_bad_numbers_keys_and_provenance_are_usage_errors(
    monkeypatch: pytest.MonkeyPatch, files: Path, extra: list[str]
) -> None:
    rid = new_run()
    assert main(monkeypatch, *eval_args(files, rid, "--resamples", "0", *extra)) == 2


def test_nan_class_values_and_boxes_on_empty_rows_are_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    write_gts(files / "g.parquet", cls=pa.array([0, 1, float("nan"), None], type=pa.float64()))
    code, err, _ = failing(monkeypatch, capsys, eval_args(files, rid, "--resamples", "0"))
    assert code == 1 and "whole numbers" in err
    write_gts(files / "g.parquet", x1=pa.array([A[0], B[0], A[0], 5.0]))  # null cls, real box
    code, err, _ = failing(monkeypatch, capsys, eval_args(files, rid, "--resamples", "0"))
    assert code == 1 and "null cls" in err


def test_cli_events_keep_the_per_invocation_request_id_and_name_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], files: Path
) -> None:
    rid = new_run()
    capsys.readouterr()
    assert main(monkeypatch, *eval_args(files, rid, "--resamples", "0")) == 0
    done = next(e for e in events(capsys.readouterr().err) if e["message"] == "eval_completed")
    assert done["run_id"] == rid and not done["request_id"].startswith(rid)  # a UUID
