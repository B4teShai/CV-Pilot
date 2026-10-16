"""A resume never falls through to an ancestor or an older run because a newer member's checkpoints
were refused, and the lineage it trusts is the one the tag-covered manifests agree with."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import resume
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError
from cvpilot.core.hashing import new_run_id

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
KEY = "7c" * 32


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def make_chain(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, key: str | None = KEY
) -> tuple[str, str]:
    if key:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", key)
    else:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    first = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(first, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False)
    mgr.save_now(1, model=weights(1.0))
    mgr.close()
    first.finish("preempted")
    child_meta = runs.resume_from(
        runs.get_run_view("p", first.run_id, store_dir=tmp / "store").meta,
        run_id=new_run_id(), host="h", name="job",
    )  # fmt: skip
    _, repro2 = run_meta("p")
    second = runs.start_run(
        child_meta, repro2.model_copy(update={"run_id": child_meta.run_id}), store_dir=tmp / "store"
    )
    cmgr = CheckpointManager(
        second, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    cmgr.save_now(2, model=weights(2.0))
    cmgr.close()
    second.finish("preempted")
    return first.run_id, second.run_id


def find(tmp: Path, **kw: Any) -> Any:
    return resume.find_resume_point("p", checkpoint_dir=tmp / "ckpt", store_dir=tmp / "store", **kw)


def test_an_honest_lineage_resumes_from_the_newest_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = make_chain(tmp_path, monkeypatch)
    point = find(tmp_path)
    assert point is not None and point.checkpoint_run_id == second


def test_a_store_that_misstates_the_parent_cannot_splice_in_another_runs_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = make_chain(tmp_path, monkeypatch)
    real = runs.list_runs

    def tampered(project: str, **kw: Any) -> Any:
        out = []
        for meta in real(project, **kw):
            if meta.run_id == second:  # the store now says this run resumed a stranger
                meta = meta.model_copy(update={"parent_run_id": new_run_id()})
            out.append(meta)
        return out

    monkeypatch.setattr(runs, "list_runs", tampered)
    with pytest.raises(CheckpointUntrustedError) as caught:
        find(tmp_path)
    assert caught.value.reason in {"binding_mismatch", "checkpoint_missing"}


def test_a_refused_newer_member_is_not_replaced_by_an_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = make_chain(tmp_path, monkeypatch)
    # the newest member's manifest loses its tag (a writer of the directory): refused, not skipped
    import json

    path = tmp_path / "ckpt" / second / "last" / "step_000000002.manifest.json"
    doc = json.loads(path.read_text())
    doc["auth_tag"] = None
    path.chmod(0o600)
    path.write_text(json.dumps(doc) + "\n")
    with pytest.raises(CheckpointUntrustedError):
        find(tmp_path)
