"""Seeded truncated uploads against fsspec's in-memory filesystem: 200 uploads cut after a random
number of bytes, either as a dropped connection (the store raises ``OSError`` after a partial
write) or as a process that dies mid-upload (nothing of it runs afterwards). A node that has
nothing must then restore exactly the newest checkpoint whose upload completed, byte for byte,
never a partial object, and every manifest the remote holds must describe a data object that
verifies.

Each iteration varies, from ``random.Random(f"{SEED}:{i}")``: the number of saves (2 to 5), which
upload is cut, whether the data object or the manifest is cut, after how many bytes (0 to all but
one), the mirror's retries (1 or 3: a retry resumes from the partial object when its bytes match)
and whether the cut is a dropped connection or a death.
"""

from __future__ import annotations

import collections
import hashlib
import random
import sys
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import remote as remote_mod
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.uri import parse_remote_uri

fsspec = pytest.importorskip("fsspec")
from cvpilot.remote.fsspec_store import FsspecStore, open_remote  # noqa: E402

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds"),
]
ITERATIONS = 200
SEED = 20261026


class Died(BaseException):
    """The training process died: nothing after this runs in it."""


@dataclass
class Cut:
    upload: int  # which upload call (1-based) is cut
    target: str  # "data" or "manifest"
    fraction: float  # how much of the object reaches the store before the cut
    death: bool


class TruncatingStore:
    """The real memory store, with one write cut after a number of bytes."""

    def __init__(self, inner: FsspecStore, cut: Cut | None) -> None:
        self.inner, self.cut = inner, cut
        self.uploads = 0  # upload calls started, counted by the test's wrapper
        self.cut_done = False
        self.dead = False
        self.appends = 0

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.inner, name)

        def call(*args: Any, **kw: Any) -> Any:
            if self.dead:
                raise Died
            return method(*args, **kw)

        return call

    def write(self, key: str, chunks: Iterable[bytes], *, append: bool = False) -> None:
        if self.dead:
            raise Died
        cut = self.cut
        is_target = cut is not None and (
            key.endswith(".manifest.json.part")
            if cut.target == "manifest"
            else key.endswith(".safetensors.part")
        )
        if append:
            self.appends += 1
        if cut is None or self.cut_done or not is_target or self.uploads != cut.upload:
            self.inner.write(key, chunks, append=append)
            return
        data = b"".join(chunks)
        keep = min(int(cut.fraction * len(data)), max(len(data) - 1, 0))
        self.inner.write(key, [data[:keep]], append=append)  # the bytes that made it
        self.cut_done = True
        if cut.death:
            self.dead = True
            raise Died
        raise ConnectionResetError("connection reset by peer")


def weights(tag: str, step: int) -> Any:
    rng = np.random.default_rng([int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16), step])
    return rng.standard_normal(rng.integers(1, 70_000)).astype(np.float32)


@pytest.fixture
def completed(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[Any, int]]]:
    """Records ``(store, step)`` for every upload call that returned, and counts upload calls on
    the store so a cut can target one of them."""
    done: list[tuple[Any, int]] = []
    real = remote_mod.Mirror.upload

    def upload(self: remote_mod.Mirror, rc: Any, manifest: Any) -> bool:
        store = self._store  # noqa: SLF001
        if isinstance(store, TruncatingStore):
            store.uploads += 1
        sent = real(self, rc, manifest)
        done.append((store, manifest.step))
        return sent

    monkeypatch.setattr(remote_mod.Mirror, "upload", upload)
    yield done


def scenario(tmp_path: Path, i: int, completed: list[tuple[Any, int]]) -> dict[str, Any]:
    rng = random.Random(f"{SEED}:{i}")  # noqa: S311  # reproducible test input
    saves = rng.randint(2, 5)
    cut = Cut(
        upload=rng.randint(1, saves),
        target=rng.choice(["data", "data", "data", "manifest"]),
        fraction=rng.random(),
        death=rng.random() < 0.5,
    )
    retries = rng.choice([1, 3])
    uri = f"memory://chaos-{uuid.uuid4().hex}/runs"
    location = parse_remote_uri(uri)
    store = TruncatingStore(open_remote(location), cut)
    tag = f"{SEED}:{i}"
    work = tmp_path / f"it-{i:04d}"
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=work / "store-a")
    manager = CheckpointManager(
        run, dir=work / "ckpt-a", remote=uri, remote_store=store, remote_retries=retries,
        remote_sleep=lambda s: None, every_n_steps=1, keep_last=10,
        install_signal_handlers=False,
    )  # fmt: skip
    died_at = None
    for step in range(1, saves + 1):
        try:
            manager.save_now(step, model={"w": weights(tag, step)})
        except Died:
            died_at = step
            break
    if died_at is None:
        manager.close()
        run.finish()
    # a process that died runs nothing more; its run, lease and files stay as they are
    ok_steps = sorted({s for st, s in completed if st is store})
    # a node with no store and no checkpoint directory looks at the remote only
    meta_b, repro_b = run_meta()
    fresh_run = runs.start_run(meta_b, repro_b, store_dir=work / "store-b")
    fresh = CheckpointManager(
        fresh_run, dir=work / "ckpt-b", remote=uri, remote_store=open_remote(location),
        every_n_steps=1, install_signal_handlers=False,
    )  # fmt: skip
    restored = fresh.restore(discover=True)
    out: dict[str, Any] = {
        "i": i, "saves": saves, "cut": cut, "retries": retries, "died_at": died_at,
        "uploaded": ok_steps, "appends": store.appends, "cut_done": store.cut_done,
    }  # fmt: skip
    if ok_steps:
        assert restored is not None, out
        assert restored.run_id == run.run_id
        assert restored.manifest.step == ok_steps[-1], out
        assert np.array_equal(restored.state["model"]["w"], weights(tag, ok_steps[-1])), out
        out["restored"] = restored.manifest.step
    else:
        assert restored is None, out  # nothing complete: nothing restored, nothing partial
        out["restored"] = None
    # every manifest on the remote describes a data object of exactly that size and digest
    fs = fsspec.filesystem("memory")
    root = location.key()
    names = [str(p) for p in fs.find(f"/{root}")]
    out["part_objects"] = sum(1 for n in names if n.endswith(".part"))
    manifests = [n for n in names if n.endswith(".manifest.json")]
    for name in manifests:
        import json

        manifest = json.loads(fs.cat_file(name))
        data = fs.cat_file(name.rsplit("/", 1)[0] + "/" + manifest["path"].split("/", 1)[1])
        assert len(data) == manifest["size_bytes"], name
        assert hashlib.sha256(data).hexdigest() == manifest["sha256"], name
    assert len(manifests) == len(ok_steps), out
    fresh.close()
    fresh_run.finish()
    fs.rm(f"/{root.split('/')[0]}", recursive=True)
    return out


def test_a_cut_upload_is_never_restored_and_the_newest_complete_one_is(
    tmp_path: Path, completed: list[tuple[Any, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_ENV", "dev")  # memory:// is for tests and development only
    results = [scenario(tmp_path, i, completed) for i in range(ITERATIONS)]
    assert len(results) == ITERATIONS
    assert all(r["cut_done"] for r in results)  # every iteration really cut an upload
    deaths = [r for r in results if r["died_at"] is not None]
    resets = [r for r in results if r["died_at"] is None]
    assert deaths and resets
    print(
        f"truncated uploads: {len(results)} cuts ({len(deaths)} deaths, {len(resets)} dropped "
        f"connections); targets {dict(collections.Counter(r['cut'].target for r in results))}; "
        f"restored the newest completed upload in {sum(r['restored'] is not None for r in results)}"
        f"; nothing restored because no upload had completed in "
        f"{sum(r['restored'] is None for r in results)}; resumed appends "
        f"{sum(r['appends'] for r in results)}; partial objects left on the remote "
        f"{sum(r['part_objects'] for r in results)}"
    )
