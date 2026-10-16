"""Helpers for checkpoint tests: real safetensors payloads and a tiny run directory builder."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.hashing import checkpoint_chain, pointer_digest
from cvpilot.core.schema import CheckpointManifest
from cvpilot.core.ulid import generate_ulid

DATASET = "merkle:" + "c" * 64


def blob(seed: int = 0, n: int = 64) -> bytes:
    """A real safetensors file whose content depends on ``seed``."""
    state = {"model": {"w": np.arange(n, dtype=np.float32) + seed}, "step": seed}
    return get_serializer("safetensors").dumps(state)


def make(tmp_path: Path, *, run_id: str | None = None, **kw: Any) -> RunCheckpoints:
    return RunCheckpoints(tmp_path / "ckpt", run_id or generate_ulid(), create=True, **kw)


def save(rc: RunCheckpoints, step: int, *, kind: str = "last", seed: int | None = None, **kw: Any):  # type: ignore[no-untyped-def]
    if kind == "best":
        kw.setdefault("metric_key", "mAP50")
        kw.setdefault("metric_value", 0.5)
        kw.setdefault("metric_mode", "max")
    return rc.save(
        kind=kind,  # type: ignore[arg-type]
        step=step,
        data=blob(step if seed is None else seed),
        format="safetensors",
        versions={"cvpilot": "0.1.0"},
        dataset_hash=DATASET,
        **kw,
    )


def kind_dir(rc: RunCheckpoints, kind: str = "last") -> Path:
    return rc.path / kind


def data_file(rc: RunCheckpoints, step: int, kind: str = "last") -> Path:
    return kind_dir(rc, kind) / f"step_{step:09d}.safetensors"


def manifest_file(rc: RunCheckpoints, step: int, kind: str = "last") -> Path:
    return kind_dir(rc, kind) / f"step_{step:09d}.manifest.json"


def make_writable(path: Path) -> None:
    os.chmod(path, 0o644)


def forge_manifest(rc: RunCheckpoints, step: int, *, kind: str = "last", **changes: Any) -> bytes:
    """Rewrite a manifest with ``changes`` and a recomputed chain value, as an adversary who can
    write the directory would. Returns the new manifest bytes."""
    path = manifest_file(rc, step, kind)
    make_writable(path)
    fields = json.loads(path.read_text())
    fields.update(changes)
    fields["chain_sha256"] = checkpoint_chain(
        fields["prev_chain_sha256"],
        fields["run_id"],
        fields["kind"],
        fields["step"],
        fields["sha256"],
        fields["size_bytes"],
        fields["format"] if fields.get("schema_version") else None,
    )
    data = (CheckpointManifest(**fields).model_dump_json() + "\n").encode()
    path.write_bytes(data)
    return data


def forge_pointer(rc: RunCheckpoints, manifest_bytes: bytes, step: int, kind: str = "last") -> None:
    """Point ``LATEST`` at the manifest bytes just forged, with a matching pointer digest."""
    fields = json.loads(manifest_file(rc, step, kind).read_text())
    name = f"{kind}/step_{step:09d}.manifest.json"
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    pointer = {
        "run_id": fields["run_id"],
        "kind": kind,
        "step": step,
        "manifest": name,
        "manifest_sha256": digest,
        "chain_sha256": fields["chain_sha256"],
    }
    pointer["pointer_sha256"] = pointer_digest(
        pointer["run_id"], kind, step, name, digest, fields["chain_sha256"]
    )
    target = rc.path / ("LATEST" if kind == "last" else "BEST")
    make_writable(target)
    target.write_text(json.dumps(pointer))


def run_meta(project: str = "p") -> tuple[Any, Any]:
    """``(RunMeta, Repro)`` of a fresh running detection run (dataset hash ``merkle:ccc...``)."""
    from datetime import datetime, timezone

    from cvpilot.core.schema import SCHEMA_VERSION, Repro, RunMeta

    rid = generate_ulid()
    meta = RunMeta(
        run_id=rid, project=project, name="n", status="running", task="detection",
        framework="python==3.12.1", host="h", started_at=datetime.now(timezone.utc),
        schema_version=SCHEMA_VERSION,
    )  # fmt: skip
    repro = Repro(
        run_id=rid, git_dirty=False, packages_lock_sha256="c" * 64, python_version="3.12",
        gpu_names=[], dataset_uri="", dataset_hash="merkle:" + "c" * 64, config={},
        command_line="t.py",
    )  # fmt: skip
    return meta, repro
