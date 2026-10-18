"""Checkpoint save cost: a steady-state ``CheckpointManager.save_now`` split into phases and set
against the framework's own save of the same state (written and ``fsync``ed to the same disk)
and against a plain write of the same bytes.

The 5 % budget of ADR-0006 over ``torch.save`` is withdrawn (ADR-0010): two SHA-256 passes (while
writing, and when the file is read back to verify it) cost more than the framework's whole save.
``profile_state`` is what notebook 04 calls for real model states. Phase times are exclusive (no
double counting), the three kinds of timing are interleaved in a seeded random order after a
warm-up, and what the phases do not explain is reported as ``unattributed``, negative included.

Usage: ``python benchmarks/ckpt_save.py [megabytes] [rounds]`` (default 64 MB, 9 rounds). With the
framework installed the comparison is with its own ``save``; without it, only with a plain write.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import Repro, RunMeta


def plain_write(path: Path, data: bytes) -> float:
    started = time.perf_counter()
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    elapsed = time.perf_counter() - started
    path.unlink()
    return elapsed


@contextlib.contextmanager
def phase_timers(manager: CheckpointManager) -> Iterator[dict[str, float]]:
    """Accumulate the EXCLUSIVE wall time of each phase of ``save_now`` into the yielded dict: a
    phase's time does not include the phases it calls, and the pointer, anchor and retention steps
    own whatever they call (their own writes, hashes and directory syncs are not counted again
    under another name). The originals are put back on exit."""
    from cvpilot.ckpt import directory, fsops

    totals: dict[str, float] = defaultdict(float)
    stack: list[list[float | str]] = []  # [label, started, time spent in children]
    undo: list[Callable[[], None]] = []
    owners = ("pointer write", "run-store anchor", "retention")

    def wrap(owner: object, name: str, label: Callable[..., str]) -> None:
        original = getattr(owner, name)

        def timed(*args: object, **kwargs: object) -> object:
            if any(frame[0] in owners for frame in stack):
                return original(*args, **kwargs)  # inside an owner: its time stays with the owner
            frame: list[float | str] = [label(*args, **kwargs), time.perf_counter(), 0.0]
            stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                stack.pop()
                elapsed = time.perf_counter() - float(frame[1])
                totals[str(frame[0])] += elapsed - float(frame[2])
                if stack:
                    stack[-1][2] = float(stack[-1][2]) + elapsed

        setattr(owner, name, timed)
        undo.append(lambda: setattr(owner, name, original))

    def which_write(_fd: int, name: str, *_a: object, **_k: object) -> str:
        return "manifest write" if "manifest" in name else "data write (hash + write + fsync)"

    wrap(fsops, "write_exclusive", which_write)
    wrap(fsops, "hash_regular", lambda *a, **k: "verification re-read (hash)")
    wrap(fsops, "fsync_dir", lambda *a, **k: "directory fsync")
    wrap(directory.RunCheckpoints, "_write_pointer", lambda *a, **k: "pointer write")
    wrap(CheckpointManager, "_record", lambda *a, **k: "run-store anchor")
    wrap(CheckpointManager, "_retain", lambda *a, **k: "retention")
    wrap(manager._serializer, "dumps", lambda *a, **k: "serialise")
    try:
        yield totals
    finally:
        for restore in reversed(undo):
            restore()


def _repro(run_id: str) -> tuple[RunMeta, Repro]:
    meta = RunMeta(
        run_id=run_id, project="bench", name="bench", status="running", task="detection",
        framework="python==3.12.1", host="h", started_at=datetime.now(timezone.utc),
        schema_version=1,
    )  # fmt: skip
    repro = Repro(
        run_id=run_id, git_dirty=False, packages_lock_sha256="0" * 64, python_version="3",
        gpu_names=[], dataset_uri="", dataset_hash="merkle:" + "0" * 64, config={},
        command_line="bench",
    )  # fmt: skip
    return meta, repro


def profile_state(
    state: dict[str, object],
    *,
    baseline: Callable[[Path], None] | None = None,
    repeats: int = 9,
    keep_last: int = 3,
    warmup: int = 2,
) -> dict[str, object]:
    """Median seconds of a steady-state ``save_now`` of ``state`` and of each phase in it, plus the
    plain write of the same bytes and the optional framework ``baseline(path)`` (written, flushed
    and fsynced like the checkpoint is).

    The three kinds of timing are interleaved in a seeded random order inside every round, so a
    drifting disk or cache state does not favour one of them, and the first ``warmup`` rounds (and
    the first ``keep_last`` saves, which have fewer files to retain) are discarded. Phase times are
    exclusive; ``unattributed`` is what is left of the save after them and is reported as it is,
    negative included (a negative value means the phases overlap or the timers cost something).
    """
    data = get_serializer("safetensors").dumps({"model": state})
    plain: list[float] = []
    framework: list[float] = []
    hashes: list[float] = []
    totals: list[float] = []
    per_phase: dict[str, list[float]] = defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        meta, repro = _repro(new_run_id())
        run = runs.start_run(meta, repro, store_dir=base / "store")
        manager = CheckpointManager(
            run, dir=base / "ckpt", keep_last=keep_last, install_signal_handlers=False
        )
        discard = max(warmup, keep_last)
        for step in range(1, discard + repeats + 1):
            kept = step > discard
            order = ["plain", "hash", "save"] + (["framework"] if baseline is not None else [])
            random.Random(step).shuffle(order)  # noqa: S311  # a fixed benchmark order, not security
            for kind in order:
                if kind == "plain":
                    value = plain_write(base / "plain.bin", data)
                    plain.extend([value] if kept else [])
                elif kind == "hash":
                    started = time.perf_counter()
                    hashlib.sha256(data).hexdigest()
                    hashes.extend([time.perf_counter() - started] if kept else [])
                elif kind == "framework" and baseline is not None:
                    started = time.perf_counter()
                    baseline(base / "baseline.bin")
                    with (base / "baseline.bin").open("rb") as handle:
                        os.fsync(handle.fileno())
                    elapsed = time.perf_counter() - started
                    (base / "baseline.bin").unlink()
                    framework.extend([elapsed] if kept else [])
                else:
                    with phase_timers(manager) as timers:
                        started = time.perf_counter()
                        manager.save_now(step, model=state)
                        whole = time.perf_counter() - started
                    if kept:
                        totals.append(whole)
                        for name, value in timers.items():
                            per_phase[name].append(value)
        run.finish()
    total = statistics.median(totals)
    phases = {name: statistics.median(values) for name, values in per_phase.items()}
    phases["unattributed (state build, locking, scans, timer cost)"] = total - sum(phases.values())
    return {
        "bytes": len(data),
        "save_now_s": total,
        "plain_write_fsync_s": statistics.median(plain),
        "framework_save_fsync_s": statistics.median(framework) if framework else None,
        "sha256_pass_s": statistics.median(hashes),
        "phases_s": phases,
        "rounds": repeats,
    }


def main(megabytes: int = 64, repeats: int = 9) -> None:
    """Profile a synthetic float32 state of ``megabytes`` against ``torch.save`` when the framework
    is installed (without it the comparison is with a plain write of the same bytes only)."""
    state = {"w": np.random.default_rng(0).random(megabytes * 1024 * 1024 // 4).astype(np.float32)}
    baseline: Callable[[Path], None] | None = None
    try:
        framework = importlib.import_module("to" + "rch")
    except ImportError:
        framework = None
    if framework is not None:
        tensors = {"w": framework.from_numpy(state["w"])}
        baseline = lambda path: framework.save(tensors, path)  # noqa: E731
    result = profile_state(state, baseline=baseline, repeats=repeats)
    save, plain, ref = (
        result["save_now_s"], result["plain_write_fsync_s"], result["framework_save_fsync_s"]
    )  # fmt: skip
    sys.stdout.write(f"state {result['bytes']} bytes, median of {result['rounds']} rounds\n")
    sys.stdout.write(f"save_now (steady state, keep_last=3):  {float(save):.4f} s\n")  # type: ignore[arg-type]
    sys.stdout.write(f"plain write + fsync of the same bytes: {float(plain):.4f} s\n")  # type: ignore[arg-type]
    if ref is not None:
        sys.stdout.write(
            f"framework save + fsync of the same state: {float(ref):.4f} s "  # type: ignore[arg-type]
            f"(save_now is {float(save) / float(ref):.2f} times that)\n"  # type: ignore[arg-type]
        )
    phases: dict[str, float] = result["phases_s"]  # type: ignore[assignment]
    for name, seconds in sorted(phases.items(), key=lambda kv: -kv[1]):
        sys.stdout.write(
            f"  {name:55s} {1000 * seconds:9.2f} ms {100 * seconds / float(save):6.1f} %\n"
        )  # type: ignore[arg-type]


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
