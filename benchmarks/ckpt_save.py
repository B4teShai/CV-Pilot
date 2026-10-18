"""Checkpoint save overhead: a full ``CheckpointManager.save_now`` against writing the same bytes
with one ``fsync`` (what ``torch.save`` of the same state costs on the same disk, minus the
serialisation the two share).

ADR-0006 sets a budget of 5 % over ``torch.save`` for hashing and the manifest. The save also
re-reads and re-hashes the file (``verify_after_save``) and records a digest in the run store;
the numbers below separate those costs so the budget can be judged with and without the
verification pass. The framework is not installed here, so the baseline is the same bytes written
with ``open``, ``write`` and ``fsync``.

Usage: ``python benchmarks/ckpt_save.py [megabytes] [repeats]`` (default 256 MB, 3 repeats).

``profile_state`` is what notebook 04 calls for real model states: where a steady-state save spends
its time (serialise, write with hash and fsync, manifest, verification re-read, run-store anchor,
pointer, retention), next to the same bytes written plainly and, when a ``baseline`` is given, next
to the framework's own ``save`` of the same state on the same disk.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
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

keep_last_warmup = 3  # the first saves have fewer files to retain


def plain_write(path: Path, data: bytes) -> float:
    started = time.perf_counter()
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    elapsed = time.perf_counter() - started
    path.unlink()
    return elapsed


def managed_save(root: Path, store: Path, state: dict[str, np.ndarray], *, verify: bool) -> float:
    run_id = new_run_id()
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
    run = runs.start_run(meta, repro, store_dir=store)
    manager = CheckpointManager(
        run, dir=root, keep_last=1, verify_after_save=verify, install_signal_handlers=False
    )
    started = time.perf_counter()
    manager.save_now(1, model=state)
    elapsed = time.perf_counter() - started
    run.finish()
    return elapsed


def steady_saves(
    root: Path, store: Path, state: dict[str, np.ndarray], *, saves: int = 7, keep_last: int = 3
) -> list[float]:
    """Times of consecutive ``save_now`` calls on one manager with retention on: the saves a long
    job actually pays for (each one also prunes the checkpoints beyond ``keep_last``)."""
    run_id = new_run_id()
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
    run = runs.start_run(meta, repro, store_dir=store)
    manager = CheckpointManager(run, dir=root, keep_last=keep_last, install_signal_handlers=False)
    times: list[float] = []
    for step in range(1, saves + 1):
        started = time.perf_counter()
        manager.save_now(step, model=state)
        times.append(time.perf_counter() - started)
    run.finish()
    return times


@contextlib.contextmanager
def phase_timers(manager: CheckpointManager) -> Iterator[dict[str, float]]:
    """Accumulate the wall time of each phase of ``save_now`` into the yielded dict, by wrapping the
    functions each phase runs in (the originals are put back on exit)."""
    from cvpilot.ckpt import directory, fsops

    totals: dict[str, float] = defaultdict(float)
    undo: list[Callable[[], None]] = []

    def wrap(owner: object, name: str, label: Callable[..., str]) -> None:
        original = getattr(owner, name)

        def timed(*args: object, **kwargs: object) -> object:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                totals[label(*args, **kwargs)] += time.perf_counter() - started

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


def profile_state(
    state: dict[str, object],
    *,
    baseline: Callable[[Path], None] | None = None,
    repeats: int = 7,
    keep_last: int = 3,
    verify_after_save: bool = True,
) -> dict[str, object]:
    """Median seconds of a steady-state ``save_now`` of ``state`` and of each phase in it, plus the
    plain write of the same bytes and the optional framework ``baseline(path)`` (written, flushed
    and fsynced like the checkpoint is). The first ``keep_last`` saves are warm-up."""
    data = get_serializer("safetensors").dumps({"model": state})
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        plain = statistics.median(plain_write(base / "plain.bin", data) for _ in range(repeats))
        framework = None
        if baseline is not None:
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                baseline(base / "baseline.bin")
                with (base / "baseline.bin").open("rb") as handle:
                    os.fsync(handle.fileno())
                samples.append(time.perf_counter() - started)
                (base / "baseline.bin").unlink()
            framework = statistics.median(samples)
        hashes = []
        for _ in range(repeats):
            started = time.perf_counter()
            hashlib.sha256(data).hexdigest()
            hashes.append(time.perf_counter() - started)
        run_id = new_run_id()
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
        run = runs.start_run(meta, repro, store_dir=base / "store")
        manager = CheckpointManager(
            run, dir=base / "ckpt", keep_last=keep_last, verify_after_save=verify_after_save,
            install_signal_handlers=False,
        )  # fmt: skip
        totals: list[float] = []
        per_phase: dict[str, list[float]] = defaultdict(list)
        for step in range(1, keep_last + repeats + 1):
            with phase_timers(manager) as timers:
                started = time.perf_counter()
                manager.save_now(step, model=state)
                whole = time.perf_counter() - started
            if step > keep_last:
                totals.append(whole)
                for name, value in timers.items():
                    per_phase[name].append(value)
        run.finish()
    total = statistics.median(totals)
    phases = {name: statistics.median(values) for name, values in per_phase.items()}
    phases["everything else (state build, manifest, locking, scans)"] = max(
        0.0, total - sum(phases.values())
    )
    return {
        "bytes": len(data),
        "save_now_s": total,
        "plain_write_fsync_s": plain,
        "framework_save_fsync_s": framework,
        "sha256_pass_s": statistics.median(hashes),
        "phases_s": phases,
    }


def main(megabytes: int = 256, repeats: int = 3) -> None:
    state = {"w": np.random.default_rng(0).random(megabytes * 1024 * 1024 // 4).astype(np.float32)}
    data = get_serializer("safetensors").dumps({"model": state})
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        plain = [plain_write(base / "plain.bin", data) for _ in range(repeats)]
        unverified = [
            managed_save(base / f"u{i}", base / "store", state, verify=False)
            for i in range(repeats)
        ]
        verified = [
            managed_save(base / f"v{i}", base / "store", state, verify=True) for i in range(repeats)
        ]
        steady = steady_saves(base / "steady", base / "store", state)[keep_last_warmup:]
    p, u, v = (statistics.median(x) for x in (plain, unverified, verified))
    # the managed numbers include serialising the state, which the plain write does not
    serialise = []
    for _ in range(repeats):
        started = time.perf_counter()
        get_serializer("safetensors").dumps({"model": state})
        serialise.append(time.perf_counter() - started)
    s = statistics.median(serialise)
    sys.stdout.write(f"state {megabytes} MB, median of {repeats}\n")
    sys.stdout.write(f"write + fsync of the same bytes:        {p:.3f} s\n")
    sys.stdout.write(f"serialise only:                          {s:.3f} s\n")
    sys.stdout.write(f"save_now, verify_after_save=False:       {u:.3f} s\n")
    sys.stdout.write(f"save_now, verify_after_save=True:        {v:.3f} s\n")
    sys.stdout.write(
        f"save_now in steady state (keep_last=3, retention on): {statistics.median(steady):.3f} s\n"
    )
    sys.stdout.write(
        f"overhead over serialise + plain write, unverified: {(u / (s + p) - 1) * 100:.1f} %\n"
    )
    sys.stdout.write(
        f"overhead over serialise + plain write, verified:   {(v / (s + p) - 1) * 100:.1f} %\n"
    )


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
