"""Harness for the seeded kill-point chaos suite: many real ``SIGKILL``s inside checkpoint saves.

``argv``: work directory, kill point, seed, iterations.

The harness itself never touches a store or a checkpoint: for each iteration it forks a WRITER
that starts a run and saves checkpoints until, inside the save it was told to break, it stops
itself (``SIGSTOP``) at the chosen place. The harness sees the stop and sends ``SIGKILL``, so the
writer dies exactly there and nothing of it runs afterwards (no ``finally``, no handler). Then the
harness forks a VERIFIER, a fresh process that does what a restarted job does: mark the dead run
failed, resume a child run from it, restore, check the restored state byte for byte against what
that step saved, check every checkpoint the directory still admits the same way, save the next
step and restore that too. Each iteration prints one JSON line; the harness exits 0 and the test
decides.

Everything an iteration varies comes from ``random.Random(f"{seed}:{point}:{i}")``: how many saves
complete before the broken one, the tensor size (4 bytes to 1 MiB), ``keep_last``, whether an
authentication key is set, where exactly inside the point the writer stops and, for a torn write,
after how many bytes. The same seed gives the same iterations.

Kill points and the place inside them:

- ``before_temp_write``: the save decided its step and chain, before the temporary data file is
  created.
- ``mid_write``: inside the write of the temporary data file (or of the temporary manifest), after
  a random number of bytes that is at least 1 and less than the whole file; never fsynced.
- ``after_write_before_rename``: both temporary files may be complete, no final name exists yet
  (after the data file, after the manifest file, or just before the first link).
- ``after_rename_before_chain``: final names exist, the run store and the ``LATEST`` pointer are
  not updated yet (after the manifest link, after the data link, after verification but before
  the store record, after the record but before the pointer, or with the new pointer written to
  its temporary name but not renamed).
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = "chaos"
KEY = "6b" * 32  # an obviously fake 64-byte test key
ITERATION_TIMEOUT_S = 60.0
POINTS = {
    "before_temp_write": ["before_data_open"],
    "mid_write": ["data_bytes", "data_bytes", "data_bytes", "manifest_bytes"],
    "after_write_before_rename": ["data_written", "manifest_written", "before_first_link"],
    "after_rename_before_chain": [
        "manifest_linked",
        "data_linked",
        "verified_unrecorded",
        "recorded_no_pointer",
        "pointer_temp_written",
    ],
    "random_moment": ["random"],
}
RANDOM_EXTRA_SAVES = 50  # a random-moment writer keeps saving; the kill lands in one of these
RANDOM_DELAY_S = 0.004  # the kill comes 0 to 4 ms after the broken save started
# where the newest complete checkpoint is after the kill: the broken save itself, or the one before
COMPLETE_AFTER = {
    "data_linked",
    "verified_unrecorded",
    "recorded_no_pointer",
    "pointer_temp_written",
}
WIDTHS = (1, 7, 1000, 4096, 65_536, 262_144)


@dataclass
class Plan:
    point: str
    place: str
    acked: int  # saves that complete before the broken one
    width: int
    keep_last: int
    keyed: bool
    anchored: bool  # CVPILOT_CKPT_REQUIRE_TRUST=anchored (only without a key)
    cut: float  # fraction of the file written before a torn write stops (random: the delay)

    @property
    def broken_step(self) -> int:
        return self.acked + 1

    @property
    def expected(self) -> int:
        if self.anchored and self.place in ("data_linked", "verified_unrecorded"):
            return self.acked  # complete on disk, but not recorded: anchored refuses it
        return self.broken_step if self.place in COMPLETE_AFTER else self.acked


def plan_for(seed: int, point: str, i: int) -> Plan:
    rng = random.Random(f"{seed}:{point}:{i}")  # noqa: S311  # reproducible test input
    return Plan(
        point=point,
        place=rng.choice(POINTS[point]),
        acked=rng.randint(1, 4),
        width=rng.choice(WIDTHS),
        keep_last=rng.randint(1, 3),
        keyed=(keyed := rng.random() < 0.3),
        anchored=not keyed and rng.random() < 0.4,
        cut=rng.random(),
    )


def weights(seed: str, step: int, width: int) -> Any:
    import numpy as np

    rng = np.random.default_rng([zlib.crc32(seed.encode()), step])
    return rng.standard_normal(width).astype(np.float32)


def new_run(store: Path) -> Any:
    from cvpilot import runs
    from cvpilot.core.hashing import new_run_id
    from cvpilot.core.schema import Repro, RunMeta

    run_id = new_run_id()
    meta = RunMeta(
        run_id=run_id, project=PROJECT, name="chaos", status="running", task="detection",
        framework="python==3.12.1", host="h", started_at=datetime.now(timezone.utc),
        schema_version=1,
    )  # fmt: skip
    repro = Repro(
        run_id=run_id, git_dirty=False, packages_lock_sha256="0" * 64, python_version="3",
        gpu_names=[], dataset_uri="", dataset_hash="merkle:" + "0" * 64, config={},
        command_line="chaos",
    )  # fmt: skip
    return runs.start_run(meta, repro, store_dir=store)


# -- the writer --------------------------------------------------------------------------------


def writer(work: Path, plan: Plan, tag: str, report: int) -> None:
    """Save ``plan.acked`` checkpoints, then stop inside the next save at ``plan.place``."""
    from cvpilot.ckpt import fsops
    from cvpilot.ckpt.directory import RunCheckpoints
    from cvpilot.ckpt.manager import CheckpointManager

    def stop() -> None:
        os.kill(os.getpid(), signal.SIGSTOP)  # the harness kills the stopped process
        os._exit(3)  # unreachable unless the harness resumed it: report as a harness fault

    current = {"step": 0, "links": 0}
    place = plan.place

    def armed() -> bool:
        return current["step"] == plan.broken_step

    real_write, real_write_all = fsops.write_exclusive, fsops._write_all
    real_link, real_record = fsops.link_noreplace, CheckpointManager._record
    real_pointer = RunCheckpoints._write_pointer
    cut: dict[str, int | None] = {"left": None}

    def write_exclusive(dir_fd: int, name: str, data: Any, **kw: Any) -> tuple[str, int]:
        is_data = ".safetensors." in name and fsops.is_temp_name(name)
        is_manifest = ".manifest.json." in name and fsops.is_temp_name(name)
        is_pointer = name.startswith(".LATEST.") and fsops.is_temp_name(name)
        if armed() and is_data and place == "before_data_open":
            stop()
        if armed() and (
            (is_data and place == "data_bytes") or (is_manifest and place == "manifest_bytes")
        ):
            size = len(data)
            cut["left"] = min(max(1, int(plan.cut * size)), size - 1)
        result = real_write(dir_fd, name, data, **kw)
        if armed() and (
            (is_data and place == "data_written")
            or (is_manifest and place == "manifest_written")
            or (is_pointer and place == "pointer_temp_written")
        ):
            stop()
        return result

    def write_all(fd: int, data: Any) -> None:
        left = cut["left"]
        if left is not None:
            view = memoryview(data)[:left]
            while view:  # the first bytes reach the file, the rest never will
                view = view[os.write(fd, view) :]
            stop()
        real_write_all(fd, data)

    def link(dir_fd: int, source: str, target: str) -> None:
        if armed():
            current["links"] += 1
            if current["links"] == 1 and place == "before_first_link":
                stop()
        real_link(dir_fd, source, target)
        if armed() and (
            (current["links"] == 1 and place == "manifest_linked")
            or (current["links"] == 2 and place == "data_linked")
        ):
            stop()

    def record(self: Any, manifest: Any, digest: str) -> None:
        if armed() and place == "verified_unrecorded":
            stop()
        real_record(self, manifest, digest)

    def write_pointer(self: Any, *args: Any) -> None:
        if armed() and place == "recorded_no_pointer":
            stop()
        real_pointer(self, *args)

    fsops.write_exclusive = write_exclusive  # type: ignore[assignment]
    fsops._write_all = write_all  # type: ignore[assignment]
    fsops.link_noreplace = link  # type: ignore[assignment]
    CheckpointManager._record = record  # type: ignore[method-assign]
    RunCheckpoints._write_pointer = write_pointer  # type: ignore[method-assign]

    run = new_run(work / "store")
    os.write(report, f"RUN {run.run_id}\n".encode())
    manager = CheckpointManager(
        run, dir=work / "ckpt", keep_last=plan.keep_last, every_n_steps=1,
        install_signal_handlers=False,
    )  # fmt: skip
    last = plan.broken_step + (RANDOM_EXTRA_SAVES if place == "random" else 0)
    for step in range(1, last + 1):
        current["step"] = step
        os.write(report, f"SAVING {step}\n".encode())
        manager.save_now(step, model={"w": weights(tag, step, plan.width)})
        os.write(report, f"ACK {step}\n".encode())
    os._exit(4)  # the broken save completed: the stop place was never reached


# -- the verifier ------------------------------------------------------------------------------


def verifier(work: Path, plan: Plan, tag: str, dead_id: str) -> dict[str, Any]:
    """Restart after the kill: recover, resume, restore, compare, continue."""
    import numpy as np

    from cvpilot import runs
    from cvpilot.ckpt import auth
    from cvpilot.ckpt.directory import RunCheckpoints
    from cvpilot.ckpt.manager import CheckpointManager
    from cvpilot.ckpt.serializers import get_serializer
    from cvpilot.core.hashing import new_run_id
    from cvpilot.core.schema import Repro

    store = work / "store"
    out: dict[str, Any] = {}
    opened = runs.open_anchor(PROJECT, store)
    assert opened is not None
    key = auth.load_key()
    policy = "authenticated" if key is not None else ("anchored" if plan.anchored else "any")
    found = RunCheckpoints(work / "ckpt", dead_id, anchor=opened[0], key=key, require_trust=policy)
    admitted: list[int] = []
    for verified in found.iter_valid("last", load=True):
        assert verified.data is not None
        state = get_serializer("safetensors").loads(verified.data)
        step = verified.manifest.step
        if not np.array_equal(state["model"]["w"], weights(tag, step, plan.width)):
            raise AssertionError(f"step {step} loaded content that step never saved")
        admitted.append(step)
    opened[1]()
    out["admitted"] = admitted
    out["skipped"] = sorted({s.reason for s in found.skipped})
    debris = [p.name for p in (work / "ckpt" / dead_id / "last").iterdir()]
    out["temp_files"] = sum(1 for n in debris if n.startswith(".") and n.endswith(".tmp"))

    runs.recover_run(PROJECT, dead_id, store_dir=store)
    parent = runs.get_run_view(PROJECT, dead_id, store_dir=store).meta
    child_id = new_run_id()
    child_meta = runs.resume_from(parent, run_id=child_id, host="h", name=None)
    repro = Repro(
        run_id=child_id, git_dirty=False, packages_lock_sha256="0" * 64, python_version="3",
        gpu_names=[], dataset_uri="", dataset_hash="merkle:" + "0" * 64, config={},
        command_line="chaos",
    )  # fmt: skip
    child = runs.start_run(child_meta, repro, store_dir=store)
    manager = CheckpointManager(
        child, dir=work / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    restored = manager.restore()
    if restored is None:
        raise AssertionError("nothing restored although acknowledged checkpoints exist")
    step = restored.manifest.step
    out["restored"] = step
    out["trust"] = restored.trust
    if restored.run_id != dead_id:
        raise AssertionError("restored a checkpoint of another run")
    if not np.array_equal(restored.state["model"]["w"], weights(tag, step, plan.width)):
        raise AssertionError(f"the restored state is not what step {step} saved")
    following = weights(tag, step + 1, plan.width)
    manager.save_now(step + 1, model={"w": following})
    manager.close()
    child.finish("preempted")  # so the next restart may resume it
    again = runs.start_run(
        runs.resume_from(
            runs.get_run_view(PROJECT, child_id, store_dir=store).meta,
            run_id=(grandchild := new_run_id()), host="h", name=None,
        ),
        repro.model_copy(update={"run_id": grandchild}),
        store_dir=store,
    )  # fmt: skip
    second = CheckpointManager(
        again, dir=work / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    resumed = second.restore()
    if resumed is None or resumed.manifest.step != step + 1:
        raise AssertionError("the step saved after the restart does not restore")
    if not np.array_equal(resumed.state["model"]["w"], following):
        raise AssertionError("the step saved after the restart restored other content")
    second.close()
    again.finish()
    return out


# -- the harness -------------------------------------------------------------------------------


def wait(pid: int, deadline: float, options: int = 0) -> int | None:
    """``waitpid`` with a deadline: the status, or None when the deadline passed."""
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, options | os.WNOHANG)
        if done == pid:
            return status
        time.sleep(0.001)
    return None


def read_until(fd: int, deadline: float) -> str | None:
    """Everything written to ``fd`` until the writer closes it, or None at the deadline."""
    import select

    chunks: list[bytes] = []
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                return None
            chunk = os.read(fd, 65536)
            if not chunk:
                return b"".join(chunks).decode()
            chunks.append(chunk)
    finally:
        os.close(fd)


def kill_at_random(pid: int, fd: int, plan: Plan, deadline: float) -> int | None:
    """Wait until the writer starts the broken save, wait a seeded 0 to 4 ms, kill it."""
    import select

    seen = b""
    marker = f"SAVING {plan.broken_step}\n".encode()
    while marker not in seen and time.monotonic() < deadline:
        if select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            seen += chunk
    time.sleep(plan.cut * RANDOM_DELAY_S)
    os.kill(pid, signal.SIGKILL)
    status = wait(pid, deadline)
    while select.select([fd], [], [], 0)[0] and (chunk := os.read(fd, 65536)):
        seen += chunk  # the writer is dead: its pipe ends here
    rest = os.pipe()  # hand what was already read back to the caller through a fresh pipe
    os.write(rest[1], seen)
    os.dup2(rest[0], fd)
    os.close(rest[0])
    os.close(rest[1])
    return status


def quiet_child() -> None:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def one(work: Path, plan: Plan, tag: str) -> dict[str, Any]:
    result: dict[str, Any] = {**asdict(plan), "expected": plan.expected}
    deadline = time.monotonic() + ITERATION_TIMEOUT_S
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # the writer
        os.close(read_fd)
        quiet_child()
        try:
            writer(work, plan, tag, write_fd)
        except BaseException:  # noqa: BLE001  # report and die; the harness records it
            os.write(write_fd, b"ERROR " + traceback.format_exc()[-2000:].encode() + b"\n")
            os._exit(5)
    os.close(write_fd)
    if plan.place == "random":
        status = kill_at_random(pid, read_fd, plan, deadline)
    else:
        status = wait(pid, deadline, os.WUNTRACED)
    if status is not None and plan.place == "random":
        result["killed"] = os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
    elif status is not None and os.WIFSTOPPED(status):
        os.kill(pid, signal.SIGKILL)
        status = wait(pid, deadline)
        killed = status is not None and os.WIFSIGNALED(status)
        result["killed"] = killed and os.WTERMSIG(status) == signal.SIGKILL  # type: ignore[arg-type]
    else:
        if status is None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        result["killed"] = False
    lines = (read_until(read_fd, deadline) or "").splitlines()
    run_ids = [x.split()[1] for x in lines if x.startswith("RUN ")]
    acks = [int(x.split()[1]) for x in lines if x.startswith("ACK ")]
    result["acks"] = max(acks, default=0)
    started_saves = [int(x.split()[1]) for x in lines if x.startswith("SAVING ")]
    # the kill landed inside a save when the newest save the writer began was not yet acknowledged
    result["in_save"] = max(started_saves, default=0) > result["acks"]
    if not result["killed"] or not run_ids:
        result["ok"] = False
        result["error"] = "writer did not stop at the kill point: " + " | ".join(lines)[-1500:]
        return result
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # the verifier
        os.close(read_fd)
        quiet_child()
        try:
            found = verifier(work, plan, tag, run_ids[0])
            message = json.dumps({"ok": True, **found})
        except BaseException:  # noqa: BLE001
            message = json.dumps({"ok": False, "error": traceback.format_exc()[-3000:]})
        os.write(write_fd, message.encode())
        os._exit(0)
    os.close(write_fd)
    text = read_until(read_fd, deadline)
    if text is None or wait(pid, deadline) is None:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        result.update(ok=False, error="verifier timed out")
        return result
    result.update(json.loads(text) if text else {"ok": False, "error": "verifier died"})
    if result["ok"]:
        problems = []
        # a random kill may land between two saves: everything acknowledged must survive, and only
        # when a save was in flight may that one have completed too
        acked = result["acks"] if plan.place == "random" else plan.acked
        if plan.place != "random":
            allowed = {plan.expected}
        else:
            allowed = {acked, acked + 1} if result["in_save"] else {acked}
        if plan.place != "random" and result["acks"] != plan.acked:
            problems.append(f"acknowledged {result['acks']} saves, planned {plan.acked}")
        if result["restored"] not in allowed:
            problems.append(f"restored step {result['restored']}, expected {sorted(allowed)}")
        if result["admitted"][:1] != [result["restored"]]:
            problems.append(f"newest admitted {result['admitted'][:1]} is not the restored step")
        if problems:
            result.update(ok=False, error="; ".join(problems))
    return result


def main(work: str, point: str, seed: int, iterations: int) -> None:
    root = Path(work)
    # imported once here, before any fork, so each child starts warm; importing opens no store
    # and starts no thread, so forking stays safe
    import numpy  # noqa: F401

    import cvpilot.ckpt.manager  # noqa: F401
    import cvpilot.ckpt.serializers  # noqa: F401
    import cvpilot.runs  # noqa: F401

    for name in [n for n in os.environ if n.startswith("CVPILOT_")]:
        del os.environ[name]  # nothing of the caller's configuration leaks into an iteration
    for i in range(iterations):
        plan = plan_for(seed, point, i)
        for name in ("CVPILOT_CKPT_KEY", "CVPILOT_CKPT_REQUIRE_TRUST"):
            os.environ.pop(name, None)
        if plan.keyed:
            os.environ["CVPILOT_CKPT_KEY"] = KEY
        if plan.anchored:
            os.environ["CVPILOT_CKPT_REQUIRE_TRUST"] = "anchored"
        started = time.monotonic()
        where = root / f"{point}-{i:04d}"
        where.mkdir(parents=True)
        result = one(where, plan, f"{seed}:{point}:{i}")
        result["i"] = i
        result["seconds"] = round(time.monotonic() - started, 3)
        print(json.dumps(result), flush=True)
        if result.get("ok"):
            import shutil

            shutil.rmtree(where, ignore_errors=True)  # keep only what failed, for a post-mortem


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
