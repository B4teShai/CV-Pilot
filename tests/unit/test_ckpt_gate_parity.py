"""One gate for every path.

Every public entry point that can hand out checkpoint state, a path to it, or a verdict about it
is found by introspection and must be registered below, either as gated (it is run against the
attack matrix and must give the same outcome as every other) or as exempt with a reason. A new
entry point fails ``test_every_entry_point_is_registered`` until someone decides which it is.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta
from conftest import CliResult

import cvpilot
import cvpilot.ckpt as ckpt_pkg
from cvpilot import runs
from cvpilot.ckpt import resume
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.ckpt.remote import Mirror
from cvpilot.cli.app import app
from cvpilot.core.errors import CheckpointCorruptError, CheckpointUntrustedError
from cvpilot.core.hashing import checkpoint_chain
from cvpilot.core.uri import parse_remote_uri

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
KEY = "ef" * 32
URI = "s3://bucket-name/parity"
Invoke = Callable[..., CliResult]

# -- the registry ---------------------------------------------------------------------------------

GATED = {
    "CheckpointManager.maybe_save": "write: floor, chain, no replace (test_ckpt_save_rules.py)",
    "CheckpointManager.save_now": "write: floor, chain, no replace (test_ckpt_save_rules.py)",
    "CheckpointManager.adopt": "write: floor, chain, no replace (test_ckpt_save_rules.py)",
    "RunCheckpoints.save": "write: floor, chain, no replace (test_ckpt_save_rules.py)",
    "CheckpointManager.restore": "state",
    "CheckpointManager.restore_latest_valid": "state",
    "cvpilot.resume_path": "path",
    "cvpilot.ckpt.resume_path": "path",
    "cvpilot.ckpt.find_resume_point": "path",
    "RunCheckpoints.iter_valid": "state",
    "RunCheckpoints.inspect": "verdict",
    "RunCheckpoints.prune": "verdict",
    "RunCheckpoints.install": "write",
    "Mirror.fetch": "write",
    "cli ckpt verify": "verdict",
    "cli ckpt prune": "verdict",
    "cli ckpt list": "listing",
}
EXEMPT = {
    # they do not decide what may be returned; the reason says why
    "CheckpointManager.close": "releases resources",
    "CheckpointManager.list": "lists the manager's own run through RunCheckpoints.inspect (gated)",
    "CheckpointManager.verify": "answers through RunCheckpoints.inspect (gated)",
    "CheckpointManager.preempted": "a flag",
    "CheckpointManager.remote_url": "a validated location string",
    "cvpilot.ckpt.CheckpointManager": "the class itself",
    "cvpilot.ckpt.Restored": "a result type",
    "cvpilot.ckpt.ResumePoint": "a result type",
    "cvpilot.ckpt.apply_rng": "applies a validated state; hands out nothing",
    "cvpilot.ckpt.capture_rng": "reads generators",
    "cvpilot.CheckpointManager": "the class itself",
    "cvpilot.Run": "the run handle",
    "cvpilot.autolog": "attaches hooks",
    "cvpilot.finish": "ends a run",
    "cvpilot.init": "starts a run",
    "cvpilot.log": "logs metrics",
    "cvpilot.log_artifact": "stores a file",
    "cvpilot.log_eval": "evaluates predictions",
    "cvpilot.__version__": "a string",
    "RunCheckpoints.admission": "is the gate",
    "RunCheckpoints.freshness_floor": "an input of the gate",
    "RunCheckpoints.trusted_steps": "an input of the gate",
    "RunCheckpoints.judge_single": "an input of the gate (trust verdict of one candidate)",
    "RunCheckpoints.read_pointer": "returns a tag-checked pointer record, never data",
    "RunCheckpoints.existing": "internal helper for adopt (manifest only, never data)",
    "RunCheckpoints.read_range": "the mirror's reader of the local file it is uploading",
    "RunCheckpoints.read_manifest_bytes": "the mirror's reader of the local manifest",
    "RunCheckpoints.mark_uploaded": "writes a marker",
    "RunCheckpoints.quarantined": "lists what a save set aside; hands out no state",
    "RunCheckpoints.clear_quarantine": "an operator's acknowledgement; deletes, hands out nothing",
    "cli ckpt quarantine": "lists or clears the quarantine; hands out no state",
    "RunCheckpoints.is_uploaded": "reads a marker (authenticated when a key is set)",
    "RunCheckpoints.hold_lease": "liveness",
    "RunCheckpoints.lease_state": "liveness",
    "RunCheckpoints.last_activity": "liveness",
    "Mirror.budget": "a time limit for remote calls",
    "Mirror.begin_restore": "resets a budget",
    "Mirror.label": "scheme and bucket",
    "Mirror.list_runs": "a bounded listing of run ids; candidates are fetched through the gate",
    "Mirror.list_steps": "a listing of step numbers; each is then fetched through the gate",
    "Mirror.upload": "writes to the remote from a verified local checkpoint",
}


def discovered() -> set[str]:
    found: set[str] = set()
    for name in cvpilot.__all__:
        found.add(f"cvpilot.{name}")
    for name in ckpt_pkg.__all__:
        found.add(f"cvpilot.ckpt.{name}")
    for cls in (CheckpointManager, RunCheckpoints, Mirror):
        for name, member in vars(cls).items():
            if not name.startswith("_") and (
                callable(member) or isinstance(member, (property, staticmethod, classmethod))
            ):
                found.add(f"{cls.__name__}.{name}")
    for command in app.registered_groups:
        if command.name == "ckpt":
            for sub in command.typer_instance.registered_commands:  # type: ignore[union-attr]
                found.add(f"cli ckpt {sub.name or sub.callback.__name__.replace('ckpt_', '')}")
    return found


def test_every_entry_point_is_registered() -> None:
    found = discovered()
    registered = set(GATED) | set(EXEMPT)
    assert found - registered == set(), (
        f"decide whether these hand out checkpoints: {found - registered}"
    )
    assert registered - found == set(), f"stale registry entries: {registered - found}"
    assert len(found) >= 40


# -- a world to attack ---------------------------------------------------------------------------


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(4, value, dtype=np.float32)}


@dataclass
class World:
    tmp: Path
    victim: str  # the run whose checkpoints are attacked
    child: runs.RunHandle  # a new run (a resumed child unless `lineage` is false)
    fake: FakeRemote | None
    kind: str = "last"

    @property
    def store(self) -> Path:
        return self.tmp / "store"

    @property
    def ckpt(self) -> Path:
        return self.tmp / "ckpt"

    def folder(self, kind: str | None = None) -> Path:
        return self.ckpt / self.victim / (kind or self.kind)


def build(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, *, policy: str, keyed: bool, remote: bool = False,
    lineage: bool = True,
) -> World:  # fmt: skip
    if keyed:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    else:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp / "store"))
    monkeypatch.setenv("CVPILOT_CHECKPOINT_DIR", str(tmp / "ckpt"))
    monkeypatch.setenv("CVPILOT_PROJECT", "p")
    monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", policy)
    fake = FakeRemote() if remote else None
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    writer = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, keep_best=20, monitor="m",
        install_signal_handlers=False, remote=URI if remote else None, remote_store=fake,
        remote_sleep=lambda s: None,
    )  # fmt: skip
    for step in (1, 2, 3):
        writer.save_now(step, model=weights(float(step)))
        writer.save_now(step, kind="best", model=weights(float(step)), metrics={"m": float(step)})
    victim = run.run_id
    writer.close()
    run.finish("preempted")
    if lineage:
        parent = runs.get_run_view("p", victim, store_dir=tmp / "store").meta
        child_meta = runs.resume_from(parent, run_id=cvpilot_new_id(), host="h", name="job")
        child_repro = repro.model_copy(update={"run_id": child_meta.run_id})
    else:
        child_meta, child_repro = run_meta("p")
    child = runs.start_run(child_meta, child_repro, store_dir=tmp / "store")
    return World(tmp, victim, child, fake)


def cvpilot_new_id() -> str:
    from cvpilot.core.hashing import new_run_id

    return new_run_id()


def remove_steps(world: World, steps: tuple[int, ...], kind: str | None = None) -> None:
    for step in steps:
        for suffix in (".manifest.json", ".safetensors"):
            os.unlink(world.folder(kind) / f"step_{step:09d}{suffix}")


def attack_rolled_back(world: World) -> None:
    remove_steps(world, (3, 2))


def attack_rolled_back_no_pointer(world: World) -> None:
    remove_steps(world, (3, 2))
    os.unlink(world.ckpt / world.victim / "LATEST")


def attack_forged_untagged(world: World) -> None:
    shutil.rmtree(world.ckpt / world.victim)
    forger = RunCheckpoints(world.ckpt, world.victim, create=True)  # no key: untagged
    from ckpt_support import blob

    for step in (1, 2, 3):
        forger.save(
            kind="last", step=step, data=blob(step), format="safetensors", versions={},
            dataset_hash="merkle:" + "c" * 64,
        )  # fmt: skip


def attack_store_ahead(world: World) -> None:
    ahead = RunCheckpoints(world.ckpt, world.victim).existing("last", 3)
    assert ahead is not None
    five = ahead.model_copy(update={"step": 5, "path": "last/step_000000005.safetensors"})
    five = five.model_copy(
        update={
            "chain_sha256": checkpoint_chain(
                five.prev_chain_sha256, five.run_id, "last", 5, five.sha256, five.size_bytes
            )
        }
    )
    world.child.record_checkpoint(five, "d" * 64)


def attack_remote_rolled_back(world: World) -> None:
    assert world.fake is not None
    shutil.rmtree(world.ckpt / world.victim)
    for suffix in (".manifest.json", ".safetensors"):
        del world.fake.objects[f"{world.victim}/last/step_000000003{suffix}"]
        del world.fake.objects[f"{world.victim}/last/step_000000002{suffix}"]


def attack_remote_forged(world: World) -> None:
    assert world.fake is not None
    shutil.rmtree(world.ckpt / world.victim)
    world.fake.objects.clear()
    scratch = world.tmp / "forger"
    forger = RunCheckpoints(scratch, world.victim, create=True)  # no key
    from ckpt_support import blob

    for step in (1, 2, 3):
        manifest = forger.save(
            kind="last", step=step, data=blob(step), format="safetensors", versions={},
            dataset_hash="merkle:" + "c" * 64,
        )  # fmt: skip
        base = f"{world.victim}/last/"
        name = manifest.path.split("/", 1)[1]
        world.fake.objects[base + name] = (scratch / world.victim / manifest.path).read_bytes()
        world.fake.objects[base + f"step_{step:09d}.manifest.json"] = (
            scratch / world.victim / "last" / f"step_{step:09d}.manifest.json"
        ).read_bytes()


@dataclass(frozen=True)
class Attack:
    name: str
    setup: Callable[[World], None]
    policy: str
    keyed: bool
    kind: str = "last"
    remote: bool = False
    lineage: bool = True
    accept: int | None = None
    reasons: frozenset[str] = frozenset({"rolled_back"})
    weaken: bool = False  # the caller asks for ``any`` although the environment says otherwise
    env: tuple[tuple[str, str], ...] = ()  # operator settings applied before the call


ATTACKS = [
    Attack("rolled back, pointer left", attack_rolled_back, "authenticated", True),
    Attack("rolled back, pointer deleted", attack_rolled_back_no_pointer, "authenticated", True),
    Attack(
        "forged untagged coherent directory",
        attack_forged_untagged,
        "authenticated",
        True,
        reasons=frozenset({"auth_missing", "rolled_back", "untrusted", "store_mismatch"}),
    ),
    Attack("store ahead of the directory", attack_store_ahead, "anchored", False),
    Attack(
        "best checkpoints rolled back",
        lambda w: remove_steps(w, (3, 2), "best"),
        "authenticated",
        True,
        kind="best",
    ),
    Attack("an ancestor's directory rolled back", attack_rolled_back, "authenticated", True),
    Attack(
        "remote rolled back",
        attack_remote_rolled_back,
        "authenticated",
        True,
        remote=True,
    ),
    Attack(
        "remote forged",
        attack_remote_forged,
        "authenticated",
        True,
        remote=True,
        reasons=frozenset({"auth_missing", "untrusted", "rolled_back", "store_mismatch"}),
    ),
    Attack("accept override too high", attack_rolled_back, "authenticated", True, accept=2),
    Attack(
        "weakened per call: untagged after a key was configured",
        lambda w: None,
        "anchored",
        False,
        reasons=frozenset({"auth_missing", "untrusted", "rolled_back"}),
        weaken=True,
        env=(("CVPILOT_CKPT_KEY", KEY), ("CVPILOT_CKPT_REQUIRE_TRUST", "authenticated")),
    ),
    Attack(
        "weakened per call: remote untagged after a key was configured",
        lambda w: shutil.rmtree(w.ckpt / w.victim),
        "anchored",
        False,
        remote=True,
        reasons=frozenset({"auth_missing", "untrusted", "rolled_back"}),
        weaken=True,
        env=(("CVPILOT_CKPT_KEY", KEY), ("CVPILOT_CKPT_REQUIRE_TRUST", "authenticated")),
    ),
    Attack(
        "every checkpoint deleted",
        lambda w: remove_steps(w, (3, 2, 1)),
        "authenticated",
        True,
        reasons=frozenset({"rolled_back", "checkpoint_missing"}),
    ),
    Attack(
        "directory replaced by an empty one",
        lambda w: (shutil.rmtree(w.ckpt / w.victim), (w.ckpt / w.victim).mkdir()),
        "authenticated",
        True,
        reasons=frozenset({"rolled_back", "checkpoint_missing"}),
    ),
    Attack(
        "operator floor lowered per call",
        lambda w: remove_steps(w, (3, 2)),
        "authenticated",
        True,
        accept=1,
        env=(("CVPILOT_CKPT_MIN_STEP", "3"),),
    ),
]


Outcome = str  # "served" | "refused"


def judge(call: Callable[[], Any], reasons: frozenset[str]) -> Outcome:
    try:
        result = call()
    except CheckpointUntrustedError as exc:
        assert exc.reason in reasons | {"untrusted"}, exc.reason
        return "refused"
    except CheckpointCorruptError as exc:
        assert exc.reason in reasons | {"auth_missing", "untrusted"}, exc.reason
        return "refused"
    return "refused" if result is None else "served"


def weak(attack: Attack) -> str | None:
    return "any" if attack.weaken else None


def entry_manager_restore(world: World, attack: Attack) -> Outcome:
    mgr = CheckpointManager(
        world.child, dir=world.ckpt, every_n_steps=1, install_signal_handlers=False,
        remote=URI if attack.remote else None, remote_store=world.fake,
        remote_sleep=lambda s: None,
    )  # fmt: skip
    return judge(
        lambda: mgr.restore(  # type: ignore[arg-type]
            world.kind, accept_rollback_to=attack.accept, require_trust=weak(attack)
        ),
        attack.reasons,
    )


def entry_manager_restore_run_ids(world: World, attack: Attack) -> Outcome:
    mgr = CheckpointManager(
        world.child, dir=world.ckpt, every_n_steps=1, install_signal_handlers=False,
        remote=URI if attack.remote else None, remote_store=world.fake,
        remote_sleep=lambda s: None,
    )  # fmt: skip
    return judge(
        lambda: mgr.restore(  # type: ignore[arg-type]
            world.kind,
            run_ids=[world.victim],
            accept_rollback_to=attack.accept,
            require_trust=weak(attack),
        ),
        attack.reasons,
    )


def entry_manager_restore_discover(world: World, attack: Attack) -> Outcome:
    mgr = CheckpointManager(
        world.child, dir=world.tmp / "elsewhere", every_n_steps=1, install_signal_handlers=False,
        remote=URI, remote_store=world.fake, remote_sleep=lambda s: None,
    )  # fmt: skip
    return judge(
        lambda: mgr.restore(  # type: ignore[arg-type]
            world.kind,
            run_ids=[],
            discover=True,
            accept_rollback_to=attack.accept,
            require_trust=weak(attack),
        ),
        attack.reasons,
    )


def entry_restore_latest_valid(world: World, attack: Attack) -> Outcome:
    mgr = CheckpointManager(
        world.child, dir=world.ckpt, every_n_steps=1, install_signal_handlers=False,
        remote=URI if attack.remote else None, remote_store=world.fake,
        remote_sleep=lambda s: None,
    )  # fmt: skip
    return judge(lambda: mgr.restore_latest_valid(require_trust=weak(attack)), attack.reasons)  # type: ignore[arg-type]


def entry_api_resume_path(world: World, attack: Attack) -> Outcome:
    return judge(
        lambda: cvpilot.resume_path(
            "p",
            accept_rollback_to=attack.accept,
            require_trust=weak(attack),  # type: ignore[arg-type]
        ),
        attack.reasons,
    )


def entry_ckpt_resume_path(world: World, attack: Attack) -> Outcome:
    return judge(
        lambda: resume.resume_path(
            "p", checkpoint_dir=world.ckpt, store_dir=world.store,
            accept_rollback_to=attack.accept, require_trust=weak(attack),  # type: ignore[arg-type]
        ),
        attack.reasons,
    )  # fmt: skip


def entry_find_resume_point(world: World, attack: Attack) -> Outcome:
    return judge(
        lambda: resume.find_resume_point(
            "p", checkpoint_dir=world.ckpt, store_dir=world.store, kind=world.kind,  # type: ignore[arg-type]
            accept_rollback_to=attack.accept, force=True, require_trust=weak(attack),  # type: ignore[arg-type]
        ),
        attack.reasons,
    )  # fmt: skip


def direct_dir(world: World, attack: Attack) -> RunCheckpoints:
    opened = runs.open_anchor("p", world.store)
    assert opened is not None
    from cvpilot.ckpt import auth

    return RunCheckpoints(
        world.ckpt, world.victim, anchor=opened[0], key=auth.load_key(),
        require_trust=attack.policy,  # type: ignore[arg-type]
    )  # fmt: skip


def entry_iter_valid(world: World, attack: Attack) -> Outcome:
    from cvpilot.ckpt.directory import AdmitContext, gate_failure

    rc = direct_dir(world, attack)
    got = list(rc.iter_valid(world.kind, ctx=AdmitContext(accept_rollback_to=attack.accept)))  # type: ignore[arg-type]
    if got:
        return "served"
    assert gate_failure(rc.skipped, attack.policy) is not None or not rc.skipped
    return "refused"


def entry_inspect(world: World, attack: Attack) -> Outcome:
    entries = direct_dir(world, attack).inspect(world.kind)  # type: ignore[arg-type]
    return "served" if any(e.status == "valid" for e in entries) else "refused"


def entry_prune(world: World, attack: Attack) -> Outcome:
    before = sorted(p.name for p in world.folder().glob("*")) if world.folder().exists() else []
    direct_dir(world, attack).prune(world.kind, keep=1)  # type: ignore[arg-type]
    after = sorted(p.name for p in world.folder().glob("*")) if world.folder().exists() else []
    return "refused" if before == after else "served"  # deleting means it trusted something


def entry_fetch(world: World, attack: Attack) -> Outcome:
    assert world.fake is not None
    from cvpilot.ckpt.directory import AdmitContext

    mirror = Mirror(parse_remote_uri(URI), world.fake, sleep=lambda s: None)
    rc = RunCheckpoints(
        world.tmp / "fresh", world.victim, create=True,
        anchor=(runs.open_anchor("p", world.store) or (None,))[0],
        key=__import__("cvpilot.ckpt.auth", fromlist=["x"]).load_key(),
        require_trust=attack.policy,  # type: ignore[arg-type]
    )  # fmt: skip
    steps = mirror.list_steps(world.victim, "last") or [3]
    ctx = AdmitContext(present=frozenset(steps), accept_rollback_to=attack.accept)

    def call() -> Any:
        return mirror.fetch(rc, "last", steps[0], ctx) or None

    return judge(call, attack.reasons)


def entry_cli_verify(world: World, attack: Attack, invoke: Invoke) -> Outcome:
    args = ["ckpt", "verify", str(world.ckpt), "--run", world.victim, "--kind", world.kind]
    trust = "any" if attack.weaken else attack.policy
    res = invoke(*args, "--project", "p", "--require-trust", trust)
    return "served" if res.code == 0 else "refused"


def entry_cli_prune(world: World, attack: Attack, invoke: Invoke) -> Outcome:
    before = sorted(p.name for p in world.folder().glob("*")) if world.folder().exists() else []
    invoke(
        "ckpt", "prune", str(world.ckpt), "--run", world.victim, "--keep-last", "1",
        "--keep-best", "1", "--project", "p",
        "--require-trust", "any" if attack.weaken else attack.policy,
    )  # fmt: skip
    after = sorted(p.name for p in world.folder().glob("*")) if world.folder().exists() else []
    return "refused" if before == after else "served"


ENTRIES: dict[str, tuple[Callable[..., Outcome], set[str]]] = {
    # name: (adapter, attacks it does not apply to)
    "CheckpointManager.restore": (entry_manager_restore, set()),
    "CheckpointManager.restore(run_ids=)": (entry_manager_restore_run_ids, set()),
    "CheckpointManager.restore(discover=)": (
        entry_manager_restore_discover,
        {a.name for a in ATTACKS if not a.remote},
    ),
    "CheckpointManager.restore_latest_valid": (
        entry_restore_latest_valid,
        {"best checkpoints rolled back", "accept override too high"},
    ),
    "cvpilot.resume_path": (
        entry_api_resume_path,
        {"best checkpoints rolled back", "remote rolled back", "remote forged"},
    ),
    "cvpilot.ckpt.resume_path": (
        entry_ckpt_resume_path,
        {"best checkpoints rolled back", "remote rolled back", "remote forged"},
    ),
    "cvpilot.ckpt.find_resume_point": (
        entry_find_resume_point,
        {"remote rolled back", "remote forged"},
    ),
    "RunCheckpoints.iter_valid": (
        entry_iter_valid,
        {"remote rolled back", "remote forged", "accept override too high"},
    ),
    "RunCheckpoints.inspect": (
        entry_inspect,
        {"remote rolled back", "remote forged", "accept override too high"},
    ),
    "RunCheckpoints.prune": (
        entry_prune,
        {"remote rolled back", "remote forged", "accept override too high"},
    ),
    "Mirror.fetch": (
        entry_fetch,
        {a.name for a in ATTACKS if not a.remote},
    ),
    "cli ckpt verify": (
        entry_cli_verify,
        {"remote rolled back", "remote forged", "accept override too high"},
    ),
    "cli ckpt prune": (
        entry_cli_prune,
        {"remote rolled back", "remote forged", "accept override too high"},
    ),
}

LOW_LEVEL = {
    "RunCheckpoints.iter_valid",
    "RunCheckpoints.inspect",
    "RunCheckpoints.prune",
    "Mirror.fetch",
}
WEAK_LOCAL = "weakened per call: untagged after a key was configured"
WEAK_REMOTE = "weakened per call: remote untagged after a key was configured"
FLOOR = "operator floor lowered per call"
for _name, (_adapter, _skip) in list(ENTRIES.items()):
    extra: set[str] = set()
    if _name in LOW_LEVEL:  # constructed with an explicit policy: there is no environment to weaken
        extra |= {WEAK_LOCAL, WEAK_REMOTE, FLOOR}
    if _name.startswith("cli ckpt"):
        extra |= {FLOOR}  # the command line has no rollback acceptance and no floor of its own
    if _name != "CheckpointManager.restore" and not _name.startswith("CheckpointManager."):
        extra |= {WEAK_REMOTE}
    ENTRIES[_name] = (_adapter, _skip | extra)


def test_every_gated_entry_point_has_an_adapter() -> None:
    names = {n.split("(")[0] for n in ENTRIES}
    write = {n for n, why in GATED.items() if why.startswith("write")}
    gated = {n for n in GATED if n not in {"cli ckpt list", "RunCheckpoints.install"} | write}
    assert gated <= names, gated - names


PAIRS = [
    pytest.param(entry, attack, id=f"{entry} / {attack.name}")
    for entry, (_, skipped_for) in ENTRIES.items()
    for attack in ATTACKS
    if attack.name not in skipped_for
]


def test_the_matrix_is_wide_enough() -> None:
    assert len(PAIRS) >= 70
    assert {p.values[0] for p in PAIRS} == set(ENTRIES)  # type: ignore[union-attr]


@pytest.mark.parametrize(("entry", "attack"), PAIRS)
def test_the_same_attack_gets_the_same_answer_from_every_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_root: Path,
    invoke: Invoke,
    entry: str,
    attack: Attack,
) -> None:
    adapter, _ = ENTRIES[entry]
    world = build(
        tmp_path / "w", monkeypatch, policy=attack.policy, keyed=attack.keyed,
        remote=attack.remote, lineage=True,
    )  # fmt: skip
    world.kind = attack.kind
    attack.setup(world)
    if attack.name == "an ancestor's directory rolled back":
        pass  # the child's lineage reaches the victim as an ancestor
    for name, value in attack.env:
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(world.store))
    params = inspect.signature(adapter).parameters
    outcome = adapter(world, attack, invoke) if "invoke" in params else adapter(world, attack)
    assert outcome == "refused", f"{entry} served an attacked checkpoint: {attack.name}"
    world.child.finish()


@pytest.mark.parametrize(
    "entry",
    [e for e in ENTRIES if e not in {"Mirror.fetch", "CheckpointManager.restore(discover=)"}],
)
def test_the_control_group_every_gated_entry_point_serves_an_honest_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_root: Path,
    invoke: Invoke,
    entry: str,
) -> None:
    """If the adapters could not succeed, 'refused' everywhere would prove nothing."""
    adapter, _ = ENTRIES[entry]
    attack = Attack("honest", lambda w: None, "authenticated", True)
    world = build(tmp_path / "w", monkeypatch, policy="authenticated", keyed=True)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(world.store))
    params = inspect.signature(adapter).parameters
    outcome = adapter(world, attack, invoke) if "invoke" in params else adapter(world, attack)
    expected = "refused" if "prune" in entry else "served"
    if "prune" in entry:  # an honest directory is pruned: that is what "served" means for prune
        expected = "served"
    assert outcome == expected, entry
    world.child.finish()
    assert json  # keep the import used


def test_the_control_group_for_discovery_serves_an_honest_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store_root: Path
) -> None:
    attack = Attack("honest", lambda w: None, "authenticated", True, remote=True)
    world = build(tmp_path / "w", monkeypatch, policy="authenticated", keyed=True, remote=True)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(world.store))
    assert entry_manager_restore_discover(world, attack) == "served"
    world.child.finish()
