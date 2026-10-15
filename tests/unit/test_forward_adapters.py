"""MLflow, W&B and Trackio adapters against injected fake SDK modules (no network, no SDK)."""

from __future__ import annotations

import ast
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fake_forwarders import LEAK_MARKER
from support import batch, meta_repro

from cvpilot.core.errors import CvpilotError, MissingExtraError
from cvpilot.core.schema import MetricPoint
from cvpilot.forward import _common
from cvpilot.forward.mlflow import MAX_BATCH_METRICS, MlflowForwarder
from cvpilot.forward.trackio import TrackioForwarder
from cvpilot.forward.wandb import WandbForwarder

SRC = Path(__file__).resolve().parents[2] / "src" / "cvpilot" / "forward"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_INSECURE_TLS",
        "WANDB_BASE_URL",
        "WANDB_INSECURE_DISABLE_SSL",
    ):
        monkeypatch.delenv(var, raising=False)


def points(n: int = 3, step: int = 1) -> tuple[Any, list[MetricPoint]]:
    meta, repro = meta_repro()
    return (meta, repro), batch(meta, step, n)


# -- fake SDKs -------------------------------------------------------------------------------


def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    calls: list[tuple[str, Any]] = []

    class Metric:
        def __init__(self, key: str, value: float, timestamp: int, step: int) -> None:
            self.key, self.value, self.timestamp, self.step = key, value, timestamp, step

    class Client:
        def __init__(self) -> None:
            calls.append(("client", None))

        def get_experiment_by_name(self, name: str) -> None:
            calls.append(("get_experiment", name))
            return None

        def create_experiment(self, name: str) -> str:
            calls.append(("create_experiment", name))
            return "exp1"

        def create_run(self, experiment_id: str, **kw: Any) -> Any:
            calls.append(("create_run", (experiment_id, kw)))
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id="mlrun1"))

        def log_batch(self, run_id: str, metrics: Sequence[Metric] = ()) -> None:
            calls.append(("log_batch", (run_id, list(metrics))))

        def set_terminated(self, run_id: str, status: str) -> None:
            calls.append(("set_terminated", (run_id, status)))

    module = types.SimpleNamespace(
        MlflowClient=Client, entities=types.SimpleNamespace(Metric=Metric), calls=calls
    )
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return module


def fake_wandb(
    monkeypatch: pytest.MonkeyPatch, *, existing_run: bool = False
) -> types.SimpleNamespace:
    calls: list[tuple[str, Any]] = []

    class Run:
        def log(self, data: dict[str, float], step: int | None = None) -> None:
            calls.append(("log", (dict(data), step)))

        def finish(self, exit_code: int = 0) -> None:
            calls.append(("finish", exit_code))

    module = types.SimpleNamespace(calls=calls, run=Run() if existing_run else None)

    def init(**kw: Any) -> Run:
        calls.append(("init", kw))
        if module.run is None:
            module.run = Run()
        return module.run  # type: ignore[no-any-return]

    module.init = init
    module.Settings = lambda **kw: ("settings", kw)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def fake_trackio(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    calls: list[tuple[str, Any]] = []
    module = types.SimpleNamespace(
        calls=calls,
        init=lambda **kw: calls.append(("init", kw)),
        log=lambda data, step=None: calls.append(("log", (dict(data), step))),
        finish=lambda: calls.append(("finish", None)),
    )
    monkeypatch.setitem(sys.modules, "trackio", module)
    return module


# -- MLflow ----------------------------------------------------------------------------------


def test_mlflow_creates_the_run_and_forwards_metrics_without_config_or_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_mlflow(monkeypatch)
    (meta, repro), pts = points()
    adapter = MlflowForwarder()
    adapter.open(meta, repro)
    adapter.log_points(pts)
    adapter.close("finished")
    kinds = [c[0] for c in sdk.calls]
    assert kinds == [
        "client",
        "get_experiment",
        "create_experiment",
        "create_run",
        "log_batch",
        "set_terminated",
    ]
    _, (exp_id, kw) = sdk.calls[3]
    assert exp_id == "exp1" and kw["run_name"] == meta.name
    assert kw["tags"]["cvpilot.run_id"] == meta.run_id
    flat = repr(kw)
    assert repro.command_line not in flat and repro.dataset_uri not in flat and "config" not in flat
    _, (run_id, metrics) = sdk.calls[4]
    assert run_id == "mlrun1" and [(m.key, m.step, m.value) for m in metrics] == [
        (p.key, p.step, p.value) for p in pts
    ]
    assert sdk.calls[-1][1] == ("mlrun1", "FINISHED")


@pytest.mark.parametrize(
    ("status", "word"),
    [
        ("finished", "FINISHED"),
        ("failed", "FAILED"),
        ("preempted", "KILLED"),
        ("running", "KILLED"),
    ],
)
def test_mlflow_terminal_status_mapping(
    monkeypatch: pytest.MonkeyPatch, status: str, word: str
) -> None:
    sdk = fake_mlflow(monkeypatch)
    (meta, repro), _ = points()
    adapter = MlflowForwarder()
    adapter.open(meta, repro)
    adapter.close(status)  # type: ignore[arg-type]
    assert sdk.calls[-1][1][1] == word


def test_mlflow_batches_are_chunked_and_keys_sanitised(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = fake_mlflow(monkeypatch)
    (meta, repro), _ = points()
    adapter = MlflowForwarder()
    adapter.open(meta, repro)
    many = [
        MetricPoint(run_id=meta.run_id, key="val/AP:cls#1 x", step=1, value=1.0, wall_time=2.5)
        for _ in range(MAX_BATCH_METRICS + 5)
    ]
    adapter.log_points(many)
    sizes = [len(c[1][1]) for c in sdk.calls if c[0] == "log_batch"]
    assert sizes == [MAX_BATCH_METRICS, 5]
    first = next(c for c in sdk.calls if c[0] == "log_batch")[1][1][0]
    assert first.key == "val/AP_cls_1 x" and first.timestamp == 2500


def test_mlflow_without_the_sdk_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # import raises ImportError
    (meta, repro), _ = points()
    with pytest.raises(MissingExtraError, match=r"cvpilot\[mlflow\]"):
        MlflowForwarder().open(meta, repro)


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("MLFLOW_TRACKING_URI", "http://tracker.example:5000"),
        ("MLFLOW_TRACKING_URI", "https://user:pw@tracker.example"),
        ("MLFLOW_TRACKING_URI", "postgresql://user:pw@db/mlflow"),
        ("MLFLOW_TRACKING_URI", "https://tracker.example/\x01"),
        ("MLFLOW_TRACKING_INSECURE_TLS", "true"),
        ("MLFLOW_TRACKING_INSECURE_TLS", "1"),
    ],
)
def test_mlflow_refuses_insecure_or_credentialed_endpoints_without_echoing_them(
    monkeypatch: pytest.MonkeyPatch, env: str, value: str
) -> None:
    monkeypatch.setenv(env, value)
    with pytest.raises(CvpilotError) as exc:
        MlflowForwarder()
    assert value not in str(exc.value) and "pw" not in str(exc.value) and env in str(exc.value)


@pytest.mark.parametrize(
    "uri",
    ["https://tracker.example", "file:///tmp/mlruns", "sqlite:///m.db", "databricks", "./mlruns"],
)
def test_mlflow_accepts_secure_or_local_endpoints(
    monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    MlflowForwarder()


def test_dev_may_use_plain_http_but_never_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MlflowForwarder(allow_insecure=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://user:pw@localhost:5000")
    with pytest.raises(CvpilotError):
        MlflowForwarder(allow_insecure=True)


# -- W&B -------------------------------------------------------------------------------------


def test_wandb_logs_per_step_in_order_and_finishes_a_run_it_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_wandb(monkeypatch)
    (meta, repro), _ = points()
    adapter = WandbForwarder()
    adapter.open(meta, repro)
    adapter.log_points(batch(meta, 2, 2) + batch(meta, 1, 2))
    adapter.close("failed")
    init = sdk.calls[0][1]
    assert init["project"] == meta.project and init["name"] == meta.name
    assert init["id"] == meta.run_id.lower() and init["settings"][0] == "settings"
    assert "init_timeout" in init["settings"][1]
    assert repro.command_line not in repr(init)
    logs = [c[1] for c in sdk.calls if c[0] == "log"]
    assert [step for _, step in logs] == [1, 2] and logs[0][0]["k0"] == 1.0
    assert sdk.calls[-1] == ("finish", 1)


def test_wandb_counts_points_older_than_the_last_forwarded_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb(monkeypatch)
    (meta, repro), _ = points()
    adapter = WandbForwarder()
    adapter.open(meta, repro)
    adapter.log_points(batch(meta, 5, 2))
    adapter.log_points(batch(meta, 3, 2))
    assert adapter.dropped_points == 2


def test_wandb_does_not_finish_a_run_it_did_not_create(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = fake_wandb(monkeypatch, existing_run=True)
    (meta, repro), _ = points()
    adapter = WandbForwarder()
    adapter.open(meta, repro)
    adapter.close("finished")
    assert [c[0] for c in sdk.calls].count("finish") == 0


def test_wandb_without_the_sdk_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    (meta, repro), _ = points()
    with pytest.raises(MissingExtraError, match=r"cvpilot\[wandb\]"):
        WandbForwarder().open(meta, repro)


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("WANDB_BASE_URL", "http://wandb.example"),
        ("WANDB_BASE_URL", "https://key:secret@wandb.example"),
        ("WANDB_INSECURE_DISABLE_SSL", "true"),
    ],
)
def test_wandb_refuses_insecure_or_credentialed_endpoints(
    monkeypatch: pytest.MonkeyPatch, env: str, value: str
) -> None:
    monkeypatch.setenv(env, value)
    with pytest.raises(CvpilotError) as exc:
        WandbForwarder()
    assert value not in str(exc.value)


# -- Trackio ---------------------------------------------------------------------------------


def test_trackio_forwards_per_step_and_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = fake_trackio(monkeypatch)
    (meta, repro), _ = points()
    adapter = TrackioForwarder()
    adapter.open(meta, repro)
    adapter.log_points(batch(meta, 2, 2) + batch(meta, 1, 2))
    adapter.close("finished")
    assert sdk.calls[0][1]["project"] == meta.project
    assert [c[1][1] for c in sdk.calls if c[0] == "log"] == [1, 2]
    assert sdk.calls[-1][0] == "finish"


def test_trackio_without_the_sdk_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "trackio", None)
    (meta, repro), _ = points()
    with pytest.raises(MissingExtraError, match=r"cvpilot\[trackio\]"):
        TrackioForwarder().open(meta, repro)


# -- structural guarantees ---------------------------------------------------------------------


def test_adapters_take_no_token_argument_and_read_no_credential_variable() -> None:
    import inspect

    for cls in (MlflowForwarder, WandbForwarder, TrackioForwarder):
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        assert params == {"allow_insecure"}, cls
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                upper = node.value.upper()
                if node.value == upper and node.value.replace("_", "").isalnum():
                    for word in ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "CREDENTIAL"):
                        assert word not in upper, f"{path.name} mentions {node.value}"


def test_adapters_never_disable_tls_verification() -> None:
    text = "\n".join(p.read_text() for p in SRC.glob("*.py")).lower()
    for needle in ("verify=false", "_create_unverified", "cert_none", "check_hostname = false"):
        assert needle not in text


def test_adapters_import_their_sdk_only_inside_functions() -> None:
    for name in ("mlflow", "wandb", "trackio"):
        tree = ast.parse((SRC / f"{name}.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in top:
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any(n.split(".")[0] == name for n in names), (name, names)


def test_check_endpoint_never_puts_the_url_in_its_message() -> None:
    with pytest.raises(CvpilotError) as exc:
        _common.check_endpoint(f"https://u:{LEAK_MARKER}@h", allow_insecure=False, what="X_URL")
    assert LEAK_MARKER not in str(exc.value)


def test_requirement_alternatives_are_accepted() -> None:
    from cvpilot.runs.forwarding import check_requires

    class Either:
        requires = ("definitely-missing|pydantic",)

    check_requires("either", Either)

    class Neither:
        requires = ("missing-a|missing-b",)

    with pytest.raises(MissingExtraError, match=r"cvpilot\[neither\]"):
        check_requires("neither", Neither)
