"""The host allow-list and endpoint checks against parsing differentials and bypasses."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fake_forwarders import LEAK_MARKER
from support import batch, meta_repro

from cvpilot.core.errors import CvpilotError
from cvpilot.forward import _common
from cvpilot.forward.mlflow import MlflowForwarder
from cvpilot.forward.wandb import DEFAULT_URL, WandbForwarder
from cvpilot.settings import SettingsError, load_settings

ALLOWED = ["allowed.example"]
HOSTILE = [
    "https://allowed.example\\@evil.example",
    "https://allowed.example.evil.example",
    "https://evil-allowed.example",
    "https://sub.allowed.example",
    "https://evil.example/@allowed.example",
    "https://evil.example/allowed.example",
    "https://evil.example?x=@allowed.example",
    "https://allowed.example:8443",
    "https://allowed.example:0",
    "https://allowed.example:99999",
    "https://allowed.example:abc",
    "https://[::1]/",
    "https://[2001:db8::1]:443/",
    "https://ａllowed.example",  # fullwidth letter
    "https://allowed.еxample",  # Cyrillic lookalike
    "https://xn--allowed-9ya.example",  # punycode lookalike, not the listed host
    "https://allowed.example%2eevil.example",
    "https://allowed%2eexample",
    "https://allowed.example evil.example",
    "https://allowed.example\t",
    "https://allowed..example",
    "https://.allowed.example",
    "https://allowed.example/?x=1",
    "https://allowed.example/#frag",
    "https://user@allowed.example",
    "https://:pw@allowed.example",
    "https://@allowed.example",
    "https://allowed.example@evil.example",
    "https://",
    "https:///path",
    "https:allowed.example",
    "//allowed.example",
    "allowed.example",
    "http://allowed.example",
]


@pytest.mark.parametrize("url", HOSTILE)
def test_hostile_urls_are_refused_with_an_allow_list(url: str) -> None:
    with pytest.raises(CvpilotError) as exc:
        _common.check_endpoint(url, allow_insecure=False, what="X", allowed_hosts=ALLOWED)
    assert "evil.example" not in str(exc.value) and "pw" not in str(exc.value)


@pytest.mark.parametrize("url", HOSTILE)
def test_the_structural_rules_hold_without_an_allow_list_too(url: str) -> None:
    structurally_bad = {
        "https://allowed.example\\@evil.example", "https://[::1]/", "https://[2001:db8::1]:443/",
        "https://ａllowed.example", "https://allowed.еxample",
        "https://allowed.example%2eevil.example", "https://allowed%2eexample",
        "https://allowed.example evil.example", "https://allowed.example\t",
        "https://allowed..example", "https://.allowed.example", "https://allowed.example/?x=1",
        "https://allowed.example/#frag", "https://user@allowed.example",
        "https://:pw@allowed.example", "https://@allowed.example",
        "https://allowed.example@evil.example", "https://", "https:///path",
        "https:allowed.example", "//allowed.example", "allowed.example",
        "http://allowed.example", "https://evil.example/@allowed.example",
        "https://evil.example?x=@allowed.example", "https://allowed.example:0",
        "https://allowed.example:99999", "https://allowed.example:abc",
        "https://allowed.example:8443",
    }  # fmt: skip
    if url not in structurally_bad:
        return
    with pytest.raises(CvpilotError):
        _common.check_endpoint(url, allow_insecure=False, what="X")


def test_the_allow_list_matches_whole_hosts_only() -> None:
    for suffix_trick in ("https://evil-allowed.example", "https://allowed.example.evil.test",
                         "https://sub.allowed.example", "https://allowed.examples"):  # fmt: skip
        with pytest.raises(CvpilotError, match="CVPILOT_FORWARD_HOSTS"):
            _common.check_endpoint(
                suffix_trick, allow_insecure=False, what="X", allowed_hosts=ALLOWED
            )


def test_a_host_is_normalised_and_the_normalised_url_is_returned() -> None:
    def pinned(url: str, hosts: list[str] | None = None) -> str | None:
        return _common.check_endpoint(
            url, allow_insecure=False, what="X", allowed_hosts=hosts or ALLOWED
        )

    assert pinned("https://ALLOWED.Example") == "https://allowed.example"
    assert pinned("https://allowed.example.:443/p") == "https://allowed.example/p"
    assert pinned("https://allowed.example./") == "https://allowed.example/"
    assert pinned("https://allowed.example:8443/p", ["allowed.example:8443"]) == (
        "https://allowed.example:8443/p"
    )
    assert pinned("https://xn--allowed-9ya.example", ["xn--allowed-9ya.example"]) is not None
    assert _common.check_endpoint(None, allow_insecure=False, what="X") is None


def test_a_non_default_port_needs_an_explicit_host_port_entry() -> None:
    url = "https://allowed.example:8443/x"
    with pytest.raises(CvpilotError, match="port"):
        _common.check_endpoint(url, allow_insecure=False, what="X", allowed_hosts=ALLOWED)
    with pytest.raises(CvpilotError, match="port"):
        _common.check_endpoint(url, allow_insecure=False, what="X")  # no list: only 443
    with pytest.raises(CvpilotError):
        _common.check_endpoint(
            url, allow_insecure=False, what="X", allowed_hosts=["allowed.example:9443"]
        )
    _common.check_endpoint(
        url, allow_insecure=False, what="X", allowed_hosts=["allowed.example:8443"]
    )
    # an entry with a port does not open the default port
    with pytest.raises(CvpilotError, match="CVPILOT_FORWARD_HOSTS"):
        _common.check_endpoint(
            "https://allowed.example/", allow_insecure=False, what="X",
            allowed_hosts=["allowed.example:8443"],
        )  # fmt: skip
    # development may use any port on an http or https server it names
    assert (
        _common.check_endpoint("http://localhost:5000/", allow_insecure=True, what="X")
        == "http://localhost:5000/"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "*",
        "*.example",
        ",",
        "a,",
        ",a",
        "a,,b",
        "a:0",
        "a:70000",
        "a:x",
        "a:",
        ":443",
        "a..b",
        ".a",
        "a.",
        "https://a",
        "a/b",
        "a@b",
        "[::1]",
    ],
)
def test_bad_allow_list_entries_are_settings_errors(raw: str) -> None:
    with pytest.raises(SettingsError, match="CVPILOT_FORWARD_HOSTS"):
        load_settings({"CVPILOT_FORWARD_HOSTS": raw})


def test_allow_list_entries_may_carry_a_port_and_are_lower_cased() -> None:
    parsed = load_settings({"CVPILOT_FORWARD_HOSTS": "Tracker.Example:8443, api.wandb.ai"})
    assert parsed.forward_hosts == ("tracker.example:8443", "api.wandb.ai")
    assert load_settings({"CVPILOT_FORWARD_HOSTS": ""}).forward_hosts == ()  # empty means unset


# -- W&B: the endpoint is always pinned --


def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    calls: list[tuple[str, Any]] = []

    class Run:
        def log(self, data: dict[str, float], step: int | None = None) -> None:
            calls.append(("log", (dict(data), step)))

        def finish(self, exit_code: int = 0) -> None:
            calls.append(("finish", exit_code))

    module = types.SimpleNamespace(calls=calls, run=None)

    def init(**kw: Any) -> Run:
        calls.append(("init", kw))
        module.run = Run()
        return module.run  # type: ignore[no-any-return]

    module.init = init
    module.Settings = lambda **kw: ("settings", kw)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "WANDB_BASE_URL", "WANDB_INSECURE_DISABLE_SSL", "MLFLOW_TRACKING_URI",
        "MLFLOW_REGISTRY_URI", "MLFLOW_TRACKING_INSECURE_TLS", "WANDB_API_KEY",
        "MLFLOW_TRACKING_TOKEN", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE",
        "SSL_CERT_DIR", "MLFLOW_TRACKING_SERVER_CERT_PATH",
    ):  # fmt: skip
        monkeypatch.delenv(var, raising=False)


def settings_received(sdk: types.SimpleNamespace) -> dict[str, Any]:
    init = next(c[1] for c in sdk.calls if c[0] == "init")
    marker, kw = init["settings"]
    assert marker == "settings"
    return dict(kw)


def test_an_unset_endpoint_is_pinned_to_the_default_not_left_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_wandb(monkeypatch)
    adapter = WandbForwarder(allowed_hosts=["api.wandb.ai"])
    (meta, repro), _ = meta_repro(), None
    adapter.open(meta, repro)
    received = settings_received(sdk)
    assert received["base_url"] == DEFAULT_URL == "https://api.wandb.ai"


def test_the_pinned_settings_cover_every_network_relevant_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_wandb(monkeypatch)
    WandbForwarder().open(*meta_repro())
    received = settings_received(sdk)
    assert set(received) == {"init_timeout", "base_url"}  # nothing else is left to settings files
    assert received["base_url"].startswith("https://")


def test_a_validated_url_is_passed_normalised_and_unchanged_by_later_environment_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_BASE_URL", "https://WANDB.example.:443")
    adapter = WandbForwarder(allowed_hosts=["wandb.example"])
    monkeypatch.setenv("WANDB_BASE_URL", "https://evil.example")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "ca/evil.pem")
    adapter.open(*meta_repro())
    assert settings_received(sdk)["base_url"] == "https://wandb.example"


# -- MLflow: tracking and registry URIs --


def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    calls: list[tuple[str, Any]] = []

    class Client:
        def __init__(
            self, tracking_uri: str | None = None, registry_uri: str | None = None
        ) -> None:
            calls.append(("client", (tracking_uri, registry_uri)))

        def get_experiment_by_name(self, name: str) -> None:
            return None

        def create_experiment(self, name: str) -> str:
            return "e"

        def create_run(self, experiment_id: str, **kw: Any) -> Any:
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id="r"))

        def set_terminated(self, run_id: str, status: str) -> None:
            calls.append(("set_terminated", status))

    module = types.SimpleNamespace(
        MlflowClient=Client, entities=types.SimpleNamespace(Metric=lambda *a: a), calls=calls
    )
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return module


def test_mlflow_registry_uri_gets_the_same_checks_and_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = fake_mlflow(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://Tracker.example.")
    adapter = MlflowForwarder(allowed_hosts=["tracker.example"])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://evil.example")
    adapter.open(*meta_repro())
    assert sdk.calls[0] == ("client", ("https://tracker.example", "https://tracker.example"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracker.example")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "https://evil.example")
    with pytest.raises(CvpilotError, match="MLFLOW_REGISTRY_URI"):
        MlflowForwarder(allowed_hosts=["tracker.example"])
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "file:///tmp/registry")
    with pytest.raises(CvpilotError, match="MLFLOW_REGISTRY_URI"):
        MlflowForwarder()


# -- nothing re-reads the environment, nothing reads a credential --


class SpyEnviron(dict[str, str]):
    def __init__(self, base: Any) -> None:
        super().__init__(base)
        self.reads: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        self.reads.append(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> str:
        self.reads.append(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.reads.append(str(key))
        return super().__contains__(key)


@pytest.mark.parametrize("which", ["mlflow", "wandb"])
def test_construction_reads_only_endpoint_and_tls_variables_and_use_reads_nothing(
    monkeypatch: pytest.MonkeyPatch, which: str, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", LEAK_MARKER)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", LEAK_MARKER)
    spy = SpyEnviron(os.environ)
    monkeypatch.setattr(os, "environ", spy)
    if which == "mlflow":
        fake_mlflow(monkeypatch)
        adapter: Any = MlflowForwarder()
    else:
        fake_wandb(monkeypatch)
        adapter = WandbForwarder()
    allowed = {
        "MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI", "MLFLOW_TRACKING_INSECURE_TLS",
        "WANDB_BASE_URL", "WANDB_INSECURE_DISABLE_SSL", *_common.CA_OVERRIDE_VARIABLES,
    }  # fmt: skip
    assert set(spy.reads) <= allowed
    assert "WANDB_API_KEY" not in spy.reads and "MLFLOW_TRACKING_TOKEN" not in spy.reads
    spy.reads.clear()
    meta, repro = meta_repro()
    adapter.open(meta, repro)
    if which == "wandb":
        adapter.log_points(batch(meta, 1))
    adapter.close("finished")
    assert spy.reads == []  # open, log and close never touch the environment
