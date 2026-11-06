"""An externally hashed dataset: ``dataset_uri`` names it, ``dataset_hash`` is ``sha256:<hex>``.

A dataset registry identifies a version as ``<name>@<semver>`` plus the SHA-256 of its manifest
file. Both reach the run record and every checkpoint manifest unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import structlog
from pydantic import ValidationError
from test_ckpt_schema import manifest_fields

from cvpilot import api, runs
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import CheckpointManifest, Repro, check_dataset_hash
from cvpilot.logging import clear_request_context
from cvpilot.repro import capture_repro, external_dataset

DIGEST = "sha256:" + hashlib.sha256(b"manifest.parquet bytes").hexdigest()


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    yield
    clear_request_context()
    structlog.reset_defaults()


def test_init_records_the_name_and_manifest_digest_without_loss(tmp_path: Path) -> None:
    run = api.init("proj", task="obb", dataset_uri="fridge-sku@1.4.0", dataset_hash=DIGEST)
    run.finish()
    repro = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store").repro
    assert (repro.dataset_uri, repro.dataset_hash) == ("fridge-sku@1.4.0", DIGEST)


def test_a_url_or_path_name_is_kept_as_given_and_never_read() -> None:
    got = capture_repro(new_run_id(), "s3://bucket/datasets/fridge@1.4.0", {}, None,
                        argv=["t.py"], dataset_hash=DIGEST)  # fmt: skip
    assert got.repro.dataset_uri == "s3://bucket/datasets/fridge@1.4.0"
    assert got.repro.dataset_hash == DIGEST


@pytest.mark.parametrize(
    ("uri", "digest", "match"),
    [
        ("fridge@1.4.0", "sha256:" + "A" * 64, "sha256"),
        ("fridge@1.4.0", "sha256:abc", "sha256"),
        ("fridge@1.4.0", "md5:" + "a" * 32, "sha256"),
        ("fridge@1.4.0", "merkle:" + "a" * 64, "external dataset_hash"),
        ("fridge@1.4.0", "dcp:1.0", "external dataset_hash"),
        (None, DIGEST, "must name the dataset"),
        ("", DIGEST, "must name the dataset"),
        ("x" * 1025, DIGEST, "longer than"),
        ("fridge\n@1.4.0", DIGEST, "control characters"),
        ("https://user:pw@example.com/fridge@1.4.0", DIGEST, "credentials"),
        ("s3://key@bucket/fridge", DIGEST, "credentials"),
    ],
)
def test_invalid_external_identities_are_refused(uri: str | None, digest: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        external_dataset(uri, digest)


def test_init_refuses_an_invalid_identity_before_creating_a_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        api.init("proj", task="detection", dataset_uri="fridge@1.4.0", dataset_hash="sha256:x")
    assert api._active is None
    assert runs.list_runs("proj", store_dir=tmp_path / "store") == []


def test_every_accepted_form_and_the_records_that_carry_it() -> None:
    for value in ("dcp:2026.10.1", "merkle:" + "a" * 64, DIGEST):
        assert check_dataset_hash(value) == value
    for value in ("sha256:", "sha256:" + "a" * 63, "sha1:" + "a" * 40, "plain", "dcp:"):
        with pytest.raises(ValueError, match="sha256:<64 hex>"):
            check_dataset_hash(value)
    fields = manifest_fields(dataset_hash=DIGEST)
    assert CheckpointManifest.model_validate(fields).dataset_hash == DIGEST
    with pytest.raises(ValidationError, match="longer than 200"):
        CheckpointManifest.model_validate(manifest_fields(dataset_hash="dcp:" + "1" * 200))
    repro = Repro(
        run_id=new_run_id(), git_dirty=False, packages_lock_sha256="b" * 64,
        python_version="3.12.1", gpu_names=[], dataset_uri="fridge@1.4.0", dataset_hash=DIGEST,
        config={}, command_line="t.py",
    )  # fmt: skip
    assert repro.dataset_hash == DIGEST
