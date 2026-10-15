"""CheckpointManifest, CheckpointPointer and the remote location grammar."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from cvpilot.core.hashing import checkpoint_chain, pointer_digest
from cvpilot.core.schema import GENESIS_DIGEST, CheckpointManifest, CheckpointPointer
from cvpilot.core.uri import parse_remote_uri

RUN = "01JBXQ5S7H8E9K2M3N4P5Q6R7S"
SHA = "a" * 64


def manifest_fields(**over: Any) -> dict[str, Any]:
    chain = checkpoint_chain(GENESIS_DIGEST, RUN, "last", 7, SHA, 10)
    fields: dict[str, Any] = {
        "run_id": RUN,
        "step": 7,
        "kind": "last",
        "path": "last/step_000000007.safetensors",
        "sha256": SHA,
        "size_bytes": 10,
        "format": "safetensors",
        "versions": {"cvpilot": "0.1.0", "python": "3.12.1"},
        "dataset_hash": "merkle:" + SHA,
        "created_at": datetime(2026, 10, 15, tzinfo=timezone.utc),
        "prev_manifest_sha256": GENESIS_DIGEST,
        "prev_chain_sha256": GENESIS_DIGEST,
        "chain_sha256": chain,
    }
    fields.update(over)
    return fields


def test_a_valid_manifest_round_trips_through_json() -> None:
    manifest = CheckpointManifest(**manifest_fields())
    again = CheckpointManifest.model_validate_json(manifest.model_dump_json())
    assert again == manifest


@pytest.mark.parametrize(
    "over",
    [
        {"path": "last/step_000000008.safetensors"},  # step disagrees with the path
        {"path": "best/step_000000007.safetensors"},  # kind disagrees
        {"path": "last/step_000000007.pt"},  # suffix disagrees with the format
        {"path": "last/../step_000000007.safetensors"},
        {"path": "/etc/passwd"},
        {"path": "last/step_7.safetensors"},  # not zero-padded
        {"sha256": "A" * 64},
        {"size_bytes": 0},
        {"kind": "best", "path": "best/step_000000007.safetensors"},  # best needs a metric
        {"metric_key": "mAP50"},  # key without value
        {"metric_key": "m", "metric_value": float("nan")},
        {"versions": {f"k{i}": "v" for i in range(17)}},
        {"dataset_hash": "plain"},
        {"git_commit": "xyz"},
        {"uri": "https://example.com/x"},
        {"extra": 1},
    ],
)
def test_invalid_manifests_are_rejected(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CheckpointManifest(**manifest_fields(**over))


def test_best_manifest_with_metric_is_accepted() -> None:
    fields = manifest_fields(
        kind="best",
        path="best/step_000000007.safetensors",
        metric_key="mAP50",
        metric_value=0.5,
        metric_mode="max",
    )
    assert CheckpointManifest(**fields).metric_value == 0.5


def test_chain_commits_to_every_field() -> None:
    base = checkpoint_chain(GENESIS_DIGEST, RUN, "last", 1, SHA, 10)
    assert base != checkpoint_chain("1" * 64, RUN, "last", 1, SHA, 10)
    assert base != checkpoint_chain(GENESIS_DIGEST, RUN, "best", 1, SHA, 10)
    assert base != checkpoint_chain(GENESIS_DIGEST, RUN, "last", 2, SHA, 10)
    assert base != checkpoint_chain(GENESIS_DIGEST, RUN, "last", 1, "b" * 64, 10)
    assert base != checkpoint_chain(GENESIS_DIGEST, RUN, "last", 1, SHA, 11)


def test_pointer_validates_its_manifest_name() -> None:
    digest = pointer_digest(RUN, "last", 1, "last/step_000000001.manifest.json", SHA, SHA)
    fields = {
        "run_id": RUN,
        "kind": "last",
        "step": 1,
        "manifest": "last/step_000000001.manifest.json",
        "manifest_sha256": SHA,
        "chain_sha256": SHA,
        "pointer_sha256": digest,
    }
    assert CheckpointPointer(**fields).step == 1
    with pytest.raises(ValidationError):
        CheckpointPointer(**{**fields, "manifest": "../../etc/passwd"})


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket-name/runs/exp1",
        "gs://my.bucket/a",
        "az://container/x/y",
        "memory://ckpt/a",
        "s3://bucket-name",
        "s3://bucket-name/",
    ],
)
def test_valid_remote_locations(uri: str) -> None:
    location = parse_remote_uri(uri)
    assert location.url.startswith(uri.split("://")[0])


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "ftp://host/x",
        "http://bucket-name/x",
        "s3://user:secret@bucket-name/x",
        "s3://bucket-name/x?X-Amz-Signature=abc",
        "s3://bucket-name/x#frag",
        "s3://bucket-name/%2e%2e/x",
        "s3://bucket-name/../x",
        "s3://bucket-name/./x",
        "s3://bucket-name//x",
        "s3://bucket-name/a b",
        "s3://bucket-name/a\\b",
        "s3://bucket-name/é",
        "s3://B/x",
        "s3:///x",
        "memory://",
        "s3://bucket-name/" + "a/" * 40,
        "s3://bucket-name/" + "a" * 2000,
        "s3://bucket-name/x\n",
        "file:///etc",
    ],
)
def test_hostile_remote_locations_are_rejected(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_remote_uri(uri)


def test_remote_errors_do_not_echo_a_secret() -> None:
    with pytest.raises(ValueError) as info:
        parse_remote_uri("s3://user:supersecretpassword-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx@b/x")
    assert "supersecretpassword" not in str(info.value)


def test_remote_key_joins_bucket_prefix_and_parts() -> None:
    location = parse_remote_uri("s3://bucket-name/runs/exp")
    assert location.key("a", "b") == "bucket-name/runs/exp/a/b"
    assert parse_remote_uri("memory://x/y").key("z") == "x/y/z"
