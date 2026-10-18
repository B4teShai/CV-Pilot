"""The strict ``.pt`` reader accepts the layout the framework's own writer produces, and still
refuses every deviation from it. The archives come from ``zip_support.framework_archive``, which
a marked test (``tests/smoke/test_real_checkpoint_files.py``) compares byte for byte with files
the installed framework wrote."""

from __future__ import annotations

import pickle
import struct
from collections import OrderedDict
from typing import Any

import pytest
from zip_support import framework_archive

from cvpilot.ckpt import serializers
from cvpilot.core.errors import CheckpointCorruptError

GOOD = pickle.dumps(OrderedDict(epoch=3, lr=[0.1]), protocol=2)
MEMBERS = {
    "archive/data.pkl": GOOD,
    "archive/.format_version": b"1",
    "archive/byteorder": b"little",
    "archive/data/0": bytes(range(32)),
    "archive/version": b"3\n",
}


def scan(archive: bytes) -> int:
    return serializers.scan_pt_archive(archive)


@pytest.mark.parametrize(
    "shape",
    [
        {},
        {"descriptor": False},
        {"zip64_end": False},
        {"descriptor": False, "zip64_end": False},
        {"deflate": True},
        {"descriptor_for": {"archive/data.pkl", "archive/data/0"}},
    ],
    ids=["framework", "no-descriptor", "no-zip64-end", "plain", "deflated", "mixed-descriptors"],
)
def test_every_layout_the_framework_or_a_plain_writer_produces_is_accepted(
    shape: dict[str, Any],
) -> None:
    assert scan(framework_archive(MEMBERS, **shape)) == 32


def test_a_framework_layout_archive_reaches_the_loader_with_weights_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    seen: list[dict[str, Any]] = []
    module = types.ModuleType("to" + "rch")
    module.load = lambda handle, **kw: (seen.append(kw), {"epoch": 3})[1]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "to" + "rch", module)
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    state = serializers.get_serializer("pt").loads(framework_archive(MEMBERS))
    assert state == {"epoch": 3} and seen == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize(
    "shape",
    [
        {"descriptor_crc": 1},
        {"descriptor_sizes": (1, 1)},
        {"descriptor_sizes": (2**31, 2**31)},
        {"local_sizes_in_header": True},
        {"drop_descriptor_signature": True},
        {"zip64_end_offset_delta": 1},
        {"zip64_extensible": b"\x00" * 8},
        {"locator_offset_delta": 4},
        {"trailing": b"\x00"},
    ],
    ids=[
        "descriptor-crc", "descriptor-sizes", "descriptor-huge", "sizes-in-local-and-descriptor",
        "descriptor-without-signature", "zip64-offset-differs", "zip64-extensible-data",
        "locator-points-elsewhere", "trailing-bytes",
    ],
)  # fmt: skip
def test_every_deviation_from_the_framework_layout_is_refused(shape: dict[str, Any]) -> None:
    with pytest.raises(CheckpointCorruptError) as info:
        scan(framework_archive(MEMBERS, **shape))
    assert info.value.reason == "unsafe_archive"


def patched(archive: bytes, offset: int, fmt: str, *values: int) -> bytes:
    data = bytearray(archive)
    struct.pack_into(fmt, data, offset, *values)
    return bytes(data)


def test_a_zip64_end_record_that_disagrees_with_the_classic_one_is_refused() -> None:
    archive = framework_archive(MEMBERS)
    at = archive.rfind(b"PK\x06\x06")
    for field, fmt, value in (
        (16, "<I", 1),  # this disk
        (20, "<I", 1),  # directory disk
        (24, "<Q", 6),  # entries on this disk
        (32, "<Q", 6),  # entries
        (40, "<Q", 1),  # directory size
        (4, "<Q", 45),  # record size
    ):
        with pytest.raises(CheckpointCorruptError):
            scan(patched(archive, at + field, fmt, value))


def test_a_locator_for_another_disk_or_disk_count_is_refused() -> None:
    archive = framework_archive(MEMBERS)
    at = archive.rfind(b"PK\x06\x07")
    for field, value in ((4, 1), (16, 2)):
        with pytest.raises(CheckpointCorruptError):
            scan(patched(archive, at + field, "<I", value))


def test_a_data_descriptor_cannot_hide_bytes_between_members() -> None:
    """A member whose flag says descriptor but whose descriptor is 24 bytes (zip64) leaves a gap."""
    archive = framework_archive(MEMBERS, descriptor_sizes=(len(GOOD), len(GOOD)))
    with pytest.raises(CheckpointCorruptError):
        scan(archive.replace(b"PK\x07\x08", b"PK\x07\x08\x00\x00\x00\x00\x00\x00\x00\x00", 1))
