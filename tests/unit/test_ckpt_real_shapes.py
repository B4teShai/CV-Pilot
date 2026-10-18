"""The strict ``.pt`` reader accepts the layout the framework's own writer produces, and still
refuses every deviation from it. The archives come from ``zip_support.framework_archive``, which
a marked test (``tests/smoke/test_real_checkpoint_files.py``) compares byte for byte with files
the installed framework wrote."""

from __future__ import annotations

import pickle
import struct
from collections import OrderedDict
from pathlib import Path
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


DESCRIPTOR = "a data descriptor is missing or differs"
LOCAL = "a local header disagrees"
ZIP64 = "the zip64 end record disagrees"


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ({"descriptor_crc": 1}, DESCRIPTOR),
        ({"descriptor_sizes": (1, 1)}, DESCRIPTOR),
        ({"descriptor_sizes": (2**31, 2**31)}, DESCRIPTOR),
        ({"drop_descriptor_signature": True}, DESCRIPTOR),
        ({"local_sizes_in_header": True}, LOCAL),
        ({"deflate": True, "local_sizes_in_header": True}, LOCAL),
        ({"zip64_end_offset_delta": 1}, ZIP64),
        ({"zip64_extensible": b"\x00" * 8}, ZIP64),
        ({"locator_offset_delta": 4}, ZIP64),
        ({"trailing": b"\x00"}, "comment-free end-of-central-directory"),
        ({"zip64_end": False, "gap_after_first_member": b"\x00" * 8}, "overlap, leave gaps"),
        ({"zip64_end": False, "descriptor_crc": 1}, DESCRIPTOR),
        ({"zip64_end": False, "gap_before_end": b"\x00" * 4}, "between the members and the"),
        ({"gap_before_end": b"\x00" * 4}, "between the members and the|zip64 end record"),
        ({"zip64_end": False, "short_last_descriptor": True}, DESCRIPTOR),
        ({"lie_usize": 1 << 20}, "different compressed and uncompressed sizes"),
        ({"extra_flags": 0x1}, "extra fields, comments, zip64 or encryption"),
        ({"extra_flags": 0x40}, "extra fields, comments, zip64 or encryption"),
    ],
    ids=[
        "descriptor-crc", "descriptor-sizes", "descriptor-huge", "descriptor-without-signature",
        "sizes-in-local-and-descriptor", "deflated-sizes-in-local", "zip64-offset-differs",
        "zip64-extensible-data", "locator-elsewhere", "trailing-bytes", "hidden-gap-8-bytes",
        "descriptor-crc-no-zip64",
        "bytes-before-the-end-record", "bytes-before-zip64-end", "descriptor-into-directory",
        "stored-size-lie", "encrypted-flag", "strong-encryption-flag",
    ],
)  # fmt: skip
def test_every_deviation_from_the_framework_layout_is_refused_by_its_own_check(
    shape: dict[str, Any], message: str
) -> None:
    with pytest.raises(CheckpointCorruptError, match=message) as info:
        scan(framework_archive(MEMBERS, **shape))
    assert info.value.reason == "unsafe_archive"


def test_an_archive_whose_last_descriptor_ends_exactly_at_the_central_directory_is_accepted() -> (
    None
):
    assert scan(framework_archive({"archive/data.pkl": GOOD}, zip64_end=False)) == 0


def test_a_member_that_needs_a_newer_zip_version_than_the_writer_uses_is_refused() -> None:
    archive = framework_archive(MEMBERS)
    at = archive.index(b"PK\x01\x02")
    with pytest.raises(CheckpointCorruptError, match="extra fields, comments, zip64 or encryption"):
        scan(patched(archive, at + 6, "<H", 45))


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


def test_a_stored_member_with_a_wrong_checksum_or_unequal_sizes_is_refused() -> None:
    archive = bytearray(framework_archive(MEMBERS))
    tensor_at = archive.index(bytes(range(32)))
    archive[tensor_at] ^= 1  # a tensor byte changes: central and descriptor checksums no longer fit
    with pytest.raises(CheckpointCorruptError, match="checksum"):
        scan(bytes(archive))
    base = framework_archive(MEMBERS)
    at = base.index(b"PK\x01\x02")  # the first central record: csize is at offset 20
    patched_sizes = patched(base, at + 20, "<I", 99)
    with pytest.raises(CheckpointCorruptError):
        scan(patched_sizes)


def test_pickle_controlled_text_in_a_message_is_plain_and_short() -> None:
    data = b"\x80\x02cevil\x1b[31m\nmodule" + b"m" * 100 + b"\nCl\x07ass\nq\x00."
    with pytest.raises(CheckpointCorruptError) as info:
        scan(framework_archive({"archive/data.pkl": data}))
    text = str(info.value)
    assert all(" " <= c < "\x7f" for c in text) and len(text) < 120, text


def test_a_development_build_of_the_first_fixed_release_is_not_a_safe_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as md

    for text, accepted in (("2.6.0", True), ("2.6.0.dev20250101", False), ("2.6.0rc1", False),
                           ("2.6.0a0", False), ("2.7.0.dev1", True), ("2.5.9", False)):  # fmt: skip
        monkeypatch.setattr(md, "version", lambda name, t=text: t)
        version = serializers.framework_version()
        assert (version is not None and version >= serializers.MIN_PT_VERSION) is accepted, text


def test_a_loader_whose_own_version_differs_from_the_metadata_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from cvpilot.core.errors import CheckpointFormatError

    module = types.ModuleType("to" + "rch")
    module.__version__ = "2.1.0"  # type: ignore[attr-defined]
    module.load = lambda *a, **k: {"epoch": 3}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "to" + "rch", module)
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    with pytest.raises(CheckpointFormatError, match="metadata"):
        serializers.get_serializer("pt").loads(framework_archive(MEMBERS))
    module.__version__ = "2.6.1"  # type: ignore[attr-defined]
    assert serializers.get_serializer("pt").loads(framework_archive(MEMBERS)) == {"epoch": 3}


@pytest.mark.parametrize(
    "folder", ["epoch=1-step=8", "my model", "best=3", "model-v1.2+final", "a,b@c", "x(1)[2]"]
)
def test_the_folder_name_the_framework_derives_from_the_file_stem_is_accepted(
    folder: str,
) -> None:
    """A file saved to ``epoch=1-step=8.ckpt`` (Lightning's default name) has members under
    ``epoch=1-step=8/``; before this the reader refused every default-named Lightning file."""
    members = {f"{folder}/{name.split('/', 1)[1]}": body for name, body in MEMBERS.items()}
    assert scan(framework_archive(members)) == 32


@pytest.mark.parametrize(
    "folder", ["a;b", "a:b", "a|b", "a$b", "a`b", "a*b", "a?b", "a'b", 'a"b', "a<b", "é", "a\tb"]
)
def test_a_folder_name_with_shell_or_path_meaning_is_still_refused(folder: str) -> None:
    members = {f"{folder}/{name.split('/', 1)[1]}": body for name, body in MEMBERS.items()}
    with pytest.raises(CheckpointCorruptError, match="name|ASCII"):
        scan(framework_archive(members))


DATA = Path(__file__).resolve().parents[1] / "data"
REAL_FILES = sorted(DATA.glob("torch_*"))


def test_checked_in_files_from_both_framework_releases_are_present() -> None:
    names = {p.name for p in REAL_FILES}
    assert {
        "torch_2_6_0_plain.pt",
        "torch_2_14_1_plain.pt",
        "torch_2_6_0_named_epoch_step.ckpt",
        "torch_2_14_1_named_epoch_step.ckpt",
    } <= names


@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.name)
def test_files_the_real_framework_wrote_are_accepted_and_the_builder_reproduces_them(
    path: Path,
) -> None:
    """Evidence kept in the repository (written by the minimum supported release, 2.6.0, and by
    2.14.1): the scanner accepts them and ``framework_archive`` rebuilds them byte for byte."""
    import zipfile

    raw = path.read_bytes()
    assert scan(raw) > 0
    with zipfile.ZipFile(path) as archive:
        members = {i.filename: archive.read(i.filename) for i in archive.infolist()}
    assert framework_archive(members) == raw
