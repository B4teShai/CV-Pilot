"""Archives laid out the way the training framework's own writer lays them out.

The framework's ``save`` writes a zip archive that Python's ``zipfile`` never produces: every
member has the data-descriptor flag (bit 3) with zero checksum and sizes in its local header and a
16-byte descriptor after the data, a 64-byte aligned data start through an alignment padding
record in the local extra field, version fields of zero, and a zip64 end record plus locator in
front of the classic end record even for a few kilobytes. ``framework_archive`` reproduces that
layout byte for byte (a marked test compares it with a file the installed framework wrote), with
switches to produce each deviation a hostile archive could use.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping

PAD_TAG = 0x4246
ALIGN = 64


def framework_archive(
    members: Mapping[str, bytes],
    *,
    descriptor: bool = True,
    zip64_end: bool = True,
    deflate: bool = False,
    descriptor_crc: int | None = None,
    descriptor_sizes: tuple[int, int] | None = None,
    local_sizes_in_header: bool = False,
    zip64_end_offset_delta: int = 0,
    zip64_extensible: bytes = b"",
    locator_offset_delta: int = 0,
    trailing: bytes = b"",
    drop_descriptor_signature: bool = False,
    descriptor_for: set[str] | None = None,
    gap_after_first_member: bytes = b"",
    gap_before_end: bytes = b"",
    short_last_descriptor: bool = False,
    extra_flags: int = 0,
    lie_usize: int = 0,
) -> bytes:
    """A zip archive in the framework writer's layout (see the module docstring).

    ``descriptor=False`` writes ordinary headers (what Python's ``zipfile`` writes) while keeping
    the framework's alignment padding and zip64 end record.
    """
    out = bytearray()
    central = bytearray()
    names = list(members)
    for name, body in members.items():
        raw = name.encode("utf-8")
        crc = zlib.crc32(body)
        usize = len(body) + (lie_usize if name != names[0] else 0)
        if deflate:
            comp = zlib.compressobj(wbits=-15)
            blob = comp.compress(body) + comp.flush()
            method = 8
        else:
            blob, method = body, 0
        offset = len(out)
        pad = (-(offset + 30 + len(raw) + 4)) % ALIGN
        extra = struct.pack("<HH", PAD_TAG, pad) + b"Z" * pad
        with_desc = descriptor and (descriptor_for is None or name in descriptor_for)
        flags = 0x800 | (0x008 if with_desc else 0) | extra_flags
        lcrc, lcsize, lusize = (0, 0, 0) if with_desc else (crc, len(blob), usize)
        if local_sizes_in_header:
            lcrc, lcsize, lusize = crc, len(blob), usize
        out += struct.pack(
            "<4sHHHHHIIIHH", b"PK\x03\x04", 0, flags, method, 0, 0, lcrc, lcsize, lusize,
            len(raw), len(extra),
        )  # fmt: skip
        out += raw + extra + blob
        if with_desc:
            dcrc = crc if descriptor_crc is None else descriptor_crc
            dcs, dus = descriptor_sizes or (len(blob), usize)
            out += (b"" if drop_descriptor_signature else b"PK\x07\x08") + struct.pack(
                "<III", dcrc, dcs, dus
            )
            if short_last_descriptor and name == names[-1]:
                del out[-4:]  # the descriptor runs into the central directory
        if gap_after_first_member and not central:
            out += gap_after_first_member  # hidden bytes between members
        central += struct.pack(
            "<4sHHHHHHIIIHHHHHII", b"PK\x01\x02", 0, 0, flags, method, 0, 0, crc, len(blob),
            usize, len(raw), 0, 0, 0, 0, 0, offset,
        )  # fmt: skip
        central += raw
    cd_offset, cd_size, total = len(out), len(central), len(members)
    out += central + gap_before_end
    if zip64_end:
        z64_at = len(out)
        out += struct.pack(
            "<4sQHHIIQQQQ", b"PK\x06\x06", 44 + len(zip64_extensible), 0x031E, 0x2D, 0, 0, total,
            total, cd_size, cd_offset + zip64_end_offset_delta,
        )  # fmt: skip
        out += zip64_extensible
        out += struct.pack("<4sIQI", b"PK\x06\x07", 0, z64_at + locator_offset_delta, 1)
    out += struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, total, total, cd_size, cd_offset, 0)
    return bytes(out) + trailing
