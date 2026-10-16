"""Content hashes: files, datasets (Merkle over path, size, mtime) and run ids."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cvpilot.core.ulid import generate_ulid

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_MANIFEST = "manifest.json"
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def new_run_id() -> str:
    """A new ULID run id."""
    return generate_ulid()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Lower-case hex SHA-256 of the file's bytes, read in chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_chain(
    previous_chain: str,
    run_id: str,
    kind: str,
    step: int,
    sha256: str,
    size_bytes: int,
    fmt: str | None = None,
) -> str:
    """Chain value of a checkpoint manifest: commits to the file digest, its size and the
    previous manifest of the same ``(run, kind)`` (the first one chains from 64 zeros). From
    schema version 2 on it also commits to the serializer format (``fmt``), so a safetensors
    checkpoint cannot be presented as a ``pt`` one; ``fmt=None`` is the version-1 formula."""
    text = f"{previous_chain}\n{run_id}\n{kind}\n{step}\n{sha256}\n{size_bytes}"
    if fmt is not None:
        text += f"\n{fmt}"
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def pointer_digest(
    run_id: str, kind: str, step: int, manifest: str, manifest_sha256: str, chain_sha256: str
) -> str:
    """Digest over the fields of a ``LATEST``/``BEST`` pointer (see ``CheckpointPointer``)."""
    text = f"{run_id}\n{kind}\n{step}\n{manifest}\n{manifest_sha256}\n{chain_sha256}"
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def merkle_hash(root: Path) -> str:
    """Merkle hash of the regular files under ``root``.

    Dot-files and dot-directories are skipped, symlinks are not followed. Leaf is
    ``sha256(relpath NUL size NUL mtime_ns)`` over POSIX relative paths sorted ascending; the
    root is ``sha256`` of the concatenated leaf digests. It changes when a file is added,
    removed, renamed, resized or touched, so it is not portable across copies that drop mtime.
    """
    if not root.exists():
        raise FileNotFoundError(str(root))
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    entries: list[tuple[str, int, int]] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            full = os.path.join(current, name)
            st = os.lstat(full)
            if not stat.S_ISREG(st.st_mode):
                continue
            rel = Path(full).relative_to(root).as_posix()
            entries.append((rel, st.st_size, st.st_mtime_ns))
    entries.sort(key=lambda e: e[0])
    outer = hashlib.sha256()
    for rel, size, mtime_ns in entries:
        outer.update(hashlib.sha256(f"{rel}\0{size}\0{mtime_ns}".encode()).digest())
    return outer.hexdigest()


def _manifest_version(directory: Path) -> str | None:
    manifest = directory / _MANIFEST
    try:
        if not manifest.is_file() or manifest.stat().st_size > _MAX_MANIFEST_BYTES:
            return None
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return version if isinstance(version, str) and version else None


def dataset_hash(dataset_uri: str) -> str:
    """``dcp:<version>`` when a DataCopilot ``manifest.json`` is present, else ``merkle:<hex>``.

    Raises:
        ValueError: ``dataset_uri`` is a URL; only local directories can be hashed.
        FileNotFoundError: the directory does not exist.
        NotADirectoryError: the path is a file.
    """
    if "://" in dataset_uri:
        raise ValueError("only local dataset directories can be hashed, not URLs")
    root = Path(dataset_uri).expanduser()
    if not root.exists():
        raise FileNotFoundError(dataset_uri)
    if not root.is_dir():
        raise NotADirectoryError(dataset_uri)
    version = _manifest_version(root)
    return f"dcp:{version}" if version else f"merkle:{merkle_hash(root)}"


def eval_set_hash(
    class_names: Sequence[str],
    image_ids: Sequence[str],
    widths: npt.NDArray[np.int64],
    heights: npt.NDArray[np.int64],
    ann_image: npt.NDArray[np.int64],
    ann_cls: npt.NDArray[np.int64],
    ann_boxes: npt.NDArray[np.float64],
    ann_crowd: npt.NDArray[np.bool_],
) -> str:
    """Content hash of an evaluation set (ground truth and class list only).

    SHA-256 of the canonical JSON ``{"class_names": [...], "images": [[image_id, width, height,
    [[cls, x1, y1, x2, y2, iscrowd], ...]], ...]}``: images in canonical id order,
    annotations sorted lexicographically, coordinates rounded to two decimals, separators
    ``(",", ":")``, UTF-8.
    File paths and timestamps never enter it, so two machines evaluating the same set agree.

    ``image_ids`` must be unique and in canonical order (numeric when every id is a canonical
    non-negative integer, else string order); ``ann_image`` indexes into it. Unknown sizes
    are 0. Coordinates are rounded to two decimals, so boxes closer than 0.005 px hash alike.
    The JSON is streamed into the digest one image at a time.
    """
    import numpy as np  # imported here so that run ids and file hashes stay numpy-free

    n = len(image_ids)
    rounded = np.round(ann_boxes, 2) + 0.0  # + 0.0 turns -0.0 into 0.0
    order = np.lexsort(
        (ann_crowd, rounded[:, 3], rounded[:, 2], rounded[:, 1], rounded[:, 0], ann_cls, ann_image)
    )
    image_s, cls_s, box_s, crowd_s = (
        ann_image[order],
        ann_cls[order],
        rounded[order],
        ann_crowd[order],
    )
    bounds = np.searchsorted(image_s, np.arange(n + 1))

    def dump(value: object) -> bytes:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    digest = hashlib.sha256()
    digest.update(b'{"class_names":' + dump(list(class_names)) + b',"images":[')
    for i in range(n):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        rows = [
            [int(c), *b, int(k)]
            for c, b, k in zip(
                cls_s[lo:hi].tolist(), box_s[lo:hi].tolist(), crowd_s[lo:hi].tolist(), strict=True
            )
        ]
        digest.update(
            (b"," if i else b"") + dump([image_ids[i], int(widths[i]), int(heights[i]), rows])
        )
    digest.update(b"]}")
    return digest.hexdigest()


MAX_SLICE_HASH_IMAGES = 1_000_000
"""Most images ``slice_hash`` covers (the evaluation limit); bounds the hashing loop."""


def slice_hash(
    spec: Mapping[str, str], image_ids: Sequence[str], meta: Mapping[str, Any]
) -> str | None:
    """Digest of the slice membership of an evaluation set, or None without slices.

    ``eval_set_hash`` covers annotations and classes, not image metadata, so two evaluations of
    the same ground truth can slice the images differently. This digest covers, for every slice
    key in sorted order, the key, the metadata column and each image's value in canonical image
    order: ``{"slices":[{"key":..,"column":..,"values":[[image_id,value],...]},...]}`` (compact
    JSON, UTF-8, streamed per image). Equal digests mean equal membership of every slice.

    ``image_ids`` is in canonical order and ``meta`` maps a column (with or without the
    ``meta.`` prefix) to its per-image values in the same order.

    Raises:
        ValueError: more than ``MAX_SLICE_HASH_IMAGES`` images, or a column is missing or has
            another length.
    """
    if not spec:
        return None
    if len(image_ids) > MAX_SLICE_HASH_IMAGES:
        raise ValueError("too many images to hash slice membership")

    def dump(value: object) -> bytes:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    digest = hashlib.sha256()
    digest.update(b'{"slices":[')
    for n, key in enumerate(sorted(spec)):
        column = spec[key][5:] if spec[key].startswith("meta.") else spec[key]
        values = meta[column]
        if len(values) != len(image_ids):
            raise ValueError(f"column {column!r} does not cover every image")
        digest.update((b"," if n else b"") + b'{"key":' + dump(key) + b',"column":' + dump(column))
        digest.update(b',"values":[')
        for i, image_id in enumerate(image_ids):
            digest.update((b"," if i else b"") + dump([image_id, str(values[i])]))
        digest.update(b"]}")
    digest.update(b"]}")
    return digest.hexdigest()
