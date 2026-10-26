import hashlib
import itertools
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from cvpilot.core.hashing import dataset_hash, merkle_hash, sha256_file


def test_sha256_file_matches_hashlib_across_chunk_boundaries(tmp_path: Path) -> None:
    data = os.urandom(5000)
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(p) == expected
    assert sha256_file(p, chunk_size=7) == expected


def test_sha256_file_empty_and_missing(tmp_path: Path) -> None:
    p = tmp_path / "e"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "nope")


def test_sha256_file_rejects_bad_chunk_size(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(p, chunk_size=0)


def make_tree(root: Path, order: list[str] | None = None) -> None:
    files = {"a.jpg": b"1", "sub/b.jpg": b"22", "sub/deep/c.txt": b"333"}
    for rel in order or list(files):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(files[rel])
        os.utime(p, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))


def test_merkle_is_independent_of_creation_order(tmp_path: Path) -> None:
    names = ["a.jpg", "sub/b.jpg", "sub/deep/c.txt"]
    hashes = set()
    for i, order in enumerate(itertools.permutations(names)):
        root = tmp_path / f"t{i}"
        make_tree(root, list(order))
        hashes.add(merkle_hash(root))
    assert len(hashes) == 1


def test_merkle_matches_the_documented_construction(tmp_path: Path) -> None:
    make_tree(tmp_path)
    leaves = []
    for rel, size in sorted([("a.jpg", 1), ("sub/b.jpg", 2), ("sub/deep/c.txt", 3)]):
        leaves.append(
            hashlib.sha256(f"{rel}\0{size}\0{1_700_000_000_000_000_000}".encode()).digest()
        )
    assert merkle_hash(tmp_path) == hashlib.sha256(b"".join(leaves)).hexdigest()


def test_merkle_changes_on_add_remove_rename_resize_touch(tmp_path: Path) -> None:
    make_tree(tmp_path)
    base = merkle_hash(tmp_path)
    f = tmp_path / "a.jpg"

    f.write_bytes(b"1")  # same bytes, mtime changes
    assert merkle_hash(tmp_path) != base
    os.utime(f, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    assert merkle_hash(tmp_path) == base

    f.write_bytes(b"12")
    os.utime(f, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    assert merkle_hash(tmp_path) != base  # resized
    f.write_bytes(b"1")
    os.utime(f, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))

    f.rename(tmp_path / "z.jpg")
    assert merkle_hash(tmp_path) != base  # renamed
    (tmp_path / "z.jpg").rename(f)
    assert merkle_hash(tmp_path) == base

    extra = tmp_path / "new.txt"
    extra.write_bytes(b"")
    assert merkle_hash(tmp_path) != base  # added
    extra.unlink()
    assert merkle_hash(tmp_path) == base


def test_merkle_ignores_dotfiles_and_dot_directories(tmp_path: Path) -> None:
    make_tree(tmp_path)
    base = merkle_hash(tmp_path)
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_bytes(b"x")
    (tmp_path / "sub" / ".hidden").write_bytes(b"x")
    assert merkle_hash(tmp_path) == base


def test_merkle_skips_symlinks(tmp_path: Path) -> None:
    make_tree(tmp_path)
    base = merkle_hash(tmp_path)
    (tmp_path / "link.jpg").symlink_to(tmp_path / "a.jpg")
    (tmp_path / "dlink").symlink_to(tmp_path / "sub")
    assert merkle_hash(tmp_path) == base


def test_merkle_of_empty_directory_is_hash_of_nothing(tmp_path: Path) -> None:
    assert merkle_hash(tmp_path) == hashlib.sha256(b"").hexdigest()


def test_merkle_unicode_names(tmp_path: Path) -> None:
    (tmp_path / "кола 330мл.jpg").write_bytes(b"1")
    assert len(merkle_hash(tmp_path)) == 64


def test_merkle_requires_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        merkle_hash(f)
    with pytest.raises(FileNotFoundError):
        merkle_hash(tmp_path / "nope")


def test_dataset_hash_uses_datacopilot_manifest_version(tmp_path: Path) -> None:
    make_tree(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "9f8e7d", "images": 3}))
    assert dataset_hash(str(tmp_path)) == "dcp:9f8e7d"


@pytest.mark.parametrize(
    "content", ["not json", "[]", json.dumps({"version": ""}), json.dumps({"version": 3}), "{}"]
)
def test_dataset_hash_falls_back_to_merkle_for_unusable_manifest(
    tmp_path: Path, content: str
) -> None:
    make_tree(tmp_path)
    (tmp_path / "manifest.json").write_text(content)
    assert dataset_hash(str(tmp_path)).startswith("merkle:")


def test_dataset_hash_merkle_and_errors(tmp_path: Path) -> None:
    make_tree(tmp_path)
    assert dataset_hash(str(tmp_path)) == "merkle:" + merkle_hash(tmp_path)
    with pytest.raises(FileNotFoundError):
        dataset_hash(str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="local"):
        dataset_hash("s3://bucket/ds")


def test_dataset_hash_on_a_file_is_an_error(tmp_path: Path) -> None:
    f = tmp_path / "list.txt"
    f.write_bytes(b"abc")
    with pytest.raises(NotADirectoryError):
        dataset_hash(str(f))


@pytest.mark.parametrize(
    "version",
    ["a" * 197, "a" * 191, "has space", "new\nline", "a/b", "é" * 4, "../x", "a;b"],
)
def test_a_manifest_version_that_a_checkpoint_manifest_could_not_carry_falls_back(
    tmp_path: Path, version: str
) -> None:
    # a 197-character version made init succeed and every later checkpoint save fail, because the
    # stored dataset_hash ("dcp:" + version) is limited to 200 characters
    make_tree(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"version": version}))
    assert dataset_hash(str(tmp_path)).startswith("merkle:")


def test_the_longest_accepted_version_is_a_valid_checkpoint_dataset_hash(tmp_path: Path) -> None:
    from cvpilot.core.hashing import MAX_DATASET_VERSION_CHARS
    from cvpilot.core.schema import Repro

    make_tree(tmp_path)
    version = "a1_-." * 100
    version = version[:MAX_DATASET_VERSION_CHARS]
    (tmp_path / "manifest.json").write_text(json.dumps({"version": version}))
    value = dataset_hash(str(tmp_path))
    assert value == "dcp:" + version
    Repro(  # the model every checkpoint manifest copies the hash into
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV", git_dirty=False, packages_lock_sha256="0" * 64,
        python_version="3", gpu_names=[], dataset_uri="", dataset_hash=value, config={},
        command_line="x",
    )  # fmt: skip


def test_a_deeply_nested_manifest_falls_back_instead_of_raising(tmp_path: Path) -> None:
    make_tree(tmp_path)
    (tmp_path / "manifest.json").write_text("[" * 100_000 + "]" * 100_000)
    assert dataset_hash(str(tmp_path)).startswith("merkle:")


def test_a_manifest_over_the_size_cap_is_not_read(tmp_path: Path) -> None:
    from cvpilot.core.hashing import MAX_MANIFEST_BYTES

    make_tree(tmp_path)
    padding = " " * (MAX_MANIFEST_BYTES + 1)
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "abc"}) + padding)
    assert dataset_hash(str(tmp_path)).startswith("merkle:")
    assert MAX_MANIFEST_BYTES == 1024 * 1024


def test_a_symlinked_manifest_is_not_followed(tmp_path: Path) -> None:
    make_tree(tmp_path)
    elsewhere = tmp_path.parent / (tmp_path.name + "-elsewhere.json")
    elsewhere.write_text(json.dumps({"version": "planted"}))
    (tmp_path / "manifest.json").symlink_to(elsewhere)
    assert dataset_hash(str(tmp_path)).startswith("merkle:")


def test_a_manifest_that_is_not_a_regular_file_falls_back(tmp_path: Path) -> None:
    make_tree(tmp_path)
    (tmp_path / "manifest.json").mkdir()
    assert dataset_hash(str(tmp_path)).startswith("merkle:")


def _within(seconds: float, fn: Any) -> Any:
    """Run ``fn`` in a thread; a regression that blocks fails the test instead of hanging it."""
    import threading

    box: list[Any] = []
    thread = threading.Thread(target=lambda: box.append(fn()), daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "dataset_hash blocked"
    return box[0]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_a_fifo_named_manifest_json_does_not_block_init(tmp_path: Path) -> None:
    make_tree(tmp_path)
    os.mkfifo(tmp_path / "manifest.json")  # opening it for reading would wait for a writer forever
    assert _within(10, lambda: dataset_hash(str(tmp_path))).startswith("merkle:")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_a_fifo_swapped_in_after_the_check_does_not_block_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # lstat sees a regular file, then the path becomes a FIFO before the open: O_NONBLOCK and the
    # fstat on the descriptor close that window
    from cvpilot.core import hashing

    make_tree(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": "abc"}))
    real_lstat = os.lstat

    def lstat_then_swap(path: Any, *a: Any, **kw: Any) -> os.stat_result:
        result = real_lstat(path, *a, **kw)
        if str(path).endswith("manifest.json") and manifest.is_file():
            manifest.unlink()
            os.mkfifo(manifest)
        return result

    monkeypatch.setattr(hashing.os, "lstat", lstat_then_swap)
    assert _within(10, lambda: hashing._manifest_version(tmp_path)) is None


def test_a_device_file_named_manifest_json_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvpilot.core import hashing

    make_tree(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    real_lstat = os.lstat

    class Device:  # what lstat reports for /dev/zero: a character device (cannot be created here)
        st_mode = stat.S_IFCHR | 0o666
        st_size = 0

    monkeypatch.setattr(
        hashing.os,
        "lstat",
        lambda p, *a, **k: Device() if str(p).endswith("manifest.json") else real_lstat(p, *a, **k),
    )
    assert hashing._manifest_version(tmp_path) is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_sha256_file_refuses_a_fifo_instead_of_waiting_for_a_writer(tmp_path: Path) -> None:
    fifo = tmp_path / "artifact"
    os.mkfifo(fifo)

    def attempt() -> str:
        try:
            sha256_file(fifo)
        except OSError as exc:
            return type(exc).__name__
        return "read"

    assert _within(10, attempt) != "read"
