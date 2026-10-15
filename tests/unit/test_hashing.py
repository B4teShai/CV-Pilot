import hashlib
import itertools
import json
import os
from pathlib import Path

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
