"""Every published extra installs something, and every extra the docs name exists."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from cvpilot.core.errors import MissingExtraError
from cvpilot.core.uri import parse_remote_uri
from cvpilot.remote import fsspec_store

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
EXTRAS: dict[str, list[str]] = PROJECT["optional-dependencies"]


def test_no_published_extra_is_empty() -> None:
    empty = [name for name, deps in EXTRAS.items() if not deps]
    assert empty == [], (
        f"an extra that installs nothing promises something that is not there: {empty}"
    )


def test_the_hf_extra_brings_the_framework_with_a_verified_lower_bound() -> None:
    (requirement,) = EXTRAS["hf"]
    assert requirement.startswith("transformers[" + "to" + "rch]>=4.")  # the hook needs the trainer to run


def test_the_remote_extras_bring_their_backend() -> None:
    assert any(d.startswith("s3fs") for d in EXTRAS["remote-s3"])
    assert any(d.startswith("gcsfs") for d in EXTRAS["remote-gcs"])
    assert any(d.startswith("adlfs") for d in EXTRAS["remote-azure"])
    for name in ("remote-s3", "remote-gcs", "remote-azure"):
        assert any(d.startswith("fsspec") for d in EXTRAS[name])


def named_in_docs() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for page in pages:
        if "plans" in page.parts:
            continue
        text = page.read_text(encoding="utf-8")
        for match in re.finditer(r"cvpilot\[([a-z0-9_,\- \[\]]+?)\]", text):
            for name in re.split(r"[,\s\[\]]+", match.group(1)):
                if name:
                    found.setdefault(name, set()).add(str(page.relative_to(ROOT)))
    return found


def test_every_extra_the_docs_tell_you_to_install_exists() -> None:
    planned_pages = {"docs/architecture/01-overview.md", "docs/architecture/02-components.md"}
    missing = {
        name: pages
        for name, pages in named_in_docs().items()
        if name not in EXTRAS and not pages <= planned_pages
    }
    assert missing == {}, f"docs name extras that do not exist: {missing}"


@pytest.mark.parametrize(
    ("uri", "extra", "backend"),
    [
        ("s3://bucket-name/x", "remote-s3", "s3fs"),
        ("gs://bucket-name/x", "remote-gcs", "gcsfs"),
        ("az://container/x", "remote-azure", "adlfs"),
    ],
)
def test_a_missing_backend_names_the_extra_for_that_scheme(
    monkeypatch: pytest.MonkeyPatch, uri: str, extra: str, backend: str
) -> None:
    monkeypatch.setitem(sys.modules, backend, None)  # the backend is not installed

    class FakeFsspec:
        @staticmethod
        def filesystem(name: str, **options: object) -> object:
            raise ImportError(f"no {backend}")

    monkeypatch.setitem(sys.modules, "fsspec", FakeFsspec)
    with pytest.raises(MissingExtraError) as info:
        fsspec_store.open_remote(parse_remote_uri(uri))
    message = str(info.value)
    assert f"cvpilot[{extra}]" in message and backend in message
    others = {"remote-s3", "remote-gcs", "remote-azure"} - {extra}
    assert not any(f"cvpilot[{o}]" in message for o in others)


def test_a_missing_fsspec_names_the_plain_remote_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(MissingExtraError) as info:
        fsspec_store.open_remote(parse_remote_uri("memory://t/x"))
    assert "cvpilot[remote]" in str(info.value)
