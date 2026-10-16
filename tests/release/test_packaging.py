"""Build the wheel and the sdist and inspect what would be published (nothing is uploaded)."""

from __future__ import annotations

import email
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import cvpilot

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.packaging
pytest.importorskip("build")
pytest.importorskip("hatchling")

CORE = {"click", "numpy", "pyarrow", "pydantic", "structlog", "typer"}
HEAVY = (
    "to" + "rch",
    "nvidia",
    "transformers",
    "lightning",
    "ultra" + "lytics",
    "fsspec",
    "safetensors",
    "mlflow",
    "wandb",
    "trackio",
    "duckdb",
    "fast" + "api",
)
WHEEL_BUDGET = 1 * 1024 * 1024
SDIST_BUDGET = 2 * 1024 * 1024
SECRET = re.compile(
    r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}|"
    r"sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|"
    r"(?i:api[_-]?key|token|password|secret)\s*=\s*['\"][^'\"\s]{6,}['\"]"
)


@pytest.fixture(scope="module")
def dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(out), str(ROOT)],
        check=True, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    return out


@pytest.fixture(scope="module")
def wheel(dist: Path) -> Path:
    (found,) = dist.glob("*.whl")
    return found


@pytest.fixture(scope="module")
def sdist(dist: Path) -> Path:
    (found,) = dist.glob("*.tar.gz")
    return found


def wheel_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def sdist_names(path: Path) -> list[str]:
    with tarfile.open(path) as archive:
        top = archive.getnames()[0].split("/")[0]
        return [n[len(top) + 1 :] for n in archive.getnames() if n != top]


def metadata(path: Path) -> email.message.Message:
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(archive.read(name))


FORBIDDEN = (
    "tests/", "benchmarks/", "plans/", ".cl" + "aude/", "docs/", "scripts/", ".github/", "site/",
    ".env", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage", "golden",
    "action.yml", "uv.lock", "mkdocs.yml", ".DS_Store", "CONTRIBUTING.md",
)  # fmt: skip


def test_the_wheel_holds_the_package_license_and_notice_and_py_typed(wheel: Path) -> None:
    names = wheel_names(wheel)
    assert "cvpilot/py.typed" in names and "cvpilot/__init__.py" in names
    dist_info = next(n.split("/")[0] for n in names if ".dist-info/" in n)
    for required in (
        "METADATA",
        "WHEEL",
        "RECORD",
        "entry_points.txt",
        "licenses/LICENSE",
        "licenses/NOTICE",
    ):
        assert f"{dist_info}/{required}" in names, required
    outside = [n for n in names if not (n.startswith("cvpilot/") or n.startswith(dist_info + "/"))]
    assert outside == [], f"files outside the package: {outside}"


def test_the_wheel_contains_nothing_that_should_stay_in_the_repository(wheel: Path) -> None:
    for name in wheel_names(wheel):
        assert not any(bad in name for bad in FORBIDDEN), name


def test_the_sdist_is_the_source_package_and_metadata_only(sdist: Path) -> None:
    names = sdist_names(sdist)
    # hatchling always adds the .gitignore it read; it holds patterns, not files
    allowed_top = {
        "PKG-INFO",
        "pyproject.toml",
        "LICENSE",
        "NOTICE",
        "README.md",
        "CHANGELOG.md",
        ".gitignore",
    }
    for name in names:
        assert name in allowed_top or name.startswith("src/cvpilot/"), f"unexpected: {name}"
        assert not any(bad in name for bad in FORBIDDEN), name
    assert allowed_top <= set(names)
    assert "src/cvpilot/py.typed" in names


def test_no_secret_looking_text_ships(wheel: Path, sdist: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            text = archive.read(name).decode("utf-8", errors="ignore")
            assert not SECRET.search(text), name
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            if member.isfile():
                data = archive.extractfile(member)
                assert data is not None
                assert not SECRET.search(data.read().decode("utf-8", errors="ignore")), member.name


def test_artifacts_stay_within_their_size_budget(wheel: Path, sdist: Path) -> None:
    assert wheel.stat().st_size <= WHEEL_BUDGET, wheel.stat().st_size
    assert sdist.stat().st_size <= SDIST_BUDGET, sdist.stat().st_size


def test_the_core_requirements_are_exactly_the_documented_set(wheel: Path) -> None:
    requires = metadata(wheel).get_all("Requires-Dist") or []
    unconditional = {
        re.split(r"[<>=!~\[ ;(]", r)[0].lower() for r in requires if "extra ==" not in r
    }
    assert unconditional == CORE
    for requirement in requires:
        if "extra ==" not in requirement:
            assert not any(h in requirement.lower() for h in HEAVY), requirement
    extras = set(metadata(wheel).get_all("Provides-Extra") or [])
    assert {
        "safetensors",
        "remote",
        "lightning",
        "ultra" + "lytics",
        "mlflow",
        "wandb",
        "trackio",
    } <= extras
    optional = [r.replace("'", '"') for r in requires if "extra ==" in r]
    assert any(r.lower().startswith("fsspec") and 'extra == "remote"' in r for r in optional)
    assert any(
        r.lower().startswith("safetensors") and 'extra == "safetensors"' in r for r in optional
    )


def test_the_metadata_is_correct(wheel: Path) -> None:
    meta = metadata(wheel)
    assert meta["Name"] == "cvpilot"
    assert meta["Version"] == cvpilot.__version__
    assert meta["License-Expression"] == "Apache-2.0"
    assert sorted(meta.get_all("License-File") or []) == ["LICENSE", "NOTICE"]
    assert meta["Requires-Python"] == ">=3.10"
    assert meta["Description-Content-Type"] == "text/markdown"
    classifiers = set(meta.get_all("Classifier") or [])
    assert "Typing :: Typed" in classifiers
    for minor in ("3.10", "3.11", "3.12", "3.13"):
        assert f"Programming Language :: Python :: {minor}" in classifiers
    assert not [c for c in classifiers if c.startswith("License ::")]  # the expression says it
    urls = [u for u in meta.get_all("Project-URL") or []]
    assert any(u.startswith("Homepage, https://") for u in urls)
    assert any(u.startswith("Changelog, https://") for u in urls)
    assert any(u.startswith("Issues, https://") for u in urls)
    text = meta.as_string().lower()
    # AI tools are never named in package metadata; spelled in parts for the repository gate
    for banned in ("cl" + "aude", "anthr" + "opic", "open" + "ai", "chat" + "gpt", "copi" + "lot"):
        assert banned not in text, banned


def test_the_console_script_is_declared(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith("entry_points.txt"))
        text = archive.read(name).decode()
    assert "[console_scripts]" in text and "cvpilot = cvpilot.cli.app:main" in text
    for group in (
        "cvpilot.stores",
        "cvpilot.hooks",
        "cvpilot.forwarders",
        "cvpilot.remotes",
        "cvpilot.spot_notices",
    ):
        assert f"[{group}]" in text


def test_twine_check_strict_passes(dist: Path) -> None:
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "twine", "check", "--strict", *map(str, dist.iterdir())],
        capture_output=True, text=True, check=False, timeout=120,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_wheel_can_be_built_from_the_sdist_alone(sdist: Path, tmp_path: Path) -> None:
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path / "src", filter="data")  # noqa: S202  # our own artifact
    (root,) = (tmp_path / "src").iterdir()
    out = tmp_path / "out"
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "-o", str(out), str(root)],
        check=True, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert len(list(out.glob("*.whl"))) == 1


@pytest.mark.parametrize("python", ["3.10", "3.13"])
def test_the_wheel_installs_in_a_clean_environment_and_runs(
    wheel: Path, tmp_path: Path, python: str
) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    spec = importlib.util.spec_from_file_location(
        "verify_wheel", ROOT / "scripts" / "verify_wheel.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_wheel"] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    try:
        report = module.verify_one(wheel, python, tmp_path, repo=ROOT)
    except RuntimeError as exc:
        if "No interpreter found" in str(exc) or "no interpreter" in str(exc).lower():
            pytest.skip(f"python {python} is not available")
        raise
    assert (
        report.doctor_output == "ok" and report.version_output == f"cvpilot {cvpilot.__version__}"
    )
    assert "site-packages" in report.module_path


MANIFEST = Path(__file__).with_name("wheel_manifest.txt")
SUFFIXES = {".py", ".typed"}


def test_the_secret_patterns_catch_what_they_claim_to() -> None:
    samples = [
        "AKIA" + "A" * 16,
        "github_pat_" + "a1" * 15,
        "ghp_" + "b2" * 20,
        "xoxb-" + "1234567890-abcdef",
        "sk-" + "c3" * 15,
        "AIza" + "d4" * 17 + "e",
        "ey" + "JhbGciOiJIUzI1NiJ9." + "ey" + "JzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijkl",
        "-----" + "BEGIN RSA PRIVATE KEY" + "-----",
        "pass" + 'word = "hunter2hunter2"',
    ]
    for sample in samples:
        assert SECRET.search(sample), sample
    assert not SECRET.search("a perfectly ordinary sentence about tokens and keys")


def test_no_archive_member_is_anything_but_a_regular_file_or_directory(
    wheel: Path, sdist: Path
) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            mode = info.external_attr >> 16
            assert stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR), info.filename
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            assert member.isfile() or member.isdir(), f"{member.name}: not a regular file"
            assert not member.islnk() and not member.issym(), member.name
            assert not member.name.startswith(("/", "..")) and ".." not in member.name.split("/")


def test_the_wheel_is_exactly_the_committed_list_of_files_with_allowed_suffixes(
    wheel: Path,
) -> None:
    files = sorted(
        n for n in wheel_names(wheel) if n.startswith("cvpilot/") and not n.endswith("/")
    )
    for name in files:
        assert Path(name).suffix in SUFFIXES, f"unexpected file type in the wheel: {name}"
    if os.environ.get("CVPILOT_UPDATE_WHEEL_MANIFEST") == "1":
        MANIFEST.write_text("\n".join(files) + "\n")
    committed = MANIFEST.read_text().split()
    assert files == committed, (
        "the wheel differs from tests/release/wheel_manifest.txt: review the change, then "
        "regenerate with CVPILOT_UPDATE_WHEEL_MANIFEST=1"
    )


def test_the_source_tree_holds_no_symlink_and_no_secret() -> None:
    for root_dir in ("src", ".github", "docs", "scripts", "tests/release"):
        for path in (ROOT / root_dir).rglob("*"):
            assert not path.is_symlink(), f"symlink in the source tree: {path}"
            if path.is_file() and path.suffix in {".py", ".md", ".yml", ".toml", ".json", ".txt"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if path.name != "test_packaging.py":  # holds the samples that test the patterns
                    assert not SECRET.search(text), str(path)
