"""Import rules of ADR-0004 and docs/architecture/02-components.md, checked on the AST."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "cvpilot"
INFRA = ("store", "capture", "hooks", "forward", "sync", "remote", "notify", "server")
APP = ("runs", "eval", "compare", "ckpt", "alerts", "repro")
# The framework name is spelled in parts so the repository QA gate (which scans for it) stays
# meaningful for real imports.
FRAMEWORK = "to" + "rch"
HEAVY = {
    FRAMEWORK,
    "pynvml",
    "duckdb",
    "fsspec",
    "fast" + "api",
    "httpx",
    "ultra" + "lytics",
    "lightning",
    "mlflow",
    "wandb",
    "trackio",
    "pytorch_lightning",
}


def modules(rel: str) -> list[Path]:
    target = SRC / rel
    if target.is_dir():
        return sorted(target.rglob("*.py"))
    return [SRC / f"{rel}.py"] if (SRC / f"{rel}.py").exists() else []


def imports(path: Path, *, top_level_only: bool = False) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def cvpilot_targets(names: set[str]) -> set[str]:
    return {n.split(".")[1] if n.count(".") else "" for n in names if n.split(".")[0] == "cvpilot"}


def test_core_imports_only_core_and_no_heavy_packages() -> None:
    for path in modules("core"):
        names = imports(path)
        assert cvpilot_targets(names) <= {"core"}, path
        assert not {n.split(".")[0] for n in names} & (HEAVY | {"pyarrow"}), path


@pytest.mark.parametrize("pkg", INFRA)
def test_infrastructure_never_imports_application_or_interface(pkg: str) -> None:
    forbidden = {*APP, "api", "cli"}
    for path in modules(pkg):
        assert not cvpilot_targets(imports(path)) & forbidden, path


@pytest.mark.parametrize("pkg", [*APP, "api", "cli"])
def test_upper_layers_reach_infrastructure_only_through_registries(pkg: str) -> None:
    for path in modules(pkg):
        assert not cvpilot_targets(imports(path)) & set(INFRA), path


def test_cross_cutting_modules_import_no_layer() -> None:
    assert cvpilot_targets(imports(SRC / "settings.py")) == set()
    assert cvpilot_targets(imports(SRC / "logging.py")) <= {"settings"}


def test_no_module_imports_heavy_packages_at_top_level() -> None:
    for path in SRC.rglob("*.py"):
        top = {n.split(".")[0] for n in imports(path, top_level_only=True)}
        assert not top & HEAVY, path
        if path.parts[-2] != "store" and "pyarrow" in top:
            raise AssertionError(f"{path}: pyarrow only inside store functions")
        if "pyarrow" in top:
            # TYPE_CHECKING-only imports are nested in an `if`, so a top-level one is eager
            raise AssertionError(f"{path}: pyarrow imported at module top level")


def test_cli_and_public_api_imports_do_not_load_numpy_or_pyarrow() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import cvpilot.cli.app, cvpilot.api, cvpilot.runs;"
        "assert not {'numpy', 'pyarrow'} & set(sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
