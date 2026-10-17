"""Static checks of the executed notebooks: nothing is run, nothing is downloaded."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
NAMES = ("01_tutorial.ipynb", "02_edge_cases.ipynb", "03_alerts_and_hf_trainer.ipynb")
ALERTS_NB = NAMES[2]
MAX_BYTES = 1_500_000
MAX_PNG = 8
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
PATHS = re.compile(r"/(?:Users|home|private|var/folders|tmp)/")
LOADERS = ("YOLO(", "to" + "rch.load", "pickle.load", "pickle.loads")


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((NOTEBOOKS / name).read_text())
    return data


def source(cell: dict[str, Any]) -> str:
    text = cell["source"]
    return text if isinstance(text, str) else "".join(text)


def code_cells(nb: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def output_text(cell: dict[str, Any]) -> str:
    parts: list[str] = []
    for out in cell.get("outputs", []):
        text = out.get("text", "")
        parts.append(text if isinstance(text, str) else "".join(text))
    return "\n".join(parts)


@pytest.fixture(params=NAMES)
def name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_the_notebooks_exist_and_are_nbformat_4(name: str) -> None:
    nb = load(name)
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 2
    assert nb["metadata"]["kernelspec"]["language"] == "python"
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert "source" in cell and "metadata" in cell
    try:
        import nbformat
    except ImportError:  # the structure above is the part that always runs
        return
    nbformat.validate(nbformat.read(str(NOTEBOOKS / name), as_version=4))


def test_every_code_cell_was_executed_and_none_failed(name: str) -> None:
    cells = code_cells(load(name))
    assert len(cells) >= 20
    for n, cell in enumerate(cells):
        assert isinstance(cell.get("execution_count"), int), f"code cell {n} was not executed"
        assert "outputs" in cell
        assert all(o["output_type"] != "error" for o in cell["outputs"]), f"code cell {n} failed"
    counts = [c["execution_count"] for c in cells]
    assert counts == sorted(counts), "the cells were not run once, top to bottom"


def test_no_path_of_the_machine_that_ran_them_is_saved(name: str) -> None:
    text = (NOTEBOOKS / name).read_text()
    assert not PATHS.search(text), PATHS.search(text)
    assert str(Path.home()) not in text
    assert "\\u001b" not in text, "terminal colour codes in an output"


def test_no_tool_or_vendor_names_in_the_notebooks_or_their_readme(name: str) -> None:
    for path in (NOTEBOOKS / name, NOTEBOOKS / "README.md"):
        assert not re.search("clau" + "de|anthro" + "pic", path.read_text(), re.IGNORECASE), (
            path.name
        )


def test_size_and_figure_budgets(name: str) -> None:
    assert (NOTEBOOKS / name).stat().st_size < MAX_BYTES
    nb = load(name)
    pngs = sum(
        "image/png" in o.get("data", {}) for c in code_cells(nb) for o in c.get("outputs", [])
    )
    assert pngs <= MAX_PNG


def test_most_code_cells_of_the_tutorial_are_followed_by_an_explanation() -> None:
    cells = load(NAMES[0])["cells"]
    explained = sum(
        1
        for a, b in zip(cells, cells[1:], strict=False)
        if a["cell_type"] == "code"
        and b["cell_type"] == "markdown"
        and "What this shows" in source(b)
    )
    assert explained >= 0.8 * len(code_cells(load(NAMES[0])))


def test_the_readme_lists_the_notebooks_in_ten_lines() -> None:
    lines = [ln for ln in (NOTEBOOKS / "README.md").read_text().splitlines() if ln.strip()]
    assert len(lines) <= 10
    text = "\n".join(lines)
    assert all(n in text for n in NAMES)
    assert "CVPILOT_NOTEBOOK_DATA" in text


def test_the_edge_case_table_has_no_mismatch_and_matches_its_row_count() -> None:
    last = code_cells(load(NAMES[1]))[-1]
    text = output_text(last)
    assert "MISMATCH" not in text
    summary = re.search(r"rows: (\d+), mismatches: (\d+)", text)
    assert summary is not None
    rows = [
        ln
        for ln in text.splitlines()
        if re.match(r"\s*\d+\s{2}", ln) and ln.rstrip().endswith("yes")
    ]
    assert int(summary.group(1)) == len(rows) >= 30
    assert summary.group(2) == "0"


# -- the download cells: every artefact is checked before it is unpacked or loaded ----------


def download_problems(nb: dict[str, Any]) -> list[str]:
    """What is wrong with the way this notebook fetches, unpacks and loads its downloads."""
    cells = code_cells(nb)
    codes = [source(c) for c in cells]
    fetch = [i for i, s in enumerate(codes) if "urlopen" in s]
    problems: list[str] = []
    if not fetch:
        return ["no download cell"]
    text = codes[fetch[0]]
    if len(set(HEX64.findall(text))) < 2:
        problems.append("fewer than two pinned SHA-256 digests")
    check = re.search(r"if digest != \w+\[name\]:\s*(?:\w+\.unlink\(\)\s*)?raise ", text)
    if check is None:
        problems.append("the digest is not compared with a raise")
    elif text.index("part.replace(dest)") < check.start():
        problems.append("the file is renamed before its digest is checked")
    if "https://" not in text or 'startswith("https://")' not in text:
        problems.append("downloads are not restricted to https")
    if "timeout=" not in text or "MAX_DOWNLOAD_BYTES" not in text:
        problems.append("the download has no timeout or size bound")
    for s in codes:
        if re.search(r"\.extractall\(", s):
            problems.append("extractall on a downloaded archive")
    if "safe_unzip(" not in text:
        problems.append("the archive is not unpacked with the validator")
    if ".." not in text or "external_attr" not in text:
        problems.append("the archive validator checks neither paths nor links")
    loads = [i for i, s in enumerate(codes) if any(t in s for t in LOADERS)]
    if loads and min(loads) < fetch[0]:
        problems.append("a pickle is loaded before the download check")
    return problems


def test_downloads_are_pinned_checked_and_unpacked_safely(name: str) -> None:
    assert download_problems(load(name)) == []


def _edit(nb: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    doctored = copy.deepcopy(nb)
    for cell in code_cells(doctored):
        if "urlopen" in source(cell):
            cell["source"] = source(cell).replace(old, new)
    return doctored


def test_a_notebook_without_the_digest_check_is_rejected() -> None:
    nb = load(NAMES[0])
    no_check = _edit(nb, "if digest != PINNED_SHA256[name]:", "if False:")
    assert any("compared" in p for p in download_problems(no_check))


def test_a_notebook_without_pinned_digests_is_rejected() -> None:
    nb = load(NAMES[0])
    text = source(next(c for c in code_cells(nb) if "urlopen" in source(c)))
    digests = sorted(set(HEX64.findall(text)))
    doctored = nb
    for digest in digests:
        doctored = _edit(doctored, digest, "0" * 63 + "g")
    assert any("pinned" in p for p in download_problems(doctored))


def test_a_notebook_that_extracts_without_the_validator_is_rejected() -> None:
    nb = load(NAMES[0])
    doctored = _edit(
        nb, "safe_unzip(coco_zip, COCO_DIR)", "zipfile.ZipFile(coco_zip).extractall(COCO_DIR)"
    )
    assert any("extractall" in p for p in download_problems(doctored))


def test_a_notebook_that_checks_after_renaming_is_rejected() -> None:
    nb = load(NAMES[0])
    doctored = _edit(nb, "part.replace(dest)  # atomic", "part.replace(dest)\n        # atomic")
    for cell in code_cells(doctored):
        if "urlopen" in source(cell):
            s = source(cell).replace(
                "        digest = sha256_of(part)",
                "        part.replace(dest)\n        digest = sha256_of(dest)",
                1,
            )
            cell["source"] = s.replace("part.unlink()\n            raise", "raise", 1)
    assert download_problems(doctored)


def test_a_notebook_that_loads_a_model_before_the_download_is_rejected() -> None:
    nb = copy.deepcopy(load(NAMES[0]))
    first = next(c for c in nb["cells"] if c["cell_type"] == "code")
    first["source"] = source(first) + "\nmodel = YOLO('x.pt')\n"
    assert any("before" in p for p in download_problems(nb))


def test_a_notebook_without_any_download_cell_is_rejected() -> None:
    nb = copy.deepcopy(load(NAMES[1]))
    for cell in code_cells(nb):
        cell["source"] = source(cell).replace("urlopen", "open_url")
    assert download_problems(nb) == ["no download cell"]


# -- notebook 03: the Hugging Face model pin, the alert messages and the step-time section ------

HEX40 = re.compile(r"\b[0-9a-f]{40}\b")


def all_code(name: str) -> str:
    return "\n".join(source(c) for c in code_cells(load(name)))


def all_output(name: str) -> str:
    return "\n".join(output_text(c) for c in code_cells(load(name)))


def test_the_hub_model_is_pinned_to_a_commit_and_a_digest_and_loaded_safely() -> None:
    code = all_code(ALERTS_NB)
    assert 'HF_REVISION = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"' in code
    assert "/resolve/{HF_REVISION}" in code and "main" not in code.split("HF_BASE")[1][:80]
    assert "1cf00ee468998c23d084361b02cffadabacb5074105564c596b8079f77ecb126" in code
    assert "11a01bcc873444bb433140e5f30f3687356641ed923266d738eef1c3438f6d22" in code
    assert "use_safetensors=True" in code and "trust_remote_code=False" in code
    assert "trust_remote_code=True" not in code and "use_safetensors=False" not in code
    assert 'HF_HUB_OFFLINE="1"' in code
    assert "pytorch_model.bin" not in code and "tf_model.h5" not in code
    assert HEX40.search(code)


def test_the_hub_files_go_through_the_same_checked_fetch() -> None:
    code = all_code(ALERTS_NB)
    assert 'fetch(f"{HF_BASE}/model.safetensors", "resnet18-model.safetensors")' in code
    assert 'fetch(f"{HF_BASE}/config.json", "resnet18-config.json")' in code
    assert "blob != GIT_BLOB_ID" in code
    out = all_output(ALERTS_NB)
    assert out.count("(matches the pin)") >= 4


def test_production_refuses_http_and_development_allows_a_local_listener() -> None:
    out = all_output(ALERTS_NB)
    assert "must be an https URL" in out and "exit code: 2" in out
    assert "channel webhook: ready" in out and "nothing was sent" in out


def test_the_received_alert_messages_are_in_the_outputs() -> None:
    out = all_output(ALERTS_NB)
    assert "Loss train/loss is NaN at step 8" in out
    assert "has not improved for 12 logged values" in out
    assert "worse than baseline tag:prod: headline_drop on mAP50_95" in out
    assert "rule: nan_loss" in out and "mrkdwn: False" in out
    assert "equal candidate" in out and "alerts received: 0" in out


def test_the_step_time_table_covers_every_condition_with_at_least_three_repeats() -> None:
    out = all_output(ALERTS_NB)
    for label in (
        "no alerts at all",
        "healthy channel",
        "closed port",
        "never answers",
        "redirects",
    ):
        rows = [
            ln
            for ln in out.splitlines()
            if ln.startswith(label) and re.search(r"\s\d+\s+\d+\s", ln)
        ]
        assert len(rows) >= 3, (label, len(rows))
    assert "VERDICT:" in out and "ms per step" in out


def test_every_alert_condition_really_fired_alerts() -> None:
    out = all_output(ALERTS_NB)
    for ln in out.splitlines():
        match = re.match(
            r"(closed port|never answers|redirects|healthy channel)"
            r"\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+(\d+)\s",
            ln,
        )
        if match:
            assert int(match.group(2)) >= 1, ln


def test_the_alert_notebook_has_no_filler_left() -> None:
    nb = load(ALERTS_NB)
    assert "filled in after the run" not in "\n".join(source(c) for c in nb["cells"])
