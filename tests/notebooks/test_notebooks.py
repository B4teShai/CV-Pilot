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
NAMES = (
    "01_tutorial.ipynb",
    "02_edge_cases.ipynb",
    "03_alerts_and_hf_trainer.ipynb",
    "04_real_checkpoints.ipynb",
    "05_chaos_and_rfdetr.ipynb",
)
ALERTS_NB = NAMES[2]
REAL_NB = NAMES[3]
CHAOS_NB = NAMES[4]
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


def test_the_computed_checks_all_hold_and_the_verdicts_are_there() -> None:
    out = all_output(ALERTS_NB)
    checks = [ln for ln in out.splitlines() if ln.startswith("CHECK:")]
    assert len(checks) >= 6, checks
    assert not [ln for ln in checks if "False" in ln], checks
    verdicts = [ln for ln in out.splitlines() if ln.startswith("VERDICT:")]
    assert len(verdicts) >= 3
    assert any("cannot resolve a difference smaller than" in ln for ln in verdicts)


def test_the_condition_order_is_rotated_so_the_baseline_is_not_always_first() -> None:
    out = all_output(ALERTS_NB)
    positions = [
        int(ln.split()[-1])
        for ln in out.splitlines()
        if ln.startswith("no alerts at all") and re.search(r"\s\d+\s+\d+\s", ln) and "ms" not in ln
    ]
    assert len(positions) >= 3 and len(set(positions)) >= 2, positions


def test_the_per_call_cost_was_measured_while_alerts_really_fired_and_with_a_full_queue() -> None:
    out = all_output(ALERTS_NB)
    assert "alerts queued" in out and "after it overflowed" in out
    queued = [int(m) for m in re.findall(r"alerts queued\s+(\d+)", out)]
    assert queued and all(n >= 1 for n in queued)
    dropped = re.search(r"dropped because the queue was full or the call was stuck: (\d+)", out)
    assert dropped is not None and int(dropped.group(1)) >= 1
    code = all_code(ALERTS_NB)
    assert '"rule": "nan_loss"' in code.split("ALERTS_BENCH")[1][:300]


def test_the_prose_carries_no_hand_typed_measurement() -> None:
    prose = "\n".join(source(c) for c in load(ALERTS_NB)["cells"] if c["cell_type"] == "markdown")
    for phrase in ("four steps", "12 logged", "0.02", "about five seconds", "at most about five"):
        assert phrase not in prose, phrase
    assert "filled in after the run" not in prose


# -- notebook 04: real checkpoint files ---------------------------------------------------------


def layout_rows(name: str) -> dict[tuple[str, str], tuple[str, str]]:
    """``(framework, file) -> (outcome, message)`` from the summary table of the notebook."""
    rows: dict[tuple[str, str], tuple[str, str]] = {}
    out = output_text(code_cells(load(name))[-1])
    for ln in out.splitlines():
        match = re.match(r"(.{32}) (.{52}) (accepted|refused|ADOPTED)\s+(.*)$", ln)
        if match:
            rows[(match.group(1).strip(), match.group(2).strip())] = (
                match.group(3),
                match.group(4).strip(),
            )
    return rows


def test_every_computed_check_of_the_real_file_notebook_holds() -> None:
    out = all_output(REAL_NB)
    checks = [ln for ln in out.splitlines() if ln.startswith("CHECK:")]
    assert len(checks) >= 20, len(checks)
    assert not [ln for ln in checks if not ln.endswith(": True")], checks
    summary = re.search(r"checks: (\d+), failed: (\d+)", out)
    assert summary is not None and int(summary.group(1)) == len(checks) and summary.group(2) == "0"


def test_the_layout_table_records_what_each_real_file_got_and_the_exact_message() -> None:
    rows = layout_rows(REAL_NB)
    assert rows[("Lightning", "last.ckpt")][0] == "accepted"
    ultra = "ultra" + "lytics"
    yolo = next(v for (fw, _), v in rows.items() if fw == "Ultra" + "lytics")
    assert yolo[0] == "refused" and ultra + ".nn.tasks.DetectionModel" in yolo[1]
    hf = {file: v for (fw, file), v in rows.items() if fw == "Hugging Face"}
    assert any(file.endswith("(the folder)") and v[0] == "refused" for file, v in hf.items())
    assert any(file.endswith("optimizer.pt") and v[0] == "accepted" for file, v in hf.items())
    assert any(
        file.endswith("training_args.bin") and "TrainingArguments" in v[1] for file, v in hf.items()
    )
    mm = next(v for (fw, _), v in rows.items() if fw == "MMEngine")
    assert mm[0] == "refused" and "HistoryBuffer" in mm[1]
    assert not [k for k, v in rows.items() if v[0] == "ADOPTED"]


def test_the_tamper_runs_cover_flip_truncate_and_swap_and_were_all_refused() -> None:
    out = all_output(REAL_NB)
    for label in ("flip one byte", "truncate to half", "swap in a foreign checkpoint"):
        assert re.search(rf"{label}\s+verify exit 1 \|", out), label
    assert "restored original              verify exit 0" in out
    assert "CheckpointUntrustedError (checkpoint_missing)" in out
    assert "resume_path default: None (nothing handed back)" in out
    assert "a file was returned" not in "\n".join(
        ln for ln in out.splitlines() if "resume_path default" in ln or "restore in a fresh" in ln
    )


def test_the_real_file_notebook_states_measured_results_for_the_save_overhead() -> None:
    out = all_output(REAL_NB)
    verdicts = [
        ln for ln in out.splitlines() if ln.startswith("VERDICT:") and ("to" + "rch.save") in ln
    ]
    assert len(verdicts) == 3
    for ln in verdicts:
        percent = int(re.search(r"costs (\d+) % more", ln).group(1))  # type: ignore[union-attr]
        assert ("is met" in ln) == (percent <= 5), ln
    for state in ("Lightning tiny classifier", "YOLO11n weights", "ResNet-18 weights"):
        assert re.search(rf"{state}\s+\d+\s+[\d.]+ms\s+[\d.]+ms", out), state
    assert "data write (hash + write + fsync)" in out


def test_the_real_file_notebook_hooks_ran_on_real_runs() -> None:
    out = all_output(REAL_NB)
    assert "status after train(): running; after test(): finished" in out
    assert "the hook marked the killed " in out and "run failed: True" in out


def test_the_real_file_notebook_prose_carries_no_hand_typed_measurement() -> None:
    prose = "\n".join(source(c) for c in load(REAL_NB)["cells"] if c["cell_type"] == "markdown")
    unit = re.compile(r"\b\d[\d,.]*\s?(ms|MB|KB|GB|seconds|minutes|bytes|%|x)\b", re.IGNORECASE)
    assert not unit.search(prose), unit.search(prose)
    for phrase in ("47 MB", "a few kilobytes", "identical to the uninterrupted run: True"):
        assert phrase not in prose, phrase


def test_the_real_file_notebook_never_loads_a_file_before_it_was_checked() -> None:
    codes = [source(c) for c in code_cells(load(REAL_NB))]
    fetch = next(i for i, s in enumerate(codes) if "urlopen" in s)
    first_load = min(i for i, s in enumerate(codes) if "YOLO(" in s or ("to" + "rch.load") in s)
    assert fetch < first_load
    assert not any(("to" + "rch.load(") in s or "pickle.load" in s for s in codes)
    assert not any("weights_only=False" in s for s in codes)


def test_the_authenticated_level_was_exercised_on_the_real_framework_file() -> None:
    out = all_output(REAL_NB)
    assert re.search(r"last\s+0+\d+\s+valid\s+authenticated", out)
    assert "key_changed" in out and "needs the key in CVPILOT_CKPT_KEY" in out
    assert "the key never appears in any output: True" in out


# -- notebook 05: 1,000 kills, cut uploads and a real RF-DETR run ----------------------------


def test_every_computed_check_of_the_chaos_and_rfdetr_notebook_holds() -> None:
    out = all_output(CHAOS_NB)
    checks = [ln for ln in out.splitlines() if ln.startswith("CHECK:")]
    assert len(checks) >= 15, len(checks)
    assert not [ln for ln in checks if not ln.endswith(": True")], checks
    summary = re.search(r"checks: (\d+), failed: (\d+)", out)
    assert summary is not None and int(summary.group(1)) == len(checks) and summary.group(2) == "0"


def test_the_kill_table_has_two_hundred_correct_restarts_per_place() -> None:
    out = all_output(CHAOS_NB)
    rows = {
        m.group(1): (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in re.finditer(r"^(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+", out, re.M)
    }
    for point in (
        "before_temp_write",
        "mid_write",
        "after_write_before_rename",
        "after_rename_before_chain",
        "random_moment",
    ):
        kills, killed, right = rows[point]
        assert kills >= 200 and killed == kills and right == kills, (point, rows[point])
    assert "truncated uploads: 200 cuts" in out


def test_licences_are_checked_before_the_weights_are_downloaded_and_the_pins_hold() -> None:
    codes = [source(c) for c in code_cells(load(CHAOS_NB))]
    licence = next(i for i, s in enumerate(codes) if "RFDETRNanoConfig().license" in s)
    fetch = next(i for i, s in enumerate(codes) if "urlopen" in s)
    assert licence < fetch
    code = "\n".join(codes)
    assert "d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d" in code
    assert 'NANO_MD5 = "fb6504cce7fbdc783f7a46991f07639f"' in code
    for banned in ("trust_checkpoint=True", "weights_only=False", "accept_platform_model_license"):
        assert banned not in code, banned
    assert 'HF_HUB_OFFLINE="1"' in code
    out = all_output(CHAOS_NB)
    assert out.count("(matches the pin)") >= 2


def test_the_real_rfdetr_run_was_recorded_and_matches_the_frameworks_own_log() -> None:
    out = all_output(CHAOS_NB)
    assert "status: finished | framework: rfdetr==" in out
    assert "largest difference: 0.0" in out
    assert "train raised: " in out and "| run status: failed" in out
    assert "strict scan: refused unsafe_archive: the pickle refers to argparse.Namespace" in out


# -- the download cache is private: no shared temporary folder, nothing trusted by existence ------


def test_the_download_cache_is_a_private_folder_never_the_shared_temporary_one(name: str) -> None:
    code = "\n".join(source(c) for c in code_cells(load(name)))
    assert "gettempdir" not in code  # /tmp is shared and predictable on Linux
    assert "<tmp>/cvpilot-notebooks" not in code
    assert 'Path.home() / ".cache" / "cvpilot-notebooks"' in code
    assert "mode=0o700" in code and "st_uid != os.getuid()" in code and "& 0o022" in code
    assert "S_ISLNK" in code


def test_downloads_are_loaded_from_a_checked_private_copy(name: str) -> None:
    codes = [source(c) for c in code_cells(load(name))]
    fetch_cell = next(s for s in codes if "urlopen" in s)
    assert "def private_copy(" in fetch_cell and "WORK /" in fetch_cell
    recheck = r"digest = sha256_of\(dest\)\s+if digest != PINNED_SHA256\[name\]:"
    assert re.search(recheck, fetch_cell)
    # an unpacked folder is checked against the archive, never trusted because it exists
    assert re.search(r"if target\.exists\(\):\s+return\b", fetch_cell) is None
    assert "unpacked_matches(" in fetch_cell
    for code in codes:
        assert 'DATA / "resnet18-pinned"' not in code


def test_the_notebook_readme_names_the_private_default() -> None:
    text = (NOTEBOOKS / "README.md").read_text()
    assert "~/.cache/cvpilot-notebooks" in text and "<tmp>/cvpilot-notebooks" not in text
