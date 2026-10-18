import pytest

from cvpilot.settings import Settings, SettingsError, load_settings


def test_defaults() -> None:
    s = load_settings({})
    assert (s.env, s.log_level, s.effective_log_level) == ("prod", "INFO", "INFO")


def test_reads_and_normalises_case() -> None:
    s = load_settings({"CVPILOT_ENV": " DEV ", "CVPILOT_LOG_LEVEL": "debug"})
    assert (s.env, s.log_level) == ("dev", "DEBUG")


def test_debug_is_capped_to_info_outside_dev() -> None:
    assert Settings(env="prod", log_level="DEBUG").effective_log_level == "INFO"
    assert Settings(env="staging", log_level="DEBUG").effective_log_level == "INFO"
    assert Settings(env="dev", log_level="DEBUG").effective_log_level == "DEBUG"


@pytest.mark.parametrize(
    ("var", "value"),
    [("CVPILOT_ENV", "banana"), ("CVPILOT_LOG_LEVEL", "LOUD"), ("CVPILOT_ENV", "")],
)
def test_invalid_value_names_the_variable(var: str, value: str) -> None:
    with pytest.raises(SettingsError, match=var):
        load_settings({var: value})


def test_store_command_line_defaults_true_and_parses_booleans() -> None:
    assert load_settings({}).store_command_line is True
    for raw in ("false", "0", "no", "OFF", " False "):
        assert load_settings({"CVPILOT_STORE_COMMAND_LINE": raw}).store_command_line is False
    for raw in ("true", "1", "yes", "ON"):
        assert load_settings({"CVPILOT_STORE_COMMAND_LINE": raw}).store_command_line is True


@pytest.mark.parametrize("raw", ["", "maybe", "2"])
def test_store_command_line_rejects_other_values(raw: str) -> None:
    with pytest.raises(SettingsError, match="CVPILOT_STORE_COMMAND_LINE"):
        load_settings({"CVPILOT_STORE_COMMAND_LINE": raw})


# --- CVPILOT_PROJECT and CVPILOT_STORE_DIR -----------------------------------------


def test_project_and_store_dir_default_to_none() -> None:
    s = load_settings({})
    assert s.project is None and s.store_dir is None


@pytest.mark.parametrize("project", ["a", "fridge-sku", "a_b-9", "0x", "a" * 64])
def test_valid_project(project: str) -> None:
    assert load_settings({"CVPILOT_PROJECT": f" {project} "}).project == project


@pytest.mark.parametrize(
    "project", ["", " ", "Upper", "a/b", "../x", "..", "-a", "a b", "a" * 65, "é", "a\x00b", "a\nb"]
)
def test_invalid_project_is_a_settings_error_that_never_echoes_the_value(project: str) -> None:
    with pytest.raises(SettingsError, match="CVPILOT_PROJECT") as exc:
        load_settings({"CVPILOT_PROJECT": project})
    assert project.strip() == "" or project not in str(exc.value)


def test_project_pattern_matches_the_domain_pattern() -> None:
    from cvpilot.core.schema import PROJECT_PATTERN
    from cvpilot.settings import PROJECT_PATTERN as SETTINGS_PATTERN

    assert SETTINGS_PATTERN == PROJECT_PATTERN


def test_store_dir_is_expanded_and_resolved(tmp_path: pytest.TempPathFactory) -> None:
    import os
    from pathlib import Path

    s = load_settings({"CVPILOT_STORE_DIR": str(tmp_path)})
    assert s.store_dir == Path(str(tmp_path)).resolve()
    s = load_settings({"CVPILOT_STORE_DIR": "~/stores"})
    assert s.store_dir == (Path(os.path.expanduser("~")) / "stores").resolve()
    rel = load_settings({"CVPILOT_STORE_DIR": "some/rel"})
    assert rel.store_dir is not None and rel.store_dir.is_absolute()
    dotted = load_settings({"CVPILOT_STORE_DIR": f"{tmp_path}/a/../b"})
    assert dotted.store_dir == Path(str(tmp_path)).resolve() / "b"


def test_store_dir_rejects_files_root_empty_and_nul(tmp_path: pytest.TempPathFactory) -> None:
    from pathlib import Path

    f = Path(str(tmp_path)) / "afile"
    f.write_text("x")
    for bad in ("", "   ", "/", "a\x00b", str(f)):
        with pytest.raises(SettingsError, match="CVPILOT_STORE_DIR"):
            load_settings({"CVPILOT_STORE_DIR": bad})


def test_a_rank_value_that_is_not_plain_ascii_digits_counts_as_rank_zero_and_never_raises() -> None:
    from cvpilot.settings import is_rank_zero

    for raw in ("²", "٣", "1e3", "abc", "0x1", "", " ", "1" * 40, "--1"):
        assert is_rank_zero({"RANK": raw}) is True, raw
    assert is_rank_zero({"RANK": " 1 "}) is False and is_rank_zero({"LOCAL_RANK": "-1"}) is False
