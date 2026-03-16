"""Tests for local sidebar preset persistence helpers."""

from __future__ import annotations

from pathlib import Path

from ifa.presets import (
    build_default_preset_name,
    get_preset_saved_at,
    list_preset_files,
    load_preset,
    rename_preset,
    sanitize_preset_filename,
    save_preset,
)


def test_sanitize_preset_filename_handles_symbols_and_spaces() -> None:
    """Filename sanitizer should produce a stable safe stem."""
    result = sanitize_preset_filename("  Growth & Income / v1  ")
    assert result == "Growth_Income_v1"


def test_save_and_load_preset_round_trip(tmp_path: Path) -> None:
    """Saved preset should round-trip name and sidebar state."""
    sidebar_state: dict[str, str | int | float | bool | None] = {
        "model_start_age_input": 54,
        "drawdown_start_age_input": 55,
        "baseline_spending_input": 30_000.0,
        "dc_name_0": "Main Pot",
    }

    target = save_preset(tmp_path, "My Preset", sidebar_state)
    loaded_name, loaded_state = load_preset(target)

    assert loaded_name == "My Preset"
    assert loaded_state["model_start_age_input"] == 54
    assert loaded_state["drawdown_start_age_input"] == 55
    assert loaded_state["baseline_spending_input"] == 30_000.0
    assert loaded_state["dc_name_0"] == "Main Pot"


def test_list_and_rename_presets(tmp_path: Path) -> None:
    """Preset listing and rename should track renamed file."""
    first = save_preset(
        tmp_path,
        "Preset One",
        {"model_start_age_input": 51, "drawdown_start_age_input": 52},
    )
    save_preset(
        tmp_path,
        "Preset Two",
        {"model_start_age_input": 59, "drawdown_start_age_input": 60},
    )

    files = list_preset_files(tmp_path)
    assert len(files) == 2

    renamed = rename_preset(first, "Renamed Preset")
    renamed_name, _ = load_preset(renamed)

    assert renamed_name == "Renamed Preset"
    assert renamed.exists()


def test_default_preset_name_has_prefix() -> None:
    """Default generated names should start with the expected prefix."""
    assert build_default_preset_name().startswith("Preset ")


def test_get_preset_saved_at_returns_iso_string(tmp_path: Path) -> None:
    """Saved preset payload should include a readable saved-at timestamp."""
    target = save_preset(
        tmp_path,
        "Timestamp Preset",
        {"model_start_age_input": 54, "drawdown_start_age_input": 55},
    )
    saved_at = get_preset_saved_at(target)
    assert isinstance(saved_at, str)
    assert len(saved_at) > 0


def test_load_preset_migrates_legacy_start_age_key(tmp_path: Path) -> None:
    """Legacy presets should be upgraded to model/drawdown start age keys."""
    target = save_preset(tmp_path, "Legacy", {"start_age_input": 60})
    _, loaded_state = load_preset(target)

    assert "start_age_input" not in loaded_state
    assert loaded_state["drawdown_start_age_input"] == 60
    assert loaded_state["model_start_age_input"] == 59


def test_load_preset_accepts_utf8_bom(tmp_path: Path) -> None:
    """Preset loader should parse JSON files even when a UTF-8 BOM is present."""
    target = tmp_path / "bom_preset.json"
    target.write_bytes(
        (
            b"\xef\xbb\xbf"
            + b'{"name":"BOM","sidebar_state":{"start_age_input":60}}'
        )
    )

    loaded_name, loaded_state = load_preset(target)

    assert loaded_name == "BOM"
    assert loaded_state["drawdown_start_age_input"] == 60
    assert loaded_state["model_start_age_input"] == 59
