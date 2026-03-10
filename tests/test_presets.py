"""Tests for local sidebar preset persistence helpers."""

from __future__ import annotations

from pathlib import Path

from ifa.presets import (
    build_default_preset_name,
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
    sidebar_state = {
        "start_age_input": 55,
        "baseline_spending_input": 30_000.0,
        "dc_name_0": "Main Pot",
    }

    target = save_preset(tmp_path, "My Preset", sidebar_state)
    loaded_name, loaded_state = load_preset(target)

    assert loaded_name == "My Preset"
    assert loaded_state["start_age_input"] == 55
    assert loaded_state["baseline_spending_input"] == 30_000.0
    assert loaded_state["dc_name_0"] == "Main Pot"


def test_list_and_rename_presets(tmp_path: Path) -> None:
    """Preset listing and rename should track renamed file."""
    first = save_preset(tmp_path, "Preset One", {"start_age_input": 52})
    save_preset(tmp_path, "Preset Two", {"start_age_input": 60})

    files = list_preset_files(tmp_path)
    assert len(files) == 2

    renamed = rename_preset(first, "Renamed Preset")
    renamed_name, _ = load_preset(renamed)

    assert renamed_name == "Renamed Preset"
    assert renamed.exists()


def test_default_preset_name_has_prefix() -> None:
    """Default generated names should start with the expected prefix."""
    assert build_default_preset_name().startswith("Preset ")
