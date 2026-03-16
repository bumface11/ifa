"""Tests for Streamlit app state normalization helpers."""

from __future__ import annotations

from pathlib import Path

from ifa.models import LumpSumEvent, SpendingStepEvent
from ifa.presets import save_preset
from ifa_web import (
    _build_simulation_inputs_from_state,
    _capture_comparison_snapshots,
    _sanitize_compare_preset_selection,
)


def test_build_simulation_inputs_from_state_reconstructs_saved_values() -> None:
    """Saved sidebar state should rebuild the matching simulation inputs."""
    state: dict[str, str | int | float | bool | None] = {
        "model_start_age_input": 56,
        "drawdown_start_age_input": 58,
        "end_age_input": 92,
        "tax_free_pot_input": 120_000.0,
        "baseline_spending_input": 34_000.0,
        "mean_return_input": 0.035,
        "std_return_input": 0.11,
        "random_seed_input": 9,
        "num_simulations_input": 800,
        "dc_pot_count": 2,
        "dc_name_0": "Main SIPP",
        "dc_start_age_0": 57,
        "dc_initial_balance_0": 420_000.0,
        "dc_name_1": "Late Pot",
        "dc_start_age_1": 67,
        "dc_initial_balance_1": 80_000.0,
        "db_stream_count": 1,
        "db_name_0": "Teacher Pension",
        "db_age_0": 65,
        "db_amount_0": 14_500.0,
        "lump_count": 1,
        "lump_name_0": "Kitchen",
        "lump_age_0": 70,
        "lump_amount_0": 16_000.0,
        "step_count": 1,
        "step_name_0": "Care",
        "step_start_0": 82,
        "step_amount_0": 7_500.0,
        "step_has_end_0": True,
        "step_end_0": 88,
    }

    inputs = _build_simulation_inputs_from_state(state, label="Compare Me")

    assert inputs.label == "Compare Me"
    assert inputs.start_age == 56
    assert inputs.drawdown_start_age == 58
    assert inputs.end_age == 92
    assert inputs.dc_pots == ((58, 420_000.0), (67, 80_000.0))
    assert inputs.dc_pot_names == ("Main SIPP", "Late Pot")
    assert inputs.db_pensions == ((65, 14_500.0),)
    assert inputs.db_pension_names == ("Teacher Pension",)
    assert inputs.life_event_names == ("Kitchen", "Care")
    assert inputs.life_events == (
        LumpSumEvent(age=70, amount=16_000.0),
        SpendingStepEvent(start_age=82, extra_per_year=7_500.0, end_age=88),
    )


def test_build_simulation_inputs_from_state_clamps_invalid_ages() -> None:
    """Invalid saved age ranges should be normalized into safe bounds."""
    state: dict[str, str | int | float | bool | None] = {
        "model_start_age_input": 84,
        "drawdown_start_age_input": 95,
        "end_age_input": 70,
        "dc_pot_count": 1,
        "dc_start_age_0": 120,
        "lump_count": 1,
        "lump_age_0": 12,
        "lump_amount_0": 10_000.0,
        "step_count": 0,
    }

    inputs = _build_simulation_inputs_from_state(state, label="Bounds")

    assert inputs.start_age == 84
    assert inputs.end_age == 89
    assert inputs.drawdown_start_age == 89
    assert inputs.dc_pots[0][0] == 89
    assert inputs.life_events == (LumpSumEvent(age=84, amount=10_000.0),)


def test_capture_comparison_snapshots_loads_selected_presets(tmp_path: Path) -> None:
    """Comparison snapshot loading should preserve order and saved names."""
    alpha = save_preset(tmp_path, "Alpha Plan", {"start_age_input": 55})
    beta = save_preset(tmp_path, "Beta Plan", {"start_age_input": 60})
    preset_map = {alpha.stem: alpha, beta.stem: beta}

    snapshots = _capture_comparison_snapshots([beta.stem, alpha.stem], preset_map)

    assert [label for label, _ in snapshots] == ["Beta Plan", "Alpha Plan"]
    assert snapshots[0][1]["drawdown_start_age_input"] == 60
    assert snapshots[0][1]["model_start_age_input"] == 59
    assert snapshots[1][1]["drawdown_start_age_input"] == 55
    assert snapshots[1][1]["model_start_age_input"] == 54


def test_sanitize_compare_preset_selection_filters_missing_and_duplicate() -> None:
    """Comparison defaults should drop stale and duplicate preset names."""
    selection = ["Alpha", "Missing", "Alpha", 123, "Beta"]

    sanitized = _sanitize_compare_preset_selection(selection, ["Alpha", "Beta"])

    assert sanitized == ["Alpha", "Beta"]


def test_sanitize_compare_preset_selection_handles_non_list() -> None:
    """Unexpected session-state types should degrade to an empty selection."""
    assert _sanitize_compare_preset_selection("Alpha", ["Alpha"]) == []