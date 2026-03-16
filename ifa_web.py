"""Streamlit app for beginner-friendly pension drawdown exploration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ifa.config import (
    DB_PENSIONS,
    DRAWDOWN_START_AGE,
    DC_POTS,
    END_AGE,
    INITIAL_DC_POT,
    INITIAL_TAX_FREE_POT,
    MEAN_RETURN,
    MODEL_START_AGE,
    NUM_SIMULATIONS,
    RANDOM_SEED,
    STD_RETURN,
)
from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.events import (
    build_annual_spending_schedule,
    build_required_withdrawals,
    build_spending_drawdown_schedule,
)
from ifa.explain import build_plain_english_explanation
from ifa.metrics import (
    MonteCarloMetrics,
    PathMetrics,
    summarize_monte_carlo,
    summarize_path,
)
from ifa.models import LifeEvent, LumpSumEvent, SpendingStepEvent
from ifa.plotting import (
    plot_baseline_vs_scenario_balances,
    plot_individual_pots_subplots,
    plot_monte_carlo_fan_chart,
    plot_pots_stacked_area,
    plot_sequence_of_returns_scenarios,
)
from ifa.presets import (
    build_default_preset_name,
    delete_preset,
    get_preset_saved_at,
    list_preset_files,
    load_preset,
    sanitize_preset_filename,
    save_preset,
)
from ifa.strategies import create_fixed_real_drawdown_strategy

_TRACKED_STATIC_KEYS: tuple[str, ...] = (
    "model_start_age_input",
    "drawdown_start_age_input",
    "end_age_input",
    "tax_free_pot_input",
    "baseline_spending_input",
    "mean_return_input",
    "std_return_input",
    "random_seed_input",
    "num_simulations_input",
    "save_outputs_input",
    "dc_pot_count",
    "db_stream_count",
    "lump_count",
    "step_count",
)

StateValue: TypeAlias = str | int | float | bool | None
SidebarState: TypeAlias = dict[str, StateValue]
ComparisonSnapshot: TypeAlias = tuple[str, SidebarState]


@dataclass(frozen=True, slots=True)
class SimulationInputs:
    """Normalized simulation inputs for one rendered panel."""

    label: str
    start_age: int
    drawdown_start_age: int
    end_age: int
    tax_free_pot: float
    baseline_spending: float
    mean_return: float
    std_return: float
    random_seed: int
    num_simulations: int
    dc_pots: tuple[tuple[int, float], ...]
    dc_pot_names: tuple[str, ...]
    db_pensions: tuple[tuple[int, float], ...]
    db_pension_names: tuple[str, ...]
    life_events: tuple[LifeEvent, ...]
    life_event_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SimulationResults:
    """Computed results and figures for one rendered panel."""

    inputs: SimulationInputs
    ages: NDArray[np.int_]
    baseline_balances: NDArray[np.float64]
    scenario_balances: NDArray[np.float64]
    spending_drawdown_schedule: NDArray[np.float64]
    baseline_metrics: PathMetrics
    scenario_metrics: PathMetrics
    monte_carlo_metrics: MonteCarloMetrics
    explanation: str
    comparison_fig: Figure | None
    sequence_fig: Figure | None
    fan_fig: Figure | None
    stacked_fig: Figure | None
    individual_fig: Figure | None


def _ensure_sidebar_defaults() -> None:
    """Initialize sidebar widget defaults in session state once."""
    defaults: dict[str, int | float | bool | str] = {
        "model_start_age_input": MODEL_START_AGE,
        "drawdown_start_age_input": DRAWDOWN_START_AGE,
        "end_age_input": END_AGE,
        "tax_free_pot_input": float(INITIAL_TAX_FREE_POT),
        "baseline_spending_input": 30_000.0,
        "mean_return_input": MEAN_RETURN,
        "std_return_input": STD_RETURN,
        "random_seed_input": RANDOM_SEED,
        "num_simulations_input": NUM_SIMULATIONS,
        "save_outputs_input": False,
        "dc_pot_count": max(1, len(DC_POTS)),
        "db_stream_count": len(DB_PENSIONS),
        "lump_count": 1,
        "step_count": 1,
        "preset_name_input": build_default_preset_name(),
        "preset_selected": "(none)",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_pending_sidebar_updates() -> None:
    """Apply deferred sidebar state updates before widget instantiation."""
    loaded_state = st.session_state.pop("_loaded_sidebar_state", None)
    if isinstance(loaded_state, dict):
        _apply_sidebar_state(loaded_state)
        # Record what we loaded so unsaved-changes detection works.
        pending_sel = st.session_state.get("_pending_preset_selected")
        if isinstance(pending_sel, str):
            st.session_state["_last_loaded_preset_name"] = pending_sel
        st.session_state["_last_loaded_preset_state"] = loaded_state

    pending_selected = st.session_state.pop("_pending_preset_selected", None)
    if isinstance(pending_selected, str):
        st.session_state["preset_selected"] = pending_selected

    pending_name = st.session_state.pop("_pending_preset_name", None)
    if isinstance(pending_name, str):
        st.session_state["preset_name_input"] = pending_name


def _sanitize_compare_preset_selection(
    selection: object,
    available_options: Sequence[str],
) -> list[str]:
    """Return comparison preset selections that still exist in the preset list."""
    if not isinstance(selection, list):
        return []

    available_set = set(available_options)
    sanitized: list[str] = []
    for value in selection:
        if not isinstance(value, str):
            continue
        if value not in available_set or value in sanitized:
            continue
        sanitized.append(value)
    return sanitized


def _replace_compare_preset_selection(old_stem: str, new_stem: str) -> None:
    """Replace a renamed preset inside the comparison selection list."""
    current = _sanitize_compare_preset_selection(
        st.session_state.get("compare_preset_selection", []),
        [*st.session_state.get("compare_preset_selection", [])],
    )
    updated = [new_stem if value == old_stem else value for value in current]
    deduped: list[str] = []
    for value in updated:
        if value not in deduped:
            deduped.append(value)
    st.session_state["compare_preset_selection"] = deduped


def _remove_compare_preset_selection(stem: str) -> None:
    """Remove a deleted preset from the comparison selection list."""
    current = st.session_state.get("compare_preset_selection", [])
    if not isinstance(current, list):
        st.session_state["compare_preset_selection"] = []
        return
    st.session_state["compare_preset_selection"] = [
        value for value in current if isinstance(value, str) and value != stem
    ]



def _execute_preset_action(
    action: str,
    selected_preset: str,
    preset_map: dict[str, Path],
    preset_dir: Path,
) -> None:
    """Execute one requested preset action after all sidebar widgets are built.

    Actions:
        new: Generate a fresh default name and deselect the current preset.
        save: Update the selected preset (rename file if name changed); falls
            back to creating a new preset when nothing is selected.
        save_as: Always create a new preset file with the current name.
        delete: Remove the selected preset file.
        load: Load the selected preset (used by auto-load and discard-confirm).
    """
    preset_name = str(st.session_state.get("preset_name_input", "")).strip()

    if action == "new":
        st.session_state["_pending_preset_name"] = build_default_preset_name()
        st.session_state["_pending_preset_selected"] = "(none)"
        st.rerun()

    if action == "save":
        current_state = _collect_sidebar_state()
        if selected_preset != "(none)":
            save_name = preset_name or selected_preset
            old_stem = selected_preset
            new_stem = sanitize_preset_filename(save_name)
            saved_path = save_preset(preset_dir, save_name, current_state)
            if old_stem != new_stem and old_stem in preset_map:
                delete_preset(preset_map[old_stem])
                _replace_compare_preset_selection(old_stem, new_stem)
        else:
            save_name = preset_name or build_default_preset_name()
            saved_path = save_preset(preset_dir, save_name, current_state)
        st.session_state["_last_loaded_preset_name"] = saved_path.stem
        st.session_state["_last_loaded_preset_state"] = current_state
        st.session_state["_pending_preset_selected"] = saved_path.stem
        st.session_state["_pending_preset_name"] = save_name
        st.session_state["_preset_notice"] = f"Saved: {save_name}"
        st.rerun()

    if action == "save_as":
        current_state = _collect_sidebar_state()
        name = preset_name or build_default_preset_name()
        saved_path = save_preset(preset_dir, name, current_state)
        st.session_state["_last_loaded_preset_name"] = saved_path.stem
        st.session_state["_last_loaded_preset_state"] = current_state
        st.session_state["_pending_preset_selected"] = saved_path.stem
        st.session_state["_pending_preset_name"] = name
        st.session_state["_preset_notice"] = f"Saved copy: {name}"
        st.rerun()

    if action == "delete":
        if selected_preset == "(none)":
            st.session_state["_preset_notice"] = "Choose a preset before deleting."
            return
        delete_preset(preset_map[selected_preset])
        _remove_compare_preset_selection(selected_preset)
        st.session_state.pop("_last_loaded_preset_name", None)
        st.session_state.pop("_last_loaded_preset_state", None)
        st.session_state["_pending_preset_selected"] = "(none)"
        st.session_state["_pending_preset_name"] = build_default_preset_name()
        st.session_state["_preset_notice"] = f"Deleted: {selected_preset}"
        st.rerun()

    if action == "load":
        if selected_preset == "(none)":
            return
        loaded_name, loaded_state = load_preset(preset_map[selected_preset])
        st.session_state["_loaded_sidebar_state"] = loaded_state
        st.session_state["_pending_preset_name"] = loaded_name
        st.session_state["_pending_preset_selected"] = selected_preset
        st.session_state["_preset_notice"] = f"Loaded: {selected_preset}"
        st.rerun()


def _tracked_dynamic_keys() -> list[str]:
    """Build the dynamic widget-key list based on current item counts."""
    keys: list[str] = []
    pot_count = int(st.session_state.get("dc_pot_count", max(1, len(DC_POTS))))
    for index in range(max(0, pot_count)):
        keys.extend(
            [
                f"dc_name_{index}",
                f"dc_start_age_{index}",
                f"dc_initial_balance_{index}",
            ]
        )

    db_count = int(st.session_state.get("db_stream_count", len(DB_PENSIONS)))
    for index in range(max(0, db_count)):
        keys.extend([f"db_name_{index}", f"db_age_{index}", f"db_amount_{index}"])

    lump_count = int(st.session_state.get("lump_count", 1))
    for index in range(max(0, lump_count)):
        keys.extend([f"lump_name_{index}", f"lump_age_{index}", f"lump_amount_{index}"])

    step_count = int(st.session_state.get("step_count", 1))
    for index in range(max(0, step_count)):
        keys.extend(
            [
                f"step_name_{index}",
                f"step_start_{index}",
                f"step_amount_{index}",
                f"step_has_end_{index}",
                f"step_end_{index}",
            ]
        )
    return keys


def _collect_sidebar_state() -> SidebarState:
    """Collect tracked sidebar widget values from session state."""
    state: SidebarState = {}
    for key in [*_TRACKED_STATIC_KEYS, *_tracked_dynamic_keys()]:
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            state[key] = value
    return state


def _apply_sidebar_state(
    state: SidebarState,
) -> None:
    """Apply loaded sidebar values into Streamlit session state."""
    for key, value in state.items():
        st.session_state[key] = value


def _format_saved_at_label(saved_at_iso: str | None) -> str:
    """Format an ISO timestamp into a readable local label."""
    if saved_at_iso is None:
        return "Last saved: unknown"

    try:
        timestamp = datetime.fromisoformat(saved_at_iso)
    except ValueError:
        return "Last saved: unknown"

    return "Last saved: " + timestamp.astimezone().strftime("%Y-%m-%d %H:%M")


def _has_unsaved_changes() -> bool:
    """Return True if the current sidebar state differs from the last loaded preset.

    Returns:
        True when a preset has been loaded and at least one tracked parameter
        has since changed; False otherwise.
    """
    last_name = st.session_state.get("_last_loaded_preset_name")
    if not last_name or last_name == "(none)":
        return False
    last_state = st.session_state.get("_last_loaded_preset_state")
    if not isinstance(last_state, dict) or not last_state:
        return False
    return _collect_sidebar_state() != last_state


def _on_preset_selectbox_change() -> None:
    """Stage an auto-load when the user selects a different preset."""
    new_sel = st.session_state.get("preset_selected", "(none)")
    if new_sel != "(none)":
        st.session_state["_pending_preset_auto_load"] = new_sel


def _build_db_income(
    ages: np.ndarray, db_pensions: Sequence[tuple[int, float]]
) -> np.ndarray:
    """Build annual DB income aligned to ages."""
    return np.array(
        [calculate_db_pension_income(int(age), db_pensions) for age in ages],
        dtype=np.float64,
    )


def _build_dc_pots(
    drawdown_floor_age: int,
    end_age: int,
) -> tuple[list[tuple[int, float]], list[str]]:
    """Collect DC pot balances, names, and drawdown start ages."""
    st.markdown("DC Pots")
    default_count = len(DC_POTS)
    pot_count = int(
        st.number_input(
            "Number of DC pots",
            min_value=1,
            max_value=6,
            value=max(1, default_count),
            key="dc_pot_count",
            help="Each DC pot can have its own drawdown start age.",
        )
    )

    pots: list[tuple[int, float]] = []
    pot_names: list[str] = []
    for index in range(pot_count):
        if index < default_count:
            default_start_age = int(DC_POTS[index][0])
            default_balance = float(DC_POTS[index][1])
        elif index == 0:
            default_start_age = 57
            default_balance = float(INITIAL_DC_POT)
        else:
            default_start_age = max(drawdown_floor_age, 57)
            default_balance = 0.0

        pot_name = st.text_input(
            f"DC pot #{index + 1} name",
            value=f"DC Pot {index + 1}",
            key=f"dc_name_{index}",
            help="Default name can be edited to anything you prefer.",
        )
        pot_drawdown_start_age = int(
            st.number_input(
                f"DC pot #{index + 1} drawdown start age",
                min_value=drawdown_floor_age,
                max_value=end_age,
                value=min(max(default_start_age, drawdown_floor_age), end_age),
                key=f"dc_start_age_{index}",
                help="This pot can only be used for withdrawals from this age onward.",
            )
        )
        initial_balance = float(
            st.number_input(
                f"DC pot #{index + 1} initial balance £",
                min_value=0.0,
                value=default_balance,
                step=1_000.0,
                key=f"dc_initial_balance_{index}",
            )
        )
        pots.append((pot_drawdown_start_age, initial_balance))
        pot_names.append(pot_name.strip() or f"DC Pot {index + 1}")

    return pots, pot_names


def _build_life_events(
    model_start_age: int,
    end_age: int,
) -> tuple[tuple[LifeEvent, ...], list[str]]:
    """Collect life events from active Streamlit container inputs."""
    life_events: list[LifeEvent] = []
    life_event_names: list[str] = []

    st.markdown("Life Events")
    lump_count = int(
        st.number_input(
            "Lump-sum events",
            min_value=0,
            max_value=5,
            value=1,
            key="lump_count",
            help="How many one-off spending events you want to model.",
        )
    )
    for index in range(lump_count):
        event_name = st.text_input(
            f"Lump-sum event #{index + 1} name",
            value=f"Lump Sum {index + 1}",
            key=f"lump_name_{index}",
            help="Default name can be edited to anything you prefer.",
        )
        st.markdown(f"Lump sum #{index + 1}")
        lump_age = int(
            st.number_input(
                f"Age (lump #{index + 1})",
                min_value=model_start_age,
                max_value=end_age,
                value=min(model_start_age + 3 + index, end_age),
                key=f"lump_age_{index}",
                help="The age when this one-off cost happens.",
            )
        )
        lump_amount = float(
            st.number_input(
                f"Amount £ (lump #{index + 1})",
                min_value=0.0,
                value=18_000.0,
                step=1_000.0,
                key=f"lump_amount_{index}",
                help="Amount of this one-off extra cost in £.",
            )
        )
        if lump_amount > 0.0:
            life_events.append(LumpSumEvent(age=lump_age, amount=lump_amount))
            life_event_names.append(event_name.strip() or f"Lump Sum {index + 1}")

    step_count = int(
        st.number_input(
            "Spending-step events",
            min_value=0,
            max_value=5,
            value=1,
            key="step_count",
            help="How many ongoing extra yearly costs you want to model.",
        )
    )
    for index in range(step_count):
        event_name = st.text_input(
            f"Spending-step event #{index + 1} name",
            value=f"Spending Step {index + 1}",
            key=f"step_name_{index}",
            help="Default name can be edited to anything you prefer.",
        )
        st.markdown(f"Spending step #{index + 1}")
        step_start = int(
            st.number_input(
                f"Start age (step #{index + 1})",
                min_value=model_start_age,
                max_value=end_age,
                value=min(model_start_age + 10 + index, end_age),
                key=f"step_start_{index}",
                help="The age when this ongoing extra cost starts.",
            )
        )
        step_amount = float(
            st.number_input(
                f"Extra per year £ (step #{index + 1})",
                min_value=0.0,
                value=6_000.0,
                step=500.0,
                key=f"step_amount_{index}",
                help="Extra spending per year in £ for this step event.",
            )
        )
        has_end = st.checkbox(
            f"Set end age (step #{index + 1})",
            value=False,
            key=f"step_has_end_{index}",
            help="Enable this if the extra yearly cost should stop at a later age.",
        )
        step_end = None
        if has_end:
            step_end = int(
                st.number_input(
                    f"End age (step #{index + 1})",
                    min_value=step_start,
                    max_value=end_age,
                    value=end_age,
                    key=f"step_end_{index}",
                    help="The last age this extra yearly cost applies.",
                )
            )

        if step_amount > 0.0:
            life_events.append(
                SpendingStepEvent(
                    start_age=step_start,
                    extra_per_year=step_amount,
                    end_age=step_end,
                )
            )
            life_event_names.append(
                event_name.strip() or f"Spending Step {index + 1}"
            )

    return tuple(life_events), life_event_names


def _build_db_pensions(
    model_start_age: int,
    end_age: int,
) -> tuple[list[tuple[int, float]], list[str]]:
    """Collect DB pension inputs and names from the active container."""
    st.markdown("DB Pensions")
    default_streams = len(DB_PENSIONS)
    stream_count = int(
        st.number_input(
            "DB income streams",
            min_value=0,
            max_value=6,
            value=default_streams,
            key="db_stream_count",
            help="How many defined-benefit pension income streams you receive.",
        )
    )
    pensions: list[tuple[int, float]] = []
    pension_names: list[str] = []
    for index in range(stream_count):
        default_age = (
            DB_PENSIONS[index][0] if index < default_streams else model_start_age
        )
        default_amount = DB_PENSIONS[index][1] if index < default_streams else 10_000.0
        pension_name = st.text_input(
            f"DB stream name #{index + 1}",
            value=f"DB Pension {index + 1}",
            key=f"db_name_{index}",
            help="Default name can be edited to anything you prefer.",
        )
        stream_age = int(
            st.number_input(
                f"DB start age #{index + 1}",
                min_value=model_start_age,
                max_value=end_age,
                value=default_age,
                key=f"db_age_{index}",
                help="The age when this DB pension income starts.",
            )
        )
        stream_amount = float(
            st.number_input(
                f"DB annual amount £ #{index + 1}",
                min_value=0.0,
                value=float(default_amount),
                step=500.0,
                key=f"db_amount_{index}",
                help="Yearly income amount for this DB pension stream in £.",
            )
        )
        pensions.append((stream_age, stream_amount))
        pension_names.append(pension_name.strip() or f"DB Pension {index + 1}")
    return pensions, pension_names


def _coerce_int(
    value: StateValue,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce a stored widget value to an int within optional bounds."""
    if isinstance(value, bool):
        result = default
    elif isinstance(value, int):
        result = value
    elif isinstance(value, float):
        result = int(value)
    else:
        result = default

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _coerce_float(
    value: StateValue,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce a stored widget value to a float within optional bounds."""
    if isinstance(value, bool):
        result = default
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        result = default

    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _coerce_bool(value: StateValue, default: bool) -> bool:
    """Coerce a stored widget value to a bool."""
    return value if isinstance(value, bool) else default


def _coerce_str(value: StateValue, default: str) -> str:
    """Coerce a stored widget value to a stripped string."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    return default


def _build_simulation_inputs_from_state(
    state: Mapping[str, StateValue],
    label: str,
) -> SimulationInputs:
    """Build normalized simulation inputs from tracked sidebar state."""
    model_start_age = _coerce_int(
        state.get("model_start_age_input"),
        _coerce_int(state.get("start_age_input"), MODEL_START_AGE),
        minimum=40,
        maximum=85,
    )
    end_age = _coerce_int(
        state.get("end_age_input"),
        END_AGE,
        minimum=model_start_age + 5,
        maximum=110,
    )
    drawdown_start_age = _coerce_int(
        state.get("drawdown_start_age_input"),
        _coerce_int(state.get("start_age_input"), DRAWDOWN_START_AGE),
        minimum=model_start_age,
        maximum=end_age,
    )
    tax_free_pot = _coerce_float(
        state.get("tax_free_pot_input"),
        float(INITIAL_TAX_FREE_POT),
        minimum=0.0,
    )
    baseline_spending = _coerce_float(
        state.get("baseline_spending_input"),
        30_000.0,
        minimum=0.0,
    )
    mean_return = _coerce_float(
        state.get("mean_return_input"),
        MEAN_RETURN,
        minimum=-0.05,
        maximum=0.15,
    )
    std_return = _coerce_float(
        state.get("std_return_input"),
        STD_RETURN,
        minimum=0.01,
        maximum=0.30,
    )
    random_seed = _coerce_int(state.get("random_seed_input"), RANDOM_SEED, minimum=0)
    num_simulations = _coerce_int(
        state.get("num_simulations_input"),
        NUM_SIMULATIONS,
        minimum=100,
        maximum=10_000,
    )

    default_pot_count = max(1, len(DC_POTS))
    pot_count = _coerce_int(
        state.get("dc_pot_count"),
        default_pot_count,
        minimum=1,
        maximum=6,
    )
    dc_pots: list[tuple[int, float]] = []
    dc_pot_names: list[str] = []
    for index in range(pot_count):
        if index < len(DC_POTS):
            default_start_age = int(DC_POTS[index][0])
            default_balance = float(DC_POTS[index][1])
        elif index == 0:
            default_start_age = 57
            default_balance = float(INITIAL_DC_POT)
        else:
            default_start_age = max(drawdown_start_age, 57)
            default_balance = 0.0

        dc_pot_names.append(
            _coerce_str(state.get(f"dc_name_{index}"), f"DC Pot {index + 1}")
        )
        dc_pots.append(
            (
                _coerce_int(
                    state.get(f"dc_start_age_{index}"),
                    min(max(default_start_age, drawdown_start_age), end_age),
                    minimum=drawdown_start_age,
                    maximum=end_age,
                ),
                _coerce_float(
                    state.get(f"dc_initial_balance_{index}"),
                    default_balance,
                    minimum=0.0,
                ),
            )
        )

    default_db_count = len(DB_PENSIONS)
    db_count = _coerce_int(
        state.get("db_stream_count"),
        default_db_count,
        minimum=0,
        maximum=6,
    )
    db_pensions: list[tuple[int, float]] = []
    db_pension_names: list[str] = []
    for index in range(db_count):
        default_age = (
            DB_PENSIONS[index][0]
            if index < default_db_count
            else model_start_age
        )
        default_amount = DB_PENSIONS[index][1] if index < default_db_count else 10_000.0
        db_pension_names.append(
            _coerce_str(state.get(f"db_name_{index}"), f"DB Pension {index + 1}")
        )
        db_pensions.append(
            (
                _coerce_int(
                    state.get(f"db_age_{index}"),
                    default_age,
                    minimum=model_start_age,
                    maximum=end_age,
                ),
                _coerce_float(
                    state.get(f"db_amount_{index}"),
                    float(default_amount),
                    minimum=0.0,
                ),
            )
        )

    lump_count = _coerce_int(state.get("lump_count"), 1, minimum=0, maximum=5)
    step_count = _coerce_int(state.get("step_count"), 1, minimum=0, maximum=5)
    life_events: list[LifeEvent] = []
    life_event_names: list[str] = []

    for index in range(lump_count):
        amount = _coerce_float(
            state.get(f"lump_amount_{index}"),
            18_000.0,
            minimum=0.0,
        )
        if amount <= 0.0:
            continue
        age = _coerce_int(
            state.get(f"lump_age_{index}"),
            min(model_start_age + 3 + index, end_age),
            minimum=model_start_age,
            maximum=end_age,
        )
        life_events.append(LumpSumEvent(age=age, amount=amount))
        life_event_names.append(
            _coerce_str(state.get(f"lump_name_{index}"), f"Lump Sum {index + 1}")
        )

    for index in range(step_count):
        amount = _coerce_float(
            state.get(f"step_amount_{index}"),
            6_000.0,
            minimum=0.0,
        )
        if amount <= 0.0:
            continue
        step_start = _coerce_int(
            state.get(f"step_start_{index}"),
            min(model_start_age + 10 + index, end_age),
            minimum=model_start_age,
            maximum=end_age,
        )
        has_end = _coerce_bool(state.get(f"step_has_end_{index}"), False)
        step_end = None
        if has_end:
            step_end = _coerce_int(
                state.get(f"step_end_{index}"),
                end_age,
                minimum=step_start,
                maximum=end_age,
            )
        life_events.append(
            SpendingStepEvent(
                start_age=step_start,
                extra_per_year=amount,
                end_age=step_end,
            )
        )
        life_event_names.append(
            _coerce_str(
                state.get(f"step_name_{index}"),
                f"Spending Step {index + 1}",
            )
        )

    return SimulationInputs(
        label=label.strip() or "Preset",
        start_age=model_start_age,
        drawdown_start_age=drawdown_start_age,
        end_age=end_age,
        tax_free_pot=tax_free_pot,
        baseline_spending=baseline_spending,
        mean_return=mean_return,
        std_return=std_return,
        random_seed=random_seed,
        num_simulations=num_simulations,
        dc_pots=tuple(dc_pots),
        dc_pot_names=tuple(dc_pot_names),
        db_pensions=tuple(db_pensions),
        db_pension_names=tuple(db_pension_names),
        life_events=tuple(life_events),
        life_event_names=tuple(life_event_names),
    )


def _capture_comparison_snapshots(
    selected_preset_names: Sequence[str],
    preset_map: Mapping[str, Path],
) -> list[ComparisonSnapshot]:
    """Load saved preset states for the comparison workspace."""
    snapshots: list[ComparisonSnapshot] = []
    for preset_name in selected_preset_names:
        preset_path = preset_map.get(preset_name)
        if preset_path is None:
            continue
        loaded_name, loaded_state = load_preset(preset_path)
        snapshots.append((loaded_name, loaded_state))
    return snapshots


def _run_simulation_panel(
    inputs: SimulationInputs,
    *,
    save_outputs: bool,
    output_dir: Path,
) -> SimulationResults:
    """Run one full simulation panel and build all derived charts."""
    primary_dc_pot = float(inputs.dc_pots[0][1]) if len(inputs.dc_pots) > 0 else 0.0
    secondary_dc_pot = (
        float(sum(pot[1] for pot in inputs.dc_pots[1:]))
        if len(inputs.dc_pots) > 1
        else 0.0
    )
    secondary_draw_age = (
        int(inputs.dc_pots[1][0]) if len(inputs.dc_pots) > 1 else inputs.end_age
    )

    ages = np.arange(inputs.start_age, inputs.end_age + 1, dtype=np.int_)
    db_income = _build_db_income(ages, inputs.db_pensions)
    baseline_required = build_required_withdrawals(
        ages=ages,
        baseline_spending=inputs.baseline_spending,
        db_income=db_income,
        events=(),
    )
    scenario_required = build_required_withdrawals(
        ages=ages,
        baseline_spending=inputs.baseline_spending,
        db_income=db_income,
        events=inputs.life_events,
    )
    spending_drawdown_schedule = build_spending_drawdown_schedule(
        ages=ages,
        baseline_spending=inputs.baseline_spending,
        db_income=db_income,
        events=inputs.life_events,
    )
    pre_drawdown_mask = ages < inputs.drawdown_start_age
    baseline_required[pre_drawdown_mask] = 0.0
    scenario_required[pre_drawdown_mask] = 0.0
    spending_drawdown_schedule[pre_drawdown_mask] = 0.0
    annual_spending_schedule = build_annual_spending_schedule(
        ages=ages,
        baseline_spending=inputs.baseline_spending,
        events=inputs.life_events,
    )

    years = inputs.end_age - inputs.start_age
    returns = (
        np.random.default_rng(inputs.random_seed)
        .normal(inputs.mean_return, inputs.std_return, years)
        .astype(np.float64)
    )
    base_strategy = create_fixed_real_drawdown_strategy(inputs.baseline_spending)

    _, baseline_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=inputs.tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        start_age=inputs.start_age,
        end_age=inputs.end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=baseline_required,
        dc_pots=inputs.dc_pots,
    )
    _, scenario_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=inputs.tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        start_age=inputs.start_age,
        end_age=inputs.end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=scenario_required,
        dc_pots=inputs.dc_pots,
    )
    _, monte_carlo_paths = run_monte_carlo_simulation(
        tax_free_pot=inputs.tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        start_age=inputs.start_age,
        end_age=inputs.end_age,
        mean_return=inputs.mean_return,
        std_return=inputs.std_return,
        strategy_fn=base_strategy,
        num_simulations=inputs.num_simulations,
        seed=inputs.random_seed,
        withdrawals_required=scenario_required,
        dc_pots=inputs.dc_pots,
    )

    baseline_metrics = summarize_path(baseline_balances)
    scenario_metrics = summarize_path(scenario_balances)
    monte_carlo_metrics = summarize_monte_carlo(monte_carlo_paths)
    explanation = build_plain_english_explanation(
        baseline_metrics=baseline_metrics,
        scenario_metrics=scenario_metrics,
        monte_carlo_metrics=monte_carlo_metrics,
        events=inputs.life_events,
        event_names=inputs.life_event_names,
    )

    output_stem = sanitize_preset_filename(inputs.label)
    comparison_fig = plot_baseline_vs_scenario_balances(
        ages=ages,
        baseline_balances=baseline_balances,
        scenario_balances=scenario_balances,
        spending_drawdown_schedule=spending_drawdown_schedule,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        life_events=inputs.life_events,
        dc_pots=inputs.dc_pots,
        dc_pot_names=inputs.dc_pot_names,
        db_pension_names=inputs.db_pension_names,
        life_event_names=inputs.life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / f"{output_stem}_baseline_vs_scenario_streamlit.png",
    )
    sequence_fig = plot_sequence_of_returns_scenarios(
        tax_free_pot=inputs.tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        start_age=inputs.start_age,
        end_age=inputs.end_age,
        mean_return=inputs.mean_return,
        std_return=inputs.std_return,
        strategy_fn=base_strategy,
        withdrawals_required=scenario_required,
        life_events=inputs.life_events,
        spending_drawdown_schedule=spending_drawdown_schedule,
        dc_pots=inputs.dc_pots,
        dc_pot_names=inputs.dc_pot_names,
        db_pension_names=inputs.db_pension_names,
        life_event_names=inputs.life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / f"{output_stem}_sequence_scenarios_streamlit.png",
    )
    fan_fig = plot_monte_carlo_fan_chart(
        tax_free_pot=inputs.tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        start_age=inputs.start_age,
        end_age=inputs.end_age,
        mean_return=inputs.mean_return,
        std_return=inputs.std_return,
        strategy_fn=base_strategy,
        num_simulations=inputs.num_simulations,
        seed=inputs.random_seed,
        withdrawals_required=scenario_required,
        life_events=inputs.life_events,
        spending_drawdown_schedule=spending_drawdown_schedule,
        dc_pots=inputs.dc_pots,
        dc_pot_names=inputs.dc_pot_names,
        db_pension_names=inputs.db_pension_names,
        life_event_names=inputs.life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / f"{output_stem}_monte_carlo_fan_streamlit.png",
    )
    stacked_fig = plot_pots_stacked_area(
        tax_free_pot=inputs.tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        start_age=inputs.start_age,
        end_age=inputs.end_age,
        mean_return=inputs.mean_return,
        std_return=inputs.std_return,
        strategy_fn=base_strategy,
        seed=inputs.random_seed,
        withdrawals_required=scenario_required,
        life_events=inputs.life_events,
        spending_drawdown_schedule=spending_drawdown_schedule,
        dc_pots=inputs.dc_pots,
        dc_pot_names=inputs.dc_pot_names,
        db_pension_names=inputs.db_pension_names,
        life_event_names=inputs.life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / f"{output_stem}_pots_stacked_streamlit.png",
    )
    individual_fig = plot_individual_pots_subplots(
        tax_free_pot=inputs.tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=inputs.db_pensions,
        start_age=inputs.start_age,
        end_age=inputs.end_age,
        mean_return=inputs.mean_return,
        std_return=inputs.std_return,
        strategy_fn=base_strategy,
        seed=inputs.random_seed,
        withdrawals_required=scenario_required,
        life_events=inputs.life_events,
        annual_spending_schedule=annual_spending_schedule,
        dc_pots=inputs.dc_pots,
        dc_pot_names=inputs.dc_pot_names,
        db_pension_names=inputs.db_pension_names,
        life_event_names=inputs.life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / f"{output_stem}_pots_individual_streamlit.png",
    )

    return SimulationResults(
        inputs=inputs,
        ages=ages,
        baseline_balances=baseline_balances,
        scenario_balances=scenario_balances,
        spending_drawdown_schedule=spending_drawdown_schedule,
        baseline_metrics=baseline_metrics,
        scenario_metrics=scenario_metrics,
        monte_carlo_metrics=monte_carlo_metrics,
        explanation=explanation,
        comparison_fig=comparison_fig,
        sequence_fig=sequence_fig,
        fan_fig=fan_fig,
        stacked_fig=stacked_fig,
        individual_fig=individual_fig,
    )


def _format_panel_caption(inputs: SimulationInputs) -> str:
    """Build a short caption describing the assumptions for one panel."""
    return (
        f"Ages {inputs.start_age}-{inputs.end_age} | "
        f"Drawdown from {inputs.drawdown_start_age} | "
        f"Spending £{inputs.baseline_spending:,.0f} | "
        f"Mean {inputs.mean_return * 100:.1f}% | "
        f"Vol {inputs.std_return * 100:.1f}%"
    )


def _render_figure(title: str, figure: Figure | None) -> None:
    """Render one matplotlib figure if available."""
    if figure is None:
        return
    st.markdown(f"#### {title}")
    st.pyplot(figure, clear_figure=True, width="stretch")
    plt.close(figure)


def _close_simulation_figures(results: SimulationResults) -> None:
    """Close any matplotlib figures attached to one result bundle."""
    for figure in (
        results.comparison_fig,
        results.sequence_fig,
        results.fan_fig,
        results.stacked_fig,
        results.individual_fig,
    ):
        if figure is not None:
            plt.close(figure)


def _render_simulation_results(
    results: SimulationResults,
    *,
    compact: bool = False,
) -> None:
    """Render one simulation result block."""
    st.markdown(f"### {results.inputs.label}")
    st.caption(_format_panel_caption(results.inputs))

    if compact:
        st.metric("Baseline ending", f"£{results.baseline_metrics.ending_balance:,.0f}")
        st.metric("Scenario ending", f"£{results.scenario_metrics.ending_balance:,.0f}")
        st.metric(
            "Ruin probability",
            f"{results.monte_carlo_metrics.ruin_probability * 100:.1f}%",
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Baseline ending",
            f"£{results.baseline_metrics.ending_balance:,.0f}",
        )
        col2.metric(
            "Scenario ending",
            f"£{results.scenario_metrics.ending_balance:,.0f}",
        )
        col3.metric(
            "Ruin probability",
            f"{results.monte_carlo_metrics.ruin_probability * 100:.1f}%",
        )

    st.markdown("#### Plain-English Summary")
    st.write(results.explanation)
    _render_figure("Baseline vs Life-Events Scenario", results.comparison_fig)
    _render_figure("Sequence-of-Returns Teaching Chart", results.sequence_fig)
    _render_figure("Monte Carlo Fan Chart", results.fan_fig)
    st.markdown("#### Pot Charts Guide")
    st.write(
        "The next two charts show the same retirement path from two angles. "
        "The stacked chart shows how each pot contributes to your total over time. "
        "The 4-panel chart separates each pot so you can see which pot is being "
        "used first, how quickly it changes, and when DB income becomes important."
    )
    _render_figure("Pot Composition (Stacked Area)", results.stacked_fig)
    _render_figure("Pot and Income Panels (4 Panels)", results.individual_fig)


def _render_comparison_workspace(
    results: Sequence[SimulationResults],
    layout_mode: str,
) -> None:
    """Render comparison results either in columns or one panel at a time."""
    if len(results) == 0:
        st.info(
            "Select one or more saved presets in the sidebar to compare them "
            "here after the next run."
        )
        return

    if layout_mode == "Focus one preset":
        labels = [result.inputs.label for result in results]
        selected_label = st.radio(
            "Preset output",
            options=labels,
            horizontal=True,
            key="comparison_focus_label",
        )
        for result in results:
            if result.inputs.label == selected_label:
                _render_simulation_results(result)
                break
        return

    columns = st.columns(len(results))
    for column, result in zip(columns, results, strict=True):
        with column:
            _render_simulation_results(result, compact=True)


def main() -> None:
    """Render the Streamlit pension simulator UI and outputs."""
    st.set_page_config(page_title="IFA Pension Simulator", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Chivo:wght@400;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
        html, body, [class*="css"]  {
            font-family: "Chivo", sans-serif;
            color: var(--text-color) !important;
        }
        h1, h2, h3 {
            font-family: "Fraunces", serif;
            letter-spacing: 0.3px;
            color: var(--text-color) !important;
        }
        p, li, label, span, .stCaption {
            color: var(--text-color) !important;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--text-color) !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
        [data-testid="stSidebar"] details summary,
        [data-testid="stSidebar"] details summary *,
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button *,
        [data-testid="stSidebar"] svg,
        [data-testid="stSidebar"] summary svg {
            color: var(--text-color) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--text-color) !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.55rem;
            line-height: 1.1;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem;
        }
        .stMarkdown small {
            color: var(--text-color) !important;
            opacity: 0.85;
        }
        .stApp {
            background: linear-gradient(
                160deg,
                var(--background-color) 0%,
                var(--secondary-background-color) 100%
            ) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("IFA Pension Drawdown Simulator")
    st.caption(
        "Experiment with retirement assumptions and life events,"
        " then compare the impact"
        " with plain-English explanations."
    )

    run_model = st.button("Run simulation", type="primary")

    _ensure_sidebar_defaults()
    _apply_pending_sidebar_updates()

    preset_dir = Path("saved_parameters")
    preset_files = list_preset_files(preset_dir)
    preset_map = {preset_path.stem: preset_path for preset_path in preset_files}
    preset_options = ["(none)", *preset_map.keys()]
    compare_selection = _sanitize_compare_preset_selection(
        st.session_state.get("compare_preset_selection", []),
        list(preset_map.keys()),
    )
    st.session_state["compare_preset_selection"] = compare_selection
    if st.session_state.get("preset_selected") not in preset_options:
        st.session_state["preset_selected"] = "(none)"

    st.sidebar.header("Saved Parameter Sets")
    with st.sidebar.container():
        notice = st.session_state.pop("_preset_notice", None)
        if isinstance(notice, str) and len(notice) > 0:
            if notice.startswith("Choose"):
                st.warning(notice)
            else:
                st.success(notice)

        st.text_input(
            "Preset name",
            key="preset_name_input",
            help=(
                "Name for this preset. Edit the name then click Save to update "
                "and rename the selected preset in one step."
            ),
        )
        selected_preset = st.selectbox(
            "Saved presets",
            options=preset_options,
            key="preset_selected",
            on_change=_on_preset_selectbox_change,
            help="Selecting a preset loads it automatically.",
        )
        if selected_preset != "(none)":
            selected_saved_at = get_preset_saved_at(preset_map[selected_preset])
            st.caption(_format_saved_at_label(selected_saved_at))

        discard_for = st.session_state.get("_show_discard_warning_for")
        if isinstance(discard_for, str):
            st.warning(f"Unsaved changes — load \"{discard_for}\"?")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                if st.button(
                    "Load anyway",
                    key="preset_discard_ok",
                    width="stretch",
                ):
                    st.session_state.pop("_show_discard_warning_for", None)
                    st.session_state["_pending_preset_action"] = "load"
            with d_col2:
                if st.button(
                    "Cancel",
                    key="preset_discard_cancel",
                    width="stretch",
                ):
                    st.session_state.pop("_show_discard_warning_for", None)
                    last = st.session_state.get("_last_loaded_preset_name") or "(none)"
                    st.session_state["_pending_preset_selected"] = (
                        last if last in preset_map else "(none)"
                    )
                    st.rerun()

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            if st.button("New", key="preset_new_button", width="stretch"):
                st.session_state["_pending_preset_action"] = "new"
        with r1c2:
            if st.button("Save", key="preset_save_button", width="stretch"):
                st.session_state["_pending_preset_action"] = "save"

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if st.button(
                "Save As",
                key="preset_save_as_button",
                width="stretch",
            ):
                st.session_state["_pending_preset_action"] = "save_as"
        with r2c2:
            if st.button(
                "Delete",
                key="preset_delete_button",
                width="stretch",
            ):
                st.session_state["_pending_preset_action"] = "delete"

    display_mode = st.sidebar.radio(
        "Display mode",
        options=("Current inputs", "Compare saved presets"),
        index=0,
        help=(
            "Use current inputs for the editable working view. Use compare "
            "saved presets to show only saved preset outputs."
        ),
    )

    st.sidebar.header("Preset Comparison")
    comparison_layout = st.sidebar.radio(
        "Comparison layout",
        options=("Side by side", "Focus one preset"),
        index=0,
        help=(
            "Use side-by-side on larger screens. Use focus mode on smaller "
            "screens to inspect one preset output at a time."
        ),
    )
    selected_compare_presets = st.sidebar.multiselect(
        "Saved presets to compare",
        options=list(preset_map.keys()),
        default=compare_selection,
        max_selections=3,
        key="compare_preset_selection",
        help="Choose up to three saved presets for the comparison workspace.",
    )

    st.sidebar.header("Core Inputs")
    model_start_age = int(
        st.sidebar.number_input(
            "Model start age",
            min_value=40,
            max_value=85,
            value=MODEL_START_AGE,
            key="model_start_age_input",
            help="The first age included in the projection timeline.",
        )
    )
    end_age = int(
        st.sidebar.number_input(
            "End age",
            min_value=model_start_age + 5,
            max_value=110,
            value=END_AGE,
            key="end_age_input",
        )
    )
    drawdown_start_age = int(
        st.sidebar.number_input(
            "Drawdown start age",
            min_value=model_start_age,
            max_value=end_age,
            value=min(max(DRAWDOWN_START_AGE, model_start_age), end_age),
            key="drawdown_start_age_input",
            help=(
                "Withdrawals are forced to zero before this age, so pots only "
                "grow (or fall) with market returns during the gap."
            ),
        )
    )

    st.sidebar.number_input(
        "Tax-free pot £",
        min_value=0.0,
        value=float(INITIAL_TAX_FREE_POT),
        step=1_000.0,
        key="tax_free_pot_input",
    )

    st.sidebar.number_input(
        "Baseline annual spending £",
        min_value=0.0,
        value=30_000.0,
        step=1_000.0,
        key="baseline_spending_input",
        help=(
            "Your planned yearly spending in today's money before adding "
            "life events."
        ),
    )

    st.sidebar.markdown("### Market Settings")
    st.sidebar.number_input(
        "Mean real return",
        min_value=-0.05,
        max_value=0.15,
        value=MEAN_RETURN,
        step=0.005,
        format="%.3f",
        key="mean_return_input",
        help=(
            "Expected average yearly investment return after inflation. "
            "Example: 0.04 means 4%."
        ),
    )
    st.sidebar.number_input(
        "Return volatility",
        min_value=0.01,
        max_value=0.30,
        value=STD_RETURN,
        step=0.005,
        format="%.3f",
        key="std_return_input",
        help=(
            "How much returns can swing up and down each year. "
            "Higher means more uncertainty."
        ),
    )
    st.sidebar.number_input(
        "Random seed",
        min_value=0,
        value=RANDOM_SEED,
        key="random_seed_input",
        help=(
            "A fixed number that makes random scenarios repeatable so you "
            "can compare changes fairly."
        ),
    )
    st.sidebar.number_input(
        "Monte Carlo simulations",
        min_value=100,
        max_value=10_000,
        value=NUM_SIMULATIONS,
        step=100,
        key="num_simulations_input",
        help=(
            "How many random market paths to test. More paths give a more "
            "stable estimate but run slower."
        ),
    )

    save_outputs = st.sidebar.checkbox(
        "Save PNG outputs to output/",
        value=False,
        key="save_outputs_input",
    )

    with st.sidebar.expander("DC Pot Inputs", expanded=False):
        dc_pots, dc_pot_names = _build_dc_pots(
            drawdown_floor_age=drawdown_start_age,
            end_age=end_age,
        )

    with st.sidebar.expander("DB Pension Inputs", expanded=False):
        db_pensions, db_pension_names = _build_db_pensions(model_start_age, end_age)

    with st.sidebar.expander("Life Event Inputs", expanded=False):
        life_events, life_event_names = _build_life_events(model_start_age, end_age)

    pending_preset_action = st.session_state.pop("_pending_preset_action", None)
    if isinstance(pending_preset_action, str):
        _execute_preset_action(
            action=pending_preset_action,
            selected_preset=selected_preset,
            preset_map=preset_map,
            preset_dir=preset_dir,
        )
    else:
        pending_auto_load = st.session_state.pop("_pending_preset_auto_load", None)
        if isinstance(pending_auto_load, str) and pending_auto_load in preset_map:
            if _has_unsaved_changes():
                st.session_state["_show_discard_warning_for"] = pending_auto_load
                st.rerun()
            else:
                _execute_preset_action(
                    action="load",
                    selected_preset=pending_auto_load,
                    preset_map=preset_map,
                    preset_dir=preset_dir,
                )

    if run_model:
        st.session_state["_last_run_current_state"] = _collect_sidebar_state()
        st.session_state["_last_run_comparison_snapshots"] = (
            _capture_comparison_snapshots(
                selected_compare_presets,
                preset_map,
            )
        )
        st.session_state["_last_run_save_outputs"] = save_outputs

    last_run_state = st.session_state.get("_last_run_current_state")
    if not isinstance(last_run_state, dict):
        st.info("Set assumptions in the sidebar, then click Run simulation.")
        return

    comparison_snapshots = st.session_state.get("_last_run_comparison_snapshots", [])
    if not isinstance(comparison_snapshots, list):
        comparison_snapshots = []
    save_outputs_for_run = bool(st.session_state.get("_last_run_save_outputs", False))

    output_dir = Path("output")
    if save_outputs_for_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if display_mode == "Compare saved presets":
        st.markdown("## Saved Preset Comparison")
        comparison_results = [
            _run_simulation_panel(
                _build_simulation_inputs_from_state(saved_state, label=label),
                save_outputs=save_outputs_for_run,
                output_dir=output_dir,
            )
            for label, saved_state in comparison_snapshots
        ]
        if len(comparison_results) == 0:
            st.info(
                "Choose one or more saved presets in the sidebar, then click "
                "Run simulation to enter comparison mode."
            )
            return
        try:
            _render_comparison_workspace(comparison_results, comparison_layout)
        finally:
            for result in comparison_results:
                _close_simulation_figures(result)
        return

    current_results = _run_simulation_panel(
        _build_simulation_inputs_from_state(last_run_state, label="Current inputs"),
        save_outputs=save_outputs_for_run,
        output_dir=output_dir,
    )
    _render_simulation_results(current_results)


if __name__ == "__main__":
    main()
