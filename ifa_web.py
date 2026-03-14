"""Streamlit app for beginner-friendly pension drawdown exploration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ifa.config import (
    DB_PENSIONS,
    DC_POTS,
    END_AGE,
    INITIAL_DC_POT,
    INITIAL_TAX_FREE_POT,
    MEAN_RETURN,
    NUM_SIMULATIONS,
    RANDOM_SEED,
    START_AGE,
    STD_RETURN,
)
from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.events import build_required_withdrawals, build_spending_drawdown_schedule
from ifa.explain import build_plain_english_explanation
from ifa.metrics import summarize_monte_carlo, summarize_path
from ifa.models import LifeEvent, LumpSumEvent, SpendingStepEvent
from ifa.plotting import (
    plot_baseline_vs_scenario_balances,
    plot_individual_pots_subplots,
    plot_monte_carlo_fan_chart,
    plot_pots_stacked_area,
    plot_sequence_of_returns_scenarios,
)
from ifa.url_presets import (
    decode_comparison_presets,
    decode_preset_url,
    encode_comparison_url,
    encode_preset_url,
)
from ifa.url_presets import JsonMap
from ifa.strategies import create_fixed_real_drawdown_strategy

_TRACKED_STATIC_KEYS: tuple[str, ...] = (
    "start_age_input",
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


def _ensure_sidebar_defaults() -> None:
    """Initialize sidebar widget defaults in session state once."""
    defaults: dict[str, int | float | bool | str] = {
        "start_age_input": START_AGE,
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
    }
    for key, value in defaults.items():
        # Only set default if key doesn't already exist (e.g., from URL preset)
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_pending_sidebar_updates() -> None:
    """Apply deferred sidebar state updates before widget instantiation."""
    loaded_state = st.session_state.pop("_loaded_sidebar_state", None)
    if isinstance(loaded_state, dict):
        _apply_sidebar_state(loaded_state)
        # Record what we loaded so unsaved-changes detection works.
        st.session_state["_last_loaded_preset_state"] = loaded_state



def _execute_preset_action(action: str) -> None:
    """Execute one requested preset action after all sidebar widgets are built.

    Actions:
        new: Reset to defaults and clear any loaded state.
    """
    if action == "new":
        # Reset sidebar to defaults
        _ensure_sidebar_defaults()
        st.session_state.pop("_last_loaded_preset_state", None)
        st.session_state["_preset_notice"] = "Reset to defaults"
        st.rerun()


def _load_comparison_presets() -> list[tuple[int, JsonMap]]:
    """Load all comparison presets from URL parameters.

    Returns:
        List of (preset_number, preset_state) tuples for preset1-preset5.
    """
    query_dict = st.query_params.to_dict()
    presets_list = decode_comparison_presets(query_dict)

    # Return as list of (index, preset) for easy enumeration
    return list(enumerate(presets_list, start=1))


def _render_comparison_controls() -> None:
    """Render UI controls for comparison mode in the sidebar."""
    with st.sidebar.expander("📊 Comparison Mode", expanded=False):
        st.markdown("Add presets to compare side-by-side.")

        # Show current comparison count
        query_dict = st.query_params.to_dict()
        active_presets = decode_comparison_presets(query_dict)
        active_count = len(active_presets)

        if active_count > 0:
            st.info(f"Comparing {active_count} preset(s)")

            if active_count < 5:
                if st.button(
                    "Add Another Preset",
                    key="add_preset_button",
                    use_container_width=True,
                ):
                    # Paste a preset URL or code here
                    st.session_state["_show_add_preset_dialog"] = True

            if st.button(
                "Clear Comparison",
                key="clear_compare_button",
                use_container_width=True,
            ):
                # Clear all comparison presets
                for i in range(1, 6):
                    st.query_params.pop(f"preset{i}", None)
                st.rerun()

        else:
            st.write("No comparison presets loaded yet.")
            if st.button(
                "Start Comparison",
                key="start_compare_button",
                use_container_width=True,
            ):
                st.session_state["_show_add_preset_dialog"] = True


def _reorder_comparison_presets(secondary_idx: int) -> None:
    """Move a secondary preset to primary position by reordering URL parameters.
    
    Args:
        secondary_idx: 0-based index within secondary presets (first secondary is 0)
    """
    # Collect current presets in order (preset1, preset2, ..., preset5)
    preset_values = []
    for i in range(1, 6):
        param_name = f"preset{i}"
        if param_name in st.query_params:
            preset_values.append(st.query_params[param_name])
    
    if secondary_idx >= len(preset_values) - 1:
        # Can't select the main preset (index 0)
        return
    
    # The secondary preset to promote is at index (1 + secondary_idx)
    selected_value = preset_values.pop(1 + secondary_idx)
    
    # Clear all presets and re-set with new order
    for i in range(1, 6):
        param_name = f"preset{i}"
        st.query_params.pop(param_name, None)
    
    # Set with reordered values: selected goes to preset1, rest shift
    st.query_params["preset1"] = selected_value
    for i, value in enumerate(preset_values, start=2):
        st.query_params[f"preset{i}"] = value


def _render_comparison_results(
    comparison_presets: list[tuple[int, JsonMap]], save_outputs: bool
) -> None:
    """Render comparison of multiple presets with main on left, secondary on right.

    Args:
        comparison_presets: List of (preset_number, preset_state) tuples
        save_outputs: Whether to save charts to output/
    """
    st.markdown("## 📊 Preset Comparison")
    st.caption(f"Comparing {len(comparison_presets)} preset(s)")

    # Run simulations for each preset in parallel (simulated via caching)
    results = []
    for preset_num, preset_state in comparison_presets:
        result = _run_preset_simulation(preset_num, preset_state)
        if result:
            results.append((preset_num, preset_state, result))

    if not results:
        st.error("Failed to run simulations for comparison presets.")
        return

    # Split results: first is main (left), rest are secondary (right)
    main_result = results[0]
    secondary_results = results[1:] if len(results) > 1 else []

    # Create two-column layout: 60% main on left, 40% secondary on right
    left_col, right_col = st.columns([0.60, 0.40], gap="medium")

    # Left column: Main preset with full-size charts
    with left_col:
        preset_num, preset_state, sim_result = main_result
        preset_name = preset_state.get("_preset_name", f"Preset {preset_num}")
        
        # Extract all needed parameters from preset for plotting functions
        start_age = int(preset_state.get("start_age_input", START_AGE))
        end_age = int(preset_state.get("end_age_input", END_AGE))
        tax_free_pot = float(preset_state.get("tax_free_pot_input", INITIAL_TAX_FREE_POT))
        baseline_spending = float(preset_state.get("baseline_spending_input", 30_000.0))
        mean_return = float(preset_state.get("mean_return_input", MEAN_RETURN))
        std_return = float(preset_state.get("std_return_input", STD_RETURN))
        random_seed = int(preset_state.get("random_seed_input", RANDOM_SEED))
        num_simulations = int(preset_state.get("num_simulations_input", NUM_SIMULATIONS))
        
        primary_dc_pot = float(preset_state.get("dc_initial_balance_0", INITIAL_DC_POT))
        db_pensions = []
        for i in range(3):
            age_key = f"db_age_{i}"
            amount_key = f"db_amount_{i}"
            if age_key in preset_state and amount_key in preset_state:
                db_pensions.append((int(preset_state[age_key]), float(preset_state[amount_key])))
        
        st.markdown(f"### {preset_name}")

        # Metrics for main preset
        baseline_metrics = sim_result.get("baseline_metrics")
        scenario_metrics = sim_result.get("scenario_metrics")
        monte_carlo_metrics = sim_result.get("monte_carlo_metrics")

        if baseline_metrics and scenario_metrics and monte_carlo_metrics:
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric(
                    "Baseline ending",
                    f"£{baseline_metrics.ending_balance:,.0f}",
                )
            with metric_cols[1]:
                st.metric(
                    "Scenario ending",
                    f"£{scenario_metrics.ending_balance:,.0f}",
                )
            with metric_cols[2]:
                st.metric(
                    "Ruin probability",
                    f"{monte_carlo_metrics.ruin_probability * 100:.1f}%",
                )

        # Use plotting functions for proper charts
        st.markdown("#### Balance Trajectory")
        baseline_balances = sim_result.get("baseline_balances")
        scenario_balances = sim_result.get("scenario_balances")
        ages = sim_result.get("ages")
        
        if baseline_balances is not None and scenario_balances is not None and ages is not None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(ages, baseline_balances, linewidth=2.5, color="#1F77B4", marker="o", markersize=4, label="Baseline")
            if not np.array_equal(baseline_balances, scenario_balances):
                ax.plot(ages, scenario_balances, linewidth=2.5, color="#FF7F0E", marker="s", markersize=4, label="Scenario")
            ax.set_xlabel("Age", fontsize=10)
            ax.set_ylabel("Balance (£)", fontsize=10)
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"£{x/1e6:.1f}M" if x >= 1e6 else f"£{x/1e3:.0f}K"))
            fig.tight_layout()
            st.pyplot(fig, width='stretch')

        # Monte Carlo fan chart for main preset
        st.markdown("#### Monte Carlo Distribution")
        monte_carlo_paths = sim_result.get("monte_carlo_paths")
        
        if monte_carlo_paths is not None and len(monte_carlo_paths) > 0:
            try:
                fan_fig = plot_monte_carlo_fan_chart(
                    tax_free_pot=tax_free_pot,
                    dc_pot=primary_dc_pot,
                    secondary_dc_pot=0.0,
                    secondary_dc_drawdown_age=end_age,
                    db_pensions=db_pensions,
                    start_age=start_age,
                    end_age=end_age,
                    mean_return=mean_return,
                    std_return=std_return,
                    strategy_fn=create_fixed_real_drawdown_strategy(baseline_spending),
                    num_simulations=min(num_simulations, 500),
                    seed=random_seed,
                    withdrawals_required=sim_result.get("baseline_required"),
                    life_events=(),
                    spending_drawdown_schedule=np.zeros(end_age - start_age + 1),
                    dc_pots=[(57, primary_dc_pot)] if primary_dc_pot > 0 else [],
                    dc_pot_names=["Primary DC"],
                    db_pension_names=[f"DB {i+1}" for i in range(len(db_pensions))],
                    life_event_names=[],
                    save_output=save_outputs,
                    return_figure=True,
                    output_file=Path("output") / "comparison_fan.png" if save_outputs else None,
                )
                if fan_fig is not None:
                    st.pyplot(fan_fig, width='stretch')
            except Exception as e:
                # Fallback to simple histogram if fan chart fails
                final_balances = monte_carlo_paths[:, -1]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(final_balances, bins=30, color="#2CA02C", alpha=0.7, edgecolor="black")
                ax.axvline(np.median(final_balances), color="red", linestyle="--", linewidth=2, label=f"Median: £{np.median(final_balances):,.0f}")
                ax.set_xlabel("Final Balance (£)", fontsize=10)
                ax.set_ylabel("Simulations", fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"£{x/1e6:.1f}M" if x >= 1e6 else f"£{x/1e3:.0f}K"))
                fig.tight_layout()
                st.pyplot(fig, width='stretch')

    # Right column: Secondary presets in a grid layout
    if secondary_results:
        with right_col:
            st.markdown("### Other Presets")
            
            # Display secondary presets in 2 columns
            for idx in range(0, len(secondary_results), 2):
                cols = st.columns(2) if idx + 1 < len(secondary_results) else [st.columns(1)[0]]
                
                for col_idx, col in enumerate(cols):
                    if idx + col_idx < len(secondary_results):
                        preset_num, preset_state, sim_result = secondary_results[idx + col_idx]
                        preset_name = preset_state.get("_preset_name", f"Preset {preset_num}")
                        
                        with col:
                            # Clickable header to swap to main
                            if st.button(
                                f"📌 {preset_name}",
                                key=f"swap_preset_{idx + col_idx}",
                                use_container_width=True,
                            ):
                                _reorder_comparison_presets(idx + col_idx)

                            # Compact metrics for secondary preset
                            baseline_metrics = sim_result.get("baseline_metrics")
                            scenario_metrics = sim_result.get("scenario_metrics")
                            monte_carlo_metrics = sim_result.get("monte_carlo_metrics")

                            if baseline_metrics and scenario_metrics and monte_carlo_metrics:
                                metric_cols = st.columns(2)
                                with metric_cols[0]:
                                    st.metric(
                                        "Baseline",
                                        f"£{baseline_metrics.ending_balance/1e3:.0f}K",
                                    )
                                with metric_cols[1]:
                                    st.metric(
                                        "Ruin",
                                        f"{monte_carlo_metrics.ruin_probability * 100:.0f}%",
                                    )

                            # Shrunk balance trajectory
                            ages = sim_result.get("ages")
                            baseline_balances = sim_result.get("baseline_balances")
                            scenario_balances = sim_result.get("scenario_balances")

                            if ages is not None and baseline_balances is not None:
                                fig, ax = plt.subplots(figsize=(4, 2.5))
                                ax.plot(ages, baseline_balances, linewidth=1.5, color="#1F77B4", marker="o", markersize=2)
                                if scenario_balances is not None and not np.array_equal(baseline_balances, scenario_balances):
                                    ax.plot(ages, scenario_balances, linewidth=1.5, color="#FF7F0E", marker="s", markersize=2)
                                ax.set_xlabel("Age", fontsize=8)
                                ax.set_ylabel("Balance (£)", fontsize=8)
                                ax.grid(True, alpha=0.2)
                                ax.tick_params(labelsize=7)
                                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"£{x/1e6:.1f}M" if x >= 1e6 else f"£{x/1e3:.0f}K"))
                                fig.tight_layout()
                                st.pyplot(fig, width='stretch')

                            # Shrunk Monte Carlo distribution
                            monte_carlo_paths = sim_result.get("monte_carlo_paths")

                            if monte_carlo_paths is not None and len(monte_carlo_paths) > 0:
                                final_balances = monte_carlo_paths[:, -1]
                                fig, ax = plt.subplots(figsize=(4, 2.5))
                                ax.hist(final_balances, bins=20, color="#2CA02C", alpha=0.7, edgecolor="black")
                                ax.axvline(np.median(final_balances), color="red", linestyle="--", linewidth=1)
                                ax.set_xlabel("Final Balance (£)", fontsize=8)
                                ax.set_ylabel("Sims", fontsize=8)
                                ax.grid(True, alpha=0.2, axis="y")
                                ax.tick_params(labelsize=7)
                                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"£{x/1e6:.1f}M" if x >= 1e6 else f"£{x/1e3:.0f}K"))
                                fig.tight_layout()
                                st.pyplot(fig, width='stretch')

                            st.divider()


@st.cache_data
def _run_preset_simulation(preset_num: int, preset_state: JsonMap) -> dict:
    """Run simulation for a single preset and cache results.

    Args:
        preset_num: Preset number (1-5) for cache keying
        preset_state: Preset parameter state

    Returns:
        Dictionary with simulation results (metrics, charts data, etc.)
    """
    try:
        # Extract parameters from preset
        start_age = int(preset_state.get("start_age_input", START_AGE))
        end_age = int(preset_state.get("end_age_input", END_AGE))
        tax_free_pot = float(preset_state.get("tax_free_pot_input", INITIAL_TAX_FREE_POT))
        baseline_spending = float(preset_state.get("baseline_spending_input", 30_000.0))
        mean_return = float(preset_state.get("mean_return_input", MEAN_RETURN))
        std_return = float(preset_state.get("std_return_input", STD_RETURN))
        random_seed = int(preset_state.get("random_seed_input", RANDOM_SEED))
        num_simulations = int(preset_state.get("num_simulations_input", NUM_SIMULATIONS))

        # Build DC pots from preset (simplified - assumes single pot for comparison)
        primary_dc_pot = float(
            preset_state.get("dc_initial_balance_0", INITIAL_DC_POT)
        )

        # Build DB pensions (simplified)
        db_pensions = []
        for i in range(3):
            age_key = f"db_age_{i}"
            amount_key = f"db_amount_{i}"
            if age_key in preset_state and amount_key in preset_state:
                db_pensions.append(
                    (
                        int(preset_state[age_key]),
                        float(preset_state[amount_key]),
                    )
                )

        # Run simplified simulation (baseline only for comparison)
        ages = np.arange(start_age, end_age + 1, dtype=np.int_)
        db_income = np.array(
            [calculate_db_pension_income(int(age), db_pensions) for age in ages],
            dtype=np.float64,
        )

        baseline_required = build_required_withdrawals(
            ages=ages,
            baseline_spending=baseline_spending,
            db_income=db_income,
            events=(),
        )

        years = end_age - start_age
        returns = (
            np.random.default_rng(random_seed)
            .normal(mean_return, std_return, years)
            .astype(np.float64)
        )

        base_strategy = create_fixed_real_drawdown_strategy(baseline_spending)
        _, baseline_balances, *_ = simulate_multi_pot_pension_path(
            tax_free_pot=tax_free_pot,
            dc_pot=primary_dc_pot,
            secondary_dc_pot=0.0,
            secondary_dc_drawdown_age=end_age,
            db_pensions=db_pensions,
            start_age=start_age,
            end_age=end_age,
            returns=returns,
            drawdown_fn=base_strategy,
            withdrawals_required=baseline_required,
            dc_pots=[(57, primary_dc_pot)] if primary_dc_pot > 0 else [],
        )

        baseline_metrics = summarize_path(baseline_balances)

        # Run Monte Carlo for scenario metrics
        _, monte_carlo_paths = run_monte_carlo_simulation(
            tax_free_pot=tax_free_pot,
            dc_pot=primary_dc_pot,
            secondary_dc_pot=0.0,
            secondary_dc_drawdown_age=end_age,
            db_pensions=db_pensions,
            start_age=start_age,
            end_age=end_age,
            mean_return=mean_return,
            std_return=std_return,
            strategy_fn=base_strategy,
            num_simulations=min(num_simulations, 500),  # Lower for faster comparison
            seed=random_seed,
            withdrawals_required=baseline_required,
            dc_pots=[(57, primary_dc_pot)] if primary_dc_pot > 0 else [],
        )

        monte_carlo_metrics = summarize_monte_carlo(monte_carlo_paths)

        # For comparison, scenario = baseline
        scenario_metrics = baseline_metrics
        scenario_balances = baseline_balances

        return {
            "baseline_metrics": baseline_metrics,
            "scenario_metrics": scenario_metrics,
            "monte_carlo_metrics": monte_carlo_metrics,
            "ages": ages,
            "baseline_balances": baseline_balances,
            "scenario_balances": scenario_balances,
            "monte_carlo_paths": monte_carlo_paths,
            "db_pensions": db_pensions,
            "baseline_spending": baseline_spending,
            "baseline_required": baseline_required,
        }

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Comparison simulation error for preset {preset_num}: {e}", exc_info=True)
        st.error(f"Failed to simulate preset {preset_num}: {type(e).__name__}: {str(e)}")
        return None


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


def _collect_sidebar_state() -> dict[str, str | int | float | bool | None]:
    """Collect tracked sidebar widget values from session state."""
    state: dict[str, str | int | float | bool | None] = {}
    for key in [*_TRACKED_STATIC_KEYS, *_tracked_dynamic_keys()]:
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            state[key] = value
    return state


def _apply_sidebar_state(
    state: dict[str, str | int | float | bool | None],
) -> None:
    """Apply loaded sidebar values into Streamlit session state."""
    for key, value in state.items():
        st.session_state[key] = value





def _has_unsaved_changes() -> bool:
    """Return True if the current sidebar state differs from the last loaded preset.

    Returns:
        True when a preset has been loaded and at least one tracked parameter
        has since changed; False otherwise.
    """
    last_state = st.session_state.get("_last_loaded_preset_state")
    if not isinstance(last_state, dict) or not last_state:
        return False
    return _collect_sidebar_state() != last_state





def _build_db_income(
    ages: np.ndarray, db_pensions: Sequence[tuple[int, float]]
) -> np.ndarray:
    """Build annual DB income aligned to ages."""
    return np.array(
        [calculate_db_pension_income(int(age), db_pensions) for age in ages],
        dtype=np.float64,
    )


def _build_dc_pots(
    start_age: int,
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
            default_start_age = max(start_age, 57)
            default_balance = 0.0

        pot_name = st.text_input(
            f"DC pot #{index + 1} name",
            value=f"DC Pot {index + 1}",
            key=f"dc_name_{index}",
            help="Default name can be edited to anything you prefer.",
        )
        drawdown_start_age = int(
            st.number_input(
                f"DC pot #{index + 1} drawdown start age",
                min_value=start_age,
                max_value=end_age,
                value=min(max(default_start_age, start_age), end_age),
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
        pots.append((drawdown_start_age, initial_balance))
        pot_names.append(pot_name.strip() or f"DC Pot {index + 1}")

    return pots, pot_names


def _build_life_events(
    start_age: int,
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
                min_value=start_age,
                max_value=end_age,
                value=min(start_age + 3 + index, end_age),
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
                min_value=start_age,
                max_value=end_age,
                value=min(start_age + 10 + index, end_age),
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
    start_age: int,
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
        default_age = DB_PENSIONS[index][0] if index < default_streams else start_age
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
                min_value=start_age,
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

    # Load and apply URL preset on first visit or when preset parameter changes
    # This allows users to paste a new shared URL to load different presets
    preset_param = st.query_params.get("preset", "")
    last_loaded_preset = st.session_state.get("_last_url_preset_param", "")
    
    # Load if: (1) not loaded yet, OR (2) preset param changed (new shared URL pasted)
    if not st.session_state.get("_url_preset_loaded") or (
        preset_param and preset_param != last_loaded_preset
    ):
        if preset_param:
            url_preset_state = decode_preset_url(f"preset={preset_param}")
            if url_preset_state:
                # Apply URL preset values directly to session state before defaults
                for key, value in url_preset_state.items():
                    st.session_state[key] = value
                st.session_state["_last_loaded_preset_state"] = url_preset_state
        # Mark that we've loaded the URL preset and record which param we loaded
        st.session_state["_url_preset_loaded"] = True
        st.session_state["_last_url_preset_param"] = preset_param

    _ensure_sidebar_defaults()
    _apply_pending_sidebar_updates()

    st.sidebar.header("Parameter Sets")
    with st.sidebar.container():
        notice = st.session_state.pop("_preset_notice", None)
        if isinstance(notice, str) and len(notice) > 0:
            st.success(notice)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Share Link", key="preset_share_button",
                         use_container_width=True, help="Copy the current URL with your parameters"):
                st.session_state["_show_share_url"] = True
        with col2:
            if st.button("Reset", key="preset_new_button",
                         use_container_width=True, help="Reset to default parameters"):
                st.session_state["_pending_preset_action"] = "new"

        # Show shareable URL when requested
        if st.session_state.pop("_show_share_url", False):
            st.divider()
            st.markdown("#### 📋 Share Your Preset")
            st.caption("Copy this URL and share it with others to let them see your parameters:")
            
            # Reconstruct the current page URL with the preset parameter
            # The preset is already in st.query_params, so we can just show the current URL
            import os
            server_url = os.getenv("STREAMLIT_SERVER_BASEURL", "http://localhost:8501")
            preset_param = st.query_params.get("preset", "")
            if preset_param:
                shareable_url = f"{server_url}?preset={preset_param}"
                st.code(shareable_url, language="text", line_numbers=False)
            else:
                st.warning("Adjust some parameters first to generate a shareable URL.")
            st.divider()

    # Render comparison mode controls
    _render_comparison_controls()

    st.sidebar.header("Core Inputs")
    start_age = int(
        st.sidebar.number_input(
            "Start age",
            min_value=40,
            max_value=85,
            value=START_AGE,
            key="start_age_input",
        )
    )
    end_age = int(
        st.sidebar.number_input(
            "End age",
            min_value=start_age + 5,
            max_value=110,
            value=END_AGE,
            key="end_age_input",
        )
    )

    tax_free_pot = float(
        st.sidebar.number_input(
            "Tax-free pot £",
            min_value=0.0,
            value=float(INITIAL_TAX_FREE_POT),
            step=1_000.0,
            key="tax_free_pot_input",
        )
    )

    baseline_spending = float(
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
    )

    st.sidebar.markdown("### Market Settings")
    mean_return = float(
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
    )
    std_return = float(
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
    )
    random_seed = int(
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
    )
    num_simulations = int(
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
    )

    save_outputs = st.sidebar.checkbox(
        "Save PNG outputs to output/",
        value=False,
        key="save_outputs_input",
    )

    with st.sidebar.expander("DC Pot Inputs", expanded=False):
        dc_pots, dc_pot_names = _build_dc_pots(start_age, end_age)

    with st.sidebar.expander("DB Pension Inputs", expanded=False):
        db_pensions, db_pension_names = _build_db_pensions(start_age, end_age)

    with st.sidebar.expander("Life Event Inputs", expanded=False):
        life_events, life_event_names = _build_life_events(start_age, end_age)

    # Update URL with current parameters every time they change
    current_state = _collect_sidebar_state()
    
    # Generate a new preset URL value
    dummy_url = encode_preset_url("http://temp", current_state)
    # Extract just the preset parameter value
    match = re.search(r"preset=([^&]*)", dummy_url)
    new_preset_value = match.group(1) if match else ""
    
    # Track the last URL preset value we set to avoid cascading updates
    # This prevents lag from repeated reruns when comparing with st.query_params
    if "_last_url_preset_value" not in st.session_state:
        st.session_state["_last_url_preset_value"] = st.query_params.get("preset", "")
    
    last_url_preset = st.session_state["_last_url_preset_value"]
    
    # Only update if the parameters have actually changed (not a rerun echo)
    if new_preset_value and new_preset_value != last_url_preset:
        st.query_params["preset"] = new_preset_value
        st.session_state["_last_url_preset_value"] = new_preset_value
        # Mark that we've already loaded this preset value to prevent reloading
        # on the same rerun that updates the URL
        st.session_state["_last_url_preset_param"] = new_preset_value

    pending_preset_action = st.session_state.pop("_pending_preset_action", None)
    if isinstance(pending_preset_action, str):
        _execute_preset_action(action=pending_preset_action)

    if not run_model:
        st.info("Set assumptions in the sidebar, then click Run simulation.")
        return

    # Check if we're in comparison mode
    comparison_presets = _load_comparison_presets()
    if comparison_presets:
        # Render comparison mode with multiple presets
        _render_comparison_results(comparison_presets, save_outputs)
        return

    output_dir = Path("output")
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    primary_dc_pot = float(dc_pots[0][1]) if len(dc_pots) > 0 else 0.0
    secondary_dc_pot = (
        float(sum(pot[1] for pot in dc_pots[1:])) if len(dc_pots) > 1 else 0.0
    )
    secondary_draw_age = int(dc_pots[1][0]) if len(dc_pots) > 1 else end_age

    ages = np.arange(start_age, end_age + 1, dtype=np.int_)
    db_income = _build_db_income(ages, db_pensions)
    baseline_required = build_required_withdrawals(
        ages=ages,
        baseline_spending=baseline_spending,
        db_income=db_income,
        events=(),
    )
    scenario_required = build_required_withdrawals(
        ages=ages,
        baseline_spending=baseline_spending,
        db_income=db_income,
        events=life_events,
    )
    spending_drawdown_schedule = build_spending_drawdown_schedule(
        ages=ages,
        baseline_spending=baseline_spending,
        db_income=db_income,
        events=life_events,
    )

    years = end_age - start_age
    returns = (
        np.random.default_rng(random_seed)
        .normal(mean_return, std_return, years)
        .astype(np.float64)
    )

    base_strategy = create_fixed_real_drawdown_strategy(baseline_spending)
    _, baseline_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=baseline_required,
        dc_pots=dc_pots,
    )
    _, scenario_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=scenario_required,
        dc_pots=dc_pots,
    )

    _, monte_carlo_paths = run_monte_carlo_simulation(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        mean_return=mean_return,
        std_return=std_return,
        strategy_fn=base_strategy,
        num_simulations=num_simulations,
        seed=random_seed,
        withdrawals_required=scenario_required,
        dc_pots=dc_pots,
    )

    baseline_metrics = summarize_path(baseline_balances)
    scenario_metrics = summarize_path(scenario_balances)
    monte_carlo_metrics = summarize_monte_carlo(monte_carlo_paths)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline ending", f"£{baseline_metrics.ending_balance:,.0f}")
    col2.metric("Scenario ending", f"£{scenario_metrics.ending_balance:,.0f}")
    col3.metric(
        "Ruin probability", f"{monte_carlo_metrics.ruin_probability * 100:.1f}%"
    )

    explanation = build_plain_english_explanation(
        baseline_metrics=baseline_metrics,
        scenario_metrics=scenario_metrics,
        monte_carlo_metrics=monte_carlo_metrics,
        events=life_events,
        event_names=life_event_names,
    )
    st.markdown("### Plain-English Summary")
    st.write(explanation)

    comparison_fig = plot_baseline_vs_scenario_balances(
        ages=ages,
        baseline_balances=baseline_balances,
        scenario_balances=scenario_balances,
        spending_drawdown_schedule=spending_drawdown_schedule,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        life_events=life_events,
        dc_pots=dc_pots,
        dc_pot_names=dc_pot_names,
        db_pension_names=db_pension_names,
        life_event_names=life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "baseline_vs_scenario_streamlit.png",
    )
    if comparison_fig is not None:
        st.markdown("### Baseline vs Life-Events Scenario")
        st.pyplot(comparison_fig, clear_figure=True)

    sequence_fig = plot_sequence_of_returns_scenarios(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        mean_return=mean_return,
        std_return=std_return,
        strategy_fn=base_strategy,
        withdrawals_required=scenario_required,
        life_events=life_events,
        spending_drawdown_schedule=spending_drawdown_schedule,
        dc_pots=dc_pots,
        dc_pot_names=dc_pot_names,
        db_pension_names=db_pension_names,
        life_event_names=life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "sequence_scenarios_streamlit.png",
    )
    if sequence_fig is not None:
        st.markdown("### Sequence-of-Returns Teaching Chart")
        st.pyplot(sequence_fig, clear_figure=True)

    fan_fig = plot_monte_carlo_fan_chart(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        mean_return=mean_return,
        std_return=std_return,
        strategy_fn=base_strategy,
        num_simulations=num_simulations,
        seed=random_seed,
        withdrawals_required=scenario_required,
        life_events=life_events,
        spending_drawdown_schedule=spending_drawdown_schedule,
        dc_pots=dc_pots,
        dc_pot_names=dc_pot_names,
        db_pension_names=db_pension_names,
        life_event_names=life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "monte_carlo_fan_streamlit.png",
    )
    if fan_fig is not None:
        st.markdown("### Monte Carlo Fan Chart")
        st.pyplot(fan_fig, clear_figure=True)

    st.markdown("### Pot Charts Guide")
    st.write(
        "The next two charts show the same retirement path from two angles. "
        "The stacked chart shows how each pot contributes to your total over time. "
        "The 4-panel chart separates each pot so you can see which pot is being "
        "used first, how quickly it changes, and when DB income becomes important."
    )

    stacked_fig = plot_pots_stacked_area(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        mean_return=mean_return,
        std_return=std_return,
        strategy_fn=base_strategy,
        seed=random_seed,
        withdrawals_required=scenario_required,
        life_events=life_events,
        spending_drawdown_schedule=spending_drawdown_schedule,
        dc_pots=dc_pots,
        dc_pot_names=dc_pot_names,
        db_pension_names=db_pension_names,
        life_event_names=life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "pots_stacked_streamlit.png",
    )
    if stacked_fig is not None:
        st.markdown("### Pot Composition (Stacked Area)")
        st.pyplot(stacked_fig, clear_figure=True)

    individual_fig = plot_individual_pots_subplots(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        mean_return=mean_return,
        std_return=std_return,
        strategy_fn=base_strategy,
        seed=random_seed,
        withdrawals_required=scenario_required,
        life_events=life_events,
        dc_pots=dc_pots,
        dc_pot_names=dc_pot_names,
        db_pension_names=db_pension_names,
        life_event_names=life_event_names,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "pots_individual_streamlit.png",
    )
    if individual_fig is not None:
        st.markdown("### Pot and Income Panels (4 Panels)")
        st.pyplot(individual_fig, clear_figure=True)


if __name__ == "__main__":
    main()
