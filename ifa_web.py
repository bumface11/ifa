"""Streamlit app for beginner-friendly pension drawdown exploration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import streamlit as st

from ifa.config import (
    DB_PENSIONS,
    END_AGE,
    INITIAL_DC_POT,
    INITIAL_TAX_FREE_POT,
    MEAN_RETURN,
    NUM_SIMULATIONS,
    RANDOM_SEED,
    SECONDARY_DC_DRAWDOWN_AGE,
    SECONDARY_DC_POT,
    START_AGE,
    STD_RETURN,
)
from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.events import build_required_withdrawals
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
from ifa.strategies import create_fixed_real_drawdown_strategy


def _build_db_income(
    ages: np.ndarray, db_pensions: Sequence[tuple[int, float]]
) -> np.ndarray:
    """Build annual DB income aligned to ages."""
    return np.array(
        [calculate_db_pension_income(int(age), db_pensions) for age in ages],
        dtype=np.float64,
    )


def _build_life_events(
    start_age: int,
    end_age: int,
) -> tuple[LifeEvent, ...]:
    """Collect life events from sidebar inputs and return a typed tuple."""
    life_events: list[LifeEvent] = []

    st.sidebar.markdown("### Life Events")
    lump_count = int(
        st.sidebar.number_input("Lump-sum events", min_value=0, max_value=5, value=1)
    )
    for index in range(lump_count):
        st.sidebar.markdown(f"Lump sum #{index + 1}")
        lump_age = int(
            st.sidebar.number_input(
                f"Age (lump #{index + 1})",
                min_value=start_age,
                max_value=end_age,
                value=min(start_age + 3 + index, end_age),
                key=f"lump_age_{index}",
            )
        )
        lump_amount = float(
            st.sidebar.number_input(
                f"Amount GBP (lump #{index + 1})",
                min_value=0.0,
                value=18_000.0,
                step=1_000.0,
                key=f"lump_amount_{index}",
            )
        )
        if lump_amount > 0.0:
            life_events.append(LumpSumEvent(age=lump_age, amount=lump_amount))

    step_count = int(
        st.sidebar.number_input(
            "Spending-step events", min_value=0, max_value=5, value=1
        )
    )
    for index in range(step_count):
        st.sidebar.markdown(f"Spending step #{index + 1}")
        step_start = int(
            st.sidebar.number_input(
                f"Start age (step #{index + 1})",
                min_value=start_age,
                max_value=end_age,
                value=min(start_age + 10 + index, end_age),
                key=f"step_start_{index}",
            )
        )
        step_amount = float(
            st.sidebar.number_input(
                f"Extra per year GBP (step #{index + 1})",
                min_value=0.0,
                value=6_000.0,
                step=500.0,
                key=f"step_amount_{index}",
            )
        )
        has_end = st.sidebar.checkbox(
            f"Set end age (step #{index + 1})",
            value=False,
            key=f"step_has_end_{index}",
        )
        step_end = None
        if has_end:
            step_end = int(
                st.sidebar.number_input(
                    f"End age (step #{index + 1})",
                    min_value=step_start,
                    max_value=end_age,
                    value=end_age,
                    key=f"step_end_{index}",
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

    return tuple(life_events)


def _build_db_pensions(start_age: int, end_age: int) -> list[tuple[int, float]]:
    """Collect DB pension inputs from the sidebar."""
    st.sidebar.markdown("### DB Pensions")
    default_streams = len(DB_PENSIONS)
    stream_count = int(
        st.sidebar.number_input(
            "DB income streams",
            min_value=0,
            max_value=6,
            value=default_streams,
        )
    )
    pensions: list[tuple[int, float]] = []
    for index in range(stream_count):
        default_age = DB_PENSIONS[index][0] if index < default_streams else start_age
        default_amount = DB_PENSIONS[index][1] if index < default_streams else 10_000.0
        stream_age = int(
            st.sidebar.number_input(
                f"DB start age #{index + 1}",
                min_value=start_age,
                max_value=end_age,
                value=default_age,
                key=f"db_age_{index}",
            )
        )
        stream_amount = float(
            st.sidebar.number_input(
                f"DB annual amount GBP #{index + 1}",
                min_value=0.0,
                value=float(default_amount),
                step=500.0,
                key=f"db_amount_{index}",
            )
        )
        pensions.append((stream_age, stream_amount))
    return pensions


def main() -> None:
    """Render the Streamlit pension simulator UI and outputs."""
    st.set_page_config(page_title="IFA Pension Simulator", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Chivo:wght@400;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
        html, body, [class*="css"]  {
            font-family: "Chivo", sans-serif;
        }
        h1, h2, h3 {
            font-family: "Fraunces", serif;
            letter-spacing: 0.3px;
        }
        .stApp {
            background: radial-gradient(
                circle at 20% 15%,
                #fff7df 0%,
                #f6fbff 45%,
                #eef6f2 100%
            );
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

    st.sidebar.header("Core Inputs")
    start_age = int(
        st.sidebar.number_input(
            "Start age", min_value=40, max_value=75, value=START_AGE
        )
    )
    end_age = int(
        st.sidebar.number_input(
            "End age", min_value=start_age + 5, max_value=110, value=END_AGE
        )
    )

    tax_free_pot = float(
        st.sidebar.number_input(
            "Tax-free pot GBP",
            min_value=0.0,
            value=float(INITIAL_TAX_FREE_POT),
            step=1_000.0,
        )
    )
    main_dc_pot = float(
        st.sidebar.number_input(
            "Main DC pot GBP",
            min_value=0.0,
            value=float(INITIAL_DC_POT),
            step=1_000.0,
        )
    )
    secondary_dc_pot = float(
        st.sidebar.number_input(
            "Secondary DC pot GBP",
            min_value=0.0,
            value=float(SECONDARY_DC_POT),
            step=1_000.0,
        )
    )
    use_secondary = st.sidebar.checkbox("Use secondary DC drawdown age", value=True)
    secondary_draw_age = None
    if use_secondary:
        secondary_draw_age = int(
            st.sidebar.number_input(
                "Secondary DC drawdown age",
                min_value=start_age,
                max_value=end_age,
                value=min(max(SECONDARY_DC_DRAWDOWN_AGE, start_age), end_age),
            )
        )

    baseline_spending = float(
        st.sidebar.number_input(
            "Baseline annual spending GBP",
            min_value=0.0,
            value=30_000.0,
            step=1_000.0,
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
        )
    )
    random_seed = int(
        st.sidebar.number_input("Random seed", min_value=0, value=RANDOM_SEED)
    )
    num_simulations = int(
        st.sidebar.number_input(
            "Monte Carlo simulations",
            min_value=100,
            max_value=10_000,
            value=NUM_SIMULATIONS,
            step=100,
        )
    )

    save_outputs = st.sidebar.checkbox("Save PNG outputs to output/", value=False)

    db_pensions = _build_db_pensions(start_age, end_age)
    life_events = _build_life_events(start_age, end_age)

    run_model = st.sidebar.button("Run simulation", type="primary")
    if not run_model:
        st.info("Set assumptions in the sidebar, then click Run simulation.")
        return

    output_dir = Path("output")
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

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

    years = end_age - start_age
    returns = (
        np.random.default_rng(random_seed)
        .normal(mean_return, std_return, years)
        .astype(np.float64)
    )

    base_strategy = create_fixed_real_drawdown_strategy(baseline_spending)
    _, baseline_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=tax_free_pot,
        dc_pot=main_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=baseline_required,
    )
    _, scenario_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=tax_free_pot,
        dc_pot=main_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=scenario_required,
    )

    _, monte_carlo_paths = run_monte_carlo_simulation(
        tax_free_pot=tax_free_pot,
        dc_pot=main_dc_pot,
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
    )

    baseline_metrics = summarize_path(baseline_balances)
    scenario_metrics = summarize_path(scenario_balances)
    monte_carlo_metrics = summarize_monte_carlo(monte_carlo_paths)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline ending", f"GBP{baseline_metrics.ending_balance:,.0f}")
    col2.metric("Scenario ending", f"GBP{scenario_metrics.ending_balance:,.0f}")
    col3.metric(
        "Ruin probability", f"{monte_carlo_metrics.ruin_probability * 100:.1f}%"
    )

    explanation = build_plain_english_explanation(
        baseline_metrics=baseline_metrics,
        scenario_metrics=scenario_metrics,
        monte_carlo_metrics=monte_carlo_metrics,
        events=life_events,
    )
    st.markdown("### Plain-English Summary")
    st.write(explanation)

    comparison_fig = plot_baseline_vs_scenario_balances(
        ages=ages,
        baseline_balances=baseline_balances,
        scenario_balances=scenario_balances,
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "baseline_vs_scenario_streamlit.png",
    )
    if comparison_fig is not None:
        st.markdown("### Baseline vs Life-Events Scenario")
        st.pyplot(comparison_fig, clear_figure=True)

    sequence_fig = plot_sequence_of_returns_scenarios(
        tax_free_pot=tax_free_pot,
        dc_pot=main_dc_pot,
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
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "sequence_scenarios_streamlit.png",
    )
    if sequence_fig is not None:
        st.markdown("### Sequence-of-Returns Teaching Chart")
        st.pyplot(sequence_fig, clear_figure=True)

    fan_fig = plot_monte_carlo_fan_chart(
        tax_free_pot=tax_free_pot,
        dc_pot=main_dc_pot,
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
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "monte_carlo_fan_streamlit.png",
    )
    if fan_fig is not None:
        st.markdown("### Monte Carlo Fan Chart")
        st.pyplot(fan_fig, clear_figure=True)

    stacked_fig = plot_pots_stacked_area(
        tax_free_pot=tax_free_pot,
        dc_pot=main_dc_pot,
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
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "pots_stacked_streamlit.png",
    )
    if stacked_fig is not None:
        st.markdown("### Pot Composition (Stacked Area)")
        st.pyplot(stacked_fig, clear_figure=True)

    individual_fig = plot_individual_pots_subplots(
        tax_free_pot=tax_free_pot,
        dc_pot=main_dc_pot,
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
        save_output=save_outputs,
        return_figure=True,
        output_file=output_dir / "pots_individual_streamlit.png",
    )
    if individual_fig is not None:
        st.markdown("### Individual Pots (4 Panels)")
        st.pyplot(individual_fig, clear_figure=True)


if __name__ == "__main__":
    main()
