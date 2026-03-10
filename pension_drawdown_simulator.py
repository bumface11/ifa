"""Pension Drawdown Simulator.

Simulates pension pot evolution under various withdrawal strategies and market
return scenarios. Calculations are in real (inflation-adjusted) terms.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ifa.config import (
    ANNUAL_DRAWDOWNS,
    DB_PENSIONS,
    DC_POTS,
    END_AGE,
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
from ifa.market import generate_random_returns
from ifa.metrics import summarize_monte_carlo, summarize_path
from ifa.models import LifeEvent, LumpSumEvent, SpendingStepEvent
from ifa.plotting import (
    plot_baseline_vs_scenario_balances,
    plot_individual_pots_subplots,
    plot_monte_carlo_fan_chart,
    plot_multiple_drawdown_levels,
    plot_pots_stacked_area,
    plot_sequence_of_returns_scenarios,
)
from ifa.strategies import create_db_aware_strategy, create_fixed_real_drawdown_strategy

LOGGER = logging.getLogger(__name__)

BASELINE_SPENDING = 30_000.0
SCENARIO_EVENTS: tuple[LifeEvent, ...] = (
    LumpSumEvent(age=55, amount=200_000.0),
    SpendingStepEvent(start_age=70, extra_per_year=24_000.0),
)


def _format_gbp(amount: float) -> str:
    """Format a currency amount for beginner-friendly logs."""
    return f"£{amount:,.0f}"


def _build_db_income_by_age(ages: np.ndarray) -> np.ndarray:
    """Calculate DB pension income at each age in the projection horizon."""
    return np.array(
        [calculate_db_pension_income(int(age), DB_PENSIONS) for age in ages],
        dtype=np.float64,
    )


def _build_required_withdrawals_for_events(
    baseline_spending: float,
    events: Sequence[LifeEvent],
) -> np.ndarray:
    """Build pot-withdrawal needs from baseline spending and life events.

    This schedule already nets off DB income to avoid double-counting.
    """
    ages = np.arange(START_AGE, END_AGE + 1, dtype=np.int_)
    db_income = _build_db_income_by_age(ages)
    return build_required_withdrawals(
        ages=ages,
        baseline_spending=baseline_spending,
        db_income=db_income,
        events=events,
    )


def _build_spending_drawdown_schedule_for_events(
    baseline_spending: float,
    events: Sequence[LifeEvent],
) -> np.ndarray:
    """Build net spending drawdown schedule for chart secondary axes."""
    ages = np.arange(START_AGE, END_AGE + 1, dtype=np.int_)
    db_income = _build_db_income_by_age(ages)
    return build_spending_drawdown_schedule(
        ages=ages,
        baseline_spending=baseline_spending,
        db_income=db_income,
        events=events,
    )


def run_life_events_comparison(
    output_dir: Path,
    baseline_spending: float,
    scenario_events: Sequence[LifeEvent],
) -> None:
    """Run and plot baseline vs life-events scenario on the same returns path."""
    baseline_required = _build_required_withdrawals_for_events(
        baseline_spending=baseline_spending,
        events=(),
    )
    scenario_required = _build_required_withdrawals_for_events(
        baseline_spending=baseline_spending,
        events=scenario_events,
    )

    returns = generate_random_returns(
        END_AGE - START_AGE,
        mean=MEAN_RETURN,
        std=STD_RETURN,
        seed=RANDOM_SEED,
    )

    (
        comparison_ages,
        baseline_balances,
        _,
        _,
        _,
        _,
        _,
    ) = simulate_multi_pot_pension_path(
        tax_free_pot=INITIAL_TAX_FREE_POT,
        dc_pot=float(DC_POTS[0][1]),
        secondary_dc_pot=float(sum(pot[1] for pot in DC_POTS[1:])),
        secondary_dc_drawdown_age=int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        db_pensions=DB_PENSIONS,
        start_age=START_AGE,
        end_age=END_AGE,
        returns=returns,
        withdrawals_required=baseline_required,
        dc_pots=DC_POTS,
    )
    (
        _,
        scenario_balances,
        _,
        _,
        _,
        _,
        _,
    ) = simulate_multi_pot_pension_path(
        tax_free_pot=INITIAL_TAX_FREE_POT,
        dc_pot=float(DC_POTS[0][1]),
        secondary_dc_pot=float(sum(pot[1] for pot in DC_POTS[1:])),
        secondary_dc_drawdown_age=int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        db_pensions=DB_PENSIONS,
        start_age=START_AGE,
        end_age=END_AGE,
        returns=returns,
        withdrawals_required=scenario_required,
        dc_pots=DC_POTS,
    )

    plot_baseline_vs_scenario_balances(
        ages=comparison_ages,
        baseline_balances=baseline_balances,
        scenario_balances=scenario_balances,
        spending_drawdown_schedule=_build_spending_drawdown_schedule_for_events(
            baseline_spending=baseline_spending,
            events=scenario_events,
        ),
        secondary_dc_drawdown_age=(
            int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE
        ),
        db_pensions=DB_PENSIONS,
        life_events=scenario_events,
        dc_pots=DC_POTS,
        output_file=output_dir / "baseline_vs_scenario.png",
    )

    baseline_metrics = summarize_path(baseline_balances)
    scenario_metrics = summarize_path(scenario_balances)

    LOGGER.info("Life Events Comparison")
    for event in scenario_events:
        if isinstance(event, LumpSumEvent):
            LOGGER.info(
                "You added a one-off cost of %s at age %d.",
                _format_gbp(event.amount),
                event.age,
            )
            continue

        if event.end_age is None:
            LOGGER.info(
                "You added an ongoing extra %s/year from age %d.",
                _format_gbp(event.extra_per_year),
                event.start_age,
            )
        else:
            LOGGER.info(
                "You added an extra %s/year from age %d to %d.",
                _format_gbp(event.extra_per_year),
                event.start_age,
                event.end_age,
            )
    LOGGER.info(
        "Baseline ending balance: £%.0f | Scenario ending balance: £%.0f",
        baseline_metrics.ending_balance,
        scenario_metrics.ending_balance,
    )
    LOGGER.info(
        "Baseline min balance: £%.0f | Scenario min balance: £%.0f",
        baseline_metrics.min_balance,
        scenario_metrics.min_balance,
    )
    LOGGER.info(
        "Sequence-of-returns risk matters: poor returns early in retirement can "
        "cause bigger long-term damage than poor returns later."
    )


def main() -> None:
    """Run all simulations and generate charts for the multi-pot scenario."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    LOGGER.info("%s", "=" * 80)
    LOGGER.info("PENSION DRAWDOWN SIMULATOR - Multi-Pot Retirement")
    LOGGER.info("%s", "=" * 80)
    LOGGER.info("\nConfiguration:")
    LOGGER.info("  Tax-Free Pot: %s", _format_gbp(INITIAL_TAX_FREE_POT))
    LOGGER.info("  DC Pots: %d", len(DC_POTS))
    for pot_index, (drawdown_start, balance) in enumerate(DC_POTS, start=1):
        LOGGER.info(
            "    - DC Pot %d: %s (drawdown starts age %d)",
            pot_index,
            _format_gbp(balance),
            drawdown_start,
        )
    LOGGER.info("  DB Pension Streams: %d streams", len(DB_PENSIONS))
    for start_age, amount in DB_PENSIONS:
        LOGGER.info("    - %s/year from age %d", _format_gbp(amount), start_age)
    LOGGER.info("\n  Simulation Period: age %d-%d", START_AGE, END_AGE)
    LOGGER.info("  Mean real return: %.1f%%", MEAN_RETURN * 100)
    LOGGER.info("  Std dev: %.1f%%", STD_RETURN * 100)
    LOGGER.info("  Random seed: %d\n", RANDOM_SEED)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("1. Running sequence-of-returns scenarios...")
    base_strategy = create_fixed_real_drawdown_strategy(18_000)
    strategy = create_db_aware_strategy(base_strategy, DB_PENSIONS)
    scenario_required = _build_required_withdrawals_for_events(
        baseline_spending=BASELINE_SPENDING,
        events=SCENARIO_EVENTS,
    )
    scenario_spending_drawdown = _build_spending_drawdown_schedule_for_events(
        baseline_spending=BASELINE_SPENDING,
        events=SCENARIO_EVENTS,
    )
    plot_sequence_of_returns_scenarios(
        INITIAL_TAX_FREE_POT,
        float(DC_POTS[0][1]),
        float(sum(pot[1] for pot in DC_POTS[1:])),
        int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        withdrawals_required=scenario_required,
        life_events=SCENARIO_EVENTS,
        spending_drawdown_schedule=scenario_spending_drawdown,
        dc_pots=DC_POTS,
        output_file=output_dir / "sequence_scenarios.png",
    )
    LOGGER.info("")

    LOGGER.info("2. Running Monte Carlo simulation...")
    plot_monte_carlo_fan_chart(
        INITIAL_TAX_FREE_POT,
        float(DC_POTS[0][1]),
        float(sum(pot[1] for pot in DC_POTS[1:])),
        int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        NUM_SIMULATIONS,
        RANDOM_SEED,
        withdrawals_required=scenario_required,
        life_events=SCENARIO_EVENTS,
        spending_drawdown_schedule=scenario_spending_drawdown,
        dc_pots=DC_POTS,
        output_file=output_dir / "monte_carlo_fan.png",
    )
    LOGGER.info("")
    _, monte_carlo_paths = run_monte_carlo_simulation(
        tax_free_pot=INITIAL_TAX_FREE_POT,
        dc_pot=float(DC_POTS[0][1]),
        secondary_dc_pot=float(sum(pot[1] for pot in DC_POTS[1:])),
        secondary_dc_drawdown_age=int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        db_pensions=DB_PENSIONS,
        start_age=START_AGE,
        end_age=END_AGE,
        mean_return=MEAN_RETURN,
        std_return=STD_RETURN,
        strategy_fn=strategy,
        num_simulations=NUM_SIMULATIONS,
        seed=RANDOM_SEED,
        withdrawals_required=scenario_required,
        dc_pots=DC_POTS,
    )
    mc_metrics = summarize_monte_carlo(monte_carlo_paths)
    LOGGER.info(
        "Monte Carlo learning summary: ruin probability %.1f%%, median ending "
        "balance £%.0f, p10 £%.0f, p90 £%.0f.",
        mc_metrics.ruin_probability * 100,
        mc_metrics.median_ending_balance,
        mc_metrics.p10_ending_balance,
        mc_metrics.p90_ending_balance,
    )
    LOGGER.info("")

    LOGGER.info("3. Comparing multiple DC drawdown levels...")
    plot_multiple_drawdown_levels(
        INITIAL_TAX_FREE_POT,
        float(DC_POTS[0][1]),
        float(sum(pot[1] for pot in DC_POTS[1:])),
        int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        ANNUAL_DRAWDOWNS,
        RANDOM_SEED,
        dc_pots=DC_POTS,
        output_file=output_dir / "multiple_drawdowns.png",
    )
    LOGGER.info("")

    LOGGER.info("4. Plotting pot composition over time (stacked areas)...")
    plot_pots_stacked_area(
        INITIAL_TAX_FREE_POT,
        float(DC_POTS[0][1]),
        float(sum(pot[1] for pot in DC_POTS[1:])),
        int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        RANDOM_SEED,
        withdrawals_required=scenario_required,
        life_events=SCENARIO_EVENTS,
        spending_drawdown_schedule=scenario_spending_drawdown,
        dc_pots=DC_POTS,
        output_file=output_dir / "pots_stacked_area.png",
    )
    LOGGER.info("")

    LOGGER.info("5. Plotting individual pots dynamics (4-panel subplots)...")
    plot_individual_pots_subplots(
        INITIAL_TAX_FREE_POT,
        float(DC_POTS[0][1]),
        float(sum(pot[1] for pot in DC_POTS[1:])),
        int(DC_POTS[1][0]) if len(DC_POTS) > 1 else END_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        RANDOM_SEED,
        withdrawals_required=scenario_required,
        life_events=SCENARIO_EVENTS,
        dc_pots=DC_POTS,
        output_file=output_dir / "pots_individual.png",
    )
    LOGGER.info("")

    LOGGER.info("6. Running life-events baseline vs scenario comparison...")
    run_life_events_comparison(
        output_dir=output_dir,
        baseline_spending=BASELINE_SPENDING,
        scenario_events=SCENARIO_EVENTS,
    )
    LOGGER.info("")

    LOGGER.info("%s", "=" * 80)
    LOGGER.info("All simulations complete. Charts saved to: %s", output_dir.resolve())
    LOGGER.info("%s", "=" * 80)


if __name__ == "__main__":
    main()
