"""Pension Drawdown Simulator.

Simulates pension pot evolution under various withdrawal strategies and market
return scenarios. Calculations are in real (inflation-adjusted) terms.
"""

from __future__ import annotations

from pathlib import Path

from ifa.config import (
    ANNUAL_DRAWDOWNS,
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
from ifa.plotting import (
    plot_individual_pots_subplots,
    plot_monte_carlo_fan_chart,
    plot_multiple_drawdown_levels,
    plot_pots_stacked_area,
    plot_sequence_of_returns_scenarios,
)
from ifa.strategies import create_db_aware_strategy, create_fixed_real_drawdown_strategy


def main() -> None:
    """Run all simulations and generate charts for the multi-pot scenario."""
    print("=" * 80)
    print("PENSION DRAWDOWN SIMULATOR - Multi-Pot Retirement")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Tax-Free Pot: GBP{INITIAL_TAX_FREE_POT:,.0f}")
    print(f"  Main DC Pot: GBP{INITIAL_DC_POT:,.0f}")
    print(
        "  Secondary DC Pot: "
        f"GBP{SECONDARY_DC_POT:,.0f} (starts drawing at age "
        f"{SECONDARY_DC_DRAWDOWN_AGE})"
    )
    print(f"  DB Pension Streams: {len(DB_PENSIONS)} streams")
    for start_age, amount in DB_PENSIONS:
        print(f"    - GBP{amount:,.0f}/year from age {start_age}")
    print(f"\n  Simulation Period: age {START_AGE}-{END_AGE}")
    print(f"  Mean real return: {MEAN_RETURN * 100:.1f}%")
    print(f"  Std dev: {STD_RETURN * 100:.1f}%")
    print(f"  Random seed: {RANDOM_SEED}\n")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Running sequence-of-returns scenarios...")
    base_strategy = create_fixed_real_drawdown_strategy(18_000)
    strategy = create_db_aware_strategy(base_strategy, DB_PENSIONS)
    plot_sequence_of_returns_scenarios(
        INITIAL_TAX_FREE_POT,
        INITIAL_DC_POT,
        SECONDARY_DC_POT,
        SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        output_file=output_dir / "sequence_scenarios.png",
    )
    print()

    print("2. Running Monte Carlo simulation...")
    plot_monte_carlo_fan_chart(
        INITIAL_TAX_FREE_POT,
        INITIAL_DC_POT,
        SECONDARY_DC_POT,
        SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        NUM_SIMULATIONS,
        RANDOM_SEED,
        output_file=output_dir / "monte_carlo_fan.png",
    )
    print()

    print("3. Comparing multiple DC drawdown levels...")
    plot_multiple_drawdown_levels(
        INITIAL_TAX_FREE_POT,
        INITIAL_DC_POT,
        SECONDARY_DC_POT,
        SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        ANNUAL_DRAWDOWNS,
        RANDOM_SEED,
        output_file=output_dir / "multiple_drawdowns.png",
    )
    print()

    print("4. Plotting pot composition over time (stacked areas)...")
    plot_pots_stacked_area(
        INITIAL_TAX_FREE_POT,
        INITIAL_DC_POT,
        SECONDARY_DC_POT,
        SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        RANDOM_SEED,
        output_file=output_dir / "pots_stacked_area.png",
    )
    print()

    print("5. Plotting individual pots dynamics (4-panel subplots)...")
    plot_individual_pots_subplots(
        INITIAL_TAX_FREE_POT,
        INITIAL_DC_POT,
        SECONDARY_DC_POT,
        SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS,
        START_AGE,
        END_AGE,
        MEAN_RETURN,
        STD_RETURN,
        strategy,
        RANDOM_SEED,
        output_file=output_dir / "pots_individual.png",
    )
    print()

    print("=" * 80)
    print(f"All simulations complete. Charts saved to: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
