"""Tests for core simulation engine behavior."""

from __future__ import annotations

import numpy as np

from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.events import build_required_withdrawals
from ifa.market import generate_random_returns
from ifa.models import LumpSumEvent
from ifa.strategies import create_fixed_real_drawdown_strategy


def test_calculate_db_pension_income_sums_active_streams() -> None:
    """DB income should include only streams active at the queried age."""
    # Arrange
    db_pensions: list[tuple[int, float]] = [
        (60, 10_000.0),
        (67, 12_000.0),
        (72, 5_000.0),
    ]

    # Act
    at_59 = calculate_db_pension_income(59, db_pensions)
    at_67 = calculate_db_pension_income(67, db_pensions)
    at_75 = calculate_db_pension_income(75, db_pensions)

    # Assert
    assert at_59 == 0.0
    assert at_67 == 22_000.0
    assert at_75 == 27_000.0


def test_simulate_multi_pot_pension_path_invariants_hold() -> None:
    """Balances should remain non-negative and totals should stay consistent."""
    # Arrange
    start_age = 60
    end_age = 65
    returns = np.zeros(end_age - start_age, dtype=np.float64)
    strategy = create_fixed_real_drawdown_strategy(15_000.0)

    # Act
    (
        ages,
        total_balances,
        dc_balances,
        secondary_dc_balances,
        tax_free_balances,
        _,
        total_withdrawals,
    ) = simulate_multi_pot_pension_path(
        tax_free_pot=50_000.0,
        dc_pot=80_000.0,
        secondary_dc_pot=20_000.0,
        secondary_dc_drawdown_age=63,
        db_pensions=[],
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=strategy,
    )

    # Assert
    assert len(ages) == (end_age - start_age + 1)
    assert np.all(dc_balances >= 0.0)
    assert np.all(secondary_dc_balances >= 0.0)
    assert np.all(tax_free_balances >= 0.0)
    assert np.all(total_withdrawals >= 0.0)
    assert np.allclose(
        total_balances, dc_balances + secondary_dc_balances + tax_free_balances
    )


def test_determinism_with_fixed_seed_and_fixed_returns() -> None:
    """Repeated runs with same inputs should produce identical outputs."""
    # Arrange
    fixed_strategy = create_fixed_real_drawdown_strategy(12_000.0)
    returns_a = generate_random_returns(10, mean=0.03, std=0.07, seed=123)
    returns_b = generate_random_returns(10, mean=0.03, std=0.07, seed=123)

    # Act
    path_a = simulate_multi_pot_pension_path(
        tax_free_pot=30_000.0,
        dc_pot=120_000.0,
        secondary_dc_pot=10_000.0,
        secondary_dc_drawdown_age=66,
        db_pensions=[(67, 8_000.0)],
        start_age=60,
        end_age=70,
        returns=returns_a,
        drawdown_fn=fixed_strategy,
    )
    path_b = simulate_multi_pot_pension_path(
        tax_free_pot=30_000.0,
        dc_pot=120_000.0,
        secondary_dc_pot=10_000.0,
        secondary_dc_drawdown_age=66,
        db_pensions=[(67, 8_000.0)],
        start_age=60,
        end_age=70,
        returns=returns_b,
        drawdown_fn=fixed_strategy,
    )

    mc_ages_a, mc_paths_a = run_monte_carlo_simulation(
        tax_free_pot=40_000.0,
        dc_pot=100_000.0,
        secondary_dc_pot=20_000.0,
        secondary_dc_drawdown_age=65,
        db_pensions=[],
        start_age=60,
        end_age=70,
        mean_return=0.04,
        std_return=0.09,
        strategy_fn=fixed_strategy,
        num_simulations=8,
        seed=2026,
    )
    mc_ages_b, mc_paths_b = run_monte_carlo_simulation(
        tax_free_pot=40_000.0,
        dc_pot=100_000.0,
        secondary_dc_pot=20_000.0,
        secondary_dc_drawdown_age=65,
        db_pensions=[],
        start_age=60,
        end_age=70,
        mean_return=0.04,
        std_return=0.09,
        strategy_fn=fixed_strategy,
        num_simulations=8,
        seed=2026,
    )

    # Assert
    assert np.array_equal(returns_a, returns_b)
    assert np.array_equal(path_a[0], path_b[0])
    assert np.allclose(path_a[1], path_b[1])
    assert np.allclose(path_a[2], path_b[2])
    assert np.allclose(path_a[3], path_b[3])
    assert np.allclose(path_a[4], path_b[4])
    assert np.allclose(path_a[5], path_b[5])
    assert np.allclose(path_a[6], path_b[6])
    assert np.array_equal(mc_ages_a, mc_ages_b)
    assert np.allclose(mc_paths_a, mc_paths_b)


def test_lump_sum_event_reduces_balances_vs_baseline_on_same_returns() -> None:
    """A life-event lump sum should lower balances versus baseline on same path."""
    # Arrange
    start_age = 60
    end_age = 66
    ages = np.arange(start_age, end_age + 1, dtype=np.int_)
    returns = np.zeros(end_age - start_age, dtype=np.float64)
    db_income = np.zeros_like(ages, dtype=np.float64)

    baseline_required = build_required_withdrawals(
        ages=ages,
        baseline_spending=10_000.0,
        db_income=db_income,
        events=(),
    )
    scenario_required = build_required_withdrawals(
        ages=ages,
        baseline_spending=10_000.0,
        db_income=db_income,
        events=(LumpSumEvent(age=63, amount=15_000.0),),
    )

    # Act
    _, baseline_total, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=25_000.0,
        dc_pot=120_000.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=None,
        db_pensions=[],
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        withdrawals_required=baseline_required,
    )
    _, scenario_total, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=25_000.0,
        dc_pot=120_000.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=None,
        db_pensions=[],
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        withdrawals_required=scenario_required,
    )

    # Assert
    assert scenario_total[-1] < baseline_total[-1]
