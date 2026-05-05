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
        _,
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


def test_dc_pot_keeps_growing_after_drawdown_start_when_not_withdrawn() -> None:
    """DC pot should compound after drawdown age if no withdrawals are needed."""
    start_age = 60
    end_age = 62
    returns = np.full(end_age - start_age, 0.10, dtype=np.float64)

    (
        _ages,
        _total_balances,
        dc_balances,
        _secondary_dc_balances,
        _tax_free_balances,
        _db_income,
        _withdrawals,
        _,
    ) = simulate_multi_pot_pension_path(
        tax_free_pot=0.0,
        dc_pot=100.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=60,
        db_pensions=[],
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=create_fixed_real_drawdown_strategy(0.0),
    )

    assert np.allclose(dc_balances, np.array([100.0, 110.0, 121.0]))


def test_annual_tax_is_zero_without_tax_regime() -> None:
    """annual_tax should be all zeros when no tax_regime is provided."""
    returns = np.zeros(5, dtype=np.float64)
    *_, annual_tax = simulate_multi_pot_pension_path(
        tax_free_pot=0.0,
        dc_pot=100_000.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=60,
        db_pensions=[(60, 8_000.0)],
        start_age=60,
        end_age=65,
        returns=returns,
        withdrawals_required=np.full(6, 5_000.0, dtype=np.float64),
    )

    assert np.all(annual_tax == 0.0)


def test_annual_tax_positive_with_tax_regime_and_dc_withdrawals() -> None:
    """annual_tax should be positive when tax_regime is set and DC is withdrawn."""
    from ifa.tax import TaxRegime

    returns = np.zeros(5, dtype=np.float64)
    # Net spending £25,000 well above personal allowance; DB income £0
    # so DC withdrawals attract tax.
    *_, annual_tax = simulate_multi_pot_pension_path(
        tax_free_pot=0.0,
        dc_pot=500_000.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=60,
        db_pensions=[],
        start_age=60,
        end_age=65,
        returns=returns,
        withdrawals_required=np.full(6, 25_000.0, dtype=np.float64),
        tax_regime=TaxRegime.REST_OF_UK,
    )

    # Annual spending £25,000 > personal allowance £12,570, so tax > 0
    assert np.all(annual_tax[1:] > 0.0)
    assert annual_tax[0] == 0.0  # index 0 is the starting snapshot, no withdrawal


def test_annual_tax_equals_income_tax_on_db_plus_gross_dc() -> None:
    """annual_tax should match calculate_income_tax(db + gross_dc, regime)."""
    from ifa.tax import TaxRegime, calculate_income_tax, gross_up_dc_withdrawal

    db_income = 8_000.0
    net_needed = 15_000.0
    regime = TaxRegime.REST_OF_UK

    gross_dc = gross_up_dc_withdrawal(net_needed, db_income, regime)
    expected_tax = calculate_income_tax(db_income + gross_dc, regime)

    returns = np.zeros(1, dtype=np.float64)
    *_, annual_tax = simulate_multi_pot_pension_path(
        tax_free_pot=0.0,
        dc_pot=500_000.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=60,
        db_pensions=[(60, db_income)],
        start_age=60,
        end_age=61,
        returns=returns,
        withdrawals_required=np.array([net_needed, net_needed], dtype=np.float64),
        tax_regime=regime,
    )

    assert np.isclose(annual_tax[1], expected_tax)
