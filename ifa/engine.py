"""Core simulation engine functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ifa.models import DbPension, DcPot

DbPensionInput = DbPension | tuple[int, float]
DcPotInput = DcPot | tuple[int, float]
DrawdownFn = Callable[[int, float, dict[str, float]], float]


def _normalize_dc_pots(
    start_age: int,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    dc_pots: Sequence[DcPotInput] | None,
) -> list[tuple[int, float]]:
    """Normalize DC pot inputs into ``(drawdown_start_age, balance)`` tuples."""
    if dc_pots is not None:
        normalized: list[tuple[int, float]] = []
        for pot in dc_pots:
            if isinstance(pot, DcPot):
                normalized.append((pot.drawdown_start_age, pot.initial_balance))
            else:
                normalized.append((int(pot[0]), float(pot[1])))
        return normalized

    secondary_start = (
        start_age if secondary_dc_drawdown_age is None else secondary_dc_drawdown_age
    )
    return [
        (start_age, float(dc_pot)),
        (int(secondary_start), float(secondary_dc_pot)),
    ]


def calculate_db_pension_income(
    age: int, db_pensions: Sequence[DbPensionInput]
) -> float:
    """Calculate total DB pension income for a given age.

    Args:
        age: Current age.
        db_pensions: Pension streams as ``DbPension`` or ``(start_age, annual_amount)``.

    Returns:
        Total annual DB pension income active at ``age``.
    """
    total = 0.0
    for pension in db_pensions:
        if isinstance(pension, DbPension):
            start_age = pension.start_age
            annual_amount = pension.annual_amount
        else:
            start_age, annual_amount = pension
        if age >= start_age:
            total += annual_amount
    return total


def simulate_multi_pot_pension_path(
    tax_free_pot: float,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
    returns: NDArray[np.float64],
    drawdown_fn: DrawdownFn | None = None,
    withdrawals_required: NDArray[np.float64] | None = None,
    dc_pots: Sequence[DcPotInput] | None = None,
) -> tuple[
    NDArray[np.int_],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Simulate pension evolution with multiple pots and income streams.

    Args:
        tax_free_pot: Initial tax-free investment pot.
        dc_pot: Main DC pot balance.
        secondary_dc_pot: Secondary DC pot balance.
        secondary_dc_drawdown_age: Age when secondary DC can be drawn.
        db_pensions: DB pension streams.
        start_age: Starting age for simulation.
        end_age: Ending age for simulation.
        returns: Annual returns for each year in the projection.
        drawdown_fn: Strategy function returning desired annual withdrawal.
        withdrawals_required: Optional required withdrawals aligned with ages.
            When provided, these values are used directly (DB-adjusted spending).
        dc_pots: Optional list of DC pots as ``(drawdown_start_age, balance)``.

    Returns:
        A tuple containing arrays for ages, balances, incomes, and withdrawals.

    Raises:
        ValueError: If ``returns`` does not contain exactly one value per year.
    """
    ages = np.arange(start_age, end_age + 1)
    num_years = len(ages)

    if len(returns) != num_years - 1:
        raise ValueError(
            "returns length must equal (end_age - start_age); "
            f"got {len(returns)} for {num_years - 1} years"
        )

    if withdrawals_required is not None and len(withdrawals_required) != num_years:
        raise ValueError(
            "withdrawals_required length must equal (end_age - start_age + 1); "
            f"got {len(withdrawals_required)} for {num_years} ages"
        )

    dc_pot_config = _normalize_dc_pots(
        start_age=start_age,
        dc_pot=dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_dc_drawdown_age,
        dc_pots=dc_pots,
    )
    num_dc_pots = len(dc_pot_config)

    total_balances = np.zeros(num_years, dtype=np.float64)
    dc_balances = np.zeros(num_years, dtype=np.float64)
    secondary_dc_balances = np.zeros(num_years, dtype=np.float64)
    tax_free_balances = np.zeros(num_years, dtype=np.float64)
    db_income_array = np.zeros(num_years, dtype=np.float64)
    total_withdrawals = np.zeros(num_years, dtype=np.float64)
    dc_balance_matrix = np.zeros((num_dc_pots, num_years), dtype=np.float64)

    drawdown_start_ages = np.array([pot[0] for pot in dc_pot_config], dtype=np.int_)
    initial_balances = np.array([pot[1] for pot in dc_pot_config], dtype=np.float64)

    if num_dc_pots > 0:
        dc_balance_matrix[:, 0] = initial_balances

    dc_balances[0] = dc_balance_matrix[0, 0] if num_dc_pots > 0 else 0.0
    secondary_dc_balances[0] = (
        float(np.sum(dc_balance_matrix[1:, 0])) if num_dc_pots > 1 else 0.0
    )
    tax_free_balances[0] = tax_free_pot
    total_balances[0] = float(np.sum(dc_balance_matrix[:, 0])) + tax_free_pot

    state_dict: dict[str, float] = {}

    for index in range(1, num_years):
        current_age = int(ages[index - 1])
        annual_return = returns[index - 1]

        prior_tax_free = tax_free_balances[index - 1]
        if prior_tax_free > 0:
            tax_free_balances[index] = prior_tax_free * (1.0 + annual_return)

        for pot_index in range(num_dc_pots):
            prior_balance = dc_balance_matrix[pot_index, index - 1]
            if prior_balance <= 0.0:
                dc_balance_matrix[pot_index, index] = 0.0
                continue

            # DC pots continue compounding while positive, even after drawdown
            # eligibility begins.
            dc_balance_matrix[pot_index, index] = prior_balance * (1.0 + annual_return)

        db_income = calculate_db_pension_income(current_age, db_pensions)
        db_income_array[index] = db_income

        combined_dc = float(np.sum(dc_balance_matrix[:, index]))
        if withdrawals_required is not None:
            desired_withdrawal = float(withdrawals_required[index - 1])
        elif drawdown_fn is not None:
            desired_withdrawal = drawdown_fn(
                current_age,
                float(combined_dc),
                state_dict,
            )
        else:
            desired_withdrawal = 0.0

        current_withdrawal = 0.0

        tax_free_withdrawal = min(
            desired_withdrawal - current_withdrawal, tax_free_balances[index]
        )
        tax_free_balances[index] -= tax_free_withdrawal
        current_withdrawal += tax_free_withdrawal

        if current_withdrawal < desired_withdrawal:
            for pot_index in range(num_dc_pots):
                if current_age < int(drawdown_start_ages[pot_index]):
                    continue
                if current_withdrawal >= desired_withdrawal:
                    break

                withdrawal = min(
                    desired_withdrawal - current_withdrawal,
                    float(dc_balance_matrix[pot_index, index]),
                )
                dc_balance_matrix[pot_index, index] -= withdrawal
                current_withdrawal += withdrawal

        if num_dc_pots > 0:
            dc_balance_matrix[:, index] = np.maximum(dc_balance_matrix[:, index], 0.0)

        dc_balances[index] = dc_balance_matrix[0, index] if num_dc_pots > 0 else 0.0
        secondary_dc_balances[index] = (
            float(np.sum(dc_balance_matrix[1:, index])) if num_dc_pots > 1 else 0.0
        )
        tax_free_balances[index] = max(0.0, tax_free_balances[index])

        total_balances[index] = (
            float(np.sum(dc_balance_matrix[:, index])) + tax_free_balances[index]
        )
        total_withdrawals[index] = current_withdrawal

    return (
        ages,
        total_balances,
        dc_balances,
        secondary_dc_balances,
        tax_free_balances,
        db_income_array,
        total_withdrawals,
    )


def run_monte_carlo_simulation(
    tax_free_pot: float,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
    mean_return: float,
    std_return: float,
    strategy_fn: DrawdownFn,
    num_simulations: int,
    seed: int,
    withdrawals_required: NDArray[np.float64] | None = None,
    dc_pots: Sequence[DcPotInput] | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Run Monte Carlo simulation for one strategy.

    Args:
        tax_free_pot: Initial tax-free investment pot.
        dc_pot: Main DC pot balance.
        secondary_dc_pot: Secondary DC pot balance.
        secondary_dc_drawdown_age: Age when secondary DC can be drawn.
        db_pensions: DB pension streams.
        start_age: Starting age for simulation.
        end_age: Ending age for simulation.
        mean_return: Mean annual return.
        std_return: Annual return standard deviation.
        strategy_fn: Strategy function for annual withdrawals.
        num_simulations: Number of simulation paths.
        seed: RNG seed.
        withdrawals_required: Optional required withdrawals aligned with ages.
        dc_pots: Optional list of DC pots as ``(drawdown_start_age, balance)``.

    Returns:
        Ages array and matrix of path balances with shape
        ``(num_simulations, num_years + 1)``.
    """
    np.random.seed(seed)
    num_years = end_age - start_age
    ages = np.arange(start_age, end_age + 1)

    paths = np.zeros((num_simulations, num_years + 1), dtype=np.float64)

    for simulation_index in range(num_simulations):
        returns = np.random.normal(mean_return, std_return, num_years).astype(
            np.float64
        )
        _, total_balances, _, _, _, _, _ = simulate_multi_pot_pension_path(
            tax_free_pot,
            dc_pot,
            secondary_dc_pot,
            secondary_dc_drawdown_age,
            db_pensions,
            start_age,
            end_age,
            returns,
            strategy_fn,
            withdrawals_required=withdrawals_required,
            dc_pots=dc_pots,
        )
        paths[simulation_index, :] = total_balances

    return ages, paths
