"""Drawdown strategy functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ifa.engine import DbPensionInput, calculate_db_pension_income

DrawdownFn = Callable[[int, float, dict[str, float]], float]


def fixed_real(annual_withdrawal: float) -> DrawdownFn:
    """Create a fixed real withdrawal strategy."""

    def strategy(age: int, current_pot: float, state_dict: dict[str, float]) -> float:
        del age, current_pot, state_dict
        return annual_withdrawal

    return strategy


def percent_of_pot(percentage: float) -> DrawdownFn:
    """Create a percentage-of-pot withdrawal strategy."""

    def strategy(age: int, current_pot: float, state_dict: dict[str, float]) -> float:
        del age, state_dict
        return current_pot * percentage

    return strategy


def guardrails(
    target_income: float,
    lower_band: float,
    upper_band: float,
    adjustment: float,
    start_age: int,
    end_age: int,
) -> DrawdownFn:
    """Create a guardrails strategy using a simple linear glidepath."""

    def strategy(age: int, current_pot: float, state_dict: dict[str, float]) -> float:
        if "initial_pot" not in state_dict:
            state_dict["initial_pot"] = current_pot
            state_dict["initial_income"] = target_income
            state_dict["current_income"] = target_income

        initial_pot = state_dict["initial_pot"]
        years_elapsed = age - start_age
        total_years = max(1, end_age - start_age)

        expected_fraction = max(0.0, 1.0 - (years_elapsed / total_years) * 0.5)
        expected_pot = initial_pot * expected_fraction

        current_income = state_dict["current_income"]
        lower_threshold = expected_pot * lower_band
        upper_threshold = expected_pot * upper_band

        if current_pot < lower_threshold:
            current_income = current_income * (1.0 - adjustment)
        elif current_pot > upper_threshold:
            current_income = current_income * (1.0 + adjustment)

        state_dict["current_income"] = current_income
        return current_income

    return strategy


def no_withdrawal() -> DrawdownFn:
    """Create a no-withdrawal baseline strategy."""

    def strategy(age: int, current_pot: float, state_dict: dict[str, float]) -> float:
        del age, current_pot, state_dict
        return 0.0

    return strategy


def db_aware(
    base_strategy: DrawdownFn, db_pensions: Sequence[DbPensionInput]
) -> DrawdownFn:
    """Wrap a strategy so DB pensions reduce DC withdrawal need."""

    def strategy(age: int, current_pot: float, state_dict: dict[str, float]) -> float:
        dc_withdrawal = base_strategy(age, current_pot, state_dict)
        db_income = calculate_db_pension_income(age, db_pensions)
        return max(0.0, dc_withdrawal - db_income)

    return strategy


def create_fixed_real_drawdown_strategy(annual_withdrawal: float) -> DrawdownFn:
    """Backwards-compatible alias for fixed_real."""
    return fixed_real(annual_withdrawal)


def create_percentage_of_pot_strategy(percentage: float) -> DrawdownFn:
    """Backwards-compatible alias for percent_of_pot."""
    return percent_of_pot(percentage)


def create_guardrails_strategy(
    target_income: float,
    lower_band: float,
    upper_band: float,
    adjustment: float,
    start_age: int,
    end_age: int,
) -> DrawdownFn:
    """Backwards-compatible alias for guardrails."""
    return guardrails(
        target_income,
        lower_band,
        upper_band,
        adjustment,
        start_age,
        end_age,
    )


def create_no_withdrawal_strategy() -> DrawdownFn:
    """Backwards-compatible alias for no_withdrawal."""
    return no_withdrawal()


def create_db_aware_strategy(
    base_strategy: DrawdownFn,
    db_pensions: Sequence[DbPensionInput],
) -> DrawdownFn:
    """Backwards-compatible alias for db_aware."""
    return db_aware(base_strategy, db_pensions)
