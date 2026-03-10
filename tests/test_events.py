"""Tests for life-event withdrawal schedule construction."""

from __future__ import annotations

import numpy as np
import pytest

from ifa.events import build_required_withdrawals, build_spending_drawdown_schedule
from ifa.models import LumpSumEvent, SpendingStepEvent


def test_lump_sum_increases_only_one_year() -> None:
    """Lump sum should only affect the specified age."""
    ages = np.arange(60, 65, dtype=np.int_)
    db_income = np.zeros_like(ages, dtype=np.float64)

    result = build_required_withdrawals(
        ages=ages,
        baseline_spending=20_000.0,
        db_income=db_income,
        events=(LumpSumEvent(age=62, amount=12_000.0),),
    )

    expected = np.array([20_000.0, 20_000.0, 32_000.0, 20_000.0, 20_000.0])
    assert np.allclose(result, expected)


def test_step_up_applies_from_start_age_onward() -> None:
    """Step event should increase spending from start age through scenario end."""
    ages = np.arange(60, 65, dtype=np.int_)
    db_income = np.zeros_like(ages, dtype=np.float64)

    result = build_required_withdrawals(
        ages=ages,
        baseline_spending=20_000.0,
        db_income=db_income,
        events=(SpendingStepEvent(start_age=62, extra_per_year=3_500.0),),
    )

    expected = np.array([20_000.0, 20_000.0, 23_500.0, 23_500.0, 23_500.0])
    assert np.allclose(result, expected)


def test_required_withdrawals_are_clamped_at_zero() -> None:
    """Required withdrawals should not go below zero when DB income is high."""
    ages = np.arange(60, 64, dtype=np.int_)
    db_income = np.full(ages.shape[0], 25_000.0, dtype=np.float64)

    result = build_required_withdrawals(
        ages=ages,
        baseline_spending=20_000.0,
        db_income=db_income,
        events=(),
    )

    assert np.allclose(result, np.zeros(ages.shape[0], dtype=np.float64))


def test_invalid_event_age_raises_value_error() -> None:
    """Out-of-range event age should raise a clear validation error."""
    ages = np.arange(60, 65, dtype=np.int_)
    db_income = np.zeros_like(ages, dtype=np.float64)

    with pytest.raises(ValueError, match="out of scenario range"):
        build_required_withdrawals(
            ages=ages,
            baseline_spending=20_000.0,
            db_income=db_income,
            events=(LumpSumEvent(age=75, amount=5_000.0),),
        )


def test_spending_drawdown_schedule_offsets_db_income() -> None:
    """Spending drawdown schedule should be spending minus DB income."""
    ages = np.arange(60, 65, dtype=np.int_)
    db_income = np.array([12_000.0, 12_000.0, 20_000.0, 25_000.0, 30_000.0])

    result = build_spending_drawdown_schedule(
        ages=ages,
        baseline_spending=20_000.0,
        db_income=db_income,
        events=(SpendingStepEvent(start_age=62, extra_per_year=3_000.0),),
    )

    expected = np.array([8_000.0, 8_000.0, 3_000.0, 0.0, 0.0])
    assert np.allclose(result, expected)
