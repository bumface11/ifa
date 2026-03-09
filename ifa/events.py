"""Life event schedule helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ifa.models import LifeEvent, SpendingStepEvent


def build_spending_schedule(
    start_age: int,
    end_age: int,
    baseline_spending: float,
) -> NDArray[np.float64]:
    """Build a constant baseline spending schedule.

    Args:
        start_age: Starting age for schedule.
        end_age: Ending age for schedule.
        baseline_spending: Annual spending target.

    Returns:
        Spending array with one value per age from start to end inclusive.
    """
    total_years = end_age - start_age + 1
    return np.full(total_years, baseline_spending, dtype=np.float64)


def build_required_withdrawals(
    ages: NDArray[np.int_],
    baseline_spending: float,
    db_income: NDArray[np.float64],
    events: Sequence[LifeEvent],
) -> NDArray[np.float64]:
    """Build required annual withdrawals from pots in real terms.

    Args:
        ages: Inclusive age sequence for the simulation horizon.
        baseline_spending: Baseline annual spending in real terms.
        db_income: Annual DB income values aligned with ``ages``.
        events: Life events that increase required spending.

    Returns:
        Required withdrawals by age after DB income offset and clamping at zero.

    Raises:
        ValueError: If array lengths differ, ages are empty, or events are out of range.
    """
    if ages.size == 0:
        raise ValueError("ages cannot be empty")

    if db_income.shape[0] != ages.shape[0]:
        raise ValueError(
            "db_income length must match ages length; "
            f"got {db_income.shape[0]} and {ages.shape[0]}"
        )

    min_age = int(ages[0])
    max_age = int(ages[-1])
    spending = np.full(ages.shape[0], baseline_spending, dtype=np.float64)

    for event in events:
        if isinstance(event, SpendingStepEvent):
            if event.start_age < min_age or event.start_age > max_age:
                raise ValueError(
                    "SpendingStepEvent start_age out of scenario range: "
                    f"{event.start_age} not in [{min_age}, {max_age}]"
                )
            effective_end = max_age if event.end_age is None else event.end_age
            if effective_end < event.start_age:
                raise ValueError(
                    "SpendingStepEvent end_age must be >= start_age; "
                    f"got end_age={effective_end}, start_age={event.start_age}"
                )
            if effective_end < min_age or effective_end > max_age:
                raise ValueError(
                    "SpendingStepEvent end_age out of scenario range: "
                    f"{effective_end} not in [{min_age}, {max_age}]"
                )

            mask = (ages >= event.start_age) & (ages <= effective_end)
            spending[mask] += event.extra_per_year
            continue

        if event.age < min_age or event.age > max_age:
            raise ValueError(
                "LumpSumEvent age out of scenario range: "
                f"{event.age} not in [{min_age}, {max_age}]"
            )
        year_idx = int(event.age - min_age)
        spending[year_idx] += event.amount

    return np.maximum(0.0, spending - db_income)
