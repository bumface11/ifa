"""Life event schedule helpers.

Phase 1 keeps event behavior unchanged; this module exposes a baseline schedule
builder that Phase 2 extends with life events.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
