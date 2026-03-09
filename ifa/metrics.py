"""Learning-focused summary metrics for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PathMetrics:
    """Summary metrics for one simulated balance path."""

    ruin: bool
    ending_balance: float
    min_balance: float


@dataclass(frozen=True)
class MonteCarloMetrics:
    """Summary metrics across Monte Carlo path outcomes."""

    ruin_probability: float
    median_ending_balance: float
    p10_ending_balance: float
    p90_ending_balance: float


def summarize_path(total_balances: NDArray[np.float64]) -> PathMetrics:
    """Compute beginner-friendly metrics for a single path.

    Args:
        total_balances: Total pot balances by age.

    Returns:
        Path-level metrics including ruin, ending balance, and minimum balance.

    Raises:
        ValueError: If ``total_balances`` is empty.
    """
    if total_balances.size == 0:
        raise ValueError("total_balances cannot be empty")

    return PathMetrics(
        ruin=bool(np.any(total_balances[:-1] <= 0.0)),
        ending_balance=float(total_balances[-1]),
        min_balance=float(np.min(total_balances)),
    )


def summarize_monte_carlo(paths: NDArray[np.float64]) -> MonteCarloMetrics:
    """Compute beginner-friendly metrics across Monte Carlo simulations.

    Args:
        paths: Matrix of simulated total balances with shape ``(n_paths, n_ages)``.

    Returns:
        Monte Carlo metrics including ruin probability and ending-balance percentiles.

    Raises:
        ValueError: If ``paths`` is not a non-empty 2D array.
    """
    if paths.ndim != 2 or paths.shape[0] == 0 or paths.shape[1] == 0:
        raise ValueError("paths must be a non-empty 2D array")

    ruined = np.any(paths[:, :-1] <= 0.0, axis=1)
    endings = paths[:, -1]

    return MonteCarloMetrics(
        ruin_probability=float(np.mean(ruined)),
        median_ending_balance=float(np.median(endings)),
        p10_ending_balance=float(np.percentile(endings, 10)),
        p90_ending_balance=float(np.percentile(endings, 90)),
    )
