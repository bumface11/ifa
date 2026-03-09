"""Market return generation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_random_returns(
    num_years: int, mean: float, std: float, seed: int | None = None
) -> NDArray[np.float64]:
    """Generate normally-distributed annual returns.

    Args:
        num_years: Number of annual returns to generate.
        mean: Mean annual return.
        std: Return standard deviation.
        seed: Optional random seed.

    Returns:
        Array of annual returns.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, std, num_years).astype(np.float64)


def generate_deterministic_sequences(
    num_years: int, mean: float, std: float, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate deterministic sequence-of-returns scenarios.

    Args:
        num_years: Projection years.
        mean: Mean annual return.
        std: Return standard deviation.
        seed: Seed used to construct deterministic sorted sequences.

    Returns:
        Tuple containing ``early_bad``, ``early_good``, and ``constant`` arrays.
    """
    returns = generate_random_returns(num_years, mean, std, seed=seed)
    sorted_returns = np.sort(returns)

    midpoint = num_years // 2

    early_bad = np.concatenate(
        [
            sorted_returns[:midpoint],
            sorted_returns[midpoint:][::-1],
        ]
    ).astype(np.float64)

    early_good = np.concatenate(
        [
            sorted_returns[midpoint:][::-1],
            sorted_returns[:midpoint],
        ]
    ).astype(np.float64)

    constant = np.full(num_years, mean, dtype=np.float64)
    return early_bad, early_good, constant


def deterministic_presets(
    num_years: int, mean: float, std: float, seed: int = 42
) -> dict[str, NDArray[np.float64]]:
    """Return named deterministic market presets.

    Args:
        num_years: Projection years.
        mean: Mean annual return.
        std: Return standard deviation.
        seed: Seed used to generate sorted scenarios.

    Returns:
        Mapping of preset names to annual return arrays.
    """
    early_bad, early_good, constant = generate_deterministic_sequences(
        num_years, mean, std, seed=seed
    )
    typical = generate_random_returns(num_years, mean, std, seed=seed)

    return {
        "Typical": typical,
        "Bad start": early_bad,
        "Good start": early_good,
        "Constant": constant,
    }


def monte_carlo_returns(
    seed: int,
    mean: float,
    std: float,
    n_sims: int,
    n_years: int,
) -> NDArray[np.float64]:
    """Generate Monte Carlo return matrix.

    Args:
        seed: Random seed.
        mean: Mean annual return.
        std: Return standard deviation.
        n_sims: Number of simulations.
        n_years: Number of years per simulation.

    Returns:
        Return matrix with shape ``(n_sims, n_years)``.
    """
    np.random.seed(seed)
    return np.random.normal(mean, std, size=(n_sims, n_years)).astype(np.float64)
