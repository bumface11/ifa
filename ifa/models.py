"""Typed data models for the pension simulator engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pot:
    """Represents an investment pot used in the simulation."""

    name: str
    initial_balance: float


@dataclass(frozen=True)
class DbPension:
    """Represents a defined-benefit pension stream."""

    start_age: int
    annual_amount: float


@dataclass(frozen=True)
class DcPot:
    """Represents a DC pot with a drawdown start age."""

    drawdown_start_age: int
    initial_balance: float


@dataclass(frozen=True)
class LumpSumEvent:
    """One-off real-terms spending increase at a specific age."""

    age: int
    amount: float


@dataclass(frozen=True)
class SpendingStepEvent:
    """Persistent real-terms spending increase over an age range."""

    start_age: int
    extra_per_year: float
    end_age: int | None = None


LifeEvent = LumpSumEvent | SpendingStepEvent


@dataclass(frozen=True)
class Scenario:
    """Represents one simulation scenario configuration."""

    start_age: int
    end_age: int
    tax_free_pot: float
    main_dc_pot: float
    secondary_dc_pot: float
    secondary_dc_drawdown_age: int | None
    db_pensions: tuple[DbPension, ...]
    dc_pots: tuple[DcPot, ...] = ()
    baseline_spending: float | None = None
    events: tuple[LifeEvent, ...] = ()
    withdrawals_required: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class MarketConfig:
    """Represents market assumptions and simulation controls."""

    mean_return: float
    std_return: float
    random_seed: int
    num_simulations: int


@dataclass(frozen=True)
class Results:
    """Container for one full simulated path."""

    ages: NDArray[np.int_]
    total_balances: NDArray[np.float64]
    dc_balances: NDArray[np.float64]
    secondary_dc_balances: NDArray[np.float64]
    tax_free_balances: NDArray[np.float64]
    db_income: NDArray[np.float64]
    total_withdrawals: NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloSummary:
    """Container for Monte Carlo path outputs."""

    ages: NDArray[np.int_]
    paths: NDArray[np.float64]
