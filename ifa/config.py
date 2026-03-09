"""Shared configuration defaults for the pension simulator."""

from __future__ import annotations

from typing import Final

INITIAL_TAX_FREE_POT: Final[float] = 373_890
INITIAL_DC_POT: Final[float] = 300_000
SECONDARY_DC_POT: Final[float] = 65_000
SECONDARY_DC_DRAWDOWN_AGE: Final[int] = 65

DB_PENSIONS: Final[list[tuple[int, float]]] = [
    (62, 12_510),
    (67, 11_900),
]

START_AGE: Final[int] = 52
END_AGE: Final[int] = 95

MEAN_RETURN: Final[float] = 0.04
STD_RETURN: Final[float] = 0.10
RANDOM_SEED: Final[int] = 42

ANNUAL_DRAWDOWNS: Final[list[float]] = [26_000, 30_000, 34_000]
NUM_SIMULATIONS: Final[int] = 1_000

GUARDRAILS_TARGET_INCOME: Final[float] = 30_000
GUARDRAILS_LOWER_BAND: Final[float] = 0.80
GUARDRAILS_UPPER_BAND: Final[float] = 1.20
GUARDRAILS_ADJUSTMENT: Final[float] = 0.10
