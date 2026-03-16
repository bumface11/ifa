"""Shared configuration defaults for the pension simulator."""

from __future__ import annotations

from typing import Final

INITIAL_TAX_FREE_POT: Final[float] = 173_890
INITIAL_DC_POT: Final[float] = 100_000
SECONDARY_DC_POT: Final[float] = 65_000
PRIMARY_DC_DRAWDOWN_AGE: Final[int] = 57
SECONDARY_DC_DRAWDOWN_AGE: Final[int] = 65

DC_POTS: Final[list[tuple[int, float]]] = [
    (PRIMARY_DC_DRAWDOWN_AGE, INITIAL_DC_POT),
    (SECONDARY_DC_DRAWDOWN_AGE, SECONDARY_DC_POT),
]

DB_PENSIONS: Final[list[tuple[int, float]]] = [
    (62, 12_510),
    (67, 11_900),
]

DRAWDOWN_START_AGE: Final[int] = 52
MODEL_START_AGE: Final[int] = DRAWDOWN_START_AGE - 1
# Backward compatibility for code paths that still import START_AGE.
START_AGE: Final[int] = DRAWDOWN_START_AGE
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
