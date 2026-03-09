"""IFA pension drawdown simulator package."""

from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.market import generate_deterministic_sequences, generate_random_returns

__all__ = [
    "calculate_db_pension_income",
    "simulate_multi_pot_pension_path",
    "run_monte_carlo_simulation",
    "generate_random_returns",
    "generate_deterministic_sequences",
]
