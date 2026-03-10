"""IFA pension drawdown simulator package."""

from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.events import (
    build_annual_spending_schedule,
    build_required_withdrawals,
    build_spending_drawdown_schedule,
)
from ifa.market import generate_deterministic_sequences, generate_random_returns
from ifa.metrics import summarize_monte_carlo, summarize_path
from ifa.models import LumpSumEvent, SpendingStepEvent

__all__ = [
    "calculate_db_pension_income",
    "simulate_multi_pot_pension_path",
    "run_monte_carlo_simulation",
    "generate_random_returns",
    "generate_deterministic_sequences",
    "LumpSumEvent",
    "SpendingStepEvent",
    "build_required_withdrawals",
    "build_annual_spending_schedule",
    "build_spending_drawdown_schedule",
    "summarize_path",
    "summarize_monte_carlo",
]
