"""Backward-compatible simulation module.

This module re-exports engine functions for compatibility during migration.
"""

from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)

__all__ = [
    "calculate_db_pension_income",
    "simulate_multi_pot_pension_path",
    "run_monte_carlo_simulation",
]
