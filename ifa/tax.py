"""UK income tax calculation for pension drawdown modelling.

Implements 2024/25 income tax bands for two regimes:

- **Rest of UK** (England, Wales, Northern Ireland) — uses HMRC standard rates.
- **Scotland** — uses Scottish Rate of Income Tax (SRIT) bands set by the
  Scottish Parliament.

The personal allowance taper above £100,000 is omitted as a simplifying
assumption; this keeps the model beginner-friendly and avoids a non-linear
regime that rarely affects typical pension savers.

All figures are in *real (inflation-adjusted) terms* consistent with the rest
of the simulation; the bands themselves are treated as fixed in real terms.

Notes
-----
Tax-band definitions are module-level constants so they are easy to update
each year in one place.

Simplifying assumptions
-----------------------
- The 25% Pension Commencement Lump Sum (PCLS) is modelled as the ``tax_free_pot``
  input — withdrawals from that pot are genuinely tax-free.
- DB pension income is treated as gross income that offsets spending needs at
  face value (i.e. the model does not separately deduct tax paid on DB income
  when computing the pot-withdrawal requirement).  This means the net-spending
  target the user enters is approximately correct when DB income falls mostly
  within the personal allowance, and is a slight underestimate of required
  withdrawals when DB income exceeds the allowance.
- The personal allowance taper above £100,000 is ignored.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

# ── Tax regime ────────────────────────────────────────────────────────────────


class TaxRegime(str, Enum):
    """Selectable UK income-tax regime for drawdown modelling."""

    REST_OF_UK = "rest_of_uk"
    SCOTLAND = "scotland"


# ── Band definitions (2024/25) ────────────────────────────────────────────────
# Each band is (lower_bound_inclusive, upper_bound_exclusive, marginal_rate).
# The personal allowance (0 %) is the first band.
# The top band uses float("inf") as its upper bound.

_BANDS_REST_OF_UK: Final[list[tuple[float, float, float]]] = [
    (0.0, 12_570.0, 0.00),      # Personal allowance
    (12_570.0, 50_270.0, 0.20),  # Basic rate
    (50_270.0, 125_140.0, 0.40), # Higher rate
    (125_140.0, float("inf"), 0.45),  # Additional rate
]

_BANDS_SCOTLAND: Final[list[tuple[float, float, float]]] = [
    (0.0, 12_570.0, 0.00),       # Personal allowance
    (12_570.0, 14_876.0, 0.19),  # Starter rate
    (14_876.0, 26_561.0, 0.20),  # Basic rate
    (26_561.0, 43_662.0, 0.21),  # Intermediate rate
    (43_662.0, 75_000.0, 0.42),  # Higher rate
    (75_000.0, 125_140.0, 0.45), # Advanced rate
    (125_140.0, float("inf"), 0.48),  # Top rate
]

_BANDS: Final[dict[TaxRegime, list[tuple[float, float, float]]]] = {
    TaxRegime.REST_OF_UK: _BANDS_REST_OF_UK,
    TaxRegime.SCOTLAND: _BANDS_SCOTLAND,
}

# ── Public helpers ────────────────────────────────────────────────────────────


def calculate_income_tax(gross_income: float, regime: TaxRegime) -> float:
    """Calculate total income tax on a given gross income.

    Args:
        gross_income: Total gross income subject to income tax.
        regime: Tax-band regime to apply.

    Returns:
        Total income tax payable (non-negative).
    """
    if gross_income <= 0.0:
        return 0.0

    bands = _BANDS[regime]
    tax = 0.0
    for lower, upper, rate in bands:
        if gross_income <= lower:
            break
        taxable_in_band = min(gross_income, upper) - lower
        tax += taxable_in_band * rate
    return tax


def gross_up_dc_withdrawal(
    net_needed: float,
    db_income: float,
    regime: TaxRegime,
) -> float:
    """Calculate the gross DC pot withdrawal required to fund *net_needed* after tax.

    The DC withdrawal is taxable income stacked on top of *db_income*.  This
    function finds the gross withdrawal G such that::

        G − (tax(db_income + G) − tax(db_income)) = net_needed

    It works analytically by iterating through the tax bands above the
    *db_income* level, filling each band in turn.

    Args:
        net_needed: Net amount (after income tax) still required from DC pots,
            after the tax-free pot has already been used.
        db_income: Gross DB pension income received in the same tax year.
            This determines which tax band the DC withdrawal starts in.
        regime: Tax-band regime to apply.

    Returns:
        Gross DC withdrawal required (>= *net_needed*).  Returns 0.0 when
        *net_needed* <= 0.
    """
    if net_needed <= 0.0:
        return 0.0

    bands = _BANDS[regime]
    remaining_net = net_needed
    gross = 0.0

    for lower, upper, rate in bands:
        # Skip bands entirely consumed by existing DB income.
        if db_income >= upper:
            continue

        # Available gross capacity in this band above db_income (and any DC
        # already allocated to prior bands in this call).
        band_start = max(lower, db_income)
        band_space = upper - band_start  # gross capacity

        # Net that can be funded from this band.
        net_from_band = band_space * (1.0 - rate)

        if remaining_net <= net_from_band:
            gross += remaining_net / (1.0 - rate) if rate < 1.0 else band_space
            remaining_net = 0.0
            break

        gross += band_space
        remaining_net -= net_from_band

    # If remaining net exceeds all defined bands (top-rate overflow).
    if remaining_net > 0.0 and bands:
        top_rate = bands[-1][2]
        gross += (
            remaining_net / (1.0 - top_rate) if top_rate < 1.0 else remaining_net
        )

    return gross
