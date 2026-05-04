"""Tests for UK income tax calculations and drawdown grossing-up."""

from __future__ import annotations

import pytest

from ifa.tax import TaxRegime, calculate_income_tax, gross_up_dc_withdrawal


# ── calculate_income_tax ──────────────────────────────────────────────────────


class TestCalculateIncomeTax:
    """Tests for calculate_income_tax."""

    def test_zero_income_returns_zero(self) -> None:
        """No income should produce no tax."""
        assert calculate_income_tax(0.0, TaxRegime.REST_OF_UK) == 0.0
        assert calculate_income_tax(0.0, TaxRegime.SCOTLAND) == 0.0

    def test_negative_income_returns_zero(self) -> None:
        """Negative input is clamped to zero tax."""
        assert calculate_income_tax(-1000.0, TaxRegime.REST_OF_UK) == 0.0

    def test_within_personal_allowance_no_tax_ruk(self) -> None:
        """Income within the personal allowance is tax-free (rest of UK)."""
        assert calculate_income_tax(12_570.0, TaxRegime.REST_OF_UK) == 0.0

    def test_within_personal_allowance_no_tax_scotland(self) -> None:
        """Income within the personal allowance is tax-free (Scotland)."""
        assert calculate_income_tax(12_570.0, TaxRegime.SCOTLAND) == 0.0

    def test_basic_rate_band_ruk(self) -> None:
        """£14,000 income: £1,430 above allowance at 20% = £286."""
        tax = calculate_income_tax(14_000.0, TaxRegime.REST_OF_UK)
        assert abs(tax - 286.0) < 0.01

    def test_basic_rate_band_scotland_starter(self) -> None:
        """£14,000 income in Scotland.

        £12,570 at 0% (personal allowance) = £0
        £14,000 - £12,570 = £1,430 in starter band (19%) = £271.70
        """
        tax = calculate_income_tax(14_000.0, TaxRegime.SCOTLAND)
        assert abs(tax - 271.70) < 0.01

    def test_higher_rate_ruk(self) -> None:
        """£60,000 income (Rest of UK).

        £12,570 at 0% = £0
        £50,270 - £12,570 = £37,700 at 20% = £7,540
        £60,000 - £50,270 = £9,730 at 40% = £3,892
        Total = £11,432
        """
        tax = calculate_income_tax(60_000.0, TaxRegime.REST_OF_UK)
        assert abs(tax - 11_432.0) < 0.01

    def test_at_additional_rate_boundary_ruk(self) -> None:
        """Income just above £125,140 incurs additional rate (45%) on the excess."""
        tax_at_boundary = calculate_income_tax(125_140.0, TaxRegime.REST_OF_UK)
        tax_above = calculate_income_tax(126_140.0, TaxRegime.REST_OF_UK)
        marginal = tax_above - tax_at_boundary
        assert abs(marginal - 450.0) < 0.01  # £1,000 at 45%

    def test_scotland_top_rate_marginal(self) -> None:
        """Income above £125,140 in Scotland incurs top rate (48%) on the excess."""
        tax_at_boundary = calculate_income_tax(125_140.0, TaxRegime.SCOTLAND)
        tax_above = calculate_income_tax(126_140.0, TaxRegime.SCOTLAND)
        marginal = tax_above - tax_at_boundary
        assert abs(marginal - 480.0) < 0.01  # £1,000 at 48%

    def test_scotland_advanced_rate_marginal(self) -> None:
        """Between £75,000 and £125,140 Scotland uses 45% advanced rate."""
        tax_75k = calculate_income_tax(75_000.0, TaxRegime.SCOTLAND)
        tax_76k = calculate_income_tax(76_000.0, TaxRegime.SCOTLAND)
        marginal = tax_76k - tax_75k
        assert abs(marginal - 450.0) < 0.01  # £1,000 at 45%


# ── gross_up_dc_withdrawal ─────────────────────────────────────────────────────


class TestGrossUpDcWithdrawal:
    """Tests for gross_up_dc_withdrawal."""

    def test_zero_net_returns_zero(self) -> None:
        """Nothing needed → nothing to withdraw."""
        assert gross_up_dc_withdrawal(0.0, 0.0, TaxRegime.REST_OF_UK) == 0.0

    def test_negative_net_returns_zero(self) -> None:
        """Negative net should return zero."""
        assert gross_up_dc_withdrawal(-100.0, 0.0, TaxRegime.REST_OF_UK) == 0.0

    def test_within_personal_allowance_no_grossing_needed(self) -> None:
        """If DB income + DC gross stays below personal allowance, no grossing."""
        # DB=0, net=5000 → gross should also be 5000 (all within 0% band)
        gross = gross_up_dc_withdrawal(5_000.0, 0.0, TaxRegime.REST_OF_UK)
        assert abs(gross - 5_000.0) < 0.01

    def test_all_in_basic_rate_ruk(self) -> None:
        """DB income already fills the personal allowance; DC all taxed at 20%.

        Net = £10,000; gross = £10,000 / 0.80 = £12,500.
        """
        gross = gross_up_dc_withdrawal(10_000.0, 14_000.0, TaxRegime.REST_OF_UK)
        assert abs(gross - 12_500.0) < 0.01

    def test_gross_up_result_is_consistent(self) -> None:
        """Verify that gross − marginal_tax == net for rest-of-UK."""
        db = 14_000.0
        net = 10_000.0
        gross = gross_up_dc_withdrawal(net, db, TaxRegime.REST_OF_UK)
        marginal_tax = (
            calculate_income_tax(db + gross, TaxRegime.REST_OF_UK)
            - calculate_income_tax(db, TaxRegime.REST_OF_UK)
        )
        assert abs(gross - marginal_tax - net) < 0.01

    def test_gross_up_result_is_consistent_scotland(self) -> None:
        """Verify that gross − marginal_tax == net for Scotland."""
        db = 20_000.0
        net = 8_000.0
        gross = gross_up_dc_withdrawal(net, db, TaxRegime.SCOTLAND)
        marginal_tax = (
            calculate_income_tax(db + gross, TaxRegime.SCOTLAND)
            - calculate_income_tax(db, TaxRegime.SCOTLAND)
        )
        assert abs(gross - marginal_tax - net) < 0.01

    def test_spans_multiple_bands_ruk(self) -> None:
        """Large withdrawal spans allowance boundary into basic-rate band (RUK)."""
        # DB=0, net=£30,000
        # First £12,570 at 0% from DC, then rest at 20%
        gross = gross_up_dc_withdrawal(30_000.0, 0.0, TaxRegime.REST_OF_UK)
        marginal_tax = calculate_income_tax(gross, TaxRegime.REST_OF_UK)
        assert abs(gross - marginal_tax - 30_000.0) < 0.01

    def test_high_db_income_pushes_into_higher_rate(self) -> None:
        """DB income pushing into higher rate means DC is taxed at 40%."""
        db = 60_000.0  # above the basic-rate ceiling (£50,270)
        net = 5_000.0
        gross = gross_up_dc_withdrawal(net, db, TaxRegime.REST_OF_UK)
        # All gross is in the higher-rate band (40%)
        assert abs(gross - net / 0.60) < 0.01

    def test_scotland_higher_rate_marginal(self) -> None:
        """Between £43,662–£75,000 Scotland applies 42%."""
        db = 50_000.0  # within Scotland's higher-rate band (43,662–75,000)
        net = 1_000.0
        gross = gross_up_dc_withdrawal(net, db, TaxRegime.SCOTLAND)
        assert abs(gross - net / 0.58) < 0.01  # 42% rate → keep 58%

    @pytest.mark.parametrize("regime", list(TaxRegime))
    def test_gross_up_greater_or_equal_to_net(self, regime: TaxRegime) -> None:
        """Grossed-up amount is always >= net required."""
        gross = gross_up_dc_withdrawal(15_000.0, 25_000.0, regime)
        assert gross >= 15_000.0


# ── Tax regime interaction with simulation engine ─────────────────────────────


class TestTaxAwareEngine:
    """Integration tests: tax regime affects simulated pot depletion."""

    def test_tax_reduces_pot_faster_than_no_tax(self) -> None:
        """With tax, DC pot is depleted faster because gross > net withdrawal."""
        import numpy as np

        from ifa.engine import simulate_multi_pot_pension_path

        start_age = 60
        end_age = 70
        returns = np.zeros(end_age - start_age, dtype=np.float64)
        net_spending = np.full(end_age - start_age + 1, 10_000.0, dtype=np.float64)

        _, _, dc_no_tax, *_ = simulate_multi_pot_pension_path(
            tax_free_pot=0.0,
            dc_pot=200_000.0,
            secondary_dc_pot=0.0,
            secondary_dc_drawdown_age=None,
            db_pensions=[(60, 20_000.0)],  # large DB fills allowance → DC is taxed
            start_age=start_age,
            end_age=end_age,
            returns=returns,
            withdrawals_required=net_spending,
            tax_regime=None,
        )
        _, _, dc_with_tax, *_ = simulate_multi_pot_pension_path(
            tax_free_pot=0.0,
            dc_pot=200_000.0,
            secondary_dc_pot=0.0,
            secondary_dc_drawdown_age=None,
            db_pensions=[(60, 20_000.0)],
            start_age=start_age,
            end_age=end_age,
            returns=returns,
            withdrawals_required=net_spending,
            tax_regime=TaxRegime.REST_OF_UK,
        )

        # With tax, DC pot should be lower at the end (more gross withdrawn).
        assert dc_with_tax[-1] < dc_no_tax[-1]

    def test_no_tax_when_only_tax_free_pot_used(self) -> None:
        """Tax-free pot withdrawals should not trigger any grossing-up."""
        import numpy as np

        from ifa.engine import simulate_multi_pot_pension_path

        start_age = 60
        end_age = 65
        returns = np.zeros(end_age - start_age, dtype=np.float64)
        # Spending needs small enough to be funded entirely from tax-free pot
        net_spending = np.full(end_age - start_age + 1, 5_000.0, dtype=np.float64)

        _, _, dc_no_tax, _, tf_no_tax, *_ = simulate_multi_pot_pension_path(
            tax_free_pot=50_000.0,
            dc_pot=100_000.0,
            secondary_dc_pot=0.0,
            secondary_dc_drawdown_age=None,
            db_pensions=[],
            start_age=start_age,
            end_age=end_age,
            returns=returns,
            withdrawals_required=net_spending,
            tax_regime=None,
        )
        _, _, dc_with_tax, _, tf_with_tax, *_ = simulate_multi_pot_pension_path(
            tax_free_pot=50_000.0,
            dc_pot=100_000.0,
            secondary_dc_pot=0.0,
            secondary_dc_drawdown_age=None,
            db_pensions=[],
            start_age=start_age,
            end_age=end_age,
            returns=returns,
            withdrawals_required=net_spending,
            tax_regime=TaxRegime.REST_OF_UK,
        )

        # DC pot should be identical because tax-free pot covers all spending.
        assert np.allclose(dc_no_tax, dc_with_tax)
