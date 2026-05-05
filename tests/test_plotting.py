"""Tests for plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ifa.models import SpendingStepEvent
from ifa.plotting import plot_cumulative_flows_waterfall, plot_individual_pots_subplots
from ifa.strategies import create_fixed_real_drawdown_strategy
from ifa.tax import TaxRegime


def test_individual_pots_subplot_shows_spending_against_db_income() -> None:
    """The income panel should show annual spending, DB income, and pot gap."""
    figure = plot_individual_pots_subplots(
        tax_free_pot=50_000.0,
        dc_pot=120_000.0,
        secondary_dc_pot=30_000.0,
        secondary_dc_drawdown_age=65,
        db_pensions=((67, 12_000.0),),
        start_age=60,
        end_age=70,
        mean_return=0.03,
        std_return=0.08,
        strategy_fn=create_fixed_real_drawdown_strategy(20_000.0),
        seed=7,
        withdrawals_required=np.full(11, 8_000.0, dtype=np.float64),
        life_events=(SpendingStepEvent(start_age=68, extra_per_year=4_000.0),),
        annual_spending_schedule=np.array(
            [20_000.0] * 8 + [24_000.0] * 3,
            dtype=np.float64,
        ),
        return_figure=True,
        save_output=False,
    )

    assert figure is not None
    income_axis = figure.axes[3]
    legend = income_axis.get_legend()

    assert income_axis.get_title() == "Annual Spending vs DB Income"
    assert legend is not None

    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "Annual spending" in legend_labels
    assert "DB Pension Income" in legend_labels
    assert "Needed from pots" in legend_labels

    plt.close(figure)


def test_individual_pots_subplot_validates_spending_schedule_length() -> None:
    """Annual spending schedule must align with the plotted ages."""
    with pytest.raises(ValueError, match="annual_spending_schedule length"):
        plot_individual_pots_subplots(
            tax_free_pot=50_000.0,
            dc_pot=120_000.0,
            secondary_dc_pot=30_000.0,
            secondary_dc_drawdown_age=65,
            db_pensions=((67, 12_000.0),),
            start_age=60,
            end_age=70,
            mean_return=0.03,
            std_return=0.08,
            strategy_fn=create_fixed_real_drawdown_strategy(20_000.0),
            seed=7,
            withdrawals_required=np.full(11, 8_000.0, dtype=np.float64),
            annual_spending_schedule=np.full(10, 20_000.0, dtype=np.float64),
            return_figure=True,
            save_output=False,
        )


def _make_waterfall_inputs(
    n_ages: int = 11,
    *,
    include_discretionary: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ages, returns, balances, baseline_req, scenario_req) test arrays."""
    ages = np.arange(60, 60 + n_ages, dtype=np.int_)
    annual_returns = np.full(n_ages - 1, 0.04, dtype=np.float64)
    baseline_balances = np.linspace(300_000.0, 100_000.0, n_ages)
    baseline_required = np.full(n_ages, 15_000.0, dtype=np.float64)
    extra = 5_000.0 if include_discretionary else 0.0
    scenario_required = baseline_required + extra
    return ages, annual_returns, baseline_balances, baseline_required, scenario_required


def test_waterfall_returns_figure() -> None:
    """plot_cumulative_flows_waterfall returns a Figure when return_figure=True."""
    ages, returns, balances, base_req, scen_req = _make_waterfall_inputs()
    fig = plot_cumulative_flows_waterfall(
        ages=ages,
        annual_returns=returns,
        baseline_balances=balances,
        baseline_required=base_req,
        scenario_required=scen_req,
        save_output=False,
        return_figure=True,
    )
    assert fig is not None
    plt.close(fig)


def test_waterfall_has_two_axes() -> None:
    """The waterfall figure should contain exactly two axes (flows + balance)."""
    ages, returns, balances, base_req, scen_req = _make_waterfall_inputs()
    fig = plot_cumulative_flows_waterfall(
        ages=ages,
        annual_returns=returns,
        baseline_balances=balances,
        baseline_required=base_req,
        scenario_required=scen_req,
        save_output=False,
        return_figure=True,
    )
    assert fig is not None
    assert len(fig.axes) == 2
    plt.close(fig)


def test_waterfall_legend_labels() -> None:
    """The flows axis legend should contain all three expected components."""
    ages, returns, balances, base_req, scen_req = _make_waterfall_inputs(
        include_discretionary=True
    )
    fig = plot_cumulative_flows_waterfall(
        ages=ages,
        annual_returns=returns,
        baseline_balances=balances,
        baseline_required=base_req,
        scenario_required=scen_req,
        save_output=False,
        return_figure=True,
    )
    assert fig is not None
    flows_ax = fig.axes[0]
    legend = flows_ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert "Annual growth" in labels
    assert "General drawdown" in labels
    assert "Discretionary drawdown" in labels
    plt.close(fig)


def test_waterfall_validates_returns_length() -> None:
    """A mismatched annual_returns array should raise ValueError."""
    ages, returns, balances, base_req, scen_req = _make_waterfall_inputs()
    with pytest.raises(ValueError, match="annual_returns length"):
        plot_cumulative_flows_waterfall(
            ages=ages,
            annual_returns=returns[:-1],  # one too short
            baseline_balances=balances,
            baseline_required=base_req,
            scenario_required=scen_req,
            save_output=False,
            return_figure=True,
        )


def test_waterfall_validates_baseline_required_length() -> None:
    """A mismatched baseline_required array should raise ValueError."""
    ages, returns, balances, base_req, scen_req = _make_waterfall_inputs()
    with pytest.raises(ValueError, match="baseline_required length"):
        plot_cumulative_flows_waterfall(
            ages=ages,
            annual_returns=returns,
            baseline_balances=balances,
            baseline_required=base_req[:-1],  # one too short
            scenario_required=scen_req,
            save_output=False,
            return_figure=True,
        )


def test_waterfall_validates_scenario_required_length() -> None:
    """A mismatched scenario_required array should raise ValueError."""
    ages, returns, balances, base_req, scen_req = _make_waterfall_inputs()
    with pytest.raises(ValueError, match="scenario_required length"):
        plot_cumulative_flows_waterfall(
            ages=ages,
            annual_returns=returns,
            baseline_balances=balances,
            baseline_required=base_req,
            scenario_required=scen_req[:-1],  # one too short
            save_output=False,
            return_figure=True,
        )


def test_individual_pots_subplot_shows_tax_deducted_line_when_tax_regime_set() -> None:
    """Income panel should include a 'Tax deducted' line when tax_regime is given."""
    figure = plot_individual_pots_subplots(
        tax_free_pot=0.0,
        dc_pot=500_000.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=60,
        db_pensions=((60, 5_000.0),),
        start_age=60,
        end_age=70,
        mean_return=0.03,
        std_return=0.0,
        strategy_fn=create_fixed_real_drawdown_strategy(20_000.0),
        seed=1,
        # Net spending £25,000 > personal allowance, so tax > 0
        withdrawals_required=np.full(11, 25_000.0, dtype=np.float64),
        annual_spending_schedule=np.full(11, 30_000.0, dtype=np.float64),
        tax_regime=TaxRegime.REST_OF_UK,
        return_figure=True,
        save_output=False,
    )

    assert figure is not None
    income_axis = figure.axes[3]
    legend = income_axis.get_legend()
    assert legend is not None

    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert "Tax deducted" in legend_labels

    plt.close(figure)


def test_individual_pots_subplot_no_tax_line_without_tax_regime() -> None:
    """Income panel should not include 'Tax deducted' when no tax_regime is given."""
    figure = plot_individual_pots_subplots(
        tax_free_pot=50_000.0,
        dc_pot=120_000.0,
        secondary_dc_pot=0.0,
        secondary_dc_drawdown_age=60,
        db_pensions=((67, 12_000.0),),
        start_age=60,
        end_age=70,
        mean_return=0.03,
        std_return=0.0,
        strategy_fn=create_fixed_real_drawdown_strategy(20_000.0),
        seed=1,
        annual_spending_schedule=np.full(11, 20_000.0, dtype=np.float64),
        return_figure=True,
        save_output=False,
    )

    assert figure is not None
    income_axis = figure.axes[3]
    legend = income_axis.get_legend()

    legend_labels = (
        [text.get_text() for text in legend.get_texts()] if legend is not None else []
    )
    assert "Tax deducted" not in legend_labels

    plt.close(figure)