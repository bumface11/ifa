"""Tests for plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ifa.models import SpendingStepEvent
from ifa.plotting import plot_individual_pots_subplots
from ifa.strategies import create_fixed_real_drawdown_strategy


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