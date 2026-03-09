"""Tests for plain-English explanation generation."""

from __future__ import annotations

from ifa.explain import build_plain_english_explanation
from ifa.metrics import MonteCarloMetrics, PathMetrics
from ifa.models import LumpSumEvent, SpendingStepEvent


def test_explanation_mentions_event_types_and_risk_summary() -> None:
    """Explanation should include event details and Monte Carlo risk context."""
    baseline = PathMetrics(ruin=False, ending_balance=900_000.0, min_balance=700_000.0)
    scenario = PathMetrics(ruin=False, ending_balance=650_000.0, min_balance=500_000.0)
    mc = MonteCarloMetrics(
        ruin_probability=0.22,
        median_ending_balance=600_000.0,
        p10_ending_balance=100_000.0,
        p90_ending_balance=1_200_000.0,
    )

    text = build_plain_english_explanation(
        baseline_metrics=baseline,
        scenario_metrics=scenario,
        monte_carlo_metrics=mc,
        events=(
            LumpSumEvent(age=70, amount=18_000.0),
            SpendingStepEvent(start_age=78, extra_per_year=6_000.0),
        ),
    )

    assert "one-off cost" in text
    assert "ongoing extra cost" in text
    assert "22.0%" in text
    assert "GBP600,000" in text


def test_explanation_handles_no_events() -> None:
    """Explanation should gracefully describe scenarios without life events."""
    baseline = PathMetrics(ruin=False, ending_balance=500_000.0, min_balance=400_000.0)
    scenario = PathMetrics(ruin=False, ending_balance=520_000.0, min_balance=410_000.0)
    mc = MonteCarloMetrics(
        ruin_probability=0.05,
        median_ending_balance=550_000.0,
        p10_ending_balance=300_000.0,
        p90_ending_balance=900_000.0,
    )

    text = build_plain_english_explanation(
        baseline_metrics=baseline,
        scenario_metrics=scenario,
        monte_carlo_metrics=mc,
        events=(),
    )

    assert text.startswith("No life events")
    assert "higher" in text
