"""Plain-English explanation helpers for novice-facing outputs."""

from __future__ import annotations

from collections.abc import Sequence

from ifa.metrics import MonteCarloMetrics, PathMetrics
from ifa.models import LifeEvent, LumpSumEvent


def build_plain_english_explanation(
    baseline_metrics: PathMetrics,
    scenario_metrics: PathMetrics,
    monte_carlo_metrics: MonteCarloMetrics,
    events: Sequence[LifeEvent],
    event_names: Sequence[str] | None = None,
) -> str:
    """Build a short plain-English explanation of scenario impact.

    Args:
        baseline_metrics: Deterministic baseline path summary.
        scenario_metrics: Deterministic scenario path summary.
        monte_carlo_metrics: Monte Carlo summary for the active scenario.
        events: Life events included in the scenario.
        event_names: Optional custom names aligned to ``events`` order.

    Returns:
        Human-readable explanation describing what changed and why it matters.
    """
    event_parts: list[str] = []
    for event_index, event in enumerate(events):
        event_name = (
            event_names[event_index]
            if event_names is not None and event_index < len(event_names)
            else None
        )
        if isinstance(event, LumpSumEvent):
            if event_name is None:
                event_parts.append(
                    f"a one-off cost of £{event.amount:,.0f} at age {event.age}"
                )
            else:
                event_parts.append(
                    f"{event_name}: one-off cost of £{event.amount:,.0f} "
                    f"at age {event.age}"
                )
        elif event.end_age is None:
            if event_name is None:
                event_parts.append(
                    "an ongoing extra cost of "
                    f"£{event.extra_per_year:,.0f}/year from age {event.start_age}"
                )
            else:
                event_parts.append(
                    f"{event_name}: ongoing extra cost of "
                    f"£{event.extra_per_year:,.0f}/year from age {event.start_age}"
                )
        else:
            if event_name is None:
                event_parts.append(
                    "an ongoing extra cost of "
                    f"£{event.extra_per_year:,.0f}/year from age {event.start_age} "
                    f"to {event.end_age}"
                )
            else:
                event_parts.append(
                    f"{event_name}: ongoing extra cost of "
                    f"£{event.extra_per_year:,.0f}/year from age {event.start_age} "
                    f"to {event.end_age}"
                )

    if len(event_parts) == 0:
        event_summary = "No life events were added"
    elif len(event_parts) == 1:
        event_summary = f"You added {event_parts[0]}"
    else:
        event_summary = "You added " + "; ".join(event_parts)

    deterministic_delta = (
        scenario_metrics.ending_balance - baseline_metrics.ending_balance
    )
    direction = "lower" if deterministic_delta < 0 else "higher"

    return (
        f"{event_summary}. In this deterministic run, your ending balance is "
        f"£{scenario_metrics.ending_balance:,.0f}, which is {direction} than "
        f"baseline by £{abs(deterministic_delta):,.0f}. "
        f"Across many market paths, the estimated chance of running out is "
        f"{monte_carlo_metrics.ruin_probability * 100:.1f}%, and the median ending "
        f"balance is £{monte_carlo_metrics.median_ending_balance:,.0f}. "
        "This happens because larger spending needs mean more money is withdrawn "
        "earlier, leaving less invested for later growth."
    )
