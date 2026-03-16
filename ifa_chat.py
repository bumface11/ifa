"""Streamlit chat-driven interface for exploring retirement scenarios."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import streamlit as st
from matplotlib.figure import Figure

from ifa.config import (
    DB_PENSIONS,
    DC_POTS,
    END_AGE,
    INITIAL_TAX_FREE_POT,
    MEAN_RETURN,
    NUM_SIMULATIONS,
    RANDOM_SEED,
    START_AGE,
    STD_RETURN,
)
from ifa.engine import (
    calculate_db_pension_income,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.events import build_required_withdrawals, build_spending_drawdown_schedule
from ifa.explain import build_plain_english_explanation
from ifa.market import generate_random_returns
from ifa.metrics import summarize_monte_carlo, summarize_path
from ifa.models import LumpSumEvent, SpendingStepEvent
from ifa.plotting import (
    plot_baseline_vs_scenario_balances,
    plot_individual_pots_subplots,
    plot_monte_carlo_fan_chart,
    plot_pots_stacked_area,
    plot_sequence_of_returns_scenarios,
)
from ifa.strategies import create_fixed_real_drawdown_strategy

_WELCOME_MESSAGE = """
Welcome! I can help you explore your retirement finances interactively.

Try asking questions like:

- *"I'm 55"* — set your current age
- *"Retire at 60"* — change DC pot drawdown start age
- *"End age 90"* — set the projection end age
- *"Spending £25,000/year"* — set baseline annual spending
- *"DC pot £300,000"* — update your DC pension pot balance
- *"Tax-free pot £50,000"* — update your tax-free savings
- *"DB pension £8,000/year from age 66"* — add a defined-benefit pension
- *"House repairs £18,000 at age 70"* — add a one-off spending event
- *"Care costs £6,000/year from age 80"* — add an ongoing spending step
- *"Run it"* / *"Show me"* — run the simulation with charts
- *"Show me which pot drains first"* — pot breakdown charts
- *"What if markets crash early?"* — sequence-of-returns chart
- *"How worried should I be?"* — Monte Carlo probability fan chart
- *"Compare"* / *"What changed?"* — baseline vs scenario comparison
- *"Show my assumptions"* — summarise your current scenario setup
- *"Start over"* — reset everything to defaults

What would you like to explore?
"""

_DEFAULT_SPENDING = 30_000.0

ChartType = Literal[
    "baseline_vs_scenario",
    "sequence_of_returns",
    "monte_carlo",
    "pots_stacked",
    "pots_individual",
]


@dataclass
class ChatScenario:
    """Holds the current retirement scenario being explored in the chat.

    Attributes:
        start_age: Current age / simulation start age.
        end_age: Projection end age.
        tax_free_pot: Tax-free savings pot balance (ISAs, Premium Bonds, etc.).
        dc_pots: DC pension pots as ``(drawdown_start_age, balance)`` pairs.
        db_pensions: Defined-benefit pensions as ``(start_age, annual_amount)``.
        baseline_spending: Annual baseline spending in real terms.
        life_events: Accumulated life events (lump sums and spending steps).
        mean_return: Mean annual market return assumption.
        std_return: Annual return standard deviation.
        random_seed: RNG seed for reproducible simulations.
        num_simulations: Number of Monte Carlo paths.
    """

    start_age: int = START_AGE
    end_age: int = END_AGE
    tax_free_pot: float = INITIAL_TAX_FREE_POT
    dc_pots: list[tuple[int, float]] = field(
        default_factory=lambda: list(DC_POTS)
    )
    db_pensions: list[tuple[int, float]] = field(
        default_factory=lambda: list(DB_PENSIONS)
    )
    baseline_spending: float = _DEFAULT_SPENDING
    life_events: list[LumpSumEvent | SpendingStepEvent] = field(
        default_factory=list
    )
    mean_return: float = MEAN_RETURN
    std_return: float = STD_RETURN
    random_seed: int = RANDOM_SEED
    num_simulations: int = NUM_SIMULATIONS


@dataclass
class ParsedIntent:
    """Structured result of parsing one user message.

    Attributes:
        reply: Immediate text reply to display in the chat.
        updates: Dict of scenario attribute updates to apply.
        chart_types: Which chart types to render (empty means no charts).
        run_simulation: Whether to run and display simulation results.
        reset: Whether to reset the scenario to defaults.
        show_setup: Whether to display the current scenario summary.
    """

    reply: str
    updates: dict = field(default_factory=dict)
    chart_types: list[ChartType] = field(default_factory=list)
    run_simulation: bool = False
    reset: bool = False
    show_setup: bool = False


# ---------------------------------------------------------------------------
# Amount / age extraction helpers
# ---------------------------------------------------------------------------


def _parse_amount(text: str) -> float | None:
    """Extract a monetary amount from text.

    Handles £ prefix and k/K/m/M suffix abbreviations.

    Args:
        text: Raw text fragment potentially containing a monetary amount.

    Returns:
        Parsed float value, or ``None`` if no amount is found.
    """
    match = re.search(r"£\s*([\d,]+(?:\.\d+)?)\s*([kKmM])?", text)
    if not match:
        match = re.search(r"\b([\d,]+(?:\.\d+)?)\s*([kKmM])?\b", text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    value = float(raw)
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        value *= 1_000.0
    elif suffix == "m":
        value *= 1_000_000.0
    return value


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------


def _parse_intent(text: str, scenario: ChatScenario) -> ParsedIntent:  # noqa: C901
    """Parse a user message into a structured intent.

    Uses rule-based regex matching — no external LLM is required.

    Args:
        text: Raw user message.
        scenario: Current scenario state (used for context in replies).

    Returns:
        ParsedIntent describing what to do in response.
    """
    low = text.lower().strip()

    # ------------------------------------------------------------------ reset
    if re.search(r"\b(reset|start over|begin again|clear all|restart)\b", low):
        return ParsedIntent(
            reply=(
                "OK, I've reset everything back to the defaults. "
                "What would you like to explore?"
            ),
            reset=True,
        )

    # -------------------------------------------------------------- show setup
    if re.search(
        r"\b(show.*(assumption|setup|scenario|setting)|"
        r"what.*(assumption|setup|scenario|entered)|"
        r"current (scenario|setup|assumption)|"
        r"my (assumption|setup|scenario))\b",
        low,
    ):
        return ParsedIntent(reply="", show_setup=True)

    # ---------------------------------------------------- run / simulate (generic)
    _is_chart_request = bool(
        re.search(
            r"\b(which pot|markets? crash|sequence|timing|how worried|"
            r"risky|ruin|probabilit|compare|difference|impact|"
            r"composition|drains|pot breakdown|individual pot)\b",
            low,
        )
    )
    if not _is_chart_request and re.search(
        r"\b(run|simulate|show me|go|calculate|what does it look|"
        r"what('?s| is) it look|project|forecast)\b",
        low,
    ):
        return ParsedIntent(
            reply="Running your simulation…",
            run_simulation=True,
            chart_types=["baseline_vs_scenario", "monte_carlo"],
        )

    # ---------------------------------------------------------- specific chart requests
    if re.search(
        r"\b(markets? crash|sequence|timing risk|early (bad|crash|loss)|"
        r"order of returns?)\b",
        low,
    ):
        return ParsedIntent(
            reply="Showing the sequence-of-returns chart…",
            run_simulation=True,
            chart_types=["sequence_of_returns"],
        )

    if re.search(
        r"\b(which pot|pot breakdown|composition|drains? first|"
        r"pot.*(run out|empty|deplet)|individual pot)\b",
        low,
    ):
        return ParsedIntent(
            reply="Showing pot breakdown charts…",
            run_simulation=True,
            chart_types=["pots_stacked", "pots_individual"],
        )

    if re.search(
        r"\b(how worried|how risky|ruin|run out|go broke|"
        r"probabilit|chances|fan chart|monte carlo)\b",
        low,
    ):
        return ParsedIntent(
            reply="Showing the Monte Carlo probability chart…",
            run_simulation=True,
            chart_types=["monte_carlo"],
        )

    if re.search(r"\b(compare|difference|impact|what changed|baseline)\b", low):
        return ParsedIntent(
            reply="Showing the baseline vs scenario comparison…",
            run_simulation=True,
            chart_types=["baseline_vs_scenario"],
        )

    # ----------------------------------------- DB pension (before generic spending)
    db_match = re.search(
        r"(db|defined.?benefit|final salary)"
        r".*?£?\s*([\d,]+(?:\.\d+)?)\s*([kKmM])?"
        r".*?(?:from\s+age|age|at)\s+(\d+)",
        low,
    )
    if db_match:
        raw = db_match.group(2).replace(",", "")
        suffix = (db_match.group(3) or "").lower()
        amount = float(raw) * (1_000.0 if suffix == "k" else 1.0)
        age = int(db_match.group(4))
        return ParsedIntent(
            reply=(
                f"Added a DB pension of £{amount:,.0f}/year starting at age {age}. "
                "Type *'run it'* to see the updated projection."
            ),
            updates={"_add_db_pension": (age, amount)},
        )

    # "£8,000/year from age 66" with "pension" / "income" keyword
    db_match2 = re.search(
        r"£\s*([\d,]+(?:\.\d+)?)\s*([kKmM])?"
        r"\s*(?:per year|/year|a year)"
        r".*?(?:from\s+age|from)\s+(\d+)",
        low,
    )
    if db_match2 and re.search(r"\b(pension|income)\b", low):
        raw = db_match2.group(1).replace(",", "")
        suffix = (db_match2.group(2) or "").lower()
        amount = float(raw) * (1_000.0 if suffix == "k" else 1.0)
        age = int(db_match2.group(3))
        return ParsedIntent(
            reply=(
                f"Added a DB pension of £{amount:,.0f}/year starting at age {age}. "
                "Type *'run it'* to see the updated projection."
            ),
            updates={"_add_db_pension": (age, amount)},
        )

    # ---------------------------------------------- spending step (ongoing extra cost)
    step_match = re.search(
        r"£?\s*([\d,]+(?:\.\d+)?)\s*([kKmM])?"
        r"\s*(?:per year|/year|a year|yearly|annually)"
        r".*?(?:from age|from)\s+(\d+)"
        r"(?:.*?(?:to age|until age|to|until)\s+(\d+))?",
        low,
    )
    if step_match:
        raw = step_match.group(1).replace(",", "")
        suffix = (step_match.group(2) or "").lower()
        amount = float(raw) * (1_000.0 if suffix == "k" else 1.0)
        start = int(step_match.group(3))
        end = int(step_match.group(4)) if step_match.group(4) else None
        end_desc = f" to age {end}" if end else " onwards"
        return ParsedIntent(
            reply=(
                f"Added an ongoing extra cost of £{amount:,.0f}/year "
                f"from age {start}{end_desc}. "
                "Type *'run it'* to see the updated projection."
            ),
            updates={"_add_spending_step": (start, amount, end)},
        )

    # ------------------------------------------------ lump sum event (one-off cost)
    lump_match = re.search(
        r"£?\s*([\d,]+(?:\.\d+)?)\s*([kKmM])?"
        r".*?(?:at age|age|at)\s+(\d+)",
        low,
    )
    if lump_match and re.search(
        r"\b(lump|one.?off|repair|replac|car|holiday|gift|cost|expense|spend)\b",
        low,
    ):
        raw = lump_match.group(1).replace(",", "")
        suffix = (lump_match.group(2) or "").lower()
        amount = float(raw) * (1_000.0 if suffix == "k" else 1.0)
        age = int(lump_match.group(3))
        return ParsedIntent(
            reply=(
                f"Added a one-off cost of £{amount:,.0f} at age {age}. "
                "Type *'run it'* to see the updated projection."
            ),
            updates={"_add_lump_sum": (age, amount)},
        )

    # ---------------------------------------------------------- tax-free pot
    if re.search(r"\b(tax.?free|isa|premium bond)\b", low):
        amount = _parse_amount(text)
        if amount is not None:
            return ParsedIntent(
                reply=(
                    f"Set tax-free pot to £{amount:,.0f}. "
                    "Type *'run it'* to see the updated projection."
                ),
                updates={"tax_free_pot": amount},
            )

    # --------------------------------------------------------------- DC pot
    if re.search(r"\b(dc pot|dc pension|pension pot|sipp|workplace)\b", low):
        amount = _parse_amount(text)
        if amount is not None:
            return ParsedIntent(
                reply=(
                    f"Set primary DC pot to £{amount:,.0f}. "
                    "Type *'run it'* to see the updated projection."
                ),
                updates={"_update_primary_dc_pot": amount},
            )

    # ------------------------------------------------------------------ spending
    if re.search(
        r"\b(spend(ing)?|annual spend|yearly spend|drawdown amount|living costs?)\b",
        low,
    ):
        amount = _parse_amount(text)
        if amount is not None:
            return ParsedIntent(
                reply=(
                    f"Set baseline spending to £{amount:,.0f}/year. "
                    "Type *'run it'* to see the updated projection."
                ),
                updates={"baseline_spending": amount},
            )

    # ------------------------------------------------------- retire / drawdown age
    retire_match = re.search(
        r"\b(retire|retirement|drawdown|draw down|access)\b"
        r".*?\b(?:at|from|age)\b"
        r".*?\b(\d{2,3})\b",
        low,
    )
    if retire_match:
        age = int(retire_match.group(2))
        return ParsedIntent(
            reply=(
                f"Set primary DC pot drawdown start age to {age}. "
                "Type *'run it'* to see the updated projection."
            ),
            updates={"_update_primary_dc_drawdown_age": age},
        )

    # ------------------------------------------------------------- end age
    if re.search(
        r"\b(end age|run to|until age|to age|project to)\b"
        r".*?\b(\d{2,3})\b",
        low,
    ):
        nums = re.findall(r"\b(\d{2,3})\b", low)
        if nums:
            age = int(nums[-1])
            if 60 <= age <= 120:
                return ParsedIntent(
                    reply=(
                        f"Set projection end age to {age}. "
                        "Type *'run it'* to see the updated projection."
                    ),
                    updates={"end_age": age},
                )

    # ---------------------------------------------------------- current age / start age
    age_match = re.search(
        r"\b(?:i(?:'m| am|m)\s+(\d{2,3})|"
        r"age\s+(\d{2,3})|"
        r"current(?:ly)?\s+(\d{2,3})|"
        r"start\s+age\s+(\d{2,3}))\b",
        low,
    )
    if age_match:
        nums = re.findall(r"\b(\d{2,3})\b", age_match.group(0))
        if nums:
            age = int(nums[0])
            if 18 <= age <= 100:
                return ParsedIntent(
                    reply=(
                        f"Set current age to {age}. "
                        "Type *'run it'* to see the updated projection."
                    ),
                    updates={"start_age": age},
                )

    return ParsedIntent(
        reply=(
            "I didn't quite understand that. Try something like "
            "*\"I'm 55\"*, *\"DC pot £300,000\"*, "
            "*\"Care costs £6,000/year from age 80\"*, or *\"Run it\"*."
        )
    )


# ---------------------------------------------------------------------------
# Scenario state helpers
# ---------------------------------------------------------------------------


def _default_scenario() -> ChatScenario:
    """Create a ChatScenario populated with config defaults.

    Returns:
        Fresh ChatScenario instance.
    """
    return ChatScenario()


def _apply_updates(
    scenario: ChatScenario, updates: dict
) -> tuple[ChatScenario, str]:
    """Apply a dict of updates to a ChatScenario.

    Handles both direct attribute updates (plain keys) and special prefixed
    keys (``_add_lump_sum``, ``_add_spending_step``, etc.).

    Args:
        scenario: Current scenario.
        updates: Dict of updates from ``ParsedIntent.updates``.

    Returns:
        Tuple of ``(updated_scenario, error_message)``.  On success the error
        string is empty; on validation failure the scenario is unchanged.
    """
    if not updates:
        return scenario, ""

    dc_pots = list(scenario.dc_pots)
    db_pensions = list(scenario.db_pensions)
    life_events: list[LumpSumEvent | SpendingStepEvent] = list(scenario.life_events)
    kwargs: dict = {}

    for key, value in updates.items():
        if key == "_add_lump_sum":
            age, amount = value
            if age <= scenario.start_age or age > scenario.end_age:
                return scenario, (
                    f"Age {age} is outside the simulation range "
                    f"({scenario.start_age}–{scenario.end_age}). "
                    "Please adjust start or end age first."
                )
            life_events.append(LumpSumEvent(age=age, amount=amount))

        elif key == "_add_spending_step":
            start, amount, end = value
            if start <= scenario.start_age or start > scenario.end_age:
                return scenario, (
                    f"Start age {start} is outside the simulation range "
                    f"({scenario.start_age}–{scenario.end_age}). "
                    "Please adjust start or end age first."
                )
            if end is not None and (end < start or end > scenario.end_age):
                return scenario, (
                    f"End age {end} must be between {start} and "
                    f"{scenario.end_age}."
                )
            life_events.append(
                SpendingStepEvent(
                    start_age=start, extra_per_year=amount, end_age=end
                )
            )

        elif key == "_add_db_pension":
            age, amount = value
            db_pensions.append((age, amount))

        elif key == "_update_primary_dc_pot":
            if dc_pots:
                dc_pots[0] = (dc_pots[0][0], value)
            else:
                dc_pots = [(scenario.start_age, value)]

        elif key == "_update_primary_dc_drawdown_age":
            if dc_pots:
                dc_pots[0] = (value, dc_pots[0][1])
            else:
                dc_pots = [(value, 0.0)]

        elif not key.startswith("_"):
            kwargs[key] = value

    new_scenario = ChatScenario(
        start_age=kwargs.get("start_age", scenario.start_age),
        end_age=kwargs.get("end_age", scenario.end_age),
        tax_free_pot=kwargs.get("tax_free_pot", scenario.tax_free_pot),
        dc_pots=dc_pots,
        db_pensions=db_pensions,
        baseline_spending=kwargs.get("baseline_spending", scenario.baseline_spending),
        life_events=life_events,
        mean_return=kwargs.get("mean_return", scenario.mean_return),
        std_return=kwargs.get("std_return", scenario.std_return),
        random_seed=kwargs.get("random_seed", scenario.random_seed),
        num_simulations=kwargs.get("num_simulations", scenario.num_simulations),
    )

    if new_scenario.start_age >= new_scenario.end_age:
        return scenario, (
            f"Start age ({new_scenario.start_age}) must be less than "
            f"end age ({new_scenario.end_age})."
        )

    return new_scenario, ""


def _scenario_summary(scenario: ChatScenario) -> str:
    """Build a human-readable markdown summary of the current scenario.

    Args:
        scenario: Current scenario state.

    Returns:
        Formatted multi-line markdown string.
    """
    lines = [
        "**Current scenario setup:**\n",
        f"- **Current age:** {scenario.start_age}",
        f"- **End age:** {scenario.end_age}",
        f"- **Tax-free pot:** £{scenario.tax_free_pot:,.0f}",
        f"- **Baseline spending:** £{scenario.baseline_spending:,.0f}/year",
        f"- **Mean return:** {scenario.mean_return * 100:.1f}%",
        f"- **Return std dev:** {scenario.std_return * 100:.1f}%",
    ]

    if scenario.dc_pots:
        lines.append("- **DC pots:**")
        for i, (draw_age, balance) in enumerate(scenario.dc_pots, 1):
            lines.append(
                f"  - Pot {i}: £{balance:,.0f} — drawable from age {draw_age}"
            )
    else:
        lines.append("- **DC pots:** none set")

    if scenario.db_pensions:
        lines.append("- **DB pensions:**")
        for start, amount in scenario.db_pensions:
            lines.append(f"  - £{amount:,.0f}/year from age {start}")
    else:
        lines.append("- **DB pensions:** none")

    if scenario.life_events:
        lines.append("- **Life events:**")
        for event in scenario.life_events:
            if isinstance(event, LumpSumEvent):
                lines.append(
                    f"  - One-off: £{event.amount:,.0f} at age {event.age}"
                )
            else:
                end_desc = (
                    f" to age {event.end_age}" if event.end_age else " onwards"
                )
                lines.append(
                    f"  - Ongoing: £{event.extra_per_year:,.0f}/year "
                    f"from age {event.start_age}{end_desc}"
                )
    else:
        lines.append("- **Life events:** none added yet")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _build_db_income(
    ages: np.ndarray, db_pensions: list[tuple[int, float]]
) -> np.ndarray:
    """Build a DB income array aligned with ``ages``.

    Args:
        ages: Inclusive age array for the simulation horizon.
        db_pensions: DB pension streams as ``(start_age, annual_amount)`` tuples.

    Returns:
        Annual DB income values aligned with ``ages``.
    """
    return np.array(
        [calculate_db_pension_income(int(age), db_pensions) for age in ages],
        dtype=np.float64,
    )


def _scenario_engine_params(
    scenario: ChatScenario,
) -> tuple[float, float, int | None, list[tuple[int, float]] | None]:
    """Extract DC pot engine parameters from a scenario.

    Args:
        scenario: Current scenario state.

    Returns:
        Tuple of ``(dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
        extra_dc_pots)``.  ``extra_dc_pots`` is the full DC pots list when
        there are two or more pots (so individual drawdown ages are preserved),
        or ``None`` when there is only one pot.
    """
    dc_pot = scenario.dc_pots[0][1] if scenario.dc_pots else 0.0
    secondary_dc_pot = (
        sum(b for _, b in scenario.dc_pots[1:])
        if len(scenario.dc_pots) > 1
        else 0.0
    )
    secondary_dc_drawdown_age = (
        scenario.dc_pots[1][0] if len(scenario.dc_pots) > 1 else None
    )
    # Pass dc_pots when ≥ 2 pots to preserve individual drawdown ages.
    extra_dc_pots: list[tuple[int, float]] | None = (
        scenario.dc_pots if len(scenario.dc_pots) >= 2 else None
    )
    return dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, extra_dc_pots


def _build_simulation_results(
    scenario: ChatScenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run baseline and scenario simulations on the same return path.

    Args:
        scenario: Current scenario configuration.

    Returns:
        Tuple of ``(ages, baseline_balances, scenario_balances,
        baseline_withdrawals, scenario_withdrawals)``.
    """
    ages = np.arange(scenario.start_age, scenario.end_age + 1, dtype=np.int_)
    num_years = scenario.end_age - scenario.start_age
    db_income = _build_db_income(ages, scenario.db_pensions)
    returns = generate_random_returns(
        num_years, scenario.mean_return, scenario.std_return, scenario.random_seed
    )

    baseline_withdrawals = build_required_withdrawals(
        ages, scenario.baseline_spending, db_income, []
    )
    scenario_withdrawals = build_required_withdrawals(
        ages, scenario.baseline_spending, db_income, scenario.life_events
    )

    dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, extra_dc_pots = (
        _scenario_engine_params(scenario)
    )

    _, baseline_balances, *_ = simulate_multi_pot_pension_path(
        scenario.tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        scenario.db_pensions,
        scenario.start_age,
        scenario.end_age,
        returns,
        withdrawals_required=baseline_withdrawals,
        dc_pots=extra_dc_pots,
    )
    _, scenario_balances, *_ = simulate_multi_pot_pension_path(
        scenario.tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        scenario.db_pensions,
        scenario.start_age,
        scenario.end_age,
        returns,
        withdrawals_required=scenario_withdrawals,
        dc_pots=extra_dc_pots,
    )

    return (
        ages,
        baseline_balances,
        scenario_balances,
        baseline_withdrawals,
        scenario_withdrawals,
    )


def _build_explanation(scenario: ChatScenario) -> str:
    """Build a plain-English explanation of the current scenario impact.

    Runs both a deterministic comparison and a Monte Carlo simulation to
    produce the full narrative via ``build_plain_english_explanation``.

    Args:
        scenario: Current scenario configuration.

    Returns:
        Human-readable explanation string.
    """
    ages, baseline_balances, scenario_balances, _, scenario_withdrawals = (
        _build_simulation_results(scenario)
    )

    dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, extra_dc_pots = (
        _scenario_engine_params(scenario)
    )
    strategy_fn = create_fixed_real_drawdown_strategy(scenario.baseline_spending)

    _, mc_paths = run_monte_carlo_simulation(
        scenario.tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        scenario.db_pensions,
        scenario.start_age,
        scenario.end_age,
        scenario.mean_return,
        scenario.std_return,
        strategy_fn,
        scenario.num_simulations,
        scenario.random_seed,
        withdrawals_required=scenario_withdrawals,
        dc_pots=extra_dc_pots,
    )

    baseline_metrics = summarize_path(baseline_balances)
    scenario_metrics = summarize_path(scenario_balances)
    mc_metrics = summarize_monte_carlo(mc_paths)

    return build_plain_english_explanation(
        baseline_metrics,
        scenario_metrics,
        mc_metrics,
        scenario.life_events,
    )


def _build_figures(
    scenario: ChatScenario, chart_types: list[ChartType]
) -> list[Figure]:
    """Build matplotlib figures for the requested chart types.

    Args:
        scenario: Current scenario configuration.
        chart_types: Which chart types to render.

    Returns:
        List of matplotlib Figure objects ready for ``st.pyplot()``.

    Raises:
        ValueError: If the scenario age range is invalid.
    """
    ages = np.arange(scenario.start_age, scenario.end_age + 1, dtype=np.int_)
    db_income = _build_db_income(ages, scenario.db_pensions)
    spending_schedule = build_spending_drawdown_schedule(
        ages, scenario.baseline_spending, db_income, scenario.life_events
    )

    dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, extra_dc_pots = (
        _scenario_engine_params(scenario)
    )
    strategy_fn = create_fixed_real_drawdown_strategy(scenario.baseline_spending)
    scenario_withdrawals = build_required_withdrawals(
        ages, scenario.baseline_spending, db_income, scenario.life_events
    )

    figures: list[Figure] = []

    for chart_type in chart_types:
        fig: Figure | None = None

        if chart_type == "baseline_vs_scenario":
            ages_r, baseline_balances, scenario_balances, _, _ = (
                _build_simulation_results(scenario)
            )
            fig = plot_baseline_vs_scenario_balances(
                ages_r,
                baseline_balances,
                scenario_balances,
                spending_drawdown_schedule=spending_schedule,
                secondary_dc_drawdown_age=secondary_dc_drawdown_age,
                db_pensions=scenario.db_pensions,
                life_events=scenario.life_events,
                dc_pots=extra_dc_pots,
                save_output=False,
                return_figure=True,
            )

        elif chart_type == "sequence_of_returns":
            fig = plot_sequence_of_returns_scenarios(
                scenario.tax_free_pot,
                dc_pot,
                secondary_dc_pot,
                secondary_dc_drawdown_age,
                scenario.db_pensions,
                scenario.start_age,
                scenario.end_age,
                scenario.mean_return,
                scenario.std_return,
                strategy_fn,
                withdrawals_required=scenario_withdrawals,
                life_events=scenario.life_events,
                spending_drawdown_schedule=spending_schedule,
                dc_pots=extra_dc_pots,
                save_output=False,
                return_figure=True,
            )

        elif chart_type == "monte_carlo":
            fig = plot_monte_carlo_fan_chart(
                scenario.tax_free_pot,
                dc_pot,
                secondary_dc_pot,
                secondary_dc_drawdown_age,
                scenario.db_pensions,
                scenario.start_age,
                scenario.end_age,
                scenario.mean_return,
                scenario.std_return,
                strategy_fn,
                scenario.num_simulations,
                scenario.random_seed,
                withdrawals_required=scenario_withdrawals,
                life_events=scenario.life_events,
                spending_drawdown_schedule=spending_schedule,
                dc_pots=extra_dc_pots,
                save_output=False,
                return_figure=True,
            )

        elif chart_type == "pots_stacked":
            fig = plot_pots_stacked_area(
                scenario.tax_free_pot,
                dc_pot,
                secondary_dc_pot,
                secondary_dc_drawdown_age,
                scenario.db_pensions,
                scenario.start_age,
                scenario.end_age,
                scenario.mean_return,
                scenario.std_return,
                strategy_fn,
                scenario.random_seed,
                withdrawals_required=scenario_withdrawals,
                life_events=scenario.life_events,
                spending_drawdown_schedule=spending_schedule,
                dc_pots=extra_dc_pots,
                save_output=False,
                return_figure=True,
            )

        elif chart_type == "pots_individual":
            fig = plot_individual_pots_subplots(
                scenario.tax_free_pot,
                dc_pot,
                secondary_dc_pot,
                secondary_dc_drawdown_age,
                scenario.db_pensions,
                scenario.start_age,
                scenario.end_age,
                scenario.mean_return,
                scenario.std_return,
                strategy_fn,
                scenario.random_seed,
                withdrawals_required=scenario_withdrawals,
                life_events=scenario.life_events,
                dc_pots=extra_dc_pots,
                save_output=False,
                return_figure=True,
            )

        if fig is not None:
            figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialise Streamlit session state with defaults on first load."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "scenario" not in st.session_state:
        st.session_state.scenario = _default_scenario()
    if "welcomed" not in st.session_state:
        st.session_state.welcomed = False


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------


def _store_message(
    role: str, content: str, figures: list[Figure] | None = None
) -> None:
    """Append a message to the persistent conversation history.

    Args:
        role: ``"user"`` or ``"assistant"``.
        content: Message text (markdown supported).
        figures: Optional matplotlib figures to associate with this message.
    """
    st.session_state.chat_messages.append(
        {"role": role, "content": content, "figures": figures or []}
    )


def _render_message(
    role: str, content: str, figures: list[Figure]
) -> None:
    """Render one chat message with optional inline charts.

    Args:
        role: ``"user"`` or ``"assistant"``.
        content: Message text (markdown supported).
        figures: Matplotlib figures to display below the text.
    """
    with st.chat_message(role):
        if content:
            st.markdown(content)
        for fig in figures:
            st.pyplot(fig)


def _replay_history() -> None:
    """Re-render the full conversation history from session state."""
    for message in st.session_state.chat_messages:
        _render_message(
            message["role"],
            message["content"],
            message.get("figures", []),
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the IFA chat-driven Streamlit app."""
    st.set_page_config(
        page_title="IFA Pension Chat",
        page_icon="💬",
        layout="wide",
    )
    st.title("💬 Pension Scenario Explorer")
    st.caption(
        "Type questions to explore your retirement finances. "
        "Try *'start over'* to reset or *'show my assumptions'* for a summary."
    )

    _init_session_state()

    if not st.session_state.welcomed:
        _store_message("assistant", _WELCOME_MESSAGE)
        st.session_state.welcomed = True

    _replay_history()

    user_input = st.chat_input("Ask a question or describe a scenario…")
    if not user_input:
        return

    _store_message("user", user_input)
    _render_message("user", user_input, [])

    scenario: ChatScenario = st.session_state.scenario
    intent = _parse_intent(user_input, scenario)

    # ------------------------------------------------------------------ reset
    if intent.reset:
        st.session_state.scenario = _default_scenario()
        _store_message("assistant", intent.reply)
        _render_message("assistant", intent.reply, [])
        return

    # ------------------------------------------------------------ show setup
    if intent.show_setup:
        summary = _scenario_summary(scenario)
        _store_message("assistant", summary)
        _render_message("assistant", summary, [])
        return

    # ---------------------------------------------------- apply state updates
    if intent.updates:
        updated_scenario, error = _apply_updates(scenario, intent.updates)
        if error:
            err_msg = f"⚠️ {error}"
            _store_message("assistant", err_msg)
            _render_message("assistant", err_msg, [])
            return
        st.session_state.scenario = updated_scenario
        scenario = updated_scenario

    # ---------------------------------------------- text-only response (no sim)
    if not intent.run_simulation:
        if intent.reply:
            _store_message("assistant", intent.reply)
            _render_message("assistant", intent.reply, [])
        return

    # ---------------------------------------------------- render spinner reply
    if intent.reply:
        _store_message("assistant", intent.reply)
        _render_message("assistant", intent.reply, [])

    # ------------------------------------------------------ run simulation
    with st.spinner("Running simulation…"):
        try:
            figures = _build_figures(scenario, intent.chart_types)
            explanation = _build_explanation(scenario)
        except ValueError as exc:
            err_msg = f"⚠️ Simulation error: {exc}"
            _store_message("assistant", err_msg)
            _render_message("assistant", err_msg, [])
            return

    _store_message("assistant", explanation, figures)
    with st.chat_message("assistant"):
        for fig in figures:
            st.pyplot(fig)
        st.markdown(explanation)


if __name__ == "__main__":
    main()
