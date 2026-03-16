"""Streamlit chat-based pension drawdown simulator.

Provides a conversational "what if" interface on top of the ifa simulation
engine.  Run with::

    streamlit run ifa_chat.py
"""

from __future__ import annotations

import re
from contextlib import suppress
from typing import Any

import numpy as np
import streamlit as st

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
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.events import build_required_withdrawals, build_spending_drawdown_schedule
from ifa.explain import build_plain_english_explanation
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

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_SPENDING: float = 30_000.0

_CHART_COMPARISON = "comparison"
_CHART_FAN = "fan"
_CHART_SEQUENCE = "sequence"
_CHART_STACKED = "stacked"
_CHART_INDIVIDUAL = "individual"

_WELCOME_TEXT = """\
**Welcome to the IFA Pension Drawdown Chat**

I help you explore "what if" retirement finance questions through conversation.
Describe your situation and I'll run simulations and show you the charts.

**Try asking things like:**
- *"I'm 55 with a £300k DC pot and £50k tax-free. DB pension £8k/year from 66."*
- *"I spend about £22,000 a year."*
- *"Run simulation"*
- *"What if I need £18,000 for a new roof at age 70?"*
- *"What if care costs start at £6,000/year from age 80?"*
- *"Which pot runs out first?"*
- *"What happens if markets crash early?"*
- *"What if I retire at 60 instead?"*

Or type **"run it"** to explore the default scenario right away. \
Type **"help"** for a full list of things I understand.
"""

_HELP_TEXT = """\
**Things I understand:**

**Setting up your situation:**
- *"I'm [age]"* or *"start age [age]"* — set starting age
- *"DC pot £[amount]"* or *"£[amount] DC pot"* — set primary DC pot balance
- *"Tax-free pot £[amount]"* — set tax-free pot balance
- *"DB pension £[amount]/year from age [age]"* — add (or replace) a DB pension
- *"I spend £[amount]/year"* or *"baseline spending £[amount]"* — set baseline
- *"Retire at [age]"* — update start age
- *"End age [age]"* or *"plan to age [age]"* — set end age

**Adding life events (these re-run the simulation automatically):**
- *"What if I need £[amount] at age [age]?"* — one-off lump sum cost
- *"What if care costs £[amount]/year from age [age]?"* — ongoing spending step
- *"What if costs £[amount]/year from [age] to [age]?"* — bounded spending step

**Running and viewing:**
- *"Run"* / *"Go"* / *"Simulate"* / *"Show results"* — run and show main charts
- *"Which pot runs out first?"* / *"Show pots"* — individual pot breakdown
- *"What about risk?"* / *"Monte Carlo"* — fan chart with ruin probability
- *"Sequence of returns"* / *"Market crash"* — sequence-of-returns teaching chart
- *"Show everything"* / *"All charts"* — show all 5 charts
- *"Baseline vs scenario"* — comparison chart only

**Market settings:**
- *"Mean return 5%"* or *"5% returns"* — adjust mean real return
- *"Volatility 12%"* — adjust return volatility
- *"1000 simulations"* — set Monte Carlo path count

**Other:**
- *"Reset"* / *"Start over"* — reset everything to defaults
- *"Help"* — show this message
"""

# ── Amount / age parsing helpers ──────────────────────────────────────────────

# Regex fragment that matches an optional £, digits, commas, optional decimal,
# and an optional k/m multiplier.
_AMT = r"£?([\d,]+(?:\.\d+)?)\s*([km]?)\b"

# Minimum amount (£) for a lump sum event — filters out age-like numbers.
_MIN_LUMP_AMOUNT: float = 500.0

# Characters to look back before a spend-event match to check for pension
# context keywords that would indicate the amount is a DB pension, not a cost.
_PENSION_CONTEXT_LOOKBACK: int = 40


def _parse_amount(digits: str, suffix: str) -> float:
    """Convert raw digit string and optional k/m suffix to a float.

    Args:
        digits: Digit string, possibly with commas (e.g. ``"300,000"``).
        suffix: One of ``""`` / ``"k"`` / ``"K"`` / ``"m"`` / ``"M"``.

    Returns:
        Numeric value as float.
    """
    val = float(digits.replace(",", ""))
    mul = {"k": 1_000.0, "m": 1_000_000.0}.get(suffix.lower(), 1.0)
    return val * mul


def _find_amount(text: str) -> float | None:
    """Return the first currency-style amount found in *text*.

    Args:
        text: Source text.

    Returns:
        First amount as float, or ``None`` if none found.
    """
    m = re.search(_AMT, text, re.IGNORECASE)
    if not m:
        return None
    try:
        return _parse_amount(m.group(1), m.group(2))
    except ValueError:
        return None


def _find_amounts(text: str) -> list[float]:
    """Return all currency-style amounts found in *text*.

    Args:
        text: Source text.

    Returns:
        List of amounts in order of appearance.
    """
    results: list[float] = []
    for m in re.finditer(_AMT, text, re.IGNORECASE):
        with suppress(ValueError):
            results.append(_parse_amount(m.group(1), m.group(2)))
    return results


# ── Parameter extraction from free text ───────────────────────────────────────


def _extract_start_age(text: str) -> int | None:
    """Extract a starting / retirement age from *text*.

    Args:
        text: User message text.

    Returns:
        Start age as int, or ``None`` if not detected.
    """
    patterns = [
        r"i(?:'m|'m| am)\s+(\d{2})\b",
        r"i(?:'m|'m| am)\s+currently\s+(\d{2})\b",
        r"start(?:ing)?\s+age\s+(\d{2,3})\b",
        r"start(?:ing)?\s+at\s+(\d{2,3})\b",
        r"retire(?:\s+at)?\s+(?:age\s+)?(\d{2,3})\b",
        r"retirement\s+at\s+(?:age\s+)?(\d{2,3})\b",
        r"from\s+age\s+(\d{2,3})\s+(?:onwards?|to\s+\d)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 40 <= age <= 85:
                return age
    return None


def _extract_end_age(text: str) -> int | None:
    """Extract an end / planning age from *text*.

    Args:
        text: User message text.

    Returns:
        End age as int, or ``None`` if not detected.
    """
    patterns = [
        r"end\s+age\s+(\d{2,3})\b",
        r"(?:plan|simulate|model)\s+to\s+age\s+(\d{2,3})\b",
        r"until\s+(?:age\s+)?(\d{2,3})\b",
        r"to\s+age\s+(\d{2,3})\b",
        r"age\s+(\d{2,3})\s+(?:model|plan)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 60 <= age <= 110:
                return age
    return None


def _extract_tax_free_pot(text: str) -> float | None:
    """Extract tax-free pot balance from *text*.

    Args:
        text: User message text.

    Returns:
        Balance as float, or ``None`` if not detected.
    """
    # "£50k tax-free [pot]"
    m = re.search(
        _AMT + r"\s+tax[\s\-]?free(?:\s+pot)?",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    # "tax-free pot [of] £50k"
    m = re.search(
        r"tax[\s\-]?free\s+(?:pot\s+)?(?:of\s+)?" + _AMT,
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    return None


def _extract_dc_pot(text: str) -> float | None:
    """Extract primary DC pot balance from *text*.

    Args:
        text: User message text.

    Returns:
        Balance as float, or ``None`` if not detected.
    """
    # "£300k DC pot" / "£300k pension pot"
    m = re.search(
        _AMT + r"\s+(?:dc|pension)\s+pot",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    # "DC pot [of] £300k" / "pension pot of £300k"
    m = re.search(
        r"(?:dc|pension)\s+pot\s+(?:of\s+)?" + _AMT,
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    # "invested pot of £300k" / "£300k in a DC pot"
    m = re.search(
        _AMT + r"\s+in\s+(?:a\s+)?(?:dc|pension)\s+pot",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    return None


def _extract_baseline_spending(text: str) -> float | None:
    """Extract baseline annual spending from *text*.

    Args:
        text: User message text.

    Returns:
        Annual spending as float, or ``None`` if not detected.
    """
    # "I spend [about] £22,000 [a year / per year / /year]"
    m = re.search(
        r"(?:i\s+)?spend\s+(?:about\s+|roughly\s+|around\s+)?" + _AMT
        + r"(?:\s+(?:a|per)\s+year|\s*\/year)?",
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    # "baseline [annual] spending [of] £22k"
    m = re.search(
        r"baseline\s+(?:annual\s+)?spending\s+(?:of\s+)?" + _AMT,
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    # "spending of £22k" / "budget £22k"
    m = re.search(
        r"(?:annual\s+)?(?:spending|budget)\s+(?:of\s+)?" + _AMT,
        text,
        re.IGNORECASE,
    )
    if m:
        try:
            return _parse_amount(m.group(1), m.group(2))
        except ValueError:
            pass
    return None


def _extract_db_pensions(text: str) -> list[tuple[int, float]]:
    """Extract DB pension streams (start_age, annual_amount) from *text*.

    Args:
        text: User message text.

    Returns:
        List of ``(start_age, annual_amount)`` tuples.
    """
    results: list[tuple[int, float]] = []
    # Match "DB pension [is/of] £8k/year from [age] 66" and variants.
    # Allow arbitrary words (is, was, of, pays, =, etc.) between keyword and amount.
    pension_phrases = [
        r"(?:db|defined[\s\-]benefit|final[\s\-]salary)\s+pension",
        r"pension\s+income",
        r"pension\s+of",
    ]
    for phrase in pension_phrases:
        for m in re.finditer(
            phrase
            + r"(?:\s+\w+){0,3}?\s*"  # allow up to 3 words (e.g. "is", "of")
            + _AMT
            + r"(?:\s*(?:/year|a\s+year|per\s+year|annually|p\.?a\.?))?"
            + r"\s+(?:from\s+)?(?:age\s+)?(\d{2,3})\b",
            text,
            re.IGNORECASE,
        ):
            try:
                amount = _parse_amount(m.group(1), m.group(2))
                age = int(m.group(3))
                if 50 <= age <= 90 and amount > 0:
                    results.append((age, amount))
            except (ValueError, IndexError):
                pass

    return results


def _extract_lump_events(
    text: str, start_age: int, end_age: int
) -> list[LumpSumEvent]:
    """Extract one-off lump sum events from *text*.

    Args:
        text: User message text.
        start_age: Scenario start age (for range validation).
        end_age: Scenario end age (for range validation).

    Returns:
        List of :class:`~ifa.models.LumpSumEvent` objects.
    """
    results: list[LumpSumEvent] = []

    # "need/cost/pay/spend £18,000 [for X] at [age] 70"
    for m in re.finditer(
        r"(?:need|cost|pay|spend)\s+(?:a\s+)?"
        + _AMT
        + r"(?:\s+for\s+[^.?\n,]+?)?"
        + r"\s+at\s+(?:age\s+)?(\d{2,3})\b",
        text,
        re.IGNORECASE,
    ):
        try:
            amount = _parse_amount(m.group(1), m.group(2))
            age = int(m.group(3))
            if start_age <= age <= end_age and amount > 0:
                results.append(LumpSumEvent(age=age, amount=amount))
        except (ValueError, IndexError):
            pass

    # "£18,000 [for X] at [age] 70" (without need/cost/pay prefix)
    for m in re.finditer(
        _AMT
        + r"(?:\s+for\s+[^.?\n,]+?)?"
        + r"\s+at\s+(?:age\s+)?(\d{2,3})\b",
        text,
        re.IGNORECASE,
    ):
        try:
            amount = _parse_amount(m.group(1), m.group(2))
            age = int(m.group(3))
            if start_age <= age <= end_age and amount > _MIN_LUMP_AMOUNT:
                event = LumpSumEvent(age=age, amount=amount)
                if event not in results:
                    results.append(event)
        except (ValueError, IndexError):
            pass

    return results


def _extract_spend_events(
    text: str, start_age: int, end_age: int
) -> list[SpendingStepEvent]:
    """Extract ongoing spending step events from *text*.

    Args:
        text: User message text.
        start_age: Scenario start age (for range validation).
        end_age: Scenario end age (for range validation).

    Returns:
        List of :class:`~ifa.models.SpendingStepEvent` objects.
    """
    results: list[SpendingStepEvent] = []
    per_year_suffix = r"(?:\s*(?:/year|a\s+year|per\s+year|annually|p\.?a\.?))?"
    # Keywords that indicate the amount is a DB pension, not a spending step.
    _pension_ctx = re.compile(
        r"(?:db|defined[\s\-]benefit|final[\s\-]salary|pension)\b",
        re.IGNORECASE,
    )

    # "£6,000/year from [age] 80 [to [age] 90]"
    for m in re.finditer(
        _AMT
        + per_year_suffix
        + r"\s+from\s+(?:age\s+)?(\d{2,3})\b"
        + r"(?:\s+to\s+(?:age\s+)?(\d{2,3})\b)?",
        text,
        re.IGNORECASE,
    ):
        # Skip if within 40 chars before the match there's a pension keyword.
        preceding = text[max(0, m.start() - _PENSION_CONTEXT_LOOKBACK) : m.start()]
        if _pension_ctx.search(preceding):
            continue
        try:
            amount = _parse_amount(m.group(1), m.group(2))
            s_age = int(m.group(3))
            e_age = int(m.group(4)) if m.group(4) else None
            if start_age <= s_age <= end_age and amount > 0:
                if e_age is not None and not (s_age <= e_age <= end_age):
                    e_age = None
                results.append(
                    SpendingStepEvent(
                        start_age=s_age,
                        extra_per_year=amount,
                        end_age=e_age,
                    )
                )
        except (ValueError, IndexError):
            pass

    return results


def _extract_market_params(text: str) -> dict[str, Any]:
    """Extract market settings from *text*.

    Args:
        text: User message text.

    Returns:
        Dict with any of: ``mean_return``, ``std_return``, ``num_simulations``.
    """
    updates: dict[str, Any] = {}

    # "mean return [of] X%" or "X% [mean] return"
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s+(?:mean\s+)?returns?", text, re.IGNORECASE)
    if not m:
        m = re.search(
            r"(?:mean\s+)?returns?\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%",
            text,
            re.IGNORECASE,
        )
    if m:
        updates["mean_return"] = float(m.group(1)) / 100.0

    # "volatility [of] X%"
    m = re.search(
        r"volatility\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%\s+volatility", text, re.IGNORECASE)
    if m:
        updates["std_return"] = float(m.group(1)) / 100.0

    # "X simulations" / "X paths"
    m = re.search(r"(\d{3,5})\s+(?:simulations?|paths?|runs?)\b", text, re.IGNORECASE)
    if m:
        updates["num_simulations"] = int(m.group(1))

    return updates


# ── Intent detection ───────────────────────────────────────────────────────────


def _any(text: str, *patterns: str) -> bool:
    """Return True if any of the regex patterns match *text* case-insensitively.

    Args:
        text: Source text.
        *patterns: Regex patterns to test.

    Returns:
        True if at least one pattern matches.
    """
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _is_reset(text: str) -> bool:
    """Detect reset / start-over intent."""
    return _any(
        text, r"\breset\b", r"\bstart over\b", r"\bclear all\b", r"\bbegin again\b"
    )


def _is_help(text: str) -> bool:
    """Detect help intent."""
    return _any(text, r"\bhelp\b", r"\bwhat can you\b", r"\bwhat do you understand\b")


def _is_run(text: str) -> bool:
    """Detect 'run simulation' intent."""
    return _any(
        text,
        r"\brun\b",
        r"\bgo\b",
        r"\bsimulat",
        r"\bcalculat",
        r"\bshow results\b",
        r"\bshow me\b",
        r"\bshow the results\b",
        r"\bresults please\b",
    )


def _is_show_all(text: str) -> bool:
    """Detect 'show all charts' intent."""
    return _any(
        text,
        r"\bshow (?:all|every)",
        r"\ball charts\b",
        r"\beverything\b",
        r"\ball views\b",
    )


def _is_show_pots(text: str) -> bool:
    """Detect 'show pot breakdown' intent."""
    return _any(
        text,
        r"\bwhich pot\b",
        r"\bpot (?:runs out|breakdown|chart|view)\b",
        r"\bshow (?:the )?pots\b",
        r"\bindividual pots\b",
        r"\bpots first\b",
        r"\bpot composition\b",
    )


def _is_show_risk(text: str) -> bool:
    """Detect 'show Monte Carlo / risk' intent."""
    return _any(
        text,
        r"\bmonte carlo\b",
        r"\brisk\b",
        r"\bfan chart\b",
        r"\bprobability\b",
        r"\bworried\b",
        r"\bruin\b",
        r"\buncertain",
    )


def _is_show_sequence(text: str) -> bool:
    """Detect 'show sequence-of-returns' intent."""
    return _any(
        text,
        r"\bsequence of returns\b",
        r"\bsequence.returns\b",
        r"\bmarket crash\b",
        r"\nearly crash\b",
        r"\bcrash early\b",
        r"\bbad early\b",
        r"\bsequence risk\b",
        r"\bcrashe?s? early\b",
    )


def _is_show_comparison(text: str) -> bool:
    """Detect 'show baseline vs scenario comparison' intent."""
    return _any(
        text,
        r"\bbaseline vs\b",
        r"\bcompar",
        r"\bbaseline chart\b",
        r"\bscenario chart\b",
    )


# ── Main message parser ────────────────────────────────────────────────────────


def _parse_message(
    text: str, start_age: int, end_age: int
) -> dict[str, Any]:
    """Parse a user message and return a structured intent dict.

    Extracts parameter updates, life events to add, which charts to show,
    and whether to auto-run the simulation.

    Args:
        text: Raw user message.
        start_age: Current scenario start age (used for range validation).
        end_age: Current scenario end age (used for range validation).

    Returns:
        Dict with keys:

        - ``"action"`` — primary intent string
        - ``"updates"`` — parameter changes to apply
        - ``"lump_events"`` — list of :class:`~ifa.models.LumpSumEvent`
        - ``"spend_events"`` — list of :class:`~ifa.models.SpendingStepEvent`
        - ``"charts"`` — which chart types to render
        - ``"auto_run"`` — whether to run the simulation before responding
    """
    t = text.strip()

    # ── Hard overrides first ──────────────────────────────────────────────
    if _is_reset(t):
        return {
            "action": "reset",
            "updates": {},
            "lump_events": [],
            "spend_events": [],
            "charts": [],
            "auto_run": False,
        }

    if _is_help(t):
        return {
            "action": "help",
            "updates": {},
            "lump_events": [],
            "spend_events": [],
            "charts": [],
            "auto_run": False,
        }

    # ── Extract parameter updates from the whole message ─────────────────
    updates: dict[str, Any] = {}
    s_age = _extract_start_age(t)
    if s_age is not None:
        updates["start_age"] = s_age

    e_age = _extract_end_age(t)
    if e_age is not None:
        updates["end_age"] = e_age

    tfp = _extract_tax_free_pot(t)
    if tfp is not None:
        updates["tax_free_pot"] = tfp

    dc = _extract_dc_pot(t)
    if dc is not None:
        updates["dc_pot_balance"] = dc

    spending = _extract_baseline_spending(t)
    if spending is not None:
        updates["baseline_spending"] = spending

    new_db_pensions = _extract_db_pensions(t)
    if new_db_pensions:
        updates["db_pensions_add"] = new_db_pensions

    market = _extract_market_params(t)
    updates.update(market)

    # Use effective start/end ages (possibly just updated) for event validation
    eff_start = updates.get("start_age", start_age)
    eff_end = updates.get("end_age", end_age)

    lump_events = _extract_lump_events(t, eff_start, eff_end)
    spend_events = _extract_spend_events(t, eff_start, eff_end)

    has_events = bool(lump_events or spend_events)
    has_params = bool(updates)

    # ── Determine primary intent from keywords ────────────────────────────
    if _is_show_all(t):
        return {
            "action": "show_all",
            "updates": updates,
            "lump_events": lump_events,
            "spend_events": spend_events,
            "charts": [
                _CHART_COMPARISON,
                _CHART_FAN,
                _CHART_SEQUENCE,
                _CHART_STACKED,
                _CHART_INDIVIDUAL,
            ],
            "auto_run": True,
        }

    if _is_show_pots(t):
        return {
            "action": "show_pots",
            "updates": updates,
            "lump_events": lump_events,
            "spend_events": spend_events,
            "charts": [_CHART_STACKED, _CHART_INDIVIDUAL],
            "auto_run": True,
        }

    if _is_show_sequence(t):
        return {
            "action": "show_sequence",
            "updates": updates,
            "lump_events": lump_events,
            "spend_events": spend_events,
            "charts": [_CHART_SEQUENCE],
            "auto_run": True,
        }

    if _is_show_risk(t):
        return {
            "action": "show_risk",
            "updates": updates,
            "lump_events": lump_events,
            "spend_events": spend_events,
            "charts": [_CHART_FAN],
            "auto_run": True,
        }

    if _is_show_comparison(t):
        return {
            "action": "show_comparison",
            "updates": updates,
            "lump_events": lump_events,
            "spend_events": spend_events,
            "charts": [_CHART_COMPARISON],
            "auto_run": True,
        }

    if _is_run(t):
        return {
            "action": "run",
            "updates": updates,
            "lump_events": lump_events,
            "spend_events": spend_events,
            "charts": [_CHART_COMPARISON, _CHART_FAN, _CHART_SEQUENCE],
            "auto_run": True,
        }

    # ── Life events with implicit "what if" run ───────────────────────────
    if has_events:
        return {
            "action": "add_events",
            "updates": updates,
            "lump_events": lump_events,
            "spend_events": spend_events,
            "charts": [_CHART_COMPARISON, _CHART_FAN],
            "auto_run": True,
        }

    # ── Pure parameter updates, no run ───────────────────────────────────
    if has_params:
        return {
            "action": "set_params",
            "updates": updates,
            "lump_events": [],
            "spend_events": [],
            "charts": [],
            "auto_run": False,
        }

    return {
        "action": "unknown",
        "updates": {},
        "lump_events": [],
        "spend_events": [],
        "charts": [],
        "auto_run": False,
    }


# ── Session state ──────────────────────────────────────────────────────────────


def _default_state() -> dict[str, Any]:
    """Return the default initial session state values.

    Returns:
        Dict of default parameter values.
    """
    return {
        "start_age": START_AGE,
        "end_age": END_AGE,
        "tax_free_pot": float(INITIAL_TAX_FREE_POT),
        "baseline_spending": _DEFAULT_SPENDING,
        "dc_pots": list(DC_POTS),
        "dc_pot_names": [f"DC Pot {i + 1}" for i in range(len(DC_POTS))],
        "db_pensions": list(DB_PENSIONS),
        "db_pension_names": [f"DB Pension {i + 1}" for i in range(len(DB_PENSIONS))],
        "life_events": [],
        "life_event_names": [],
        "mean_return": MEAN_RETURN,
        "std_return": STD_RETURN,
        "random_seed": RANDOM_SEED,
        "num_simulations": NUM_SIMULATIONS,
        "sim_run": False,
        "sim_version": 0,
        "sim_cache": None,
    }


def _init_session_state() -> None:
    """Initialize session state on first run.

    Sets all scenario parameters, chat history, and simulation cache to
    their defaults if they have not yet been set.
    """
    if "chat_initialized" in st.session_state:
        return
    defaults = _default_state()
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state["messages"] = []
    st.session_state["chat_initialized"] = True


def _reset_state() -> None:
    """Reset scenario parameters and simulation cache to defaults.

    Preserves the existing message history so the conversation is not lost.
    """
    defaults = _default_state()
    for key, value in defaults.items():
        st.session_state[key] = value


def _apply_updates(updates: dict[str, Any]) -> list[str]:
    """Apply extracted parameter updates to session state.

    Args:
        updates: Dict of parameter changes as produced by :func:`_parse_message`.

    Returns:
        List of human-readable confirmation strings for each applied change.
    """
    confirmations: list[str] = []

    if "start_age" in updates:
        old = st.session_state["start_age"]
        new = updates["start_age"]
        st.session_state["start_age"] = new
        confirmations.append(f"Start age: {old} → **{new}**")
        # Invalidate sim
        st.session_state["sim_run"] = False

    if "end_age" in updates:
        old = st.session_state["end_age"]
        new = updates["end_age"]
        st.session_state["end_age"] = new
        confirmations.append(f"End age: {old} → **{new}**")
        st.session_state["sim_run"] = False

    if "tax_free_pot" in updates:
        new = updates["tax_free_pot"]
        st.session_state["tax_free_pot"] = new
        confirmations.append(f"Tax-free pot: **£{new:,.0f}**")
        st.session_state["sim_run"] = False

    if "dc_pot_balance" in updates:
        new_bal = updates["dc_pot_balance"]
        pots = list(st.session_state["dc_pots"])
        if pots:
            pots[0] = (pots[0][0], new_bal)
        else:
            pots = [(57, new_bal)]
        st.session_state["dc_pots"] = pots
        confirmations.append(f"Primary DC pot balance: **£{new_bal:,.0f}**")
        st.session_state["sim_run"] = False

    if "baseline_spending" in updates:
        new = updates["baseline_spending"]
        st.session_state["baseline_spending"] = new
        confirmations.append(f"Baseline annual spending: **£{new:,.0f}**")
        st.session_state["sim_run"] = False

    if "db_pensions_add" in updates:
        new_pensions: list[tuple[int, float]] = updates["db_pensions_add"]
        existing: list[tuple[int, float]] = list(st.session_state["db_pensions"])
        names: list[str] = list(st.session_state["db_pension_names"])
        for age, amount in new_pensions:
            existing.append((age, amount))
            names.append(f"DB Pension {len(existing)}")
            confirmations.append(
                f"Added DB pension: **£{amount:,.0f}/year from age {age}**"
            )
        st.session_state["db_pensions"] = existing
        st.session_state["db_pension_names"] = names
        st.session_state["sim_run"] = False

    if "mean_return" in updates:
        new = updates["mean_return"]
        st.session_state["mean_return"] = new
        confirmations.append(f"Mean real return: **{new * 100:.1f}%**")
        st.session_state["sim_run"] = False

    if "std_return" in updates:
        new = updates["std_return"]
        st.session_state["std_return"] = new
        confirmations.append(f"Return volatility: **{new * 100:.1f}%**")
        st.session_state["sim_run"] = False

    if "num_simulations" in updates:
        new = updates["num_simulations"]
        st.session_state["num_simulations"] = new
        confirmations.append(f"Monte Carlo simulations: **{new:,}**")
        st.session_state["sim_run"] = False

    return confirmations


def _apply_events(
    lump_events: list[LumpSumEvent],
    spend_events: list[SpendingStepEvent],
) -> list[str]:
    """Append new life events to session state.

    Args:
        lump_events: One-off lump sum events to add.
        spend_events: Ongoing spending step events to add.

    Returns:
        List of human-readable confirmation strings.
    """
    confirmations: list[str] = []
    existing: list[LumpSumEvent | SpendingStepEvent] = list(
        st.session_state["life_events"]
    )
    names: list[str] = list(st.session_state["life_event_names"])

    for ev in lump_events:
        existing.append(ev)
        names.append(f"Lump Sum {len(existing)}")
        confirmations.append(
            f"Added one-off cost: **£{ev.amount:,.0f} at age {ev.age}**"
        )

    for ev in spend_events:
        existing.append(ev)
        names.append(f"Spending Step {len(existing)}")
        if ev.end_age is not None:
            confirmations.append(
                f"Added ongoing cost: **£{ev.extra_per_year:,.0f}/year from age "
                f"{ev.start_age} to {ev.end_age}**"
            )
        else:
            confirmations.append(
                f"Added ongoing cost: **£{ev.extra_per_year:,.0f}/year from age "
                f"{ev.start_age}**"
            )

    if confirmations:
        st.session_state["life_events"] = existing
        st.session_state["life_event_names"] = names
        st.session_state["sim_run"] = False

    return confirmations


# ── Simulation runner ──────────────────────────────────────────────────────────


def _run_simulation() -> dict[str, Any] | None:
    """Run the simulation with current session state parameters.

    Caches results in ``st.session_state["sim_cache"]`` and bumps
    ``sim_version``.  Returns cached results immediately if parameters have
    not changed since the last run.

    Returns:
        Simulation result dict, or ``None`` on error.
    """
    if st.session_state.get("sim_run"):
        return st.session_state.get("sim_cache")

    s = st.session_state
    start_age: int = s["start_age"]
    end_age: int = s["end_age"]
    tax_free_pot: float = s["tax_free_pot"]
    baseline_spending: float = s["baseline_spending"]
    dc_pots: list[tuple[int, float]] = s["dc_pots"]
    dc_pot_names: list[str] = s["dc_pot_names"]
    db_pensions: list[tuple[int, float]] = s["db_pensions"]
    db_pension_names: list[str] = s["db_pension_names"]
    life_events: list[LumpSumEvent | SpendingStepEvent] = s["life_events"]
    life_event_names: list[str] = s["life_event_names"]
    mean_return: float = s["mean_return"]
    std_return: float = s["std_return"]
    random_seed: int = s["random_seed"]
    num_simulations: int = s["num_simulations"]

    if not dc_pots:
        dc_pots = [(57, 0.0)]

    primary_dc_pot = float(dc_pots[0][1])
    secondary_dc_pot = (
        float(sum(p[1] for p in dc_pots[1:])) if len(dc_pots) > 1 else 0.0
    )
    secondary_draw_age: int = int(dc_pots[1][0]) if len(dc_pots) > 1 else end_age

    ages = np.arange(start_age, end_age + 1, dtype=np.int_)
    db_income = np.array(
        [
            sum(
                float(amt)
                for (db_age, amt) in db_pensions
                if int(age) >= int(db_age)
            )
            for age in ages
        ],
        dtype=np.float64,
    )

    try:
        baseline_required = build_required_withdrawals(
            ages=ages,
            baseline_spending=baseline_spending,
            db_income=db_income,
            events=(),
        )
        scenario_required = build_required_withdrawals(
            ages=ages,
            baseline_spending=baseline_spending,
            db_income=db_income,
            events=life_events,
        )
    except ValueError:
        return None

    spending_drawdown = build_spending_drawdown_schedule(
        ages=ages,
        baseline_spending=baseline_spending,
        db_income=db_income,
        events=life_events,
    )

    years = end_age - start_age
    # Fresh RNG seeded with random_seed ensures deterministic, comparable results
    # across re-runs with the same parameters.
    returns = (
        np.random.default_rng(random_seed)
        .normal(mean_return, std_return, years)
        .astype(np.float64)
    )

    base_strategy = create_fixed_real_drawdown_strategy(baseline_spending)

    _, baseline_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=baseline_required,
        dc_pots=dc_pots,
    )
    _, scenario_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        returns=returns,
        drawdown_fn=base_strategy,
        withdrawals_required=scenario_required,
        dc_pots=dc_pots,
    )

    _, monte_carlo_paths = run_monte_carlo_simulation(
        tax_free_pot=tax_free_pot,
        dc_pot=primary_dc_pot,
        secondary_dc_pot=secondary_dc_pot,
        secondary_dc_drawdown_age=secondary_draw_age,
        db_pensions=db_pensions,
        start_age=start_age,
        end_age=end_age,
        mean_return=mean_return,
        std_return=std_return,
        strategy_fn=base_strategy,
        num_simulations=num_simulations,
        seed=random_seed,
        withdrawals_required=scenario_required,
        dc_pots=dc_pots,
    )

    baseline_metrics = summarize_path(baseline_balances)
    scenario_metrics = summarize_path(scenario_balances)
    mc_metrics = summarize_monte_carlo(monte_carlo_paths)

    cache: dict[str, Any] = {
        "ages": ages,
        "baseline_balances": baseline_balances,
        "scenario_balances": scenario_balances,
        "monte_carlo_paths": monte_carlo_paths,
        "spending_drawdown": spending_drawdown,
        "baseline_metrics": baseline_metrics,
        "scenario_metrics": scenario_metrics,
        "mc_metrics": mc_metrics,
        "primary_dc_pot": primary_dc_pot,
        "secondary_dc_pot": secondary_dc_pot,
        "secondary_draw_age": secondary_draw_age,
        "tax_free_pot": tax_free_pot,
        "dc_pots": dc_pots,
        "dc_pot_names": dc_pot_names,
        "db_pensions": db_pensions,
        "db_pension_names": db_pension_names,
        "life_events": life_events,
        "life_event_names": life_event_names,
        "start_age": start_age,
        "end_age": end_age,
        "mean_return": mean_return,
        "std_return": std_return,
        "random_seed": random_seed,
        "num_simulations": num_simulations,
        "baseline_spending": baseline_spending,
        "base_strategy": base_strategy,
        "baseline_required": baseline_required,
        "scenario_required": scenario_required,
    }

    st.session_state["sim_cache"] = cache
    st.session_state["sim_run"] = True
    st.session_state["sim_version"] = s.get("sim_version", 0) + 1

    return cache


# ── Chart rendering ────────────────────────────────────────────────────────────


def _render_chart(chart_type: str, cache: dict[str, Any]) -> None:
    """Render a single chart from the simulation cache.

    Args:
        chart_type: One of the ``_CHART_*`` constants.
        cache: Simulation result cache as returned by :func:`_run_simulation`.
    """
    c = cache

    if chart_type == _CHART_COMPARISON:
        st.markdown("**Baseline vs Life-Events Scenario**")
        fig = plot_baseline_vs_scenario_balances(
            ages=c["ages"],
            baseline_balances=c["baseline_balances"],
            scenario_balances=c["scenario_balances"],
            spending_drawdown_schedule=c["spending_drawdown"],
            secondary_dc_drawdown_age=c["secondary_draw_age"],
            db_pensions=c["db_pensions"],
            life_events=c["life_events"],
            dc_pots=c["dc_pots"],
            dc_pot_names=c["dc_pot_names"],
            db_pension_names=c["db_pension_names"],
            life_event_names=c["life_event_names"],
            save_output=False,
            return_figure=True,
        )
        if fig is not None:
            st.pyplot(fig, clear_figure=True)

    elif chart_type == _CHART_FAN:
        st.markdown("**Monte Carlo Fan Chart**")
        fig = plot_monte_carlo_fan_chart(
            tax_free_pot=c["tax_free_pot"],
            dc_pot=c["primary_dc_pot"],
            secondary_dc_pot=c["secondary_dc_pot"],
            secondary_dc_drawdown_age=c["secondary_draw_age"],
            db_pensions=c["db_pensions"],
            start_age=c["start_age"],
            end_age=c["end_age"],
            mean_return=c["mean_return"],
            std_return=c["std_return"],
            strategy_fn=c["base_strategy"],
            num_simulations=c["num_simulations"],
            seed=c["random_seed"],
            withdrawals_required=c["scenario_required"],
            life_events=c["life_events"],
            spending_drawdown_schedule=c["spending_drawdown"],
            dc_pots=c["dc_pots"],
            dc_pot_names=c["dc_pot_names"],
            db_pension_names=c["db_pension_names"],
            life_event_names=c["life_event_names"],
            save_output=False,
            return_figure=True,
        )
        if fig is not None:
            st.pyplot(fig, clear_figure=True)

    elif chart_type == _CHART_SEQUENCE:
        st.markdown("**Sequence-of-Returns Teaching Chart**")
        fig = plot_sequence_of_returns_scenarios(
            tax_free_pot=c["tax_free_pot"],
            dc_pot=c["primary_dc_pot"],
            secondary_dc_pot=c["secondary_dc_pot"],
            secondary_dc_drawdown_age=c["secondary_draw_age"],
            db_pensions=c["db_pensions"],
            start_age=c["start_age"],
            end_age=c["end_age"],
            mean_return=c["mean_return"],
            std_return=c["std_return"],
            strategy_fn=c["base_strategy"],
            withdrawals_required=c["scenario_required"],
            life_events=c["life_events"],
            spending_drawdown_schedule=c["spending_drawdown"],
            dc_pots=c["dc_pots"],
            dc_pot_names=c["dc_pot_names"],
            db_pension_names=c["db_pension_names"],
            life_event_names=c["life_event_names"],
            save_output=False,
            return_figure=True,
        )
        if fig is not None:
            st.pyplot(fig, clear_figure=True)

    elif chart_type == _CHART_STACKED:
        st.markdown("**Pot Composition (Stacked Area)**")
        fig = plot_pots_stacked_area(
            tax_free_pot=c["tax_free_pot"],
            dc_pot=c["primary_dc_pot"],
            secondary_dc_pot=c["secondary_dc_pot"],
            secondary_dc_drawdown_age=c["secondary_draw_age"],
            db_pensions=c["db_pensions"],
            start_age=c["start_age"],
            end_age=c["end_age"],
            mean_return=c["mean_return"],
            std_return=c["std_return"],
            strategy_fn=c["base_strategy"],
            seed=c["random_seed"],
            withdrawals_required=c["scenario_required"],
            life_events=c["life_events"],
            spending_drawdown_schedule=c["spending_drawdown"],
            dc_pots=c["dc_pots"],
            dc_pot_names=c["dc_pot_names"],
            db_pension_names=c["db_pension_names"],
            life_event_names=c["life_event_names"],
            save_output=False,
            return_figure=True,
        )
        if fig is not None:
            st.pyplot(fig, clear_figure=True)

    elif chart_type == _CHART_INDIVIDUAL:
        st.markdown("**Pot and Income Panels (4 Panels)**")
        fig = plot_individual_pots_subplots(
            tax_free_pot=c["tax_free_pot"],
            dc_pot=c["primary_dc_pot"],
            secondary_dc_pot=c["secondary_dc_pot"],
            secondary_dc_drawdown_age=c["secondary_draw_age"],
            db_pensions=c["db_pensions"],
            start_age=c["start_age"],
            end_age=c["end_age"],
            mean_return=c["mean_return"],
            std_return=c["std_return"],
            strategy_fn=c["base_strategy"],
            seed=c["random_seed"],
            withdrawals_required=c["scenario_required"],
            life_events=c["life_events"],
            dc_pots=c["dc_pots"],
            dc_pot_names=c["dc_pot_names"],
            db_pension_names=c["db_pension_names"],
            life_event_names=c["life_event_names"],
            save_output=False,
            return_figure=True,
        )
        if fig is not None:
            st.pyplot(fig, clear_figure=True)


def _render_charts_from_cache(
    chart_types: list[str], cache: dict[str, Any]
) -> None:
    """Render all requested chart types from the simulation cache.

    Args:
        chart_types: Ordered list of ``_CHART_*`` constants.
        cache: Simulation result cache as returned by :func:`_run_simulation`.
    """
    for chart_type in chart_types:
        _render_chart(chart_type, cache)


# ── Response builders ──────────────────────────────────────────────────────────


def _build_scenario_summary() -> str:
    """Build a short human-readable summary of the current scenario.

    Returns:
        Markdown string describing the current scenario parameters.
    """
    s = st.session_state
    pots_str = ", ".join(
        f"£{bal:,.0f} from age {age}"
        for age, bal in s["dc_pots"]
        if bal > 0
    )
    db_str = ", ".join(
        f"£{amt:,.0f}/year from age {age}"
        for age, amt in s["db_pensions"]
        if amt > 0
    )
    events_str = (
        f"{len(s['life_events'])} life event(s)"
        if s["life_events"]
        else "no life events"
    )
    return (
        f"**Current scenario:** ages {s['start_age']}–{s['end_age']}, "
        f"tax-free £{s['tax_free_pot']:,.0f}, "
        f"DC pots: {pots_str or 'none'}, "
        f"DB income: {db_str or 'none'}, "
        f"spending £{s['baseline_spending']:,.0f}/year, "
        f"{events_str}."
    )


def _build_sim_response(cache: dict[str, Any]) -> str:
    """Build a conversational summary of simulation results.

    Args:
        cache: Simulation result cache.

    Returns:
        Markdown string with plain-English explanation and key metrics.
    """
    bm = cache["baseline_metrics"]
    sm = cache["scenario_metrics"]
    mc = cache["mc_metrics"]
    events = cache["life_events"]
    names = cache["life_event_names"]

    explanation = build_plain_english_explanation(
        baseline_metrics=bm,
        scenario_metrics=sm,
        monte_carlo_metrics=mc,
        events=events,
        event_names=names if names else None,
    )

    next_steps = (
        "\n\n**What to explore next:** Try asking *'What if I need £X at age Y?'*, "
        "*'Which pot runs out first?'*, or *'What about risk?'*"
    )
    return explanation + next_steps


# ── Main app ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Render the Streamlit chat-based pension simulator UI."""
    st.set_page_config(page_title="IFA Pension Chat", layout="wide")
    st.title("IFA Pension Drawdown Chat")
    st.caption(
        "Ask 'what if' questions about your retirement finances and see the "
        "answers as charts and plain-English summaries."
    )

    _init_session_state()

    # ── Display message history ────────────────────────────────────────────
    current_version: int = st.session_state.get("sim_version", 0)
    sim_cache: dict[str, Any] | None = st.session_state.get("sim_cache")

    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(_WELCOME_TEXT)

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("charts") and msg["role"] == "assistant":
                msg_version: int = msg.get("sim_version", -1)
                if msg_version == current_version and sim_cache is not None:
                    _render_charts_from_cache(msg["charts"], sim_cache)
                else:
                    st.caption(
                        "*(Charts from a previous simulation — "
                        "type 'run' to refresh.)*"
                    )

    # ── Handle new user input ──────────────────────────────────────────────
    user_input = st.chat_input("Ask a 'what if' question…")
    if not user_input:
        return

    # Record and display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Parse intent
    intent = _parse_message(
        text=user_input,
        start_age=st.session_state["start_age"],
        end_age=st.session_state["end_age"],
    )
    action = intent["action"]

    # ── Handle reset ───────────────────────────────────────────────────────
    if action == "reset":
        _reset_state()
        response = (
            "Everything has been reset to the default scenario. "
            + _build_scenario_summary()
            + "\n\nWhat would you like to explore?"
        )
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": response,
                "charts": None,
                "sim_version": None,
            }
        )
        with st.chat_message("assistant"):
            st.markdown(response)
        return

    # ── Handle help ────────────────────────────────────────────────────────
    if action == "help":
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": _HELP_TEXT,
                "charts": None,
                "sim_version": None,
            }
        )
        with st.chat_message("assistant"):
            st.markdown(_HELP_TEXT)
        return

    # ── Apply parameter updates ────────────────────────────────────────────
    param_confirmations = _apply_updates(intent["updates"])
    event_confirmations = _apply_events(intent["lump_events"], intent["spend_events"])
    all_confirmations = param_confirmations + event_confirmations

    # ── Run simulation if needed ───────────────────────────────────────────
    charts_to_show: list[str] = intent.get("charts", [])
    sim_cache = None

    if intent.get("auto_run"):
        with st.spinner("Running simulation…"):
            sim_cache = _run_simulation()

    # ── Build response text ────────────────────────────────────────────────
    parts: list[str] = []

    if all_confirmations:
        parts.append("**Updated:**\n" + "\n".join(f"- {c}" for c in all_confirmations))

    if intent.get("auto_run") and sim_cache is None:
        parts.append(
            "I couldn't run the simulation — please check that your "
            "life-event ages are within your start/end age range."
        )
        charts_to_show = []
    elif sim_cache is not None:
        parts.append(_build_sim_response(sim_cache))

    if action == "unknown" and not all_confirmations:
        parts.append(
            "I didn't quite understand that. Type **help** to see what I can do, "
            "or try *'run simulation'*, *'show pots'*, or describe your situation."
        )

    if not parts:
        parts.append(_build_scenario_summary())

    response_text = "\n\n".join(parts)

    # ── Store and display assistant message ───────────────────────────────
    new_version = st.session_state.get("sim_version", 0)
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": response_text,
            "charts": charts_to_show if charts_to_show else None,
            "sim_version": new_version if charts_to_show else None,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response_text)
        if charts_to_show and sim_cache is not None:
            _render_charts_from_cache(charts_to_show, sim_cache)


if __name__ == "__main__":
    main()
