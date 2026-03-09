"""Plot creation functions for pension simulation outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ifa.engine import (
    DbPensionInput,
    run_monte_carlo_simulation,
    simulate_multi_pot_pension_path,
)
from ifa.market import generate_deterministic_sequences, generate_random_returns
from ifa.models import DbPension
from ifa.strategies import (
    DrawdownFn,
    create_db_aware_strategy,
    create_fixed_real_drawdown_strategy,
    create_no_withdrawal_strategy,
)

LOGGER = logging.getLogger(__name__)


def _to_output_path(output_file: str | Path) -> Path:
    return output_file if isinstance(output_file, Path) else Path(output_file)


def add_event_lines_to_plot(
    ax: plt.Axes,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
) -> None:
    """Add vertical lines for key pension events."""
    colors_event = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    event_index = 0

    if (
        secondary_dc_drawdown_age is not None
        and start_age <= secondary_dc_drawdown_age <= end_age
    ):
        ax.axvline(
            x=secondary_dc_drawdown_age,
            color=colors_event[event_index % len(colors_event)],
            linestyle="--",
            linewidth=2,
            alpha=0.6,
        )
        ax.text(
            secondary_dc_drawdown_age,
            ax.get_ylim()[1] * 0.95,
            f"Secondary DC\nstarts (age {secondary_dc_drawdown_age})",
            fontsize=9,
            ha="center",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
        )
        event_index += 1

    for pension in db_pensions:
        if isinstance(pension, DbPension):
            db_start_age = pension.start_age
            db_amount = pension.annual_amount
        else:
            db_start_age, db_amount = pension
        if start_age <= db_start_age <= end_age:
            ax.axvline(
                x=db_start_age,
                color=colors_event[event_index % len(colors_event)],
                linestyle="--",
                linewidth=2,
                alpha=0.6,
            )
            ax.text(
                db_start_age,
                ax.get_ylim()[1] * (0.90 - event_index * 0.05),
                f"DB Pension +GBP{db_amount // 1000:.0f}k\n(age {db_start_age})",
                fontsize=9,
                ha="center",
                bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
            )
            event_index += 1


def plot_pots_stacked_area(
    tax_free_pot: float,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
    mean_return: float,
    std_return: float,
    strategy_fn: DrawdownFn,
    seed: int,
    output_file: str | Path = "pots_stacked_area.png",
) -> None:
    """Plot stacked pot composition over time."""
    num_years = end_age - start_age
    returns = generate_random_returns(num_years, mean_return, std_return, seed)

    (
        ages,
        total_balances,
        dc_balances,
        secondary_dc_balances,
        tax_free_balances,
        _,
        _,
    ) = simulate_multi_pot_pension_path(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        returns,
        strategy_fn,
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.fill_between(
        ages, 0, tax_free_balances, alpha=0.7, label="Tax-Free Pot", color="#2ECC71"
    )
    ax.fill_between(
        ages,
        tax_free_balances,
        tax_free_balances + dc_balances,
        alpha=0.7,
        label="Main DC Pot",
        color="#3498DB",
    )
    ax.fill_between(
        ages,
        tax_free_balances + dc_balances,
        tax_free_balances + dc_balances + secondary_dc_balances,
        alpha=0.7,
        label="Secondary DC Pot",
        color="#9B59B6",
    )

    ax.plot(
        ages,
        total_balances,
        color="black",
        linewidth=2.5,
        label="Total Pot",
        marker="o",
        markersize=4,
    )

    add_event_lines_to_plot(
        ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age
    )

    ax.axhline(y=0, color="red", linestyle=":", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pot Balance (GBP)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Pension Pot Composition Over Time\n"
        "(Stacked Area - Individual Pot Contribution)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    target = _to_output_path(output_file)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", target)
    plt.close()


def plot_individual_pots_subplots(
    tax_free_pot: float,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
    mean_return: float,
    std_return: float,
    strategy_fn: DrawdownFn,
    seed: int,
    output_file: str | Path = "pots_individual.png",
) -> None:
    """Plot individual pot trajectories in four panels."""
    num_years = end_age - start_age
    returns = generate_random_returns(num_years, mean_return, std_return, seed)

    (
        ages,
        total_balances,
        dc_balances,
        secondary_dc_balances,
        tax_free_balances,
        db_income,
        _,
    ) = simulate_multi_pot_pension_path(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        returns,
        strategy_fn,
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(
        ages, tax_free_balances, linewidth=3, color="#2ECC71", marker="o", markersize=4
    )
    ax.fill_between(ages, 0, tax_free_balances, alpha=0.3, color="#2ECC71")
    ax.set_title(
        "Tax-Free Pot (ISAs, Premium Bonds, etc.)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Balance (GBP)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    ax = axes[0, 1]
    ax.plot(ages, dc_balances, linewidth=3, color="#3498DB", marker="s", markersize=4)
    ax.fill_between(ages, 0, dc_balances, alpha=0.3, color="#3498DB")
    ax.set_title("Main DC Pot (Drawing from age 55)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Balance (GBP)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    ax = axes[1, 0]
    ax.plot(
        ages,
        secondary_dc_balances,
        linewidth=3,
        color="#9B59B6",
        marker="^",
        markersize=4,
    )
    ax.fill_between(ages, 0, secondary_dc_balances, alpha=0.3, color="#9B59B6")
    if secondary_dc_drawdown_age is not None:
        ax.axvline(
            x=secondary_dc_drawdown_age,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
        )
        ax.text(
            secondary_dc_drawdown_age,
            ax.get_ylim()[1] * 0.9,
            f"Drawdown starts\nage {secondary_dc_drawdown_age}",
            fontsize=9,
            ha="center",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
        )
    ax.set_title("Secondary DC Pot (Grows then Draws)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Age", fontsize=11, fontweight="bold")
    ax.set_ylabel("Balance (GBP)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    ax = axes[1, 1]
    ax.plot(
        ages,
        total_balances,
        linewidth=3,
        color="black",
        label="Total Pot",
        marker="o",
        markersize=4,
    )
    ax.fill_between(ages, 0, total_balances, alpha=0.2, color="gray")

    if len(db_pensions) > 0:
        ax_secondary = ax.twinx()
        ax_secondary.step(
            ages,
            db_income,
            linewidth=2.5,
            color="#E74C3C",
            label="DB Pension Income",
            where="post",
        )
        ax_secondary.set_ylabel(
            "Annual DB Income (GBP)", fontsize=11, fontweight="bold", color="#E74C3C"
        )
        ax_secondary.tick_params(axis="y", labelcolor="#E74C3C")
        ax_secondary.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
        )

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_secondary.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")
    else:
        ax.legend(fontsize=10)

    ax.set_title(
        "Total Pot + DB Pension Income Timeline", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Age", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Balance (GBP)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    fig.suptitle(
        "Individual Pension Pots Evolution\n"
        "(DB pension income shown on secondary axis, right plot)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    target = _to_output_path(output_file)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", target)
    plt.close()


def plot_sequence_of_returns_scenarios(
    tax_free_pot: float,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
    mean_return: float,
    std_return: float,
    strategy_fn: DrawdownFn,
    output_file: str | Path = "sequence_scenarios.png",
) -> None:
    """Plot deterministic sequence-of-returns scenarios plus baseline."""
    num_years = end_age - start_age
    early_bad, early_good, constant = generate_deterministic_sequences(
        num_years, mean_return, std_return
    )

    ages, balances_early_bad, *_ = simulate_multi_pot_pension_path(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        early_bad,
        strategy_fn,
    )
    _, balances_early_good, *_ = simulate_multi_pot_pension_path(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        early_good,
        strategy_fn,
    )
    _, balances_constant, *_ = simulate_multi_pot_pension_path(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        constant,
        strategy_fn,
    )
    _, balances_no_withdrawal, *_ = simulate_multi_pot_pension_path(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        constant,
        create_no_withdrawal_strategy(),
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        ages,
        balances_early_bad,
        linewidth=2.5,
        label="Early bad returns",
        marker="o",
        markersize=4,
    )
    ax.plot(
        ages,
        balances_early_good,
        linewidth=2.5,
        label="Early good returns",
        marker="s",
        markersize=4,
    )
    ax.plot(
        ages,
        balances_constant,
        linewidth=2.5,
        label="Constant returns",
        marker="^",
        markersize=4,
    )
    ax.plot(
        ages,
        balances_no_withdrawal,
        linewidth=2.5,
        label="No withdrawal (baseline)",
        linestyle="--",
        marker="d",
        markersize=4,
    )

    ax.axhline(
        y=0, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="Zero balance"
    )
    add_event_lines_to_plot(
        ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age
    )

    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pension Balance (GBP)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Pension Drawdown: Sequence-of-Returns Scenarios",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    target = _to_output_path(output_file)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", target)
    plt.close()


def plot_monte_carlo_fan_chart(
    tax_free_pot: float,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
    mean_return: float,
    std_return: float,
    strategy_fn: DrawdownFn,
    num_simulations: int,
    seed: int,
    output_file: str | Path = "monte_carlo_fan.png",
) -> None:
    """Plot a Monte Carlo fan chart with percentile bands."""
    ages, paths = run_monte_carlo_simulation(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        mean_return,
        std_return,
        strategy_fn,
        num_simulations,
        seed,
    )

    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    final_balances = paths[:, -1]
    zero_count = int(np.sum(final_balances == 0))
    zero_pct = 100 * zero_count / num_simulations

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(
        ages, p10, p90, alpha=0.2, color="blue", label="10th-90th percentile"
    )
    ax.fill_between(
        ages, p25, p75, alpha=0.3, color="blue", label="25th-75th percentile"
    )
    ax.plot(
        ages,
        p50,
        linewidth=3,
        color="darkblue",
        label="Median (50th percentile)",
        marker="o",
        markersize=5,
    )
    ax.plot(ages, p10, linewidth=1, color="blue", linestyle=":", alpha=0.6)
    ax.plot(ages, p90, linewidth=1, color="blue", linestyle=":", alpha=0.6)
    ax.axhline(
        y=0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Zero balance"
    )

    add_event_lines_to_plot(
        ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age
    )

    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pension Balance (GBP)", fontsize=12, fontweight="bold")
    title = f"Monte Carlo Pension Projection ({num_simulations} simulations)"
    if zero_pct > 0:
        title += (
            f"\n({zero_pct:.1f}% of simulations exhausted pot before age {end_age})"
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    target = _to_output_path(output_file)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", target)
    LOGGER.info("Risk metric: %.1f%% of simulations ran out of money.", zero_pct)
    plt.close()


def plot_multiple_drawdown_levels(
    tax_free_pot: float,
    dc_pot: float,
    secondary_dc_pot: float,
    secondary_dc_drawdown_age: int | None,
    db_pensions: Sequence[DbPensionInput],
    start_age: int,
    end_age: int,
    mean_return: float,
    std_return: float,
    annual_drawdowns: Sequence[float],
    seed: int,
    output_file: str | Path = "multiple_drawdowns.png",
) -> None:
    """Plot outcomes for multiple fixed drawdown levels on one return path."""
    num_years = end_age - start_age
    returns = generate_random_returns(num_years, mean_return, std_return, seed)

    fig, ax = plt.subplots(figsize=(12, 7))
    color_map = plt.get_cmap("viridis")
    colors = color_map(np.linspace(0, 1, len(annual_drawdowns) + 1))

    for index, drawdown in enumerate(annual_drawdowns):
        base_strategy = create_fixed_real_drawdown_strategy(drawdown)
        strategy = create_db_aware_strategy(base_strategy, db_pensions)
        ages, balances, *_ = simulate_multi_pot_pension_path(
            tax_free_pot,
            dc_pot,
            secondary_dc_pot,
            secondary_dc_drawdown_age,
            db_pensions,
            start_age,
            end_age,
            returns,
            strategy,
        )
        ax.plot(
            ages,
            balances,
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=f"GBP{drawdown:,.0f}/year",
            color=colors[index],
        )

    ages, baseline_balances, *_ = simulate_multi_pot_pension_path(
        tax_free_pot,
        dc_pot,
        secondary_dc_pot,
        secondary_dc_drawdown_age,
        db_pensions,
        start_age,
        end_age,
        returns,
        create_no_withdrawal_strategy(),
    )
    ax.plot(
        ages,
        baseline_balances,
        linewidth=2.5,
        marker="s",
        markersize=5,
        label="No withdrawal (baseline)",
        linestyle="--",
        color="black",
    )

    ax.axhline(y=0, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    add_event_lines_to_plot(
        ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age
    )

    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pension Balance (GBP)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Impact of Different Drawdown Levels\n(Same market return sequence)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    target = _to_output_path(output_file)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", target)
    plt.close()


def plot_baseline_vs_scenario_balances(
    ages: np.ndarray,
    baseline_balances: np.ndarray,
    scenario_balances: np.ndarray,
    output_file: str | Path = "baseline_vs_scenario.png",
) -> None:
    """Plot baseline and life-event scenario balances on one comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        ages,
        baseline_balances,
        linewidth=2.5,
        color="#1F77B4",
        marker="o",
        markersize=4,
        label="Baseline (no life events)",
    )
    ax.plot(
        ages,
        scenario_balances,
        linewidth=2.5,
        color="#D62728",
        marker="s",
        markersize=4,
        label="Scenario (with life events)",
    )

    ax.axhline(y=0, color="black", linestyle=":", linewidth=1.3, alpha=0.6)
    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Balance (GBP)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Baseline vs Life-Event Scenario\n(Real-terms pension pot trajectory)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"GBP{value / 1000:.0f}k")
    )

    target = _to_output_path(output_file)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved: %s", target)
    plt.close()
