"""
Pension Drawdown Simulator
Simulates pension pot evolution under various withdrawal strategies and market return scenarios.
Works entirely in real (inflation-adjusted) terms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================================
# CONFIGURATION SECTION - Modify these parameters
# ============================================================================

# TAX-FREE INVESTMENT POT (drawn first, e.g., ISAs, premium bonds)
INITIAL_TAX_FREE_POT = 373_890     # Tax-free savings drawn down first

# DEFINED CONTRIBUTION (DC) PENSION POTS
INITIAL_DC_POT = 300_000           # Main DC pot, starts drawing at START_AGE
SECONDARY_DC_POT = 65_000         # Secondary DC pot, grows until SECONDARY_DC_DRAWDOWN_AGE
SECONDARY_DC_DRAWDOWN_AGE = 65     # Age when secondary DC pot starts being drawn

# DEFINED BENEFIT (DB) PENSION INCOME STREAMS (inflation-adjusted, up to 3)
# Format: [(start_age, annual_amount), ...]
# Leave empty [] if not applicable
DB_PENSIONS = [
    (62, 12_510),    # Example: £8k/year from age 60 (early DB from previous employer)
    (67, 11_900),   # Example: £12k/year from State Pension age
    # (70, 5_000),    # Example: additional income from age 70
]

# GENERAL SIMULATION PARAMETERS
START_AGE = 52
END_AGE = 95

# Investment return parameters (real, inflation-adjusted)
MEAN_RETURN = 0.04      # 4% real return
STD_RETURN = 0.10       # 10% standard deviation
RANDOM_SEED = 42        # For reproducibility

# Annual drawdown scenarios for fixed real withdrawal (supplementing DB pensions)
ANNUAL_DRAWDOWNS = [26_000, 30_000, 34_000]  # In real terms, from DC pot

# Monte Carlo simulation settings
NUM_SIMULATIONS = 1_000

# Guardrails strategy parameters
GUARDRAILS_TARGET_INCOME = 30_000
GUARDRAILS_LOWER_BAND = 0.80      # 20% below initial glidepath triggers cut
GUARDRAILS_UPPER_BAND = 1.20      # 20% above glidepath triggers increase
GUARDRAILS_ADJUSTMENT = 0.10      # Adjust income by 10%

# ============================================================================
# CORE SIMULATION ENGINE
# ============================================================================

def calculate_db_pension_income(age, db_pensions):
    """
    Calculate total DB pension income for a given age.
    
    Args:
        age: Current age
        db_pensions: List of tuples [(start_age, annual_amount), ...]
    
    Returns:
        Total DB pension income for this age
    """
    return sum(amount for start_age, amount in db_pensions if age >= start_age)


def simulate_multi_pot_pension_path(tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
                                    db_pensions, start_age, end_age, returns, drawdown_fn):
    """
    Simulate pension evolution with multiple pots and income streams.
    
    Args:
        tax_free_pot: Initial tax-free investment pot
        dc_pot: Main DC pension pot (starts drawing at start_age)
        secondary_dc_pot: Secondary DC pot (starts drawing at secondary_dc_drawdown_age)
        secondary_dc_drawdown_age: Age when secondary DC pot starts being drawn
        db_pensions: List of (start_age, annual_amount) tuples for DB pensions
        start_age: Starting age
        end_age: Ending age
        returns: Array of annual returns
        drawdown_fn: Function(age, current_dc_pot, state_dict) -> withdrawal_amount from DC
    
    Returns:
        Tuple of (ages, total_balances, dc_balances, secondary_dc_balances, tax_free_balances,
                  db_income, total_withdrawals)
    """
    ages = np.arange(start_age, end_age + 1)
    num_years = len(ages)
    
    # Initialize tracking arrays
    total_balances = np.zeros(num_years)
    dc_balances = np.zeros(num_years)
    secondary_dc_balances = np.zeros(num_years)
    tax_free_balances = np.zeros(num_years)
    db_income_array = np.zeros(num_years)
    total_withdrawals = np.zeros(num_years)
    
    # Set initial values
    dc_balances[0] = dc_pot
    secondary_dc_balances[0] = secondary_dc_pot
    tax_free_balances[0] = tax_free_pot
    total_balances[0] = dc_pot + secondary_dc_pot + tax_free_pot
    
    state_dict = {}  # For strategies that need to track state
    
    for i in range(1, num_years):
        current_age = ages[i - 1]
        
        # Apply investment returns to growing pots (not drawn yet)
        # Tax-free pot
        if tax_free_balances[i - 1] > 0:
            tax_free_balances[i] = tax_free_balances[i - 1] * (1 + returns[i - 1])
        
        # Secondary DC pot (only grows if not yet drawing)
        if secondary_dc_drawdown_age is not None and current_age < secondary_dc_drawdown_age:
            if secondary_dc_balances[i - 1] > 0:
                secondary_dc_balances[i] = secondary_dc_balances[i - 1] * (1 + returns[i - 1])
        else:
            secondary_dc_balances[i] = secondary_dc_balances[i - 1]
        
        # Main DC pot always grows
        dc_balances[i] = dc_balances[i - 1] * (1 + returns[i - 1])
        
        # Calculate DB pension income for this year
        db_income = calculate_db_pension_income(current_age, db_pensions)
        db_income_array[i] = db_income
        
        # Calculate DC withdrawal amount using strategy
        # Pass combined DC pot value to strategy
        combined_dc = dc_balances[i] + secondary_dc_balances[i]
        dc_withdrawal = drawdown_fn(current_age, combined_dc, state_dict)
        
        # Total desired withdrawal = DC strategy withdrawal
        # (DB pensions are received automatically, not withdrawn)
        total_withdrawal_desired = dc_withdrawal
        
        # Allocate withdrawal from pots in order: tax-free first, then primary DC, then secondary DC
        current_withdrawal = 0
        
        # First, draw from tax-free pot
        tax_free_withdrawal = min(total_withdrawal_desired - current_withdrawal, tax_free_balances[i])
        tax_free_balances[i] -= tax_free_withdrawal
        current_withdrawal += tax_free_withdrawal
        
        # Then draw from primary DC pot
        if current_withdrawal < total_withdrawal_desired:
            dc_withdrawal_amt = min(total_withdrawal_desired - current_withdrawal, dc_balances[i])
            dc_balances[i] -= dc_withdrawal_amt
            current_withdrawal += dc_withdrawal_amt
        
        # Finally draw from secondary DC pot
        if current_withdrawal < total_withdrawal_desired:
            secondary_withdrawal = min(total_withdrawal_desired - current_withdrawal, secondary_dc_balances[i])
            secondary_dc_balances[i] -= secondary_withdrawal
            current_withdrawal += secondary_withdrawal
        
        # Ensure no negative balances
        dc_balances[i] = max(0, dc_balances[i])
        secondary_dc_balances[i] = max(0, secondary_dc_balances[i])
        tax_free_balances[i] = max(0, tax_free_balances[i])
        
        # Total balance is all pots combined
        total_balances[i] = dc_balances[i] + secondary_dc_balances[i] + tax_free_balances[i]
        total_withdrawals[i] = current_withdrawal
    
    return (ages, total_balances, dc_balances, secondary_dc_balances, tax_free_balances,
            db_income_array, total_withdrawals)


def create_fixed_real_drawdown_strategy(annual_withdrawal):
    """
    Create a fixed real drawdown strategy function.
    
    Args:
        annual_withdrawal: Fixed amount to withdraw each year (real terms)
    
    Returns:
        Callable strategy function
    """
    def strategy(age, current_pot, state_dict):
        # Simply withdraw the fixed amount every year
        return annual_withdrawal
    
    return strategy


def create_percentage_of_pot_strategy(percentage):
    """
    Create a percentage-of-pot drawdown strategy (e.g., 4% rule).
    
    Args:
        percentage: Withdrawal rate, e.g., 0.04 for 4%
    
    Returns:
        Callable strategy function
    """
    def strategy(age, current_pot, state_dict):
        return current_pot * percentage
    
    return strategy


def create_guardrails_strategy(target_income, lower_band, upper_band, adjustment):
    """
    Create a guardrails-based withdrawal strategy.
    Starts with target income, adjusts if pot drifts from band.
    
    Args:
        target_income: Initial target real income
        lower_band: Threshold below which to cut withdrawals (e.g., 0.80 = 80%)
        upper_band: Threshold above which to increase (e.g., 1.20 = 120%)
        adjustment: Adjustment size, e.g., 0.10 for ±10%
    
    Returns:
        Callable strategy function
    """
    def strategy(age, current_pot, state_dict):
        # On first call, initialize
        if 'initial_pot' not in state_dict:
            state_dict['initial_pot'] = current_pot
            state_dict['initial_income'] = target_income
            state_dict['current_income'] = target_income
        
        # Compute what the balance "should be" if no withdrawals and constant return
        # For simplicity, use a static glidepath
        initial_pot = state_dict['initial_pot']
        initial_income = state_dict['initial_income']
        
        # A simple glidepath: assume pot should decrease linearly
        # (In practice you'd use a more sophisticated glidepath)
        years_elapsed = age - (START_AGE)
        total_years = END_AGE - START_AGE
        expected_fraction = max(0, 1 - (years_elapsed / total_years) * 0.5)
        expected_pot = initial_pot * expected_fraction
        
        # Check guardrails
        current_income = state_dict['current_income']
        lower_threshold = expected_pot * lower_band
        upper_threshold = expected_pot * upper_band
        
        if current_pot < lower_threshold:
            current_income = current_income * (1 - adjustment)
        elif current_pot > upper_threshold:
            current_income = current_income * (1 + adjustment)
        
        state_dict['current_income'] = current_income
        return current_income
    
    return strategy


def create_no_withdrawal_strategy():
    """
    Create a strategy with no withdrawals (for comparison baseline).
    """
    def strategy(age, current_pot, state_dict):
        return 0
    
    return strategy


def create_db_aware_strategy(base_strategy, db_pensions):
    """
    Wrap a base strategy to account for DB pension income.
    When DB pensions are available, reduce DC withdrawal needs proportionally.
    This ensures we don't over-draw from DC pots when receiving DB pension income.
    
    Args:
        base_strategy: Base drawdown strategy function
        db_pensions: List of (start_age, annual_amount) tuples
    
    Returns:
        Callable strategy function that reduces DC withdrawals by DB income
    """
    def strategy(age, current_pot, state_dict):
        # Get base DC withdrawal amount
        dc_withdrawal = base_strategy(age, current_pot, state_dict)
        
        # Calculate available DB income
        db_income = calculate_db_pension_income(age, db_pensions)
        
        # Reduce DC withdrawal by DB income received
        # (DB income supplements DC withdrawals, so we need less from DC)
        adjusted_withdrawal = max(0, dc_withdrawal - db_income)
        
        return adjusted_withdrawal
    
    return strategy


# ============================================================================
# RETURN SEQUENCE GENERATION
# ============================================================================

def generate_random_returns(num_years, mean, std, seed=None):
    """
    Generate normally distributed investment returns.
    
    Args:
        num_years: Number of years to generate
        mean: Mean return
        std: Standard deviation
        seed: Random seed for reproducibility
    
    Returns:
        Array of returns
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, std, num_years)


def generate_deterministic_sequences(num_years, mean, std):
    """
    Generate three deterministic return sequences with same mean but different ordering.
    
    Args:
        num_years: Number of years
        mean: Mean return
        std: Standard deviation
    
    Returns:
        Tuple of three return arrays:
        - Early bad returns, then good
        - Early good returns, then good
        - Constant return
    """
    # Generate a random sequence to determine the actual values
    np.random.seed(RANDOM_SEED)
    returns = np.random.normal(mean, std, num_years)
    sorted_returns = np.sort(returns)
    
    # Early bad: worst returns first, then best
    early_bad = np.concatenate([
        sorted_returns[:num_years // 2],
        sorted_returns[num_years // 2:][::-1]
    ])
    
    # Early good: best returns first, then worst
    early_good = np.concatenate([
        sorted_returns[num_years // 2:][::-1],
        sorted_returns[:num_years // 2]
    ])
    
    # Constant: repeat mean return every year
    constant = np.full(num_years, mean)
    
    return early_bad, early_good, constant


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo_simulation(tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
                               db_pensions, start_age, end_age, mean_return, std_return,
                               strategy_fn, num_simulations, seed):
    """
    Run Monte Carlo simulation for a given strategy with multiple pots.
    
    Args:
        tax_free_pot: Initial tax-free investment pot
        dc_pot: Main DC pension pot
        secondary_dc_pot: Secondary DC pot
        secondary_dc_drawdown_age: Age when secondary DC starts being drawn
        db_pensions: List of DB pension streams
        start_age: Starting age
        end_age: Ending age
        mean_return: Mean annual return
        std_return: Standard deviation of return
        strategy_fn: Callable strategy function
        num_simulations: Number of paths to simulate
        seed: Random seed
    
    Returns:
        Tuple of (ages, paths_array) where paths_array is (num_simulations, num_years)
    """
    np.random.seed(seed)
    num_years = end_age - start_age
    ages = np.arange(start_age, end_age + 1)
    
    paths = np.zeros((num_simulations, num_years + 1))
    
    for sim in range(num_simulations):
        returns = np.random.normal(mean_return, std_return, num_years)
        _, total_balances, _, _, _, _, _ = simulate_multi_pot_pension_path(
            tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
            db_pensions, start_age, end_age, returns, strategy_fn
        )
        paths[sim, :] = total_balances
    
    return ages, paths


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def add_event_lines_to_plot(ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age):
    """
    Add vertical lines for key pension events (secondary DC drawdown, DB pension starts).
    
    Args:
        ax: Matplotlib axis to add lines to
        secondary_dc_drawdown_age: Age when secondary DC pot starts drawing
        db_pensions: List of (start_age, amount) tuples for DB pensions
        start_age: Simulation start age (for context)
        end_age: Simulation end age (for context)
    """
    colors_event = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    event_idx = 0
    
    # Add line for secondary DC pot drawdown
    if secondary_dc_drawdown_age is not None and start_age <= secondary_dc_drawdown_age <= end_age:
        ax.axvline(x=secondary_dc_drawdown_age, color=colors_event[event_idx % len(colors_event)],
                   linestyle='--', linewidth=2, alpha=0.6)
        ax.text(secondary_dc_drawdown_age, ax.get_ylim()[1] * 0.95,
                f'Secondary DC\nstarts (age {secondary_dc_drawdown_age})',
                fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        event_idx += 1
    
    # Add lines for DB pension start ages
    for db_start_age, db_amount in db_pensions:
        if start_age <= db_start_age <= end_age:
            ax.axvline(x=db_start_age, color=colors_event[event_idx % len(colors_event)],
                       linestyle='--', linewidth=2, alpha=0.6)
            ax.text(db_start_age, ax.get_ylim()[1] * (0.90 - event_idx * 0.05),
                    f'DB Pension +£{db_amount//1000}k\n(age {db_start_age})',
                    fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            event_idx += 1


def plot_pots_stacked_area(tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
                           db_pensions, start_age, end_age, mean_return, std_return, 
                           strategy_fn, seed,
                           output_file="pots_stacked_area.png"):
    """
    Plot stacked area chart showing composition of all pots over time.
    Clearly visualizes how each pot contributes to total wealth.
    """
    num_years = end_age - start_age
    np.random.seed(seed)
    returns = np.random.normal(mean_return, std_return, num_years)
    
    ages, total_balances, dc_balances, secondary_dc_balances, tax_free_balances, db_income, _ = \
        simulate_multi_pot_pension_path(
            tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
            start_age, end_age, returns, strategy_fn
        )
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create stacked area chart
    ax.fill_between(ages, 0, tax_free_balances, alpha=0.7, label='Tax-Free Pot', color='#2ECC71')
    ax.fill_between(ages, tax_free_balances, tax_free_balances + dc_balances, 
                    alpha=0.7, label='Main DC Pot', color='#3498DB')
    ax.fill_between(ages, tax_free_balances + dc_balances, 
                    tax_free_balances + dc_balances + secondary_dc_balances,
                    alpha=0.7, label='Secondary DC Pot', color='#9B59B6')
    
    # Add total line on top
    ax.plot(ages, total_balances, color='black', linewidth=2.5, label='Total Pot', marker='o', markersize=4)
    
    # Add event lines
    add_event_lines_to_plot(ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age)
    
    ax.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Age", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pot Balance (£)", fontsize=12, fontweight='bold')
    ax.set_title("Pension Pot Composition Over Time\n(Stacked Area - Individual Pot Contribution)", 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_individual_pots_subplots(tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
                                  db_pensions, start_age, end_age, mean_return, std_return, 
                                  strategy_fn, seed,
                                  output_file="pots_individual.png"):
    """
    Plot individual pot evolution in separate subplots with income sources annotated.
    """
    num_years = end_age - start_age
    np.random.seed(seed)
    returns = np.random.normal(mean_return, std_return, num_years)
    
    ages, total_balances, dc_balances, secondary_dc_balances, tax_free_balances, db_income, _ = \
        simulate_multi_pot_pension_path(
            tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
            start_age, end_age, returns, strategy_fn
        )
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pot 1: Tax-Free Pot
    ax = axes[0, 0]
    ax.plot(ages, tax_free_balances, linewidth=3, color='#2ECC71', marker='o', markersize=4)
    ax.fill_between(ages, 0, tax_free_balances, alpha=0.3, color='#2ECC71')
    ax.set_title("Tax-Free Pot (ISAs, Premium Bonds, etc.)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Balance (£)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    # Pot 2: Main DC Pot
    ax = axes[0, 1]
    ax.plot(ages, dc_balances, linewidth=3, color='#3498DB', marker='s', markersize=4)
    ax.fill_between(ages, 0, dc_balances, alpha=0.3, color='#3498DB')
    ax.set_title("Main DC Pot (Drawing from age 55)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Balance (£)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    # Pot 3: Secondary DC Pot
    ax = axes[1, 0]
    ax.plot(ages, secondary_dc_balances, linewidth=3, color='#9B59B6', marker='^', markersize=4)
    ax.fill_between(ages, 0, secondary_dc_balances, alpha=0.3, color='#9B59B6')
    if secondary_dc_drawdown_age is not None:
        ax.axvline(x=secondary_dc_drawdown_age, color='red', linestyle='--', linewidth=2, alpha=0.6)
        ax.text(secondary_dc_drawdown_age, ax.get_ylim()[1] * 0.9, 
                f'Drawdown starts\nage {secondary_dc_drawdown_age}',
                fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_title("Secondary DC Pot (Grows then Draws)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Age", fontsize=11, fontweight='bold')
    ax.set_ylabel("Balance (£)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    # Plot 4: Total with DB Income overlay
    ax = axes[1, 1]
    ax.plot(ages, total_balances, linewidth=3, color='black', label='Total Pot', marker='o', markersize=4)
    ax.fill_between(ages, 0, total_balances, alpha=0.2, color='gray')
    
    # Overlay DB income as stepping function
    if len(db_pensions) > 0:
        ax2 = ax.twinx()
        ax2.step(ages, db_income, linewidth=2.5, color='#E74C3C', label='DB Pension Income', where='post')
        ax2.set_ylabel("Annual DB Income (£)", fontsize=11, fontweight='bold', color='#E74C3C')
        ax2.tick_params(axis='y', labelcolor='#E74C3C')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
        
        # Add legend for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    else:
        ax.legend(fontsize=10)
    
    ax.set_title("Total Pot + DB Pension Income Timeline", fontsize=12, fontweight='bold')
    ax.set_xlabel("Age", fontsize=11, fontweight='bold')
    ax.set_ylabel("Total Balance (£)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    fig.suptitle("Individual Pension Pots Evolution\n(DB pension income shown on secondary axis, right plot)", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_sequence_of_returns_scenarios(tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
                                       db_pensions, start_age, end_age, 
                                       mean_return, std_return, strategy_fn,
                                       output_file="sequence_scenarios.png"):
    """
    Plot three deterministic return sequences plus no-withdrawal baseline for total pot.
    """
    num_years = end_age - start_age
    early_bad, early_good, constant = generate_deterministic_sequences(
        num_years, mean_return, std_return
    )
    
    ages, balances_early_bad, *_ = simulate_multi_pot_pension_path(
        tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
        start_age, end_age, early_bad, strategy_fn
    )
    _, balances_early_good, *_ = simulate_multi_pot_pension_path(
        tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
        start_age, end_age, early_good, strategy_fn
    )
    _, balances_constant, *_ = simulate_multi_pot_pension_path(
        tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
        start_age, end_age, constant, strategy_fn
    )
    _, balances_no_withdrawal, *_ = simulate_multi_pot_pension_path(
        tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
        start_age, end_age, constant, create_no_withdrawal_strategy()
    )
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(ages, balances_early_bad, linewidth=2.5, label="Early bad returns", marker='o', markersize=4)
    ax.plot(ages, balances_early_good, linewidth=2.5, label="Early good returns", marker='s', markersize=4)
    ax.plot(ages, balances_constant, linewidth=2.5, label="Constant returns", marker='^', markersize=4)
    ax.plot(ages, balances_no_withdrawal, linewidth=2.5, label="No withdrawal (baseline)", 
            linestyle='--', marker='d', markersize=4)
    
    ax.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label="Zero balance")
    
    # Add event lines for key ages
    add_event_lines_to_plot(ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age)
    
    ax.set_xlabel("Age", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pension Balance (£)", fontsize=12, fontweight='bold')
    ax.set_title("Pension Drawdown: Sequence-of-Returns Scenarios", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_monte_carlo_fan_chart(tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
                               db_pensions, start_age, end_age, mean_return, std_return,
                               strategy_fn, num_simulations, seed,
                               output_file="monte_carlo_fan.png"):
    """
    Plot Monte Carlo fan chart with percentile bands for total pot value.
    """
    ages, paths = run_monte_carlo_simulation(
        tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
        start_age, end_age, mean_return, std_return,
        strategy_fn, num_simulations, seed
    )
    
    # Compute percentiles
    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    
    # Count simulations where pot hits zero before end_age
    final_balances = paths[:, -1]
    zero_count = np.sum(final_balances == 0)
    zero_pct = 100 * zero_count / num_simulations
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Shaded bands
    ax.fill_between(ages, p10, p90, alpha=0.2, color='blue', label='10th–90th percentile')
    ax.fill_between(ages, p25, p75, alpha=0.3, color='blue', label='25th–75th percentile')
    
    # Median line
    ax.plot(ages, p50, linewidth=3, color='darkblue', label='Median (50th percentile)', marker='o', markersize=5)
    
    # Percentile lines
    ax.plot(ages, p10, linewidth=1, color='blue', linestyle=':', alpha=0.6)
    ax.plot(ages, p90, linewidth=1, color='blue', linestyle=':', alpha=0.6)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Zero balance")
    
    # Add event lines for key ages
    add_event_lines_to_plot(ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age)
    
    ax.set_xlabel("Age", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pension Balance (£)", fontsize=12, fontweight='bold')
    title = f"Monte Carlo Pension Projection ({num_simulations} simulations)"
    if zero_pct > 0:
        title += f"\n({zero_pct:.1f}% of simulations exhausted pot before age {end_age})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    print(f"  Risk metric: {zero_pct:.1f}% of simulations ran out of money.")
    plt.close()


def plot_multiple_drawdown_levels(tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age,
                                  db_pensions, start_age, end_age, mean_return, std_return,
                                  annual_drawdowns, seed,
                                  output_file="multiple_drawdowns.png"):
    """
    Plot multiple fixed real drawdown levels using the same random return path.
    """
    num_years = end_age - start_age
    np.random.seed(seed)
    returns = np.random.normal(mean_return, std_return, num_years)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(annual_drawdowns) + 1))
    
    for idx, drawdown in enumerate(annual_drawdowns):
        base_strategy = create_fixed_real_drawdown_strategy(drawdown)
        strategy = create_db_aware_strategy(base_strategy, db_pensions)
        ages, balances, *_ = simulate_multi_pot_pension_path(
            tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
            start_age, end_age, returns, strategy
        )
        ax.plot(ages, balances, linewidth=2.5, marker='o', markersize=5,
                label=f"£{drawdown:,.0f}/year", color=colors[idx])
    
    # No withdrawal baseline
    strategy_baseline = create_no_withdrawal_strategy()
    ages, balances_baseline, *_ = simulate_multi_pot_pension_path(
        tax_free_pot, dc_pot, secondary_dc_pot, secondary_dc_drawdown_age, db_pensions,
        start_age, end_age, returns, strategy_baseline
    )
    ax.plot(ages, balances_baseline, linewidth=2.5, marker='s', markersize=5,
            label="No withdrawal (baseline)", linestyle='--', color='black')
    
    ax.axhline(y=0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Add event lines for key ages
    add_event_lines_to_plot(ax, secondary_dc_drawdown_age, db_pensions, start_age, end_age)
    
    ax.set_xlabel("Age", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pension Balance (£)", fontsize=12, fontweight='bold')
    ax.set_title("Impact of Different Drawdown Levels\n(Same market return sequence)", 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x/1000:.0f}k'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all simulations and generate charts for multi-pot retirement scenario.
    """
    print("=" * 80)
    print("PENSION DRAWDOWN SIMULATOR - Multi-Pot Retirement")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Tax-Free Pot: £{INITIAL_TAX_FREE_POT:,.0f}")
    print(f"  Main DC Pot: £{INITIAL_DC_POT:,.0f}")
    print(f"  Secondary DC Pot: £{SECONDARY_DC_POT:,.0f} (starts drawing at age {SECONDARY_DC_DRAWDOWN_AGE})")
    print(f"  DB Pension Streams: {len(DB_PENSIONS)} streams")
    for start_age, amount in DB_PENSIONS:
        print(f"    - £{amount:,.0f}/year from age {start_age}")
    print(f"\n  Simulation Period: age {START_AGE}–{END_AGE}")
    print(f"  Mean real return: {MEAN_RETURN*100:.1f}%")
    print(f"  Std dev: {STD_RETURN*100:.1f}%")
    print(f"  Random seed: {RANDOM_SEED}\n")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sequence-of-returns scenario (fixed real drawdown at £18k from DC, DB-aware)
    print("1. Running sequence-of-returns scenarios...")
    base_strategy = create_fixed_real_drawdown_strategy(18_000)
    strategy = create_db_aware_strategy(base_strategy, DB_PENSIONS)
    plot_sequence_of_returns_scenarios(
        INITIAL_TAX_FREE_POT, INITIAL_DC_POT, SECONDARY_DC_POT, SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS, START_AGE, END_AGE, MEAN_RETURN, STD_RETURN, strategy,
        output_file=output_dir / "sequence_scenarios.png"
    )
    print()
    
    # 2. Monte Carlo fan chart (fixed real drawdown at £18k from DC)
    print("2. Running Monte Carlo simulation...")
    plot_monte_carlo_fan_chart(
        INITIAL_TAX_FREE_POT, INITIAL_DC_POT, SECONDARY_DC_POT, SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS, START_AGE, END_AGE, MEAN_RETURN, STD_RETURN, strategy,
        NUM_SIMULATIONS, RANDOM_SEED,
        output_file=output_dir / "monte_carlo_fan.png"
    )
    print()
    
    # 3. Multiple drawdown levels (same return sequence)
    print("3. Comparing multiple DC drawdown levels...")
    plot_multiple_drawdown_levels(
        INITIAL_TAX_FREE_POT, INITIAL_DC_POT, SECONDARY_DC_POT, SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS, START_AGE, END_AGE, MEAN_RETURN, STD_RETURN,
        ANNUAL_DRAWDOWNS, RANDOM_SEED,
        output_file=output_dir / "multiple_drawdowns.png"
    )
    print()
    
    # 4. Stacked area showing pot composition
    print("4. Plotting pot composition over time (stacked areas)...")
    plot_pots_stacked_area(
        INITIAL_TAX_FREE_POT, INITIAL_DC_POT, SECONDARY_DC_POT, SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS, START_AGE, END_AGE, MEAN_RETURN, STD_RETURN, strategy,
        RANDOM_SEED,
        output_file=output_dir / "pots_stacked_area.png"
    )
    print()
    
    # 5. Individual pot evolution with subplots
    print("5. Plotting individual pots dynamics (4-panel subplots)...")
    plot_individual_pots_subplots(
        INITIAL_TAX_FREE_POT, INITIAL_DC_POT, SECONDARY_DC_POT, SECONDARY_DC_DRAWDOWN_AGE,
        DB_PENSIONS, START_AGE, END_AGE, MEAN_RETURN, STD_RETURN, strategy,
        RANDOM_SEED,
        output_file=output_dir / "pots_individual.png"
    )
    print()
    
    print("=" * 80)
    print(f"All simulations complete. Charts saved to: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
