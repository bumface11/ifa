# IFA Pension Drawdown Simulator

A beginner-friendly pension simulator that shows how withdrawals, market returns,
DB pension income, and life events can change a retirement plan.

## Quick Start

Install dependencies in your environment:

```bash
uv sync
```

Run the CLI simulator and save charts to `output/`:

```bash
uv run python pension_drawdown_simulator.py
```

Run the Streamlit app:

```bash
uv run streamlit run ifa_web.py
```

## Life Events

Life events model real-world spending changes in simple terms:

- `LumpSumEvent(age, amount)`: one-off extra spending at one age.
- `SpendingStepEvent(start_age, extra_per_year, end_age=None)`: ongoing extra
	yearly spending from a start age.

Examples:

- House repairs lump sum: `LumpSumEvent(age=70, amount=18000)`
- Care costs step-up: `SpendingStepEvent(start_age=78, extra_per_year=6000)`

The model compares a baseline plan (no events) against a life-events scenario on
the same return path so the impact is easier to understand.

## What The App Shows

- Baseline vs scenario line chart on the same market path.
- Sequence-of-returns teaching chart.
- Monte Carlo fan chart with ruin probability.
- Pot composition charts (stacked and per-pot subplots).
- Plain-English explanation of scenario impact.

## Output Files

- CLI saves charts to `output/` by default.
- Streamlit can optionally save its generated charts to `output/` using the
	sidebar checkbox.
