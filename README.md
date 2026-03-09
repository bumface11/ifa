# IFA Pension Drawdown Simulator

A beginner-friendly pension simulator that shows how withdrawals, market returns,
DB pension income, and life events can change a retirement plan.

## Quick Start

### Windows 11 (Recommended)

Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies with pip:

```bash
python -m pip install -e .
```

Run the CLI simulator and save charts to `output/`:

```bash
python pension_drawdown_simulator.py
```

Run the Streamlit app:

```bash
streamlit run ifa_web.py
```

### Optional: uv Workflow

If you prefer `uv`, these commands are equivalent:

```bash
uv sync
uv run python pension_drawdown_simulator.py
uv run streamlit run ifa_web.py
```

## Documentation

- Getting started guide: `docs/START_HERE.md`
- Project structure and data flow: `docs/ARCHITECTURE.md`
- Import and dependency diagrams: `docs/DEPENDENCIES.md`


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
- Pot composition charts (stacked and per-pot subplots), with a short guide
  before the charts to explain what each view does.
- Plain-English explanation of scenario impact.
- Numbered event markers with a right-side notes panel in age order, so charts
  stay cleaner and event details remain easy to read.
- Compact metrics styling for easier viewing on smaller laptop screens.
- Collapsible sidebar sections for DB pensions and life events to reduce
  scrolling in the control panel.
- Theme-aware text styling so sidebar labels and expand/collapse controls stay
  readable in dark mode.
- Streamlit theme-variable-based styling so both text contrast and background
  update consistently across main content and sidebar when switching themes.

## Output Files

- CLI saves charts to `output/` by default.
- Streamlit can optionally save its generated charts to `output/` using the
  sidebar checkbox.
