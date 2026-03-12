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

## DC Pots

The simulator now supports multiple DC pots, each with its own drawdown start age.

- Primary DC pot drawdown start defaults to age `57`.
- Additional DC pots can start later (for example age `65`).
- Both CLI and Streamlit use the same per-pot drawdown rules.
- DC pots continue to compound with market returns while they remain above zero,
  including after drawdown eligibility starts.

In Streamlit, open `DC Pot Inputs` in the sidebar to add or edit multiple DC
pots and start ages.

## What The App Shows

- Baseline vs scenario line chart on the same market path.
- Sequence-of-returns teaching chart.
- Monte Carlo fan chart with ruin probability.
- Pot composition charts (stacked and per-pot subplots), with a short guide
  before the charts to explain what each view does.
- The individual-pots view shows tax-free, combined DC, and total-pot balance
  panels, plus a dedicated DB income timeline panel.
- The first four charts include a secondary axis for spending drawdown
  (spending requirement minus DB income).
- Plain-English explanation of scenario impact.
- Numbered event markers with a right-side notes panel in age order, so charts
  stay cleaner and event details remain easy to read.
- In the individual 4-panel pot chart, event number markers are shown on all
  four subplots.
- Compact metrics styling for easier viewing on smaller laptop screens.
- Collapsible sidebar sections for DB pensions and life events to reduce
  scrolling in the control panel.
- Collapsible `DC Pot Inputs` section supporting multiple DC pots with per-pot
  drawdown start ages.
- Sidebar parameter presets that can be saved locally, loaded from a list, and
  renamed after creation.
- Editable name fields for each DC pot and each life event, with sensible
  default names.
- Editable name fields for each DB pension stream, with sensible defaults.
- Chart notes and plain-English explanations use your custom pot/event/DB names
  instead of generic labels.
- The sidebar saved-parameters section is placed at the top of the sidebar.
- The `Run simulation` button is placed near the top of the page for quicker
  access.
- Theme-aware text styling so sidebar labels and expand/collapse controls stay
  readable in dark mode.
- Streamlit theme-variable-based styling so both text contrast and background
  update consistently across main content and sidebar when switching themes.

## Output Files

- CLI saves charts to `output/` by default.
- Streamlit can optionally save its generated charts to `output/` using the
  sidebar checkbox.
