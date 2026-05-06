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

Run the Streamlit dashboard:

```bash
streamlit run ifa_web.py
```

Run the new conversational chat UI:

```bash
streamlit run ifa_chat.py
```

### Optional: uv Workflow

If you prefer `uv`, these commands are equivalent:

```bash
uv sync
uv run python pension_drawdown_simulator.py
uv run streamlit run ifa_web.py
uv run streamlit run ifa_chat.py
```

## Chat Interface

`ifa_chat.py` provides a conversational "what if" interface alongside the original
dashboard.  Ask natural-language questions and see inline charts in response:

- *"I'm 55 with a £300k DC pot and £50k tax-free. DB pension £8k/year from 66."*
- *"What if I need £18,000 for a new roof at age 70?"*
- *"What if care costs start at £6,000/year from age 80?"*
- *"Which pot runs out first?"*
- *"What happens if markets crash early?"*

Type `run` to run the simulation, `help` to see all understood phrases, or
`reset` to start over.  The original parameter-driven dashboard (`ifa_web.py`)
remains available for fine-grained control.

## Documentation

- Getting started guide: `docs/START_HERE.md`
- Project structure and data flow: `docs/ARCHITECTURE.md`
- Import and dependency diagrams: `docs/DEPENDENCIES.md`


## Chat Interface

`ifa_chat.py` provides a conversational "what if" experience alongside the
existing dashboard.  Instead of configuring parameters in a sidebar, you type
natural-language questions and the app parses your intent, updates the
scenario, runs the simulation, and renders the relevant charts inline.

Example questions you can ask:

- *"I'm 55"* — set your current age
- *"Retire at 60"* — change the DC pot drawdown start age
- *"DC pot £300,000"* — update your pension pot balance
- *"DB pension £8,000/year from age 66"* — add a defined-benefit stream
- *"House repairs £18,000 at age 70"* — add a one-off event
- *"Care costs £6,000/year from age 80"* — add an ongoing spending step
- *"Run it"* / *"Show me"* — simulate and display charts with explanation
- *"Show me which pot drains first"* — pot breakdown view
- *"What if markets crash early?"* — sequence-of-returns chart
- *"How worried should I be?"* — Monte Carlo fan chart
- *"Show my assumptions"* — display current scenario setup
- *"Start over"* — reset to defaults

Run it with:

```bash
streamlit run ifa_chat.py
```

The `ifa/` package is unchanged — `ifa_chat.py` is a thin conversational UI
layer that calls the same engine, events, metrics, explain, and plotting
functions used by `ifa_web.py`.

## Age Inputs

In the Streamlit dashboard, ages are now split into two controls:

- `Model start age`: first age shown in the timeline.
- `Drawdown start age`: first age when withdrawals are allowed.

Withdrawals are forced to zero between model start age and drawdown start age,
so this period represents pre-retirement accumulation (with market movement but
no spending drawdown).

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

- **Cumulative Flows Waterfall** — the opening chart, showing for each year:
  - Annual growth (green, above zero): investment return earned on the prior
    balance.
  - General drawdown (amber, below zero): regular spending withdrawn from pots
    (baseline spending net of DB income).
  - Discretionary drawdown (dark red, stacked below general): extra withdrawals
    driven by life events such as lump-sum costs or step-up spending.
  - A lower balance-trajectory panel provides running-total context.
- Baseline vs scenario line chart on the same market path.
- Sequence-of-returns teaching chart.
- Monte Carlo fan chart with ruin probability.
- Pot composition charts (stacked and per-pot subplots), with a short guide
  before the charts to explain what each view does.
- The individual-pots view shows tax-free, combined DC, and total-pot balance
  panels, plus an annual spending versus DB income panel showing the gap that
  still needs to come from pots.  When a tax regime is selected, the panel also
  draws a **Tax deducted** line showing the total UK income tax paid each year
  (DB income plus gross DC withdrawals), so users can see how tax bites
  alongside spending and DB income on the same annual-amount scale.
- The baseline vs scenario, sequence, fan, and stacked charts include a
  secondary axis for spending drawdown (spending requirement minus DB income).
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
- Sidebar parameter presets with streamlined controls: `New`, `Save`,
  `Save As`, and `Delete`.
- Selecting a preset from the list loads it automatically.
- If you switch presets with unsaved changes, the app prompts before loading
  the new selection.
- Editing `Preset name` and then clicking `Save` updates the selected preset
  and renames it in one step.
- Saved-preset comparison workspace: choose up to three named presets and view
  them either side by side or one at a time.
- `Compare saved presets` is a separate display mode, and in that mode the app
  shows only saved preset outputs rather than the current unsaved sidebar
  inputs.
- The focused comparison mode is intended for smaller screens where full
  side-by-side panels would be too cramped.
- The last run stays visible while you switch comparison layout or choose which
  preset to focus on.
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

## Tax on Drawdowns

The simulator models UK income tax on pension withdrawals so pot depletion
reflects real-world gross-vs-net dynamics.

### How it works

| Source | Tax treatment |
|--------|---------------|
| Tax-free pot | No tax — withdrawals fund spending directly |
| DB pension | Gross income; uses up personal allowance and lower bands |
| DC pot | Taxable — the engine grosses up withdrawals so you receive your *net* spending target after tax |

The **Baseline net spending** input is your target *take-home* amount after tax.
For every pound you need from a DC pot, the engine works out how many gross
pounds must actually leave the pot to cover both your spending and any income
tax due, taking into account how much your DB pension has already consumed of
the personal allowance and lower bands.

### Selectable tax regime

Use the **Tax bands** selector in the *3) Plan Basics* sidebar (or say
`"Scottish tax"` / `"Rest of UK tax"` in the chat) to choose:

| Option | Applies to |
|--------|------------|
| Rest of UK (England, Wales, N. Ireland) | HMRC standard bands |
| Scotland | Scottish Rate of Income Tax (SRIT) |

Both regimes use 2024/25 band boundaries and rates, defined as a simple table
in `ifa/tax.py` — easy to update each tax year.

### Tax deducted chart line

When a tax regime is selected, the **Annual Spending vs DB Income** panel (bottom-right of the individual-pots chart) also shows a **Tax deducted** line.  It plots the total UK income tax paid in each year — calculated on gross DB income plus the grossed-up DC withdrawal for that year — on the same annual-amount scale as spending and DB income.  This makes it easy to see when tax starts to bite materially as DB income and drawdown increase.

The line uses the same tax regime you chose in the sidebar (Rest of UK or Scotland).

### Simplifying assumptions

- **Personal allowance taper** (income > £100,000) is not modelled.
- **DB income** is treated as offsetting spending needs at its gross face value;
  the marginal tax paid on DB is not separately deducted from the withdrawal
  requirement.  This means the net-spending target you enter is accurate when DB
  income falls mostly within the personal allowance and is a slight under-estimate
  of required gross DC withdrawals when DB income significantly exceeds it.
- Band boundaries are held constant in real terms; no annual uprating is modelled.
- National Insurance is not included.

### Chat interface

In `ifa_chat.py`, say things like:

- *"Use Scottish tax bands"*
- *"Switch to rest-of-UK tax"*
- *"I pay Scottish income tax"*

The spending target you enter in the chat is your *net* spending need (what you
want to take home).


