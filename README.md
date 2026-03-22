# IFA Pension Drawdown Simulator

A beginner-friendly pension simulator that shows how withdrawals, market returns,
DB pension income, and life events can change a retirement plan.

## BBC Radio Drama Cache

The `radio_cache/` package provides a cloud-hostable cache of BBC Radio drama
programme metadata with a modern search interface.

### Features

- **Daily refresh** from BBC Sounds feeds via GitHub Actions.
- **Full-text search** across titles, synopses, series, and categories.
- **Series grouping** -- episodes are grouped by parent series, with episode
  numbering for serialisations.
- **Brand hierarchy** -- series are grouped under their parent brands.
- **REST API** (FastAPI) for programmatic access at `/api/search`,
  `/api/series`, `/api/programme/{pid}`, and `/api/stats`.
- **Web UI** for searching and browsing, with one-click `get_iplayer` command
  copying for local downloads.
- **Static JSON export** (`radio_cache_export.json`) for cheap static hosting
  on GitHub Pages or similar.

### Quick Start -- Radio Cache

```bash
# Install with radio extras
pip install -e ".[radio]"

# Refresh the cache (fetches from BBC feeds)
python -m radio_cache.refresh --verbose

# Or import from a JSON export
python -m radio_cache.refresh --import-json radio_cache_export.json

# Start the web search UI
uvicorn radio_cache_api:app --reload
```

Open `http://localhost:8000` to search and browse programmes.  Each programme
shows a copyable `get_iplayer` command for local download.

### Hosting Options (Cheap/Free)

| Option | Cost | Notes |
|---|---|---|
| **Render free tier** | Free | Deploy `radio_cache_api.py`; spins down on idle |
| **Fly.io** | Free tier | 3 shared VMs free |
| **Railway** | Free trial | Simple Docker deploy |
| **GitHub Pages** | Free | Host `radio_cache_export.json` as static file |
| **GitHub Actions** | Free | Daily cache refresh via cron workflow |

### Downloading Programmes

Find a programme in the web UI or JSON export, then use:

```bash
get_iplayer --pid=<PID> --type=radio
```

Requires [get_iplayer](https://github.com/get-iplayer/get_iplayer) installed
locally.

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

- Baseline vs scenario line chart on the same market path.
- Sequence-of-returns teaching chart.
- Monte Carlo fan chart with ruin probability.
- Pot composition charts (stacked and per-pot subplots), with a short guide
  before the charts to explain what each view does.
- The individual-pots view shows tax-free, combined DC, and total-pot balance
  panels, plus an annual spending versus DB income panel showing the gap that
  still needs to come from pots.
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

## Output Files

- CLI saves charts to `output/` by default.
- Streamlit can optionally save its generated charts to `output/` using the
  sidebar checkbox.
