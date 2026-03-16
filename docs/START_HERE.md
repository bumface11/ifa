# Start Here

This project has three ways to run the simulator.

## 1) Command-Line Version

Use this when you want a scripted run that always produces chart files.

```bash
uv run python pension_drawdown_simulator.py
```

Outputs are written to `output/`.

## 2) Streamlit Dashboard

Use this when you want interactive sidebar controls for assumptions and life events.

```bash
uv run streamlit run ifa_web.py
```

In the web app:

- Edit ages, pots, return assumptions, and Monte Carlo count.
- Add or remove DB pensions.
- Add lump-sum and spending-step life events.
- Use the preset controls at the top of the sidebar:
  - `New`: start a fresh preset name without writing a file yet.
  - `Save`: save current inputs to the selected preset, including rename if
    you edited `Preset name`.
  - `Save As`: always create a new preset file from current inputs.
  - `Delete`: remove the selected preset.
- Selecting a preset from the list loads it automatically.
- If current inputs differ from the loaded preset, switching presets prompts
  you before loading so unsaved edits are not lost accidentally.
- Click `Run simulation` to refresh metrics, explanations, and charts.
- Optionally enable PNG saving in the sidebar.

## 3) Chat UI Version

Use this when you want to explore scenarios through natural-language "what if"
questions with inline chart responses.

```bash
uv run streamlit run ifa_chat.py
```

In the chat app:

- Describe your retirement situation in plain English, e.g.:
  - *"I'm 55 with a £300k DC pot and £50k tax-free."*
  - *"DB pension of £8,000/year from age 66."*
  - *"I spend about £22,000 a year."*
- Type **run** to run the simulation and see the main charts inline.
- Ask **"what if"** questions to add life events:
  - *"What if I need £18,000 for a roof at age 70?"*
  - *"What if care costs start at £6,000/year from age 80?"*
- Request specific views: *"Which pot runs out first?"*,
  *"What about risk?"*, *"Sequence of returns"*, *"Show everything"*.
- Type **help** to see the full list of understood phrases.
- Type **reset** to start over with default parameters.



- Baseline scenario: spending plan with no added life events.
- Life-events scenario: baseline plus one-off or ongoing extra spending.
- Required withdrawals are always net of DB pension income to avoid
  double-counting.
