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

## 3) Chat Interface

Use this for a conversational "what if" experience where you type questions
and the app updates the scenario, runs the simulation, and renders the
relevant charts inline.

```bash
uv run streamlit run ifa_chat.py
```

Example questions:

- *"I'm 55"* — set your current age
- *"Retire at 60"* — change DC pot drawdown start age
- *"DC pot £300,000"* — update your pension pot balance
- *"DB pension £8,000/year from age 66"* — add a defined-benefit stream
- *"House repairs £18,000 at age 70"* — add a one-off spending event
- *"Care costs £6,000/year from age 80"* — add an ongoing spending step
- *"Run it"* / *"Show me"* — simulate and display charts with explanation
- *"Show me which pot drains first"* — pot breakdown view
- *"What if markets crash early?"* — sequence-of-returns chart
- *"How worried should I be?"* — Monte Carlo fan chart
- *"Show my assumptions"* — summarise current scenario
- *"Start over"* — reset to defaults

The chat interface uses the same `ifa/` engine, events, metrics, explain,
and plotting modules as the dashboard — only the UI layer is different.

## Concept Notes

- Baseline scenario: spending plan with no added life events.
- Life-events scenario: baseline plus one-off or ongoing extra spending.
- Required withdrawals are always net of DB pension income to avoid
  double-counting.
