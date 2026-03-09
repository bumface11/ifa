# Start Here

This project has two ways to run the simulator.

## 1) Command-Line Version

Use this when you want a scripted run that always produces chart files.

```bash
uv run python pension_drawdown_simulator.py
```

Outputs are written to `output/`.

## 2) Streamlit Version

Use this when you want interactive controls for assumptions and life events.

```bash
uv run streamlit run ifa_web.py
```

In the web app:

- Edit ages, pots, return assumptions, and Monte Carlo count.
- Add or remove DB pensions.
- Add lump-sum and spending-step life events.
- Click `Run simulation` to refresh metrics, explanations, and charts.
- Optionally enable PNG saving in the sidebar.

## Concept Notes

- Baseline scenario: spending plan with no added life events.
- Life-events scenario: baseline plus one-off or ongoing extra spending.
- Required withdrawals are always net of DB pension income to avoid
  double-counting.
