# IFA Pension Drawdown Simulator

A beginner-friendly pension simulator that shows how withdrawals, market returns, and life events can change your retirement plan.

## Life Events

Life events let you model real-world spending changes in simple terms:

- `LumpSumEvent(age, amount)`: one-off extra spending at one age.
- `SpendingStepEvent(start_age, extra_per_year, end_age=None)`: ongoing extra yearly spending from a start age.

Examples:

- House repairs lump sum: `LumpSumEvent(age=70, amount=18000)`
- Care costs step-up: `SpendingStepEvent(start_age=78, extra_per_year=6000)`

The simulator compares a baseline plan (no events) against a life-events scenario on the same return path so the impact is easy to understand.
