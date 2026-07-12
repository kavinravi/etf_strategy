# etf_strategy

Tax-aware sector-rotation strategy over the 11 SPDR sector ETFs
(XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE, XLC).

Long-only, always invested, weights tilted around an equal-weight anchor under a
per-sector cap (roughly 15–20%). Defensive positioning (overweighting health care /
staples / utilities in stress) is **learned by the model, not hand-coded**.

## Benchmark ladder

Strategies are built as a ladder — each rung must beat every lower rung on the same
backtest harness (Sharpe with a drawdown term, at low turnover):

| Rung | Strategy                                                | Status  |
| ---- | ------------------------------------------------------- | ------- |
| 0    | Buy-and-hold SPY                                        | baseline |
| 1    | Strategy A — gradient-boosting tactical tilt (XGBoost)  | in dev   |
| 2    | Strategy B — mean-variance + Black-Litterman (LSTM views) | planned |
| 3    | Strategy C — RL allocator                               | planned  |

Each strategy gets its own notebook reporting results plus data-leakage and
Monte-Carlo "luck" tests.

## Repository layout

```
src/etf/
  data/        # price + macro ingestion, monthly panel build
  features/    # feature engineering (price + macro), label construction
  strategy/    # allocator, constants, per-strategy models (strategy_a.py …)
  utils/       # walk-forward, backtest metrics, benchmarks, validation tests
tests/         # pytest suite mirroring src/etf/ layout
notebooks/     # per-strategy reporting notebooks (e.g. strategy_a.ipynb)
experiments/   # pre-registered H-files + JSON configs (frozen search boxes)
docs/          # design specs and rung plans
data/          # cached processed panels (gitignored)
```

## Getting started

Requires **Python 3.12+** (see `.python-version`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # exposes the `etf` package to notebooks/tests
```

## Running tests

Network-hitting tests are marked `live` and skipped by default.

```bash
pytest                      # fast, offline unit tests
pytest -m live              # also run tests that hit yfinance / FRED
```

## Running Strategy A

Reproduces the rung-1 acceptance run end-to-end (data pull → features → walk-forward
backtest → validation tests → report).

```bash
jupyter lab notebooks/strategy_a.ipynb
```

The pre-registered acceptance bars live in
[`experiments/H_strategy_a.md`](experiments/H_strategy_a.md) and the frozen config in
[`experiments/configs/strategy_a.json`](experiments/configs/strategy_a.json).
Changing either after seeing test results invalidates the pre-registration.

## Design docs

- Design spec: [`docs/specs/2026-06-29-sector-etf-rotation-design.md`](docs/specs/2026-06-29-sector-etf-rotation-design.md)
- Strategy A rung-1 plan: [`docs/plans/2026-06-29-strategy-a-rung1.md`](docs/plans/2026-06-29-strategy-a-rung1.md)
