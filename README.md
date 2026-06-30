# etf_strategy

Tax-aware sector-rotation strategy over the 11 SPDR sector ETFs (XLB, XLE, XLF, XLI,
XLK, XLP, XLU, XLV, XLY, XLRE, XLC).

Long-only, always invested, weights tilted around an equal-weight anchor under a
per-sector cap (15-20%). Defensive positioning (overweighting health care / staples /
utilities in stress) is learned by the model, not hand-coded.

Strategies are built as a **benchmark ladder** — each rung must beat every lower rung
on the same backtest harness (Sharpe with a drawdown term, at low turnover):

0. Buy-and-hold SPY
1. Strategy A — gradient-boosting tactical tilt
2. Strategy B — mean-variance + Black-Litterman (LSTM-forecast views)
3. Strategy C — RL allocator

Each strategy gets its own notebook reporting results plus data-leakage and
Monte-Carlo "luck" tests.

Design spec: [`docs/specs/2026-06-29-sector-etf-rotation-design.md`](docs/specs/2026-06-29-sector-etf-rotation-design.md)
