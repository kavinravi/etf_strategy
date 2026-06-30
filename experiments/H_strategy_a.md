# H — Strategy A (rung 1) beats buy-and-hold SPY

Pre-registered 2026-06-29, **before** any full-history run. No retroactive bar relaxation.

## Fixed search box
- Config: `experiments/configs/strategy_a.json` (xgb_params; tilt_scale=0.5,
  max_weight=0.18, min_weight=0.02, no_trade_band=0.02, cost_bps=5).
- Universe: 11 SPDR sector ETFs, expanding 9→11 (XLRE 2015, XLC 2018).
- Walks: expanding, test years 2005–2025; train start 2002; sealed test years.
- Single XGBoost forecaster → log-tilt around equal-weight anchor under the cap.

## Acceptance bars (ALL must hold, net of 5 bps)
1. OOS Sharpe(A) > Sharpe(SPY) over the full 2005–2025 series.
2. OOS max_drawdown(A) ≥ max_drawdown(SPY) (no deeper than SPY).
3. Label-shuffle median Sharpe < Sharpe(A) — the signal is real, not leakage/overfit.
4. Block-bootstrap 95% Sharpe CI lower bound > Sharpe(SPY) point estimate.
5. Dirichlet random-weights null p-value < 0.10 — allocation skill, not luck.
6. Seed stability: the **minimum** Sharpe across seeds 0–9 still > Sharpe(SPY).

## Decision rule
If all six bars hold, Strategy A becomes benchmark **rung 1** (the bar rung 2/Strategy B
must then beat). If any bar fails, A does **not** advance: record which bar failed and
diagnose before any tuning. Changing the config after seeing test results invalidates
the pre-registration — start a new H-file with a new fixed search box.

## Notes
- `first_test_year=2005` assumes ≥3 years of clean 9-sector history to train on; adjust
  only if the price source returns a shorter SPDR history (record the change here).
- Deflated-Sharpe `n_trials` = the honest number of configs evaluated across the whole
  research line, not just this run.
