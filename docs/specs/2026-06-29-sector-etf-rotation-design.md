# Sector-ETF Rotation Strategy — Design Spec

**Date:** 2026-06-29
**Status:** Draft for review
**Predecessors:** `axiom_tilt` (research paper), `axiom_tilt_strategy` (production). This repo (`etf_strategy`) is a new, tax-aware line.

---

## 1. Motivation

`axiom_tilt_strategy` (live ~5 weeks) is too volatile: ~18-20% realized vol and a ~4.9% drawdown already. It selects from a 500-name universe, churns weekly, and has no defensive escape — its market beta is ~1 by construction and its only risk control is a 10% per-name cap.

This project reframes the problem to remove the parts that hurt:

- **Selection is not the edge.** Hold a small, fixed band of **11 SPDR sector ETFs** (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE, XLC). No universe churn → structurally low turnover → tax-efficient.
- **Defense is explicit, and learned.** The "safe-haven" is a model-learned *overweight into defensive sectors* (health care / staples / utilities) when conditions warrant — not a bond sleeve (deferred) and not a hand-coded rule.
- **Not trapped in a fixed short hold.** A no-trade band + monthly cadence means the strategy mostly holds and only trades when the signal is strong, rather than forcing a weekly decision.

**Success metric:** risk-adjusted return that actively punishes deep drawdowns — Sharpe with an explicit drawdown term (Sharpe reported alongside Calmar/MDD; a Sharpe-with-MDD-penalty used for model/config selection). Not "preservation first" — upside is wanted.

## 2. The benchmark ladder (the spine of the project)

Complexity is added one rung at a time, and **each rung must beat every lower rung on the same harness** (annualized return, vol, MDD, Sharpe, with the drawdown-aware objective) or it does not ship. This makes every added layer answer: *"does the extra work justify itself?"*

| Rung | Strategy | Must beat | Status |
|---|---|---|---|
| 0 | Buy-and-hold SPY (total return) | — | benchmark |
| 1 | **Strategy A** — single gradient-boosting (XGBoost) TAA over 11 sectors | SPY | **this spec** |
| 2 | **Strategy B** — MVO + Black-Litterman optimizer, LSTM-forecast views | SPY, A | future |
| 3 | **Strategy C** — learned (RL) allocator | SPY, A, B | future |

All rungs share one backtest harness, one cost model, one metric set, and one validation protocol so the comparison is apples-to-apples.

This spec covers **rung 1 (Strategy A) in full**, plus the **shared infrastructure** (data, harness, validation) that every rung uses.

## 3. Universe & data

### 3.1 Universe — expanding 9 → 11 sectors

All 11 ETFs only coexist from 2018 (XLRE launched 2015, XLC 2018) — too short for an ML walk-forward. We therefore use the **full available history with an expanding universe**: the 9 original Select Sector SPDRs from inception (~1998-1999), with XLRE entering in 2015 and XLC in 2018. Pre-launch, those sectors simply don't exist in the band (real estate was inside XLF, comm-services inside XLK/XLY); we make **no attempt to reconstruct** synthetic pre-launch series — we use the ETFs as they actually traded.

The model is **pooled cross-sectional** (one row per sector-month, sector-agnostic features), so a changing sector count is handled transparently and training data is pooled across all sectors × months.

### 3.2 Data

- **Prices:** daily **total-return-adjusted** close for the 11 sector ETFs + SPY. Default source: yfinance / Stooq (adjusted close handles dividends & splits). Optional: Sharadar SEP (already licensed from predecessors) for vendor-clean dividend-adjusted ETF prices. *(Decision: confirm source at spec review.)*
- **Macro features:** FRED — at minimum VIXCLS, DGS3MO, DGS10, T10Y2Y (reuse `axiom_tilt/src/data/ingest_macro.py` near-verbatim). Extendable (credit spreads, etc.).
- These ETFs all still trade → no survivorship issue. PIT discipline still applies to feature/label timing (§5, §6).

## 4. Shared backtest harness & metrics

Reuse, adapting from the predecessors rather than rewriting:

- **Metrics:** `compute_strategy_metrics` from `axiom_tilt/src/utils/backtest.py` — Sharpe, Sortino, ann. return/vol, MDD, Calmar, hit-rate, turnover. **Annualization changes from 52 (weekly) to 12 (monthly).** Sharpe is geometric-ann-return / ann-vol (consistent across all rungs).
- **Weight→return application & cost:** deterministic rebalance loop; cost charged on L1 turnover. **5 bps** default (carried from predecessors), treated as a tunable; tax drag tracked separately via turnover.
- **Allocation primitive:** `project_to_simplex` (water-fill cap) from `axiom_tilt/src/utils/rl_env.py`, **extended to support a floor** so weights land in `[floor, cap]` and sum to 1.
- **Walk-forward:** expanding-window, sealed test years (template: `axiom_tilt/experiments/run_all_walks.py`, `backtest_full_period.py`). Concatenate per-walk test segments into one OOS series for headline metrics.
- **Research discipline:** pre-registered hypotheses (`H_*.md` pattern from `axiom_tilt_strategy/experiments/`) — fixed search box, acceptance bars declared *before* running, no retroactive bar relaxation.

## 5. Strategy A (rung 1) design

A single gradient-boosting model produces a tactical tilt around an equal-weight anchor; a no-trade band controls turnover.

### 5.1 Cadence & flow

Monthly (default; tunable). Each rebalance date `t`:
1. Build features for each existing sector from data **strictly ≤ t**.
2. GBM scores each sector.
3. Map scores → target weights in `[floor, cap]` (§5.4).
4. Apply no-trade band vs current weights (§5.5); trade only the sectors that breach it.
5. Hold to `t+1`; realize returns net of turnover cost.

### 5.2 Model & target

- **Model:** XGBoost (gradient-boosting family), pooled cross-sectional (row = sector-month). Single model — no LSTM, no optimizer, no Black-Litterman at this rung.
- **Target (label):** forward **risk-adjusted / rank** return over the next month (next ~21 trading days) — cross-sectional rank or excess-vs-universe-mean, *not* raw return (more stable). Label uses only forward data from the rebalance date; never an input feature.
- **Why GBM not LSTM:** ~26 yrs of monthly data ≈ ~300 obs/sector is starvation rations for a sequence model; GBM is sample-efficient on tabular features and captures temporal structure via engineered momentum/vol features.

### 5.3 Features (mechanical, minimal hand-design)

We inject *features*, not *rules* — the model learns the interactions. Per sector, computed PIT:
- **Momentum** at several lookbacks (e.g. 1, 3, 6, 12 months), 12-1 skip-month variant.
- **Realized volatility** at several windows; current drawdown.
- **Relative strength vs SPY** (sector return − market return) at several lookbacks.
- **Macro regime** (shared across sectors): VIX, term spread, short/long yields, etc., lagged.

Defensive overweighting emerges *only if* the data says defensives pay in stress — it is not hard-coded.

### 5.4 Allocation mapping (forecast → weights, stabilized)

Shrink toward the equal-weight anchor — a poor-man's Black-Litterman, most of the stability with none of the machinery:
- Anchor `w0_i = 1/N` (N = sectors existing at `t`).
- Cross-sectional score `s_i` (z-scored model output).
- **Log-tilt around anchor:** `w_raw_i = w0_i · exp(tilt_scale · s_i)` (reuses the predecessor's `baseline_anchor` idea; `tilt_scale = 0` ⇒ exact equal-weight).
- Project to `[floor, cap]`, sum 1, via the extended water-fill. Defaults: **cap 18%** (tunable in [15%, 20%]), **floor ~2%** (ensures all sectors held). `tilt_scale` and the optional vol-scaling of the tilt are tunables.

### 5.5 Turnover / tax control

- **No-trade band:** rebalance a sector only when |target − current| > band (e.g. 1-3%); otherwise hold. This is the implicit "fire detector" — the strategy acts only when the signal is strong enough.
- Monthly cadence + band + small `tilt_scale` keep turnover low → tax-efficient. Turnover and an estimated tax-drag proxy are reported every run.

## 6. Validation & reporting — one notebook per strategy

Every strategy (rung) gets its **own notebook** that prints results **and** rigorously stress-tests them. A shared `src/utils` validation library backs all notebooks (template: `axiom_tilt/notebooks/10_robustness_checks.ipynb`).

### 6.1 Headline results
Full-period OOS equity curve + per-metric table vs all lower rungs; per-walk / per-year breakdown.

### 6.2 Data-leakage tests (rigorous)
- **PIT assertions:** every feature at `t` uses only data ≤ `t`; labels strictly forward; train/test walks non-overlapping (programmatic checks that fail loudly).
- **Label-shuffle / permutation test:** retrain on shuffled labels — performance must collapse to null. Surviving performance ⇒ leakage or artifact.
- **Feature-timing audit:** shift each feature forward one period and confirm no performance *gain* (a gain implies look-ahead).

### 6.3 "Luck" tests via Monte Carlo (the explicit ask)
- **Seed stability:** train across N seeds; report the *distribution* of OOS Sharpe/MDD. Headline must not depend on a lucky seed (predecessors saw +0.315/+0.029/−0.018 across 3 seeds — exactly the trap to expose).
- **Stationary block bootstrap** (Politis-Romano, mean block ~3-6 mo, N≥5000) → confidence intervals on Sharpe/MDD; CI should exclude the relevant benchmark, not just 0.
- **Random-weights null** (Dirichlet over the 11 sectors under the same cap, incl. an HHI/concentration-matched null) → does the model's *allocation* beat random allocation in the same universe? Isolates allocation skill from universe drift.
- **Deflated / multiple-testing aware Sharpe** given we sweep configs (guard against selecting the luckiest config).

A run is only called a "win" if it beats the lower rungs **and** survives leakage + luck tests.

## 7. Code reuse plan (copy/adapt from predecessors)

| New file (approx) | Source | Change |
|---|---|---|
| `src/utils/backtest.py` | `axiom_tilt/src/utils/backtest.py` | annualization 52→12; keep metrics, min-variance (rung B) |
| `src/utils/allocate.py` (`project_to_simplex`) | `axiom_tilt/src/utils/rl_env.py` | add a floor to the water-fill |
| `src/data/ingest_macro.py` | `axiom_tilt/src/data/ingest_macro.py` | near-verbatim (FRED) |
| `src/data/ingest_prices.py` | new (thin) | yfinance/Stooq ETF adjusted close |
| walk-forward harness | `axiom_tilt/experiments/run_all_walks.py`, `backtest_full_period.py` | sectors instead of stocks; monthly |
| robustness notebook | `axiom_tilt/notebooks/10_robustness_checks.ipynb` | template for §6 |
| `H_*.md` discipline | `axiom_tilt_strategy/experiments/H_*.md` | same pattern |

The heavy stock-fundamentals stack (CRSP/WRDS/Sharadar SF1, EDGAR, FinBERT, the LightGBM ranker) is **not** reused — ETFs don't have those inputs.

## 8. Repo structure (proposed)

```
src/
  data/      ingest_prices.py, ingest_macro.py, build_panel.py (sector-month panel)
  features/  feature builders (momentum, vol, rel-strength, macro)
  strategy/  strategy_a.py (GBM tilt), allocate.py, constants.py
  utils/     backtest.py, walkforward.py, validation.py (leakage + MC), io.py
notebooks/
  strategy_a.ipynb          # results + leakage + MC for rung 1
  (strategy_b.ipynb, ...)   # future rungs
experiments/
  configs/   per-experiment JSON
  H_*.md     pre-registered hypotheses
docs/specs/                 this spec + future
data/processed/             sector-month panel, prices, macro
```

## 9. Out of scope / future

- **Rung B:** MVO + Black-Litterman allocator, with an **LSTM return-forecast engine** producing the BL "views." Tests whether the optimizer/stabilizer layer beats rung A's simple mapping.
- **Rung C:** RL allocator (reuse `rl_env.py`), reward = Sharpe − downside penalty.
- **Optional later levers:** a one-ticker cash escape (e.g. BIL) to lift the drawdown ceiling; a cap-breach-on-stress mode; weekly/biweekly cadence variants.

## 10. Open tunables / decisions to confirm

- Price data source: yfinance/Stooq (default) vs Sharadar SEP.
- Rebalance cadence: monthly (default) vs biweekly.
- Cap (default 18%, range 15-20%), floor (~2%), no-trade band (1-3%), `tilt_scale`.
- Label definition: cross-sectional rank vs excess return; horizon (~1 month).
- Cost assumption: 5 bps (default) — confirm for ETF execution.
