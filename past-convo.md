# etf_strategy — Conversation Handoff (2026-06-29)

This file hands off an in-progress build from a remote Claude Code session to a fresh
**local** Claude Code session on the Mac. Read it top to bottom, then resume at
**"How to resume"**. You can `git rm past-convo.md` once you've absorbed it.

---

## 1. What this project is

`etf_strategy` is a **tax-aware sector-ETF rotation strategy**. It's the third repo in a
lineage:

- `axiom_tilt` (research paper): text-enhanced RL portfolio (FinBERT → LightGBM ranker →
  PPO allocator). Honest result: text didn't beat plain fundamentals and RL barely beat
  trivial weighting.
- `axiom_tilt_strategy` (production successor): dropped text+RL; deterministic 2-factor
  screen (sales/price × FCF/assets) + a regime ML "concentration" selector; runs live on
  IBKR. Its weakness — and the motivation for THIS repo — is that it's **too volatile**
  (~18–20% vol, deep drawdowns) because it's always ~100% in single stocks with no
  defensive escape.

**This repo's idea** (decided with the user this session): stop making *selection* the job.
Hold a fixed band of the **11 SPDR sector ETFs** and allocate creatively. Lower vol, and
an *internal safe-haven* via a model-learned overweight into defensive sectors
(health/staples/utilities) when conditions warrant — no separate bond sleeve (deferred).

---

## 2. Strategy design (the spec)

Full spec: **`docs/specs/2026-06-29-sector-etf-rotation-design.md`**. Key points:

- **Universe:** 11 SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE,
  XLC), long-only, always invested. **Expanding 9→11**: the 9 original SPDRs go back to
  ~1999; XLRE enters 2015, XLC 2018 (so only 9 sectors exist pre-2015). Modeled as a
  **pooled cross-sectional panel** (one row per sector-month) so the changing count is
  handled and training data is pooled.
- **Per-sector weight band [~2% floor, ~18% cap].** The 10% cap was rejected because with
  11 names equal-weight is 1/11 = 9.09% — a 10% cap leaves no room for edge. 15–20% gives
  tilt room while capping concentration (a "dad-safe" guardrail). The cap is enforced by a
  water-fill simplex projection.
- **Success metric: Sharpe with a drawdown term** (not preservation-first — upside is
  wanted). Honest limitation acknowledged: with no cash/bond sleeve, market beta is ~1, so
  defensive-sector rotation softens but cannot dodge a broad crash.
- **Rebalance:** monthly, with a **no-trade band** (only trade a sector when its target
  moves > band from current) — this is the tax-friendly "fire only on real signal"
  mechanism, and frees the strategy from a forced weekly hold.
- **Costs:** 5 bps on turnover. **PIT discipline** throughout (features at month-end `t`
  use only data ≤ t; labels are forward-only; test years sealed).

### The benchmark ladder (the spine of the whole project — this was the user's idea)

Complexity is added one rung at a time; **each rung must beat every lower rung on the same
harness, or it doesn't ship.** "Does the extra work earn its keep?" answered with numbers.

| Rung | Strategy | Must beat | Status |
|---|---|---|---|
| 0 | Buy-and-hold SPY | — | benchmark |
| 1 | **Strategy A** — single XGBoost TAA tilt | SPY | **building now (this is the current job)** |
| 2 | Strategy B — MVO + Black-Litterman, **LSTM-forecast views** | SPY, A | future |
| 3 | Strategy C — RL allocator | SPY, A, B | future |

LSTM/MVO/Black-Litterman were explicitly **deferred out of rung 1**: the dataset is tiny
(~26 yrs monthly ≈ ~300 obs/sector — starvation for an LSTM), and naive MVO amplifies
forecast noise. Rung 1 stabilizes forecasts cheaply instead: predict cross-sectional
**ranks** (not raw returns), **shrink toward the equal-weight anchor** (a poor-man's
Black-Litterman: `weights = EW-anchor × exp(tilt_scale · z(score))`, projected into the
band), plus the no-trade band. BL+MVO becomes the *substance* of rung 2, where its value is
measurable against rung 1.

---

## 3. Strategy A (rung 1) — how it works

`src/etf/strategy/strategy_a.py`:
- `train_and_score(train_df, test_df, feature_cols, params, seed)`: fits one
  `XGBRegressor` on the `y_rank` label (per-date percentile rank of next-month return),
  scores the test rows.
- `run_strategy_a(features, walks, config, seed)`: per expanding walk, train on the train
  mask, score the test year; per test month: `tilt_weights(scores)` → target weights in
  the band; `apply_no_trade_band` vs the prior month; realize `sum(weight · fwd_ret_1m)`;
  accumulate turnover. Returns `{weights, returns, turnover, scores}`.

Default config: `experiments/configs/strategy_a.json` (xgb_params; tilt_scale=0.5,
max_weight=0.18, min_weight=0.02, no_trade_band=0.02, cost_bps=5, test years 2005–2025).

---

## 4. What's DONE (Tasks 1–10 + the hypothesis file)

All committed on branch **`strategy-a-rung1`** (pushed to `github.com/kavinravi/etf_strategy`).
**All 23 tests pass on the Mac in ~6s.** Tasks were executed with the
`superpowers:subagent-driven-development` flow (fresh implementer + spec/quality reviewer
per task) against the plan.

| Task | What | File(s) |
|---|---|---|
| 1 | Scaffolding + constants (universe, inception dates, band/cost defaults) | `pyproject.toml`, `src/etf/strategy/constants.py` |
| 2 | ETF price ingestion + monthly resampling (yfinance, total-return adj) | `src/etf/data/ingest_prices.py` |
| 3 | Macro ingestion (FRED via pandas-datareader) + monthly align | `src/etf/data/ingest_macro.py` |
| 4 | Month-end PIT panel + forward-return label (`fwd_ret_1m`), expanding universe | `src/etf/data/build_panel.py` |
| 5 | PIT price features (momentum, vol, drawdown, relative strength) — has a truncation test proving no look-ahead | `src/etf/features/build_features.py` |
| 6 | Feature-matrix assembly + cross-sectional rank label (`y_rank`) | `src/etf/features/build_features.py` |
| 7 | Allocation: floor+cap water-fill projection, log-tilt, no-trade band | `src/etf/strategy/allocate.py` |
| 8 | Monthly backtest metrics (Sharpe/Sortino/MDD/Calmar/…) | `src/etf/utils/backtest.py` |
| 9 | Expanding walk-forward splits + SPY/equal-weight benchmarks | `src/etf/utils/walkforward.py`, `src/etf/utils/benchmarks.py` |
| 10 | Strategy A model + walk-forward backtest runner | `src/etf/strategy/strategy_a.py` |
| — | Pre-registered hypothesis (6 acceptance bars, written before any backtest) | `experiments/H_strategy_a.md` |

**Two real engine bugs were caught in review of Task 10 and fixed** (the runner is
alignment-sensitive — keep this in mind if you touch it):
1. On a universe change, the no-trade band + renormalization could push a sector above the
   18% cap. Fixed: band runs over the current investable set only; no post-hoc renorm.
2. On a ticker *exit*, `cur` summed to <1 and the band's `hold.all()` shortcut returned a
   sub-1 book. Fixed: normalize `cur` before the band (a no-op for the real
   monotonically-expanding universe; defensive only). A universe-change test guards both.

---

## 5. What REMAINS — Tasks 11, 12, 13

**The full, ready-to-implement code is in `docs/plans/2026-06-29-strategy-a-rung1.md`**
(complete functions + tests for 11/12; notebook spec for 13). Do these with
subagent-driven-development (or directly — your call), verifying tests as you go.

- **Task 11 — Leakage tests** (`src/etf/utils/validation.py` + `tests/utils/test_validation_leakage.py`):
  `label_shuffle_sharpe` (retrain on shuffled labels → performance must collapse) and
  `feature_timing_gain` (shift a feature forward → a Sharpe gain implies look-ahead).
- **Task 12 — Monte-Carlo luck tests** (append to `validation.py` + `tests/utils/test_validation_mc.py`):
  `seed_stability` (Sharpe distribution across seeds — catches lucky-seed results),
  `block_bootstrap_ci` (stationary bootstrap CI on Sharpe), `dirichlet_null` (random
  weights under the cap → is allocation skill real?), `deflated_sharpe` (multiple-testing
  correction). `scipy` is already in `pyproject.toml` for this.
- **Task 13 — Results notebook** (`notebooks/strategy_a.ipynb`): loads real data
  (yfinance + FRED — **needs network**), runs Strategy A + SPY + equal-weight over
  2005–2025, prints the metrics table + equity curves, then runs the leakage + MC checks
  and evaluates each of the 6 acceptance bars in `H_strategy_a.md` as PASS/FAIL. Use the
  `ipynb` skill. The predecessor's `notebooks/10_robustness_checks.ipynb` is the template.

After 11–13: run the final whole-branch code review, then
`superpowers:finishing-a-development-branch` (likely open a PR `strategy-a-rung1` → `main`).

**Watch-outs for 11/12:** the plan's tests import a helper via
`from tests.strategy.test_strategy_a import _synth_features` — confirm that import resolves
under your pytest config (if not, lift `_synth_features` into a `conftest.py` fixture).
The XGBoost-dependent validation tests are slow only under CPU contention; on the Mac they're fast.

---

## 6. Working conventions & preferences (IMPORTANT — carry these forward)

- **No AI attribution anywhere.** No `Co-Authored-By`/`Generated with`/session trailers in
  commits, and no AI mentions in code/docs/PRs. (`past-convo.md` itself is a handoff
  artifact — fine to delete.)
- **Commit messages:** brief, practical, imperative, like a dev. No `feat:`/type prefixes.
- **Specs live in `docs/specs/`, plans in `docs/plans/`** (NOT `docs/superpowers/…`).
- **Pushing requires SSH:** remote is `git@github.com:kavinravi/etf_strategy.git`. The
  HTTPS token and `gh` token are dead. (On the Mac your SSH key works.)
- **Research rigor is the point** (inherited from the predecessors): strict PIT, sealed
  walk-forward, pre-registered hypotheses with fixed acceptance bars and no retroactive
  relaxation, and null/bootstrap robustness. A result only "wins" if it beats the lower
  rung AND survives the leakage + luck checks. Don't p-hack; don't tune on test years.
- Branch: **`strategy-a-rung1`** (off `main`). `main` has only the spec + plan.

---

## 7. How to resume (paste this to your local Claude Code session)

> Continue executing `docs/plans/2026-06-29-strategy-a-rung1.md` with subagent-driven
> development. Read `past-convo.md` first for full context. Tasks 1–10 and
> `experiments/H_strategy_a.md` are done, committed on branch `strategy-a-rung1`, and all
> 23 tests pass. Implement and verify **Task 11 (leakage tests), Task 12 (Monte-Carlo luck
> tests), Task 13 (results notebook)**, then run the final whole-branch review and finish
> the branch. Follow the conventions in past-convo.md §6 (no AI attribution, brief commit
> messages, specs in docs/specs, SSH remote).

Environment is already set up if `pytest` is green. Task 13 needs network for data.

---

## 8. Current git state
- Branch `strategy-a-rung1`, latest commit `Add scipy dependency (needed by Monte-Carlo
  validation)` plus this handoff commit.
- 23 tests pass (~6s on M1). Tree clean.
- Remaining: Tasks 11, 12, 13 → then review → then finish/PR.
