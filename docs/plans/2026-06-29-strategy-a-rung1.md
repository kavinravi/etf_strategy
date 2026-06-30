# Strategy A (Rung 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tax-aware sector-ETF rotation strategy (Strategy A) — a single XGBoost model that tilts weights around an equal-weight anchor over the 11 SPDR sector ETFs — plus the shared backtest harness and validation tooling, and prove it beats buy-and-hold SPY on a sealed walk-forward with leakage and Monte-Carlo "luck" checks.

**Architecture:** A linear pipeline: ingest daily ETF + macro data → build a month-end sector panel with point-in-time (PIT) features and forward-return labels → train one pooled cross-sectional XGBoost per expanding walk → map scores to weights via log-tilt-around-anchor under a `[floor, cap]` projection with a no-trade band → run a deterministic monthly backtest → validate with leakage tests and Monte-Carlo robustness. Every component is a small, independently tested library function; the notebook only orchestrates and reports.

**Tech Stack:** Python 3.12, pandas, numpy, xgboost, scikit-learn, yfinance, pandas-datareader (FRED), pyarrow, matplotlib, pytest, jupyter.

## Global Constraints

These apply to **every** task:

- **Python 3.12.** Package installed editable (`pip install -e .`).
- **No AI attribution anywhere** — no "generated with"/co-author/trailers in commits, code, docs, or notebooks. Commit messages are brief, practical, imperative (e.g. `Add price ingestion`), no `feat:`/`type:` prefixes.
- **PIT discipline (hard rule):** every feature at month-end date `t` is computable from data with timestamp `≤ t`; labels use only data `> t`; the label column (`fwd_ret_1m`) is never a feature. Walk train/val/test windows never overlap; test years are sealed (no tuning on them).
- **Annualization = 12** (monthly) everywhere in metrics.
- **Defaults (in `src/strategy/constants.py`, all tunable):** `MAX_WEIGHT=0.18`, `MIN_WEIGHT=0.02`, `NO_TRADE_BAND=0.02`, `TILT_SCALE=0.5`, `COST_BPS=5.0`, `PERIODS_PER_YEAR=12`.
- **Universe:** 11 SPDR sector ETFs, expanding 9→11 as XLRE (2015-10) and XLC (2018-06) launch. Benchmark = SPY (total-return adjusted).
- **Determinism:** every model/MC routine takes an explicit `seed`. No `Math.random`-style unseeded calls.
- **No network in unit tests.** Ingestion network calls live in thin wrappers; all parsing/transform logic is pure and tested on in-memory fixtures. Live calls are covered by `@pytest.mark.live` tests, skipped by default.

---

## File Structure

```
pyproject.toml                     # package + deps
src/etf/
  __init__.py
  data/
    ingest_prices.py               # yfinance daily adj-close download
    ingest_macro.py                # FRED via pandas-datareader
    build_panel.py                 # month-end panel + ret_1m + fwd_ret_1m label
  features/
    build_features.py              # momentum / vol / drawdown / rel-strength / macro merge / label
  strategy/
    constants.py                   # universe, inception dates, defaults
    allocate.py                    # project_to_simplex (floor+cap), tilt_weights, no_trade_band
    strategy_a.py                  # label, XGBoost score, run_strategy_a backtest
  utils/
    backtest.py                    # compute_strategy_metrics (monthly)
    walkforward.py                 # expanding Walk splits
    benchmarks.py                  # buy_hold_spy, equal_weight
    validation.py                  # leakage tests + Monte-Carlo luck tests
    io.py                          # parquet load/save helpers
tests/
  data/ features/ strategy/ utils/ # mirrors src
notebooks/
  strategy_a.ipynb                 # results + leakage + MC report
data/processed/                    # cached panel/prices/macro (small; tracked)
experiments/
  configs/strategy_a.json          # default HP/alloc config
  H_strategy_a.md                  # pre-registered hypothesis + acceptance bars
```

---

## Task 1: Project scaffolding + constants

**Files:**
- Create: `pyproject.toml`, `src/etf/__init__.py`, `src/etf/strategy/__init__.py`, `src/etf/strategy/constants.py`, `tests/__init__.py`, `tests/strategy/test_constants.py`
- Create empty `__init__.py` in each `src/etf/<subpkg>/` and `tests/<subpkg>/`.

**Interfaces:**
- Produces: `constants.SECTOR_ETFS: list[str]` (11 tickers), `constants.SECTOR_INCEPTION: dict[str,str]`, `constants.BENCHMARK="SPY"`, `MAX_WEIGHT, MIN_WEIGHT, NO_TRADE_BAND, TILT_SCALE, COST_BPS, PERIODS_PER_YEAR`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "etf"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "pandas>=2.2", "numpy>=1.26", "xgboost>=2.0", "scikit-learn>=1.4",
  "yfinance>=0.2.40", "pandas-datareader>=0.10", "pyarrow>=15",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8", "jupyter>=1.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = ["live: hits external network (skipped by default)"]
addopts = "-m 'not live'"
```

- [ ] **Step 2: Create package `__init__.py` files and `constants.py`**

`src/etf/strategy/constants.py`:
```python
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]

# First full month each ETF traded (used to build the expanding universe).
SECTOR_INCEPTION = {
    "XLB": "1998-12-31", "XLE": "1998-12-31", "XLF": "1998-12-31",
    "XLI": "1998-12-31", "XLK": "1998-12-31", "XLP": "1998-12-31",
    "XLU": "1998-12-31", "XLV": "1998-12-31", "XLY": "1998-12-31",
    "XLRE": "2015-10-31", "XLC": "2018-06-30",
}
BENCHMARK = "SPY"

MAX_WEIGHT = 0.18
MIN_WEIGHT = 0.02
NO_TRADE_BAND = 0.02
TILT_SCALE = 0.5
COST_BPS = 5.0
PERIODS_PER_YEAR = 12
```

- [ ] **Step 3: Write the test**

`tests/strategy/test_constants.py`:
```python
from etf.strategy import constants as C

def test_universe_is_eleven_sectors():
    assert len(C.SECTOR_ETFS) == 11
    assert set(C.SECTOR_INCEPTION) == set(C.SECTOR_ETFS)

def test_cap_floor_are_feasible():
    n = len(C.SECTOR_ETFS)
    assert C.MIN_WEIGHT * n <= 1.0 <= C.MAX_WEIGHT * n  # a valid simplex exists
```

- [ ] **Step 4: Install and run**

Run: `pip install -e ".[dev]" && pytest tests/strategy/test_constants.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/etf tests/__init__.py tests/strategy
git commit -m "Add package scaffolding and strategy constants"
```

---

## Task 2: Price ingestion + monthly resampling

**Files:**
- Create: `src/etf/data/ingest_prices.py`, `tests/data/test_ingest_prices.py`

**Interfaces:**
- Produces:
  - `download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame` — daily adjusted close, index `DatetimeIndex` (tz-naive), columns = tickers. Network; thin.
  - `to_month_end(daily: pd.DataFrame) -> pd.DataFrame` — last available price per calendar month, index = month-end dates. Pure.
  - `monthly_returns(month_end: pd.DataFrame) -> pd.DataFrame` — simple returns over each month (`pct_change`), first row dropped. Pure.

- [ ] **Step 1: Write failing tests for the pure transforms**

`tests/data/test_ingest_prices.py`:
```python
import numpy as np, pandas as pd
from etf.data.ingest_prices import to_month_end, monthly_returns

def _daily(prices, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({"AAA": prices}, index=idx)

def test_to_month_end_takes_last_price_of_month():
    df = _daily([10, 11, 12, 13, 20, 21])  # spans Jan->? within bdays
    me = to_month_end(df)
    # index entries are month-end dates, values are last obs in each month
    assert (me.index == me.index.to_period("M").to_timestamp("M")).all()
    assert me["AAA"].iloc[0] == df["AAA"].loc[df.index.to_period("M") == me.index.to_period("M")[0]].iloc[-1]

def test_monthly_returns_are_pct_change():
    me = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]},
                      index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))
    r = monthly_returns(me)
    assert len(r) == 2
    assert np.isclose(r["AAA"].iloc[0], 0.10)
    assert np.isclose(r["AAA"].iloc[1], -0.10)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/data/test_ingest_prices.py -v`
Expected: FAIL (ImportError / module not found).

- [ ] **Step 3: Implement**

`src/etf/data/ingest_prices.py`:
```python
from __future__ import annotations
import pandas as pd

def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Daily total-return-adjusted close. Network wrapper (not unit-tested)."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        out = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:  # single ticker
        out = raw[["Close"]].rename(columns={"Close": tickers[0]})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index().dropna(how="all")

def to_month_end(daily: pd.DataFrame) -> pd.DataFrame:
    me = daily.resample("ME").last()
    me.index = me.index.to_period("M").to_timestamp("M")
    return me.dropna(how="all")

def monthly_returns(month_end: pd.DataFrame) -> pd.DataFrame:
    return month_end.pct_change().iloc[1:]
```

- [ ] **Step 4: Add a live smoke test (skipped by default)**

Append to the test file:
```python
import pytest

@pytest.mark.live
def test_download_prices_live():
    from etf.data.ingest_prices import download_prices
    df = download_prices(["XLK", "SPY"], start="2020-01-01", end="2020-03-01")
    assert {"XLK", "SPY"}.issubset(df.columns) and len(df) > 20
```

- [ ] **Step 5: Run and commit**

Run: `pytest tests/data/test_ingest_prices.py -v` → Expected: 2 passed, 1 deselected.
```bash
git add src/etf/data/ingest_prices.py tests/data/test_ingest_prices.py
git commit -m "Add ETF price ingestion and monthly resampling"
```

---

## Task 3: Macro ingestion (FRED)

**Files:**
- Create: `src/etf/data/ingest_macro.py`, `tests/data/test_ingest_macro.py`

**Interfaces:**
- Produces:
  - `download_macro(series: list[str], start: str) -> pd.DataFrame` — daily FRED series, forward-filled, columns renamed `mac_<lower>`. Network; thin.
  - `align_macro_monthly(macro_daily: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame` — last macro value at or before each month-end (`merge_asof` backward). Pure.
- Default series: `["VIXCLS", "DGS3MO", "DGS10", "T10Y2Y"]`.

- [ ] **Step 1: Write failing test for the pure aligner**

`tests/data/test_ingest_macro.py`:
```python
import pandas as pd
from etf.data.ingest_macro import align_macro_monthly

def test_align_takes_last_value_at_or_before_month_end():
    daily = pd.DataFrame(
        {"mac_vixcls": [15.0, 16.0, 30.0]},
        index=pd.to_datetime(["2020-01-15", "2020-01-31", "2020-02-20"]),
    )
    month_ends = pd.to_datetime(["2020-01-31", "2020-02-29"])
    out = align_macro_monthly(daily, month_ends)
    assert list(out.index) == list(month_ends)
    assert out["mac_vixcls"].iloc[0] == 16.0   # value on Jan-31
    assert out["mac_vixcls"].iloc[1] == 30.0   # last <= Feb-29 is Feb-20
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/data/test_ingest_macro.py -v` → FAIL.

- [ ] **Step 3: Implement**

`src/etf/data/ingest_macro.py`:
```python
from __future__ import annotations
import pandas as pd

DEFAULT_SERIES = ["VIXCLS", "DGS3MO", "DGS10", "T10Y2Y"]

def download_macro(series: list[str] = DEFAULT_SERIES, start: str = "1998-01-01") -> pd.DataFrame:
    from pandas_datareader import data as pdr
    df = pdr.DataReader(series, "fred", start)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.ffill()
    df.columns = [f"mac_{c.lower()}" for c in df.columns]
    return df

def align_macro_monthly(macro_daily: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    left = pd.DataFrame(index=pd.DatetimeIndex(month_ends)).reset_index(names="date")
    right = macro_daily.sort_index().reset_index(names="date")
    merged = pd.merge_asof(left.sort_values("date"), right, on="date", direction="backward")
    return merged.set_index("date")
```

- [ ] **Step 4: Live smoke test + run**

```python
import pytest
@pytest.mark.live
def test_download_macro_live():
    from etf.data.ingest_macro import download_macro
    df = download_macro(start="2020-01-01")
    assert "mac_vixcls" in df.columns and len(df) > 200
```
Run: `pytest tests/data/test_ingest_macro.py -v` → 1 passed, 1 deselected.

- [ ] **Step 5: Commit**

```bash
git add src/etf/data/ingest_macro.py tests/data/test_ingest_macro.py
git commit -m "Add FRED macro ingestion and monthly alignment"
```

---

## Task 4: Build the month-end sector panel with labels

**Files:**
- Create: `src/etf/data/build_panel.py`, `tests/data/test_build_panel.py`

**Interfaces:**
- Consumes: `monthly_returns` output shape (month-end index, ticker columns).
- Produces:
  - `build_panel(month_end_prices: pd.DataFrame, sector_etfs: list[str], inception: dict[str,str]) -> pd.DataFrame`
    Long panel, one row per `(date, ticker)` for tickers that exist at `date` (month-end ≥ inception). Columns: `date, ticker, price, ret_1m, fwd_ret_1m`.
    - `ret_1m` = return over the month ending at `date`.
    - `fwd_ret_1m` = return over the **next** month (shift -1 per ticker) — the LABEL. NaN at each ticker's last month.

- [ ] **Step 1: Write failing test**

`tests/data/test_build_panel.py`:
```python
import numpy as np, pandas as pd
from etf.data.build_panel import build_panel

def _prices():
    idx = pd.to_datetime(["2015-08-31","2015-09-30","2015-10-31","2015-11-30"])
    return pd.DataFrame({
        "XLK":  [100.0, 110.0, 99.0, 108.9],   # +10%, -10%, +10%
        "XLRE": [np.nan, np.nan, 50.0, 55.0],   # launches Oct-2015
    }, index=idx)

def test_panel_has_forward_label_and_respects_inception():
    inception = {"XLK": "1998-12-31", "XLRE": "2015-10-31"}
    p = build_panel(_prices(), ["XLK", "XLRE"], inception)
    xlk = p[p.ticker == "XLK"].sort_values("date")
    # ret_1m on 2015-09-30 = +10%; fwd_ret_1m there = next month return = -10%
    sep = xlk[xlk.date == "2015-09-30"].iloc[0]
    assert np.isclose(sep.ret_1m, 0.10)
    assert np.isclose(sep.fwd_ret_1m, -0.10)
    # XLRE never appears before its inception month-end (2015-10-31)
    assert (p[p.ticker == "XLRE"].date.min() == pd.Timestamp("2015-10-31"))
    # label is forward-only: last row per ticker has NaN fwd_ret_1m
    assert np.isnan(xlk.iloc[-1].fwd_ret_1m)
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

`src/etf/data/build_panel.py`:
```python
from __future__ import annotations
import pandas as pd

def build_panel(month_end_prices: pd.DataFrame, sector_etfs: list[str],
                inception: dict[str, str]) -> pd.DataFrame:
    rows = []
    for t in sector_etfs:
        if t not in month_end_prices.columns:
            continue
        s = month_end_prices[t].dropna()
        start = pd.Timestamp(inception[t])
        s = s[s.index >= start]
        if len(s) < 2:
            continue
        ret = s.pct_change()
        fwd = ret.shift(-1)  # next-month return = label
        df = pd.DataFrame({"date": s.index, "ticker": t, "price": s.values,
                           "ret_1m": ret.values, "fwd_ret_1m": fwd.values})
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    return panel.dropna(subset=["ret_1m"]).sort_values(["date", "ticker"]).reset_index(drop=True)
```

- [ ] **Step 4: Run** — `pytest tests/data/test_build_panel.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/etf/data/build_panel.py tests/data/test_build_panel.py
git commit -m "Add month-end sector panel with forward-return labels"
```

---

## Task 5: Feature builders (momentum, vol, drawdown, relative strength)

**Files:**
- Create: `src/etf/features/build_features.py`, `tests/features/test_build_features.py`

**Interfaces:**
- Produces `compute_price_features(daily_prices: pd.DataFrame, spy_daily: pd.Series, asof_dates: pd.DatetimeIndex, ticker: str) -> pd.DataFrame` returning one row per `asof_date` with columns:
  `mom_3m, mom_6m, mom_12m, mom_12m1m, vol_63d, vol_126d, dd_252d, rs_3m, rs_6m, rs_12m`.
  All computed from daily data with index `≤ asof_date` only (PIT).
  - `mom_Km` = `price_t / price_{t-21K} - 1` (21 trading days/month).
  - `mom_12m1m` = `price_{t-21} / price_{t-252} - 1` (12-1 skip-month momentum).
  - `vol_Wd` = std of daily returns over last `W` days × sqrt(252).
  - `dd_252d` = `price_t / max(price over last 252d) - 1`.
  - `rs_Km` = ticker `mom_Km` − SPY `mom_Km`.

- [ ] **Step 1: Write failing test**

`tests/features/test_build_features.py`:
```python
import numpy as np, pandas as pd
from etf.features.build_features import compute_price_features

def _series(n=400, drift=0.0005, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    rets = rng.normal(drift, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx)

def test_features_are_pit_and_named():
    px = _series(seed=1); spy = _series(seed=2)
    daily = px.to_frame("XLK")
    asof = pd.DatetimeIndex([px.index[300], px.index[350]])
    f = compute_price_features(daily, spy, asof, "XLK")
    assert list(f.index) == list(asof)
    for c in ["mom_3m","mom_6m","mom_12m","mom_12m1m","vol_63d","vol_126d","dd_252d","rs_3m","rs_6m","rs_12m"]:
        assert c in f.columns
    # mom_3m at asof matches manual 63-day price ratio
    t = asof[0]; p = daily["XLK"]
    expected = p.loc[t] / p.loc[:t].iloc[-64] - 1
    assert np.isclose(f.loc[t, "mom_3m"], expected, atol=1e-9)

def test_features_use_only_past_data():
    # Truncating the series after asof must not change the asof-date features.
    px = _series(seed=3); spy = _series(seed=4)
    asof = pd.DatetimeIndex([px.index[300]])
    full = compute_price_features(px.to_frame("XLK"), spy, asof, "XLK")
    trunc_px = px.loc[:asof[0]]; trunc_spy = spy.loc[:asof[0]]
    trunc = compute_price_features(trunc_px.to_frame("XLK"), trunc_spy, asof, "XLK")
    pd.testing.assert_frame_equal(full, trunc)
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

`src/etf/features/build_features.py`:
```python
from __future__ import annotations
import numpy as np, pandas as pd

_MONTH = 21

def _mom(p: pd.Series, t: pd.Timestamp, days: int) -> float:
    hist = p.loc[:t]
    if len(hist) <= days:
        return np.nan
    return hist.iloc[-1] / hist.iloc[-1 - days] - 1

def _mom_skip(p: pd.Series, t: pd.Timestamp, end_days: int, start_days: int) -> float:
    hist = p.loc[:t]
    if len(hist) <= start_days:
        return np.nan
    return hist.iloc[-1 - end_days] / hist.iloc[-1 - start_days] - 1

def compute_price_features(daily_prices: pd.DataFrame, spy_daily: pd.Series,
                           asof_dates: pd.DatetimeIndex, ticker: str) -> pd.DataFrame:
    p = daily_prices[ticker].dropna()
    spy = spy_daily.dropna()
    out = {}
    for t in asof_dates:
        hist = p.loc[:t]; rets = hist.pct_change().dropna()
        row = {
            "mom_3m": _mom(p, t, 3 * _MONTH),
            "mom_6m": _mom(p, t, 6 * _MONTH),
            "mom_12m": _mom(p, t, 12 * _MONTH),
            "mom_12m1m": _mom_skip(p, t, _MONTH, 12 * _MONTH),
            "vol_63d": rets.iloc[-63:].std(ddof=1) * np.sqrt(252) if len(rets) >= 63 else np.nan,
            "vol_126d": rets.iloc[-126:].std(ddof=1) * np.sqrt(252) if len(rets) >= 126 else np.nan,
            "dd_252d": (hist.iloc[-1] / hist.iloc[-252:].max() - 1) if len(hist) >= 1 else np.nan,
            "rs_3m": _mom(p, t, 3 * _MONTH) - _mom(spy, t, 3 * _MONTH),
            "rs_6m": _mom(p, t, 6 * _MONTH) - _mom(spy, t, 6 * _MONTH),
            "rs_12m": _mom(p, t, 12 * _MONTH) - _mom(spy, t, 12 * _MONTH),
        }
        out[t] = row
    return pd.DataFrame.from_dict(out, orient="index")[
        ["mom_3m","mom_6m","mom_12m","mom_12m1m","vol_63d","vol_126d","dd_252d","rs_3m","rs_6m","rs_12m"]
    ]
```

- [ ] **Step 4: Run** — `pytest tests/features/test_build_features.py -v` → PASS (the truncation test is the leakage guarantee).

- [ ] **Step 5: Commit**

```bash
git add src/etf/features/build_features.py tests/features/test_build_features.py
git commit -m "Add PIT price features (momentum, vol, drawdown, relative strength)"
```

---

## Task 6: Assemble feature matrix + cross-sectional label

**Files:**
- Modify: `src/etf/features/build_features.py` (append functions)
- Create: `tests/features/test_assemble.py`

**Interfaces:**
- Consumes: `build_panel` output; `compute_price_features`; `align_macro_monthly`.
- Produces:
  - `assemble_features(panel, daily_prices, spy_daily, macro_daily) -> pd.DataFrame` — panel + price features + macro columns joined on `(date, ticker)`/`date`. Returns rows that have a non-NaN label (`fwd_ret_1m`) and complete core features.
  - `add_label(df) -> pd.DataFrame` — adds `y_rank`: cross-sectional percentile rank of `fwd_ret_1m` within each `date`, in [0, 1].
  - `FEATURE_COLS: list[str]` — the model's input columns (10 price + 4 macro).

- [ ] **Step 1: Write failing test**

`tests/features/test_assemble.py`:
```python
import numpy as np, pandas as pd
from etf.features.build_features import add_label

def test_label_is_within_date_percentile_rank():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31"]*3 + ["2020-02-29"]*3),
        "ticker": ["A","B","C","A","B","C"],
        "fwd_ret_1m": [0.01, 0.02, 0.03, -0.05, 0.0, 0.05],
    })
    out = add_label(df)
    jan = out[out.date == "2020-01-31"].sort_values("ticker")
    assert list(jan.y_rank) == [0.0, 0.5, 1.0]   # min..max within the date
    assert out.y_rank.between(0, 1).all()
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement (append to `build_features.py`)**

```python
from etf.data.ingest_macro import align_macro_monthly

MACRO_COLS = ["mac_vixcls", "mac_dgs3mo", "mac_dgs10", "mac_t10y2y"]
PRICE_COLS = ["mom_3m","mom_6m","mom_12m","mom_12m1m","vol_63d","vol_126d","dd_252d","rs_3m","rs_6m","rs_12m"]
FEATURE_COLS = PRICE_COLS + MACRO_COLS

def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_rank"] = df.groupby("date")["fwd_ret_1m"].rank(pct=True, method="average")
    # rescale per-date min->0, max->1 so endpoints are stable for tests/training
    g = df.groupby("date")["y_rank"]
    df["y_rank"] = (df["y_rank"] - g.transform("min")) / (g.transform("max") - g.transform("min"))
    return df

def assemble_features(panel: pd.DataFrame, daily_prices: pd.DataFrame,
                      spy_daily: pd.Series, macro_daily: pd.DataFrame) -> pd.DataFrame:
    asof = pd.DatetimeIndex(sorted(panel["date"].unique()))
    feats = []
    for t, sub in panel.groupby("ticker"):
        f = compute_price_features(daily_prices, spy_daily, pd.DatetimeIndex(sub["date"]), t)
        f = f.reset_index(names="date"); f["ticker"] = t
        feats.append(f)
    fmat = pd.concat(feats, ignore_index=True)
    macro_m = align_macro_monthly(macro_daily, asof).reset_index(names="date")
    out = panel.merge(fmat, on=["date", "ticker"], how="left").merge(macro_m, on="date", how="left")
    out = out.dropna(subset=["fwd_ret_1m"] + PRICE_COLS).reset_index(drop=True)
    return add_label(out)
```

- [ ] **Step 4: Run** — `pytest tests/features/test_assemble.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/etf/features/build_features.py tests/features/test_assemble.py
git commit -m "Assemble feature matrix and cross-sectional rank label"
```

---

## Task 7: Allocation primitives (floor+cap projection, tilt, no-trade band)

**Files:**
- Create: `src/etf/strategy/allocate.py`, `tests/strategy/test_allocate.py`

**Interfaces:**
- Produces:
  - `project_to_simplex(raw: np.ndarray, max_weight: float, min_weight: float = 0.0) -> np.ndarray` — weights in `[min_weight, max_weight]` summing to 1 (water-fill).
  - `tilt_weights(scores: np.ndarray, tilt_scale: float, max_weight: float, min_weight: float, anchor: np.ndarray | None = None) -> np.ndarray` — `anchor * exp(tilt_scale * z(scores))` then projected. `anchor=None` ⇒ equal weight. `tilt_scale=0` ⇒ exactly `anchor`.
  - `apply_no_trade_band(target: np.ndarray, current: np.ndarray, band: float, max_weight: float, min_weight: float) -> np.ndarray` — hold sectors whose `|target-current| < band`, re-project the rest.

- [ ] **Step 1: Write failing tests**

`tests/strategy/test_allocate.py`:
```python
import numpy as np
from etf.strategy.allocate import project_to_simplex, tilt_weights, apply_no_trade_band

def test_projection_respects_bounds_and_sums_to_one():
    w = project_to_simplex(np.array([5.0, 0.0, -3.0, 1.0, 0.5]), max_weight=0.4, min_weight=0.1)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0.1 - 1e-9).all() and (w <= 0.4 + 1e-9).all()

def test_tilt_scale_zero_returns_anchor():
    scores = np.array([3.0, -1.0, 0.0, 2.0])
    w = tilt_weights(scores, tilt_scale=0.0, max_weight=0.5, min_weight=0.0)
    assert np.allclose(w, 0.25, atol=1e-9)

def test_tilt_overweights_high_scores_within_cap():
    scores = np.array([2.0, 0.0, 0.0, -2.0])
    w = tilt_weights(scores, tilt_scale=0.5, max_weight=0.4, min_weight=0.05)
    assert w.argmax() == 0 and w.argmin() == 3
    assert np.isclose(w.sum(), 1.0)

def test_no_trade_band_holds_small_moves():
    cur = np.array([0.25, 0.25, 0.25, 0.25])
    tgt = np.array([0.26, 0.40, 0.10, 0.24])  # only index 1,2 exceed band 0.05
    w = apply_no_trade_band(tgt, cur, band=0.05, max_weight=0.5, min_weight=0.0)
    assert np.isclose(w[0], cur[0]) and np.isclose(w[3], cur[3])  # held
    assert np.isclose(w.sum(), 1.0)
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

`src/etf/strategy/allocate.py`:
```python
from __future__ import annotations
import numpy as np

def project_to_simplex(raw: np.ndarray, max_weight: float, min_weight: float = 0.0) -> np.ndarray:
    raw = np.asarray(raw, float)
    n = len(raw)
    assert min_weight * n <= 1.0 + 1e-12 <= max_weight * n + 1e-12, "infeasible bounds"
    w = np.exp(raw - raw.max()); w /= w.sum()          # softmax start (positive, sums to 1)
    for _ in range(1000):
        w = np.clip(w, min_weight, max_weight)
        deficit = 1.0 - w.sum()
        if abs(deficit) < 1e-12:
            break
        if deficit > 0:                                 # need to add weight to non-capped names
            room = max_weight - w
            free = room > 1e-15
        else:                                           # need to remove weight from non-floored names
            room = w - min_weight
            free = room > 1e-15
        share = room[free] / room[free].sum()
        w[free] += deficit * share
    w = np.clip(w, min_weight, max_weight)
    return w / w.sum()

def tilt_weights(scores: np.ndarray, tilt_scale: float, max_weight: float,
                 min_weight: float, anchor: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, float); n = len(scores)
    if anchor is None:
        anchor = np.full(n, 1.0 / n)
    sd = scores.std()
    z = (scores - scores.mean()) / sd if sd > 0 else np.zeros(n)
    raw = np.log(anchor) + tilt_scale * z
    return project_to_simplex(raw, max_weight, min_weight)

def apply_no_trade_band(target: np.ndarray, current: np.ndarray, band: float,
                        max_weight: float, min_weight: float) -> np.ndarray:
    target = np.asarray(target, float); current = np.asarray(current, float)
    w = target.copy()
    hold = np.abs(target - current) < band
    w[hold] = current[hold]
    if not hold.all():                                  # re-project the traded names to keep sum=1 + bounds
        return project_to_simplex(np.log(np.clip(w, 1e-9, None)), max_weight, min_weight)
    return w / w.sum()
```

- [ ] **Step 4: Run** — `pytest tests/strategy/test_allocate.py -v` → PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/etf/strategy/allocate.py tests/strategy/test_allocate.py
git commit -m "Add allocation primitives: floor/cap projection, tilt, no-trade band"
```

---

## Task 8: Backtest metrics (monthly)

**Files:**
- Create: `src/etf/utils/backtest.py`, `tests/utils/test_backtest.py`

**Interfaces:**
- Produces `compute_strategy_metrics(returns: pd.Series | np.ndarray, turnover=None, cost_bps: float = 0.0, periods_per_year: int = 12) -> dict` with keys: `total_return, ann_return, ann_vol, sharpe, sortino, max_drawdown, calmar, hit_rate, avg_turnover`. Returns are charged `cost_bps/1e4 * turnover` per period when both provided.

- [ ] **Step 1: Write failing tests**

`tests/utils/test_backtest.py`:
```python
import numpy as np, pandas as pd
from etf.utils.backtest import compute_strategy_metrics

def test_zero_turnover_means_gross_equals_net():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    m_gross = compute_strategy_metrics(r)
    m_net = compute_strategy_metrics(r, turnover=pd.Series([0,0,0,0]), cost_bps=5.0)
    assert np.isclose(m_gross["total_return"], m_net["total_return"])

def test_known_metrics():
    r = pd.Series([0.10, -0.10, 0.10, -0.10] * 6)  # 24 months
    m = compute_strategy_metrics(r)
    assert m["max_drawdown"] < 0
    assert np.isclose(m["ann_vol"], r.std(ddof=1) * np.sqrt(12))
    assert set(m) >= {"sharpe","sortino","calmar","hit_rate","ann_return","ann_vol","max_drawdown"}

def test_cost_reduces_return():
    r = pd.Series([0.02, 0.02, 0.02])
    base = compute_strategy_metrics(r, turnover=pd.Series([1.0,1.0,1.0]), cost_bps=0.0)
    costed = compute_strategy_metrics(r, turnover=pd.Series([1.0,1.0,1.0]), cost_bps=50.0)
    assert costed["total_return"] < base["total_return"]
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

`src/etf/utils/backtest.py`:
```python
from __future__ import annotations
import numpy as np, pandas as pd

def compute_strategy_metrics(returns, turnover=None, cost_bps: float = 0.0,
                             periods_per_year: int = 12) -> dict:
    r = np.asarray(returns, float)
    if turnover is not None and cost_bps:
        r = r - (cost_bps / 1e4) * np.asarray(turnover, float)
    n = len(r)
    eq = np.cumprod(1.0 + r)
    total = eq[-1] - 1.0 if n else np.nan
    ann_ret = (1.0 + total) ** (periods_per_year / n) - 1.0 if n else np.nan
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year) if n > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    down = r[r < 0]
    dvol = down.std(ddof=1) * np.sqrt(periods_per_year) if len(down) > 1 else np.nan
    sortino = ann_ret / dvol if dvol and dvol > 0 else np.nan
    peak = np.maximum.accumulate(eq) if n else np.array([np.nan])
    mdd = float((eq / peak - 1.0).min()) if n else np.nan
    calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan
    avg_to = float(np.mean(turnover)) if turnover is not None else np.nan
    return {"total_return": float(total), "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol), "sharpe": float(sharpe), "sortino": float(sortino),
            "max_drawdown": mdd, "calmar": float(calmar),
            "hit_rate": float((r > 0).mean()) if n else np.nan, "avg_turnover": avg_to}
```

- [ ] **Step 4: Run** — `pytest tests/utils/test_backtest.py -v` → PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/etf/utils/backtest.py tests/utils/test_backtest.py
git commit -m "Add monthly backtest metrics"
```

---

## Task 9: Walk-forward splits + benchmarks

**Files:**
- Create: `src/etf/utils/walkforward.py`, `src/etf/utils/benchmarks.py`, `tests/utils/test_walkforward.py`

**Interfaces:**
- Produces:
  - `Walk` dataclass: `train_end_year: int, val_year: int, test_year: int` + `def masks(self, dates: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]` returning boolean (train, val, test) masks where train = `year <= train_end_year`, val = `year == val_year`, test = `year == test_year`.
  - `expanding_walks(first_test_year: int, last_test_year: int) -> list[Walk]`.
  - `buy_hold_spy(spy_monthly_returns: pd.Series) -> pd.Series` — pass-through (rebased) monthly returns.
  - `equal_weight(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]` — `(returns, turnover)` for an equal-weight-of-available-sectors monthly rebalance, using `ret_1m` realized next period via the panel's per-date forward labels.

- [ ] **Step 1: Write failing tests**

`tests/utils/test_walkforward.py`:
```python
import numpy as np, pandas as pd
from etf.utils.walkforward import Walk, expanding_walks
from etf.utils.benchmarks import equal_weight

def test_expanding_walks_are_sealed_and_sequential():
    walks = expanding_walks(2005, 2007)
    assert [w.test_year for w in walks] == [2005, 2006, 2007]
    w = walks[0]
    assert w.val_year == 2004 and w.train_end_year == 2003  # val=test-1, train<=test-2

def test_walk_masks_are_disjoint():
    dates = pd.Series(pd.to_datetime(["2003-06-30","2004-06-30","2005-06-30"]))
    tr, va, te = expanding_walks(2005, 2005)[0].masks(dates)
    assert (tr & va).sum() == 0 and (va & te).sum() == 0 and (tr & te).sum() == 0

def test_equal_weight_uses_available_sectors():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31","2020-01-31","2020-02-29","2020-02-29"]),
        "ticker": ["A","B","A","B"],
        "fwd_ret_1m": [0.10, 0.00, -0.10, 0.20],
    })
    rets, turn = equal_weight(panel)
    assert np.isclose(rets.iloc[0], 0.05)   # mean(0.10, 0.00)
    assert np.isclose(rets.iloc[1], 0.05)   # mean(-0.10, 0.20)
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

`src/etf/utils/walkforward.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Walk:
    train_end_year: int
    val_year: int
    test_year: int
    def masks(self, dates: pd.Series):
        y = pd.to_datetime(dates).dt.year
        return (y <= self.train_end_year, y == self.val_year, y == self.test_year)

def expanding_walks(first_test_year: int, last_test_year: int) -> list[Walk]:
    return [Walk(ty - 2, ty - 1, ty) for ty in range(first_test_year, last_test_year + 1)]
```

`src/etf/utils/benchmarks.py`:
```python
from __future__ import annotations
import numpy as np, pandas as pd

def buy_hold_spy(spy_monthly_returns: pd.Series) -> pd.Series:
    return spy_monthly_returns.dropna()

def equal_weight(panel: pd.DataFrame):
    rets, turns, prev = [], [], None
    dates = sorted(panel["date"].unique())
    for d in dates:
        sub = panel[panel["date"] == d]
        tickers = list(sub["ticker"]); n = len(tickers)
        w = pd.Series(1.0 / n, index=tickers)
        r = float((w * sub.set_index("ticker")["fwd_ret_1m"]).sum())
        if prev is None:
            to = 1.0
        else:
            allt = prev.index.union(w.index)
            to = float((w.reindex(allt, fill_value=0) - prev.reindex(allt, fill_value=0)).abs().sum())
        rets.append(r); turns.append(to); prev = w
    idx = pd.to_datetime(dates)
    return pd.Series(rets, index=idx), pd.Series(turns, index=idx)
```

- [ ] **Step 4: Run** — `pytest tests/utils/test_walkforward.py -v` → PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/etf/utils/walkforward.py src/etf/utils/benchmarks.py tests/utils/test_walkforward.py
git commit -m "Add expanding walk-forward splits and benchmark portfolios"
```

---

## Task 10: Strategy A model + end-to-end backtest runner

**Files:**
- Create: `src/etf/strategy/strategy_a.py`, `tests/strategy/test_strategy_a.py`, `experiments/configs/strategy_a.json`

**Interfaces:**
- Consumes: `FEATURE_COLS`, `tilt_weights`, `apply_no_trade_band`, `expanding_walks`, `compute_strategy_metrics`.
- Produces:
  - `train_and_score(train_df, test_df, feature_cols, params, seed) -> pd.DataFrame` — adds `score` column to `test_df` (XGBoost regression on `y_rank`).
  - `run_strategy_a(features: pd.DataFrame, walks, config, seed) -> dict` returning `{"weights": DataFrame[date,ticker,weight], "returns": Series, "turnover": Series, "scores": DataFrame}`. For each test month: rank-score available sectors, `tilt_weights` → target, `apply_no_trade_band` vs prior, realize `sum(weight * fwd_ret_1m)`, accumulate turnover.

- [ ] **Step 1: Write `experiments/configs/strategy_a.json`**

```json
{
  "xgb_params": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
                 "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
                 "reg_lambda": 1.0, "objective": "reg:squarederror"},
  "tilt_scale": 0.5, "max_weight": 0.18, "min_weight": 0.02,
  "no_trade_band": 0.02, "cost_bps": 5.0,
  "first_test_year": 2005, "last_test_year": 2025
}
```

- [ ] **Step 2: Write failing test (uses a tiny synthetic feature set, no network)**

`tests/strategy/test_strategy_a.py`:
```python
import numpy as np, pandas as pd
from etf.strategy.strategy_a import run_strategy_a
from etf.utils.walkforward import expanding_walks
from etf.features.build_features import FEATURE_COLS

def _synth_features(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2002, 2007):
        for m in range(1, 13):
            date = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
            for tk in ["A","B","C","D","E"]:
                feat = {c: rng.normal() for c in FEATURE_COLS}
                # signal: high mom_3m -> high forward return
                fwd = 0.02 * feat["mom_3m"] + rng.normal(0, 0.01)
                rows.append({"date": date, "ticker": tk, "fwd_ret_1m": fwd, **feat})
    df = pd.DataFrame(rows)
    df["y_rank"] = df.groupby("date")["fwd_ret_1m"].rank(pct=True)
    return df

def test_run_strategy_a_produces_valid_weights_and_beats_nothing_trivially():
    feats = _synth_features()
    walks = expanding_walks(2005, 2006)
    cfg = {"xgb_params": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1,
            "objective": "reg:squarederror"},
           "tilt_scale": 0.5, "max_weight": 0.4, "min_weight": 0.05,
           "no_trade_band": 0.0, "cost_bps": 5.0}
    res = run_strategy_a(feats, walks, cfg, seed=0)
    w = res["weights"]
    # every rebalance sums to 1 and respects bounds
    sums = w.groupby("date")["weight"].sum()
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert (w["weight"] >= 0.05 - 1e-9).all() and (w["weight"] <= 0.4 + 1e-9).all()
    assert len(res["returns"]) == w["date"].nunique()
```

- [ ] **Step 3: Run to verify failure** — FAIL.

- [ ] **Step 4: Implement**

`src/etf/strategy/strategy_a.py`:
```python
from __future__ import annotations
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from etf.strategy.allocate import tilt_weights, apply_no_trade_band

def train_and_score(train_df, test_df, feature_cols, params, seed):
    model = XGBRegressor(random_state=seed, n_jobs=1, **params)
    model.fit(train_df[feature_cols], train_df["y_rank"])
    out = test_df.copy()
    out["score"] = model.predict(out[feature_cols])
    return out

def run_strategy_a(features: pd.DataFrame, walks, config, seed) -> dict:
    from etf.features.build_features import FEATURE_COLS
    p = config
    scored_parts, weight_rows, rets, turns = [], [], [], []
    prev = None  # pd.Series indexed by ticker
    for walk in walks:
        tr, _, te = walk.masks(features["date"])
        train_df, test_df = features[tr], features[te]
        if train_df.empty or test_df.empty:
            continue
        scored = train_and_score(train_df, test_df, FEATURE_COLS, p["xgb_params"], seed)
        scored_parts.append(scored)
        for d, sub in scored.groupby("date"):
            sub = sub.sort_values("ticker")
            tickers = list(sub["ticker"])
            target = tilt_weights(sub["score"].to_numpy(), p["tilt_scale"],
                                  p["max_weight"], p["min_weight"])
            tgt = pd.Series(target, index=tickers)
            if prev is None:
                w = tgt; to = 1.0
            else:
                allt = sorted(set(tickers) | set(prev.index))
                cur = prev.reindex(allt, fill_value=0.0)
                # only sectors available this month can hold weight
                tgt_full = tgt.reindex(allt, fill_value=0.0)
                banded = apply_no_trade_band(tgt_full.to_numpy(), cur.to_numpy(),
                                             p["no_trade_band"], p["max_weight"], p["min_weight"])
                w = pd.Series(banded, index=allt)
                w = w[w.index.isin(tickers)]; w = w / w.sum()
                to = float((w.reindex(allt, fill_value=0) - cur).abs().sum())
            r = float((w * sub.set_index("ticker")["fwd_ret_1m"].reindex(w.index)).sum())
            for tk, wt in w.items():
                weight_rows.append({"date": d, "ticker": tk, "weight": wt})
            rets.append((d, r)); turns.append((d, to)); prev = w
    weights = pd.DataFrame(weight_rows)
    ridx = pd.to_datetime([d for d, _ in rets])
    return {"weights": weights,
            "returns": pd.Series([r for _, r in rets], index=ridx),
            "turnover": pd.Series([t for _, t in turns], index=ridx),
            "scores": pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()}
```

- [ ] **Step 5: Run** — `pytest tests/strategy/test_strategy_a.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/etf/strategy/strategy_a.py tests/strategy/test_strategy_a.py experiments/configs/strategy_a.json
git commit -m "Add Strategy A model and walk-forward backtest runner"
```

---

## Task 11: Leakage tests (PIT recompute, label-shuffle, feature-timing)

**Files:**
- Create: `src/etf/utils/validation.py`, `tests/utils/test_validation_leakage.py`

**Interfaces:**
- Produces:
  - `label_shuffle_sharpe(features, walks, config, seed, n_shuffles) -> np.ndarray` — refit Strategy A with `y_rank` shuffled within each train date; returns array of resulting test Sharpes. Expectation: centered near/below the SPY-free null (~0), and far below the real run.
  - `feature_timing_gain(features, walks, config, seed, feature) -> float` — shift one feature forward one month within each date-group (giving the model *future* info); returns `Sharpe(shifted) - Sharpe(real)`. A large positive gain flags look-ahead in that feature; expectation ≈ 0 for PIT features.

- [ ] **Step 1: Write failing tests**

`tests/utils/test_validation_leakage.py`:
```python
import numpy as np, pandas as pd
from etf.utils.validation import label_shuffle_sharpe
from etf.utils.walkforward import expanding_walks
from etf.strategy.strategy_a import run_strategy_a
from etf.utils.backtest import compute_strategy_metrics
from tests.strategy.test_strategy_a import _synth_features  # reuse synthetic builder

def test_label_shuffle_destroys_performance():
    feats = _synth_features(seed=1)
    walks = expanding_walks(2005, 2006)
    cfg = {"xgb_params": {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.1,
            "objective": "reg:squarederror"},
           "tilt_scale": 0.7, "max_weight": 0.4, "min_weight": 0.05,
           "no_trade_band": 0.0, "cost_bps": 5.0}
    real = compute_strategy_metrics(run_strategy_a(feats, walks, cfg, 0)["returns"])["sharpe"]
    shuffled = label_shuffle_sharpe(feats, walks, cfg, seed=0, n_shuffles=10)
    # real signal should beat the median shuffled (null) run
    assert real > np.median(shuffled)
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement (in `src/etf/utils/validation.py`)**

```python
from __future__ import annotations
import numpy as np, pandas as pd
from etf.strategy.strategy_a import run_strategy_a
from etf.utils.backtest import compute_strategy_metrics

def label_shuffle_sharpe(features, walks, config, seed, n_shuffles=20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_shuffles):
        f = features.copy()
        f["y_rank"] = f.groupby("date")["y_rank"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        r = run_strategy_a(f, walks, config, seed)["returns"]
        out.append(compute_strategy_metrics(r)["sharpe"])
    return np.array(out)

def feature_timing_gain(features, walks, config, seed, feature) -> float:
    real = compute_strategy_metrics(run_strategy_a(features, walks, config, seed)["returns"])["sharpe"]
    f = features.copy()
    f[feature] = f.groupby("ticker")[feature].shift(-1)  # leak next month's feature
    f = f.dropna(subset=[feature])
    leaked = compute_strategy_metrics(run_strategy_a(f, walks, config, seed)["returns"])["sharpe"]
    return float(leaked - real)
```

- [ ] **Step 4: Run** — `pytest tests/utils/test_validation_leakage.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/etf/utils/validation.py tests/utils/test_validation_leakage.py
git commit -m "Add leakage validation: label-shuffle and feature-timing audit"
```

---

## Task 12: Monte-Carlo luck tests (seed stability, bootstrap CI, random-weights null, deflated Sharpe)

**Files:**
- Modify: `src/etf/utils/validation.py` (append)
- Create: `tests/utils/test_validation_mc.py`

**Interfaces:**
- Produces:
  - `seed_stability(features, walks, config, seeds: list[int]) -> pd.DataFrame` — one row per seed with `sharpe, ann_return, max_drawdown`.
  - `block_bootstrap_ci(returns: pd.Series, n_boot=5000, mean_block=6, seed=0, periods_per_year=12) -> dict` — stationary-bootstrap 95% CI for Sharpe (`{"sharpe_lo","sharpe_hi","sharpe_point"}`).
  - `dirichlet_null(features, walks, config, n_sims=2000, seed=0) -> dict` — replace the model's per-month weights with `Dirichlet(1)` draws projected to `[floor,cap]`, same realized `fwd_ret_1m`; returns `{"p_value", "null_mean", "strategy_sharpe"}` (p = fraction of null Sharpe ≥ strategy).
  - `deflated_sharpe(observed_sharpe, n_trials, n_obs, skew=0.0, kurt=3.0) -> float` — Bailey–López de Prado deflated Sharpe probability.

- [ ] **Step 1: Write failing tests**

`tests/utils/test_validation_mc.py`:
```python
import numpy as np, pandas as pd
from etf.utils.validation import block_bootstrap_ci, deflated_sharpe, dirichlet_null
from etf.utils.walkforward import expanding_walks
from tests.strategy.test_strategy_a import _synth_features

def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.04, 120))
    ci = block_bootstrap_ci(r, n_boot=500, mean_block=6, seed=0)
    assert ci["sharpe_lo"] <= ci["sharpe_point"] <= ci["sharpe_hi"]

def test_deflated_sharpe_drops_with_more_trials():
    assert deflated_sharpe(1.0, n_trials=1, n_obs=120) > deflated_sharpe(1.0, n_trials=100, n_obs=120)

def test_dirichlet_null_returns_pvalue_in_unit_interval():
    feats = _synth_features(seed=2)
    walks = expanding_walks(2005, 2005)
    cfg = {"xgb_params": {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1,
            "objective": "reg:squarederror"},
           "tilt_scale": 0.7, "max_weight": 0.4, "min_weight": 0.05,
           "no_trade_band": 0.0, "cost_bps": 5.0}
    res = dirichlet_null(feats, walks, cfg, n_sims=100, seed=0)
    assert 0.0 <= res["p_value"] <= 1.0
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement (append to `src/etf/utils/validation.py`)**

```python
from scipy.stats import norm  # add scipy to pyproject dependencies
from etf.strategy.allocate import project_to_simplex

def seed_stability(features, walks, config, seeds) -> pd.DataFrame:
    rows = []
    for s in seeds:
        m = compute_strategy_metrics(run_strategy_a(features, walks, config, s)["returns"])
        rows.append({"seed": s, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
                     "max_drawdown": m["max_drawdown"]})
    return pd.DataFrame(rows)

def block_bootstrap_ci(returns, n_boot=5000, mean_block=6, seed=0, periods_per_year=12) -> dict:
    r = np.asarray(returns, float); n = len(r); rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    def sharpe(x):
        v = x.std(ddof=1)
        return (x.mean() * periods_per_year) / (v * np.sqrt(periods_per_year)) if v > 0 else np.nan
    sims = []
    for _ in range(n_boot):
        idx = np.empty(n, int); i = rng.integers(n)
        for k in range(n):
            idx[k] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        sims.append(sharpe(r[idx]))
    sims = np.array(sims)
    return {"sharpe_point": float(sharpe(r)),
            "sharpe_lo": float(np.nanpercentile(sims, 2.5)),
            "sharpe_hi": float(np.nanpercentile(sims, 97.5))}

def dirichlet_null(features, walks, config, n_sims=2000, seed=0) -> dict:
    base = run_strategy_a(features, walks, config, seed)
    strat_sharpe = compute_strategy_metrics(base["returns"])["sharpe"]
    # realized forward returns per (date,ticker) for the test months actually traded
    sched = base["weights"][["date", "ticker"]].merge(
        features[["date", "ticker", "fwd_ret_1m"]], on=["date", "ticker"], how="left")
    rng = np.random.default_rng(seed); null = []
    dates = list(sched["date"].unique())
    for _ in range(n_sims):
        rets = []
        for d in dates:
            sub = sched[sched["date"] == d]
            w = project_to_simplex(rng.gamma(1.0, 1.0, len(sub)),
                                   config["max_weight"], config["min_weight"])
            rets.append(float((w * sub["fwd_ret_1m"].to_numpy()).sum()))
        null.append(compute_strategy_metrics(pd.Series(rets))["sharpe"])
    null = np.array(null)
    return {"strategy_sharpe": float(strat_sharpe), "null_mean": float(np.nanmean(null)),
            "p_value": float(np.mean(null >= strat_sharpe))}

def deflated_sharpe(observed_sharpe, n_trials, n_obs, skew=0.0, kurt=3.0) -> float:
    if n_trials <= 1:
        sr0 = 0.0
    else:
        e = 0.5772156649
        z1 = norm.ppf(1 - 1.0 / n_trials); z2 = norm.ppf(1 - 1.0 / (n_trials * e))
        sr0 = z1 * (1 - e) + z2 * e  # expected max Sharpe under the null (per-obs units handled below)
    sr0 = sr0 / np.sqrt(n_obs)       # scale null-max into per-observation Sharpe space
    num = (observed_sharpe - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * observed_sharpe + ((kurt - 1) / 4) * observed_sharpe ** 2)
    return float(norm.cdf(num / den))
```

Add `"scipy>=1.11"` to `dependencies` in `pyproject.toml` and re-run `pip install -e .`.

- [ ] **Step 4: Run** — `pytest tests/utils/test_validation_mc.py -v` → PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/etf/utils/validation.py tests/utils/test_validation_mc.py pyproject.toml
git commit -m "Add Monte-Carlo luck tests: seed stability, bootstrap CI, random-weights null, deflated Sharpe"
```

---

## Task 13: Pre-registered hypothesis + results notebook

**Files:**
- Create: `experiments/H_strategy_a.md`, `notebooks/strategy_a.ipynb`
- Use the `ipynb` skill to author the notebook.

**Interfaces:**
- Consumes everything above. No new library code — the notebook orchestrates and reports.

- [ ] **Step 1: Write `experiments/H_strategy_a.md` BEFORE running the real backtest**

```markdown
# H — Strategy A (rung 1) beats buy-and-hold SPY

Pre-registered 2026-06-29, before any full-history run.

**Fixed search box:** config = experiments/configs/strategy_a.json (xgb_params,
tilt_scale=0.5, max_weight=0.18, min_weight=0.02, no_trade_band=0.02, cost_bps=5).
Walks: expanding, test years 2005-2025.

**Acceptance bars (ALL must hold, net of 5 bps):**
1. OOS Sharpe(A) > Sharpe(SPY) over 2005-2025.
2. OOS max_drawdown(A) >= max_drawdown(SPY) (no deeper than SPY).
3. Label-shuffle median Sharpe < Sharpe(A) (signal is real, not leakage).
4. Block-bootstrap 95% Sharpe CI lower bound > Sharpe(SPY) point estimate.
5. Dirichlet random-weights null p-value < 0.10 (allocation skill, not luck).
6. seed_stability: min Sharpe across seeds [0..9] still > Sharpe(SPY).

**Decision rule:** if all bars hold, Strategy A becomes benchmark rung 1.
If any fail, A does not advance; record the failure and diagnose before tuning.
**No retroactive bar relaxation.**
```

- [ ] **Step 2: Author `notebooks/strategy_a.ipynb` (via the `ipynb` skill) with these cells:**

1. **Data load** — `download_prices(SECTOR_ETFS + ["SPY"], "1998-01-01")`, `download_macro()`; cache to `data/processed/{prices,macro}.parquet` (load cache if present).
2. **Panel + features** — `to_month_end` → `monthly_returns` → `build_panel` → `assemble_features`. Save `data/processed/features.parquet`.
3. **Run A + benchmarks** — `run_strategy_a(features, expanding_walks(2005,2025), config, seed=0)`; `buy_hold_spy`; `equal_weight`. Print `compute_strategy_metrics` table for SPY / equal-weight / Strategy A side by side.
4. **Equity curves** — plot cumulative `(1+r).cumprod()` for the three, log scale.
5. **Leakage** — `label_shuffle_sharpe` (histogram vs real Sharpe line); `feature_timing_gain` for each feature (bar chart; flag any > 0.2).
6. **Monte-Carlo** — `seed_stability(seeds=range(10))` (table + Sharpe dispersion); `block_bootstrap_ci` (print CI); `dirichlet_null` (histogram + p-value); `deflated_sharpe(observed, n_trials=<#configs tried>, n_obs)`.
7. **Acceptance check** — a cell that evaluates all 6 bars from `H_strategy_a.md` and prints PASS/FAIL per bar and an overall verdict.

- [ ] **Step 3: Execute the notebook top-to-bottom**

Run: `jupyter nbconvert --to notebook --execute notebooks/strategy_a.ipynb --output strategy_a.ipynb`
Expected: runs clean; the acceptance-check cell prints a verdict for each of the 6 bars.

- [ ] **Step 4: Run the full test suite**

Run: `pytest`
Expected: all tests pass (live tests deselected).

- [ ] **Step 5: Commit**

```bash
git add experiments/H_strategy_a.md notebooks/strategy_a.ipynb data/processed/.gitkeep
git commit -m "Add Strategy A hypothesis and results notebook"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** universe/expanding 9→11 (Tasks 1,4) · data ingest prices+macro (Tasks 2,3) · PIT panel+labels (Task 4) · features incl. macro (Tasks 5,6) · equal-weight anchor × log-tilt under [floor,cap] + no-trade band (Tasks 7,10) · monthly metrics with drawdown terms (Task 8) · sealed expanding walk-forward + SPY/EW benchmarks (Task 9) · single XGBoost, no LSTM/MVO/BL (Task 10) · leakage tests (Task 11) · Monte-Carlo luck tests incl. seed stability & deflated Sharpe (Task 12) · per-strategy notebook + pre-registered H-file (Task 13). All spec sections map to a task.
- **Out of scope (correctly deferred):** rung B (MVO+BL+LSTM), rung C (RL), live trading, cash/bond sleeve, cap-breach mode.
- **Type consistency:** panel columns (`date, ticker, price, ret_1m, fwd_ret_1m`), `FEATURE_COLS`, `y_rank`, `Walk.masks`, and `run_strategy_a`'s return dict are referenced identically across Tasks 4–13.
- **Open items to confirm at execution:** `first_test_year=2005` (enough 9-sector history to train; adjust if XLK/SPY history is short); deflated-Sharpe `n_trials` = number of configs actually evaluated (record honestly).
