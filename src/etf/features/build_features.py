from __future__ import annotations
import numpy as np, pandas as pd
from etf.data.ingest_macro import align_macro_monthly

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

MACRO_COLS = ["mac_vixcls", "mac_dgs3mo", "mac_dgs10", "mac_t10y2y"]
PRICE_COLS = ["mom_3m","mom_6m","mom_12m","mom_12m1m","vol_63d","vol_126d","dd_252d","rs_3m","rs_6m","rs_12m"]
FEATURE_COLS = PRICE_COLS + MACRO_COLS

def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_rank"] = df.groupby("date")["fwd_ret_1m"].rank(method="average")
    # rescale per-date min->0, max->1 so endpoints are stable for tests/training
    g = df.groupby("date")["y_rank"]
    df["y_rank"] = (df["y_rank"] - g.transform("min")) / (g.transform("max") - g.transform("min"))
    # guard against single ticker or all-equal returns on a date: fillna with 0.5 (neutral value)
    df["y_rank"] = df["y_rank"].fillna(0.5)
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
