from __future__ import annotations
import pandas as pd

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
