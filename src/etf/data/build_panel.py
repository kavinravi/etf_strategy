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
    # Keep inception rows (where ret_1m is NaN but fwd_ret_1m is not NaN)
    # Only drop rows where ret_1m is NaN and fwd_ret_1m is also NaN
    valid_rows = panel["ret_1m"].notna() | panel["fwd_ret_1m"].notna()
    return panel[valid_rows].sort_values(["date", "ticker"]).reset_index(drop=True)
