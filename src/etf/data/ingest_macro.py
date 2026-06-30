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
