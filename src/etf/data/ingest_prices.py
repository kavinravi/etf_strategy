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
