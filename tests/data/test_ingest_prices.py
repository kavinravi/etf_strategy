import numpy as np, pandas as pd
import pytest
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

@pytest.mark.live
def test_download_prices_live():
    from etf.data.ingest_prices import download_prices
    df = download_prices(["XLK", "SPY"], start="2020-01-01", end="2020-03-01")
    assert {"XLK", "SPY"}.issubset(df.columns) and len(df) > 20
