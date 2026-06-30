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
