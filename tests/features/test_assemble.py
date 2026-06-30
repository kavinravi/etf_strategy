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

def test_label_single_ticker_date_is_neutral():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31"]*1 + ["2020-02-29"]*3),
        "ticker": ["A", "A","B","C"],
        "fwd_ret_1m": [0.01, -0.05, 0.0, 0.05],
    })
    out = add_label(df)
    # single-ticker date should have y_rank = 0.5 (neutral, no cross-sectional ordering)
    jan = out[out.date == "2020-01-31"]
    assert len(jan) == 1
    assert jan.iloc[0]["y_rank"] == 0.5
    # all values should be in [0, 1] with no NaN
    assert out.y_rank.between(0, 1).all()
    assert out.y_rank.notna().all()
