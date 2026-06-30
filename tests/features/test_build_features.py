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
