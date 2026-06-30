import numpy as np
import pandas as pd
from etf.strategy.strategy_a import run_strategy_a
from etf.utils.walkforward import expanding_walks
from etf.features.build_features import FEATURE_COLS


def _synth_features(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2002, 2007):
        for m in range(1, 13):
            date = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
            for tk in ["A", "B", "C", "D", "E"]:
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
    assert res["returns"].mean() > 0.0


def test_universe_change_keeps_weights_valid():
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2002, 2007):
        for m in range(1, 13):
            date = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
            tickers = ["A", "B", "C", "D", "E"] if year <= 2005 else ["A", "B", "C", "D"]  # E exits in 2006
            for tk in tickers:
                feat = {c: rng.normal() for c in FEATURE_COLS}
                fwd = 0.02 * feat["mom_3m"] + rng.normal(0, 0.01)
                rows.append({"date": date, "ticker": tk, "fwd_ret_1m": fwd, **feat})
    df = pd.DataFrame(rows)
    df["y_rank"] = df.groupby("date")["fwd_ret_1m"].rank(pct=True)
    walks = expanding_walks(2005, 2006)  # test 2005 (E present) then 2006 (E absent)
    cfg = {"xgb_params": {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
                          "objective": "reg:squarederror"},
           "tilt_scale": 0.5, "max_weight": 0.4, "min_weight": 0.05,
           "no_trade_band": 0.05, "cost_bps": 5.0}
    res = run_strategy_a(df, walks, cfg, seed=0)
    w = res["weights"]
    sums = w.groupby("date")["weight"].sum()
    assert np.allclose(sums, 1.0, atol=1e-6)   # holds across the E-exit month
    assert (w["weight"] >= 0.05 - 1e-9).all() and (w["weight"] <= 0.4 + 1e-9).all()
    assert not ((w["date"].dt.year == 2006) & (w["ticker"] == "E")).any()  # E fully exited
