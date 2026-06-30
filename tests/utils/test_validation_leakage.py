import numpy as np
import pandas as pd
from etf.utils.validation import label_shuffle_sharpe
from etf.utils.validation import feature_timing_gain
from etf.utils.walkforward import expanding_walks
from etf.strategy.strategy_a import run_strategy_a
from etf.utils.backtest import compute_strategy_metrics
from etf.features.build_features import FEATURE_COLS
from tests.strategy.test_strategy_a import _synth_features  # reuse synthetic builder

def test_label_shuffle_destroys_performance():
    feats = _synth_features(seed=1)
    walks = expanding_walks(2005, 2006)
    cfg = {"xgb_params": {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.1,
            "objective": "reg:squarederror"},
           "tilt_scale": 0.7, "max_weight": 0.4, "min_weight": 0.05,
           "no_trade_band": 0.0, "cost_bps": 5.0}
    real = compute_strategy_metrics(run_strategy_a(feats, walks, cfg, 0)["returns"])["sharpe"]
    shuffled = label_shuffle_sharpe(feats, walks, cfg, seed=0, n_shuffles=10)
    # real signal should beat the median shuffled (null) run
    assert real > np.median(shuffled)


def _forward_dependent_features(seed=0):
    """Panel where fwd_ret_1m is driven by NEXT month's mom_3m, so leaking
    mom_3m forward (shift -1) should raise Sharpe -> positive feature_timing_gain."""
    rng = np.random.default_rng(seed)
    tickers = ["A", "B", "C", "D", "E"]
    dates = [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
             for y in range(2002, 2007) for m in range(1, 13)]
    rows = []
    for tk in tickers:
        driver = rng.normal(size=len(dates))             # this ticker's mom_3m path
        for i, d in enumerate(dates):
            feat = {c: rng.normal() for c in FEATURE_COLS}
            feat["mom_3m"] = driver[i]
            nxt = driver[i + 1] if i + 1 < len(dates) else np.nan
            fwd = 0.05 * nxt + rng.normal(0, 0.003)       # label driven by NEXT month's mom_3m
            rows.append({"date": d, "ticker": tk, "fwd_ret_1m": fwd, **feat})
    df = pd.DataFrame(rows).dropna(subset=["fwd_ret_1m"]).reset_index(drop=True)
    df["y_rank"] = df.groupby("date")["fwd_ret_1m"].rank(pct=True)
    return df

def test_feature_timing_gain_flags_forward_dependent_feature():
    feats = _forward_dependent_features(seed=0)
    walks = expanding_walks(2005, 2006)
    cfg = {"xgb_params": {"n_estimators": 80, "max_depth": 2, "learning_rate": 0.1,
            "objective": "reg:squarederror"},
           "tilt_scale": 0.7, "max_weight": 0.4, "min_weight": 0.05,
           "no_trade_band": 0.0, "cost_bps": 5.0}
    gain = feature_timing_gain(feats, walks, cfg, seed=0, feature="mom_3m")
    print(f"feature_timing_gain for mom_3m: {gain:.4f}")
    assert gain > 0.0
