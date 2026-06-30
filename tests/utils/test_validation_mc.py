import numpy as np
import pandas as pd
from etf.utils.validation import block_bootstrap_ci, deflated_sharpe, dirichlet_null
from etf.utils.walkforward import expanding_walks
from tests.strategy.test_strategy_a import _synth_features

def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.04, 120))
    ci = block_bootstrap_ci(r, n_boot=500, mean_block=6, seed=0)
    assert ci["sharpe_lo"] <= ci["sharpe_point"] <= ci["sharpe_hi"]

def test_deflated_sharpe_drops_with_more_trials():
    assert deflated_sharpe(1.0, n_trials=1, n_obs=120) > deflated_sharpe(1.0, n_trials=100, n_obs=120)

def test_dirichlet_null_returns_pvalue_in_unit_interval():
    feats = _synth_features(seed=2)
    walks = expanding_walks(2005, 2005)
    cfg = {"xgb_params": {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1,
            "objective": "reg:squarederror"},
           "tilt_scale": 0.7, "max_weight": 0.4, "min_weight": 0.05,
           "no_trade_band": 0.0, "cost_bps": 5.0}
    res = dirichlet_null(feats, walks, cfg, n_sims=100, seed=0)
    assert 0.0 <= res["p_value"] <= 1.0
