from __future__ import annotations
import numpy as np
import pandas as pd
from etf.strategy.strategy_a import run_strategy_a
from etf.utils.backtest import compute_strategy_metrics

def label_shuffle_sharpe(features, walks, config, seed, n_shuffles=20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_shuffles):
        f = features.copy()
        f["y_rank"] = f.groupby("date")["y_rank"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        r = run_strategy_a(f, walks, config, seed)["returns"]
        out.append(compute_strategy_metrics(r)["sharpe"])
    return np.array(out)

def feature_timing_gain(features, walks, config, seed, feature: str) -> float:
    leaked = features.copy()
    leaked[feature] = leaked.groupby("ticker")[feature].shift(-1)  # pull next month's value into this row
    keep = leaked[feature].notna()
    leaked = leaked[keep]
    base = features[keep]  # same (date, ticker) rows, original feature timing
    real = compute_strategy_metrics(run_strategy_a(base, walks, config, seed)["returns"])["sharpe"]
    leaked_sharpe = compute_strategy_metrics(run_strategy_a(leaked, walks, config, seed)["returns"])["sharpe"]
    return float(leaked_sharpe - real)
