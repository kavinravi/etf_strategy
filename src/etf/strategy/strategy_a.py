from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from etf.strategy.allocate import tilt_weights, apply_no_trade_band
from etf.features.build_features import FEATURE_COLS


def train_and_score(train_df, test_df, feature_cols, params, seed):
    model = XGBRegressor(random_state=seed, n_jobs=1, **params)
    model.fit(train_df[feature_cols], train_df["y_rank"])
    out = test_df.copy()
    out["score"] = model.predict(out[feature_cols])
    return out


def run_strategy_a(features: pd.DataFrame, walks, config, seed) -> dict:
    p = config
    scored_parts, weight_rows, rets, turns = [], [], [], []
    prev = None  # pd.Series indexed by ticker
    for walk in walks:
        tr, _, te = walk.masks(features["date"])
        train_df, test_df = features[tr], features[te]
        if train_df.empty or test_df.empty:
            continue
        scored = train_and_score(train_df, test_df, FEATURE_COLS, p["xgb_params"], seed)
        scored_parts.append(scored)
        for d, sub in scored.groupby("date"):
            sub = sub.sort_values("ticker")
            tickers = list(sub["ticker"])
            target = tilt_weights(sub["score"].to_numpy(), p["tilt_scale"],
                                  p["max_weight"], p["min_weight"])
            tgt = pd.Series(target, index=tickers)
            if prev is None:
                w = tgt
                to = 1.0
            else:
                cur = prev.reindex(tickers, fill_value=0.0)
                cur_sum = cur.sum()
                if cur_sum > 0:
                    cur = cur / cur_sum   # renormalize after any ticker exit so the band sees a valid portfolio
                banded = apply_no_trade_band(tgt.to_numpy(), cur.to_numpy(),
                                             p["no_trade_band"], p["max_weight"], p["min_weight"])
                w = pd.Series(banded, index=tickers)
                allt = sorted(set(tickers) | set(prev.index))
                to = float((w.reindex(allt, fill_value=0.0) - prev.reindex(allt, fill_value=0.0)).abs().sum())
            r = float((w * sub.set_index("ticker")["fwd_ret_1m"].reindex(w.index)).sum())
            for tk, wt in w.items():
                weight_rows.append({"date": d, "ticker": tk, "weight": wt})
            rets.append((d, r))
            turns.append((d, to))
            prev = w
    weights = pd.DataFrame(weight_rows)
    ridx = pd.to_datetime([d for d, _ in rets])
    return {
        "weights": weights,
        "returns": pd.Series([r for _, r in rets], index=ridx),
        "turnover": pd.Series([t for _, t in turns], index=ridx),
        "scores": pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame(),
    }
