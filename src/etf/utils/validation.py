from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from etf.strategy.strategy_a import run_strategy_a
from etf.utils.backtest import compute_strategy_metrics
from etf.strategy.allocate import project_to_simplex

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


def seed_stability(features, walks, config, seeds) -> pd.DataFrame:
    rows = []
    for s in seeds:
        m = compute_strategy_metrics(run_strategy_a(features, walks, config, s)["returns"])
        rows.append({"seed": s, "sharpe": m["sharpe"], "ann_return": m["ann_return"],
                     "max_drawdown": m["max_drawdown"]})
    return pd.DataFrame(rows)


def block_bootstrap_ci(returns, n_boot=5000, mean_block=6, seed=0, periods_per_year=12) -> dict:
    r = np.asarray(returns, float); n = len(r); rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    def sharpe(x):  # geometric annualization, mirrors compute_strategy_metrics
        v = x.std(ddof=1)
        if v <= 0 or len(x) == 0:
            return np.nan
        total = np.prod(1.0 + x) - 1.0
        ann_ret = (1.0 + total) ** (periods_per_year / len(x)) - 1.0
        return ann_ret / (v * np.sqrt(periods_per_year))
    sims = []
    for _ in range(n_boot):
        idx = np.empty(n, int); i = rng.integers(n)
        for k in range(n):
            idx[k] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        sims.append(sharpe(r[idx]))
    sims = np.array(sims)
    return {"sharpe_point": float(sharpe(r)),
            "sharpe_lo": float(np.nanpercentile(sims, 2.5)),
            "sharpe_hi": float(np.nanpercentile(sims, 97.5))}


def dirichlet_null(features, walks, config, n_sims=2000, seed=0) -> dict:
    base = run_strategy_a(features, walks, config, seed)
    strat_sharpe = compute_strategy_metrics(base["returns"])["sharpe"]
    sched = base["weights"][["date", "ticker"]].merge(
        features[["date", "ticker", "fwd_ret_1m"]], on=["date", "ticker"], how="left")
    dates = list(sched["date"].unique())
    # realized forward returns per test date, precomputed once (hot-loop hoist)
    fwd_by_date = [sched.loc[sched["date"] == d, "fwd_ret_1m"].to_numpy() for d in dates]
    rng = np.random.default_rng(seed); null = []
    for _ in range(n_sims):
        rets = []
        for fwd in fwd_by_date:
            # log(gamma(1,1)) -> project_to_simplex softmax -> true Dirichlet(1), then cap water-fill
            w = project_to_simplex(np.log(rng.gamma(1.0, 1.0, len(fwd))),
                                   config["max_weight"], config["min_weight"])
            rets.append(float((w * fwd).sum()))
        null.append(compute_strategy_metrics(pd.Series(rets))["sharpe"])
    null = np.array(null)
    return {"strategy_sharpe": float(strat_sharpe), "null_mean": float(np.nanmean(null)),
            "p_value": float(np.mean(null >= strat_sharpe))}


def deflated_sharpe(observed_sharpe, n_trials, n_obs, skew=0.0, kurt=3.0) -> float:
    """Bailey-Lopez de Prado deflated Sharpe probability.

    observed_sharpe and n_obs must be in the SAME (per-period, non-annualized)
    frequency, e.g. a monthly Sharpe with n_obs = number of months. kurt is RAW
    kurtosis (normal = 3), not excess kurtosis.
    """
    if n_trials <= 1:
        sr0 = 0.0
    else:
        e = 0.5772156649
        z1 = norm.ppf(1 - 1.0 / n_trials); z2 = norm.ppf(1 - 1.0 / (n_trials * e))
        sr0 = z1 * (1 - e) + z2 * e  # expected max Sharpe under the null (per-obs units handled below)
    sr0 = sr0 / np.sqrt(n_obs)       # scale null-max into per-observation Sharpe space
    num = (observed_sharpe - sr0) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * observed_sharpe + ((kurt - 1) / 4) * observed_sharpe ** 2)  # raw
    return float(norm.cdf(num / den))
