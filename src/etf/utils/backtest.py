from __future__ import annotations
import numpy as np

def compute_strategy_metrics(returns, turnover=None, cost_bps: float = 0.0,
                             periods_per_year: int = 12) -> dict:
    r = np.asarray(returns, float)
    if turnover is not None and cost_bps:
        r = r - (cost_bps / 1e4) * np.asarray(turnover, float)
    n = len(r)
    eq = np.cumprod(1.0 + r)
    total = eq[-1] - 1.0 if n else np.nan
    ann_ret = (1.0 + total) ** (periods_per_year / n) - 1.0 if n else np.nan
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year) if n > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    down = r[r < 0]
    dvol = down.std(ddof=1) * np.sqrt(periods_per_year) if len(down) > 1 else np.nan
    sortino = ann_ret / dvol if dvol and dvol > 0 else np.nan
    eq_dd = np.concatenate([[1.0], eq]) if n else np.array([np.nan])
    peak = np.maximum.accumulate(eq_dd)
    mdd = float((eq_dd / peak - 1.0).min()) if n else np.nan
    calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan
    avg_to = float(np.mean(turnover)) if turnover is not None else np.nan
    return {"total_return": float(total), "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol), "sharpe": float(sharpe), "sortino": float(sortino),
            "max_drawdown": mdd, "calmar": float(calmar),
            "hit_rate": float((r > 0).mean()) if n else np.nan, "avg_turnover": avg_to}
