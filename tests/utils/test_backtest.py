import numpy as np, pandas as pd
from etf.utils.backtest import compute_strategy_metrics

def test_zero_turnover_means_gross_equals_net():
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    m_gross = compute_strategy_metrics(r)
    m_net = compute_strategy_metrics(r, turnover=pd.Series([0,0,0,0]), cost_bps=5.0)
    assert np.isclose(m_gross["total_return"], m_net["total_return"])

def test_known_metrics():
    r = pd.Series([0.10, -0.10, 0.10, -0.10] * 6)  # 24 months
    m = compute_strategy_metrics(r)
    assert m["max_drawdown"] < 0
    assert np.isclose(m["ann_vol"], r.std(ddof=1) * np.sqrt(12))
    assert set(m) >= {"sharpe","sortino","calmar","hit_rate","ann_return","ann_vol","max_drawdown","total_return","avg_turnover"}

def test_cost_reduces_return():
    r = pd.Series([0.02, 0.02, 0.02])
    base = compute_strategy_metrics(r, turnover=pd.Series([1.0,1.0,1.0]), cost_bps=0.0)
    costed = compute_strategy_metrics(r, turnover=pd.Series([1.0,1.0,1.0]), cost_bps=50.0)
    assert costed["total_return"] < base["total_return"]
