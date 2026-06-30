import numpy as np, pandas as pd
from etf.utils.walkforward import Walk, expanding_walks
from etf.utils.benchmarks import equal_weight

def test_expanding_walks_are_sealed_and_sequential():
    walks = expanding_walks(2005, 2007)
    assert [w.test_year for w in walks] == [2005, 2006, 2007]
    w = walks[0]
    assert w.val_year == 2004 and w.train_end_year == 2003  # val=test-1, train<=test-2

def test_walk_masks_are_disjoint():
    dates = pd.Series(pd.to_datetime(["2003-06-30","2004-06-30","2005-06-30"]))
    tr, va, te = expanding_walks(2005, 2005)[0].masks(dates)
    assert (tr & va).sum() == 0 and (va & te).sum() == 0 and (tr & te).sum() == 0

def test_equal_weight_uses_available_sectors():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-31","2020-01-31","2020-02-29","2020-02-29"]),
        "ticker": ["A","B","A","B"],
        "fwd_ret_1m": [0.10, 0.00, -0.10, 0.20],
    })
    rets, turn = equal_weight(panel)
    assert np.isclose(rets.iloc[0], 0.05)   # mean(0.10, 0.00)
    assert np.isclose(rets.iloc[1], 0.05)   # mean(-0.10, 0.20)
