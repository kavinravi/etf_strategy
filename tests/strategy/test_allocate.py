import numpy as np
from etf.strategy.allocate import project_to_simplex, tilt_weights, apply_no_trade_band

def test_projection_respects_bounds_and_sums_to_one():
    w = project_to_simplex(np.array([5.0, 0.0, -3.0, 1.0, 0.5]), max_weight=0.4, min_weight=0.1)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0.1 - 1e-9).all() and (w <= 0.4 + 1e-9).all()

def test_tilt_scale_zero_returns_anchor():
    scores = np.array([3.0, -1.0, 0.0, 2.0])
    w = tilt_weights(scores, tilt_scale=0.0, max_weight=0.5, min_weight=0.0)
    assert np.allclose(w, 0.25, atol=1e-9)

def test_tilt_overweights_high_scores_within_cap():
    scores = np.array([2.0, 0.0, 0.0, -2.0])
    w = tilt_weights(scores, tilt_scale=0.5, max_weight=0.4, min_weight=0.05)
    assert w.argmax() == 0 and w.argmin() == 3
    assert np.isclose(w.sum(), 1.0)

def test_no_trade_band_holds_small_moves():
    cur = np.array([0.25, 0.25, 0.25, 0.25])
    tgt = np.array([0.26, 0.40, 0.10, 0.24])  # only index 1,2 exceed band 0.05
    w = apply_no_trade_band(tgt, cur, band=0.05, max_weight=0.5, min_weight=0.0)
    assert np.isclose(w[0], cur[0]) and np.isclose(w[3], cur[3])  # held
    assert np.isclose(w.sum(), 1.0)

def test_no_trade_band_preserves_held_weights_exactly():
    # Fixture where the mixed array does NOT sum to 1 before projection.
    # cur sums to 1.0 but the held subset (index 0 only) + target subset
    # would not sum to 1 if naively concatenated and re-projected.
    cur = np.array([0.20, 0.25, 0.25, 0.30])
    tgt = np.array([0.22, 0.50, 0.05, 0.23])  # |tgt[0]-cur[0]|=0.02 < band=0.05 → hold index 0
    w = apply_no_trade_band(tgt, cur, band=0.05, max_weight=0.6, min_weight=0.0)
    # Held weight must be EXACTLY preserved (not just close)
    assert np.isclose(w[0], cur[0], atol=1e-12), f"held weight drifted: {w[0]} != {cur[0]}"
    # Portfolio must still sum to 1
    assert np.isclose(w.sum(), 1.0, atol=1e-9), f"weights don't sum to 1: {w.sum()}"
    # All weights within [min_weight, max_weight]
    assert (w >= 0.0 - 1e-9).all() and (w <= 0.6 + 1e-9).all()
    # Traded sectors preserve target ordering: tgt[1]=0.50 > tgt[3]=0.23 > tgt[2]=0.05
    assert w[1] > w[3], f"w[1]={w[1]} should be > w[3]={w[3]} (targets 0.50 > 0.23)"
    assert w[3] > w[2], f"w[3]={w[3]} should be > w[2]={w[2]} (targets 0.23 > 0.05)"
