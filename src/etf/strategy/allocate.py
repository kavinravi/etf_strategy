from __future__ import annotations
import numpy as np

def project_to_simplex(raw: np.ndarray, max_weight: float, min_weight: float = 0.0, total: float = 1.0) -> np.ndarray:
    raw = np.asarray(raw, float)
    n = len(raw)
    assert min_weight * n <= total + 1e-12 <= max_weight * n + 1e-12, "infeasible bounds"
    w = np.exp(raw - raw.max())
    w = w / w.sum() * total
    for _ in range(1000):
        w = np.clip(w, min_weight, max_weight)
        deficit = total - w.sum()
        if abs(deficit) < 1e-12:
            break
        if deficit > 0:
            room = max_weight - w
            free = room > 1e-15
        else:
            room = w - min_weight
            free = room > 1e-15
        if not free.any():
            break
        share = room[free] / room[free].sum()
        w[free] += deficit * share
    w = np.clip(w, min_weight, max_weight)
    s = w.sum()
    return w / s * total if s > 0 else w

def tilt_weights(scores: np.ndarray, tilt_scale: float, max_weight: float,
                 min_weight: float, anchor: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, float); n = len(scores)
    if anchor is None:
        anchor = np.full(n, 1.0 / n)
    sd = scores.std()
    z = (scores - scores.mean()) / sd if sd > 0 else np.zeros(n)
    raw = np.log(anchor) + tilt_scale * z
    return project_to_simplex(raw, max_weight, min_weight)

def apply_no_trade_band(target: np.ndarray, current: np.ndarray, band: float,
                        max_weight: float, min_weight: float) -> np.ndarray:
    target = np.asarray(target, float)
    current = np.asarray(current, float)
    hold = np.abs(target - current) < band
    if hold.all():
        return current.copy()
    free = ~hold
    n_free = int(free.sum())
    budget = 1.0 - current[hold].sum()
    # If the freed budget can't fit the traded sectors within [min,max], fall back
    # to projecting the whole vector (band relaxed) so we always return a valid book.
    if not (min_weight * n_free <= budget + 1e-12 <= max_weight * n_free + 1e-12):
        return project_to_simplex(np.log(np.clip(target, 1e-9, None)), max_weight, min_weight)
    w = current.copy()
    w[free] = project_to_simplex(np.log(np.clip(target[free], 1e-9, None)), max_weight, min_weight, total=budget)
    w[hold] = current[hold]   # strictly preserved
    return w
