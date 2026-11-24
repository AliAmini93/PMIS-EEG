"""Participant-level inference for large n (R6-C3): t interval and Monte Carlo sign-flip (fixed seed)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from evaluation.r6_indep_stats import participant_values

MC_DRAWS, MC_SEED = 1_000_000, 7499


def sign_flip_mc(v: np.ndarray, draws: int = MC_DRAWS, seed: int = MC_SEED) -> float:
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(seed)
    obs = abs(v.mean())
    hits = 0
    for i in range(0, draws, 100_000):
        s = rng.choice([-1.0, 1.0], size=(min(100_000, draws - i), v.size))
        hits += int(np.sum(np.abs(s @ v) / v.size >= obs - 1e-12))
    return float((hits + 1) / (draws + 1))


def summary(participant, label, value, scale: float = 100.0, test: bool = True) -> dict:
    pv = participant_values(participant, label, value)
    v = np.array(list(pv.values()))
    out = {"n_participants": int(v.size), "n_pairs": int(np.size(value))}
    if v.size < 2:
        return out
    se = v.std(ddof=1) / np.sqrt(v.size)
    q = stats.t.ppf(0.975, v.size - 1)
    out.update({"mean_points": float(v.mean() * scale),
                "ci95_points": [float((v.mean() - q * se) * scale), float((v.mean() + q * se) * scale)],
                "sd_between_participants_points": float(v.std(ddof=1) * scale),
                "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum())})
    if test:
        out["sign_flip_p_monte_carlo"] = sign_flip_mc(v)
        out["sign_flip_mc_draws"] = MC_DRAWS
    return out
