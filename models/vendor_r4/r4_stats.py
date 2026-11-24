"""Participant-level inference, frozen from Paper A Section 2.3: exact two-sided sign-flip test over all 2^n sign
patterns, paired t intervals, SESOI 2 pp labels (relevant / practically null via TOST 90 % interval /
inconclusive) and Holm adjustment."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

SESOI = 2.0


def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def sign_flip_p(d) -> float:
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n > 20:
        raise ValueError("exact enumeration limited to n <= 20")
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    null = np.abs((signs * d).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12))


def t_interval(d, level: float):
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return float(m - q * se), float(m + q * se)


def label(d) -> str:
    lo95, _ = t_interval(d, 0.95)
    lo90, hi90 = t_interval(d, 0.90)
    if np.mean(d) >= SESOI and lo95 > 0:
        return "relevant"
    if lo90 > -SESOI and hi90 < SESOI:
        return "practically_null"
    return "inconclusive"


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def summarise(d) -> dict:
    """Participants without any trial in a subset (NaN) are dropped and counted (pre-specified)."""
    d = np.asarray(d, dtype=np.float64)
    n_nan = int(np.isnan(d).sum())
    d = d[~np.isnan(d)]
    return {"n": int(d.size), "n_participants_without_trials_in_subset": n_nan, "mean_pp": round(float(d.mean()), 3),
            "ci95_pp": [round(x, 3) for x in t_interval(d, 0.95)],
            "ci90_pp": [round(x, 3) for x in t_interval(d, 0.90)],
            "n_positive": int((d > 0).sum()), "sign_flip_p": sign_flip_p(d), "label": label(d),
            "sorted_values_pp_unlabelled": [round(float(x), 3) for x in np.sort(d)]}
