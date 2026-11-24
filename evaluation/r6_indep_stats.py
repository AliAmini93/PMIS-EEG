"""Independent participant-level aggregation and inference (R6 audit), written without pmis_g2.estimands.

pair value -> mean within (participant, class) -> equal class weight -> participant value -> mean over
participants; 95 % t interval (scipy t quantile); exact two-sided sign-flip by explicit enumeration of all
2^n sign patterns (bit masks), p = #{|mean*| >= |mean_obs| - 1e-12} / 2^n.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def participant_values(participant, label, value) -> dict:
    participant = np.asarray(participant)
    label = np.asarray(label)
    value = np.asarray(value, dtype=np.float64)
    out = {}
    for p in np.unique(participant):
        m = participant == p
        labs = np.unique(label[m])
        if labs.size != 2:
            continue
        out[p] = float(np.mean([value[m & (label == c)].mean() for c in labs]))
    return out


def sign_flip_p(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    masks = np.arange(2 ** n, dtype=np.int64)
    signs = 1.0 - 2.0 * ((masks[:, None] >> np.arange(n)[None, :]) & 1)
    means = np.abs(signs @ v) / n
    return float(np.mean(means >= abs(v.mean()) - 1e-12))


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
        out["exact_sign_flip_p"] = sign_flip_p(v)
    return out
