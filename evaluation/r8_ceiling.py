"""R8-C: is the weak-decoder sensitivity a probability-ceiling artefact? (analysis only, exploratory: data examined)

For each source of pair records (model x cohort) with fields pAA, pAB, pBA, pBB (correct-class probabilities):
  d_prob   = 1/2 [(pAA - pAB) + (pBB - pBA)]                         (the frozen estimand)
  d_logit  = 1/2 [(L(pAA) - L(pAB)) + (L(pBB) - L(pBA))], L = logit of p clipped to [eps, 1 - eps]; eps 1e-3 primary,
             1e-4 and 1e-2 sensitivity (bounded-outcome sensitivity)
  baseline = partner-arm probability, q = 1/2 (pAB + pBA); headroom = 1 - q
Participant level (class-balanced means as in every R6/R7 analysis):
  r(d, BA) for d_prob and d_logit, with BA = participant partner-arm balanced accuracy (from the pAB / pBA decisions)
  partial r(d, BA | mean headroom)
  d within fixed baseline bands q in [0, 0.5), [0.5, 0.8), [0.8, 0.95), [0.95, 1] (pooled, participant means; bands with
  >= 10 participants having >= 5 pairs) and r(d_logit, BA) within the middle band [0.5, 0.8) (log-odds scale: on the
  probability scale even a narrow band confounds position-within-band with headroom; shown in tests/test_r8_c_synthetic.py)
  headroom-normalised d_rel = d_prob / headroom on pairs with q <= 0.95 only (no normalisation by tiny headroom)
Interpretation rule (declared before computing): the weak-decoder pattern is called largely a CEILING ARTEFACT if r(d_logit, BA)
and the within-band r are both > -0.3 while r(d_prob, BA) < -0.6; SUBSTANTIVE if r(d_logit, BA) < -0.5 and the within-band r
< -0.3; otherwise MIXED. Correlations are reported with 95 % CIs (Fisher z) over participants; no p-value claims.
"""

from __future__ import annotations

import numpy as np

EPS = (1e-3, 1e-4, 1e-2)
BANDS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0001))


def logit(p, eps):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cb_mean(v, subj, lab, s):
    m = subj == s
    vals = [np.nanmean(v[m & (lab == c)]) for c in (1, 2) if np.any(m & (lab == c) & np.isfinite(v))]
    return float(np.mean(vals)) if len(vals) == 2 else np.nan


def corr_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return {"n": int(n), "r": None}
    r = float(np.corrcoef(x, y)[0, 1])
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(n - 3)
    return {"n": int(n), "r": r, "ci95": [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]}


def partial_r(x, y, zc):
    x, y, zc = (np.asarray(v, float) for v in (x, y, zc))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(zc)
    x, y, zc = x[ok], y[ok], zc[ok]
    A = np.c_[np.ones(zc.size), zc]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return corr_ci(rx, ry)


def audit(rec: dict) -> dict:
    subj, lab = np.asarray(rec["subject"]), np.asarray(rec["label"])
    pAA, pAB, pBA, pBB = (np.asarray(rec[k], float) for k in ("pAA", "pAB", "pBA", "pBB"))
    parts = np.unique(subj)
    q = 0.5 * (pAB + pBA)
    dprob = 0.5 * ((pAA - pAB) + (pBB - pBA))
    ba = 0.5 * ((pAB > 0.5).astype(float) + (pBA > 0.5).astype(float))
    P = lambda v: np.array([cb_mean(v, subj, lab, s) for s in parts])  # noqa: E731
    out = {"n_participants": int(parts.size), "n_pairs": int(subj.size)}
    BA, Dp, H = P(ba), P(dprob), P(1 - q)
    out["d_prob_mean_points"] = float(np.nanmean(Dp) * 100)
    out["r_dprob_BA"] = corr_ci(Dp, BA)
    out["partial_r_dprob_BA_given_headroom"] = partial_r(Dp, BA, H)
    for eps in EPS:
        dl = 0.5 * ((logit(pAA, eps) - logit(pAB, eps)) + (logit(pBB, eps) - logit(pBA, eps)))
        Dl = P(dl)
        out[f"logit_eps{eps:g}"] = {"d_logit_mean": float(np.nanmean(Dl)), "r_dlogit_BA": corr_ci(Dl, BA),
                                     "partial_r_dlogit_BA_given_headroom": partial_r(Dl, BA, H)}
    bands = {}
    for lo, hi in BANDS:
        sel = (q >= lo) & (q < hi)
        cnt = np.array([np.sum(sel & (subj == s)) for s in parts])
        v = np.where(sel, dprob, np.nan)
        Db = P(v)
        keep = cnt >= 5
        bands[f"q_{lo:g}_{min(hi, 1):g}"] = {"pairs": int(sel.sum()), "participants_with_5_pairs": int(keep.sum()),
                                            "d_prob_mean_points": float(np.nanmean(Db[keep]) * 100) if keep.sum() >= 10 else None}
    out["baseline_bands"] = bands
    mid = (q >= 0.5) & (q < 0.8)
    cnt = np.array([np.sum(mid & (subj == s)) for s in parts])
    dl3 = 0.5 * ((logit(pAA, 1e-3) - logit(pAB, 1e-3)) + (logit(pBB, 1e-3) - logit(pBA, 1e-3)))
    Dm = P(np.where(mid, dl3, np.nan))
    out["r_dlogit_BA_within_band_0.5_0.8"] = corr_ci(np.where(cnt >= 5, Dm, np.nan), BA) if (cnt >= 5).sum() >= 10 else {"n": int((cnt >= 5).sum()), "r": None}
    ok = q <= 0.95
    out["d_rel_headroom_normalised_q_le_0.95"] = {"pairs": int(ok.sum()), "mean": float(np.nanmean(P(np.where(ok, dprob / (1 - q), np.nan)))),
                                                   "r_with_BA": corr_ci(P(np.where(ok, dprob / (1 - q), np.nan)), BA)}
    rp, rl, rb = out["r_dprob_BA"]["r"], out["logit_eps0.001"]["r_dlogit_BA"]["r"], out["r_dlogit_BA_within_band_0.5_0.8"].get("r")
    if rp is not None and rl is not None:
        if rl > -0.3 and (rb is None or rb > -0.3) and rp < -0.6:
            verdict = "LARGELY_CEILING_ARTEFACT"
        elif rl < -0.5 and rb is not None and rb < -0.3:
            verdict = "SUBSTANTIVE"
        else:
            verdict = "MIXED"
        out["verdict"] = verdict
    return out
