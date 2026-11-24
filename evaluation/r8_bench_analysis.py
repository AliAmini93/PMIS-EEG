"""R8-B analysis (frozen before execution): R4 aggregation (BA pooled per participant over scored trials, mean over seeds)."""

from __future__ import annotations

import numpy as np

from models.vendor_r4 import r4_stats as rs

FAMILY = ("H1_EEGNet_LEAKED_minus_PLACEBO_BA", "H2_EEGNet_minus_TSLR_LEAKED_minus_PLACEBO_BA",
          "H3_support_bound_below_half_of_identity")


def summ(d):
    return {k: v for k, v in rs.summarise(d).items() if k not in ("sorted_values_pp_unlabelled", "label")}


def per_part(fn, subj, parts):
    return np.array([fn(subj == p) for p in parts])


def analyse(A, dist, exposed, seeds, tslr_c7):
    """A[seed][key] over distinct trials (NaN where unscored): p_/c_ for L, P, C, LCB; exposed[seed] bool masks;
    tslr_c7: dict seed -> {'r4_pred_L','r4_pred_P'} TS+LR decisions on the same arms (C7, for H2)."""
    subj, y = dist["subject"], dist["label"]
    parts = np.unique(subj)

    def ba(s, arm, mask=None):
        def f(m):
            m = m & np.isfinite(A[s][f"c_{arm}"]) & (True if mask is None else mask)
            if not m.sum() or np.unique(y[m]).size < 2:
                return np.nan
            return 100.0 * np.mean([np.mean(A[s][f"c_{arm}"][m & (y == c)]) for c in (1, 2)])
        return per_part(f, subj, parts)

    def prob(s, arm, mask=None):
        def f(m):
            m = m & np.isfinite(A[s][f"p_{arm}"]) & (True if mask is None else mask)
            return 100.0 * np.mean([np.mean(A[s][f"p_{arm}"][m & (y == c)]) for c in (1, 2)]) if m.sum() else np.nan
        return per_part(f, subj, parts)

    B = {a: np.mean([ba(s, a) for s in seeds], axis=0) for a in ("L", "P", "C", "LCB")}
    Pp = {a: np.mean([prob(s, a) for s in seeds], axis=0) for a in ("L", "P", "C", "LCB")}
    exL = np.nanmean([ba(s, "L", exposed[s]) for s in seeds], axis=0)
    exP = np.nanmean([ba(s, "P", exposed[s]) for s in seeds], axis=0)
    unL = np.nanmean([ba(s, "L", ~exposed[s]) for s in seeds], axis=0)
    unP = np.nanmean([ba(s, "P", ~exposed[s]) for s in seeds], axis=0)
    pexL = np.nanmean([prob(s, "L", exposed[s]) for s in seeds], axis=0)
    pexP = np.nanmean([prob(s, "P", exposed[s]) for s in seeds], axis=0)
    # TS+LR (C7 frozen R4 estimator) BA on the same arms and seeds
    tb = {}
    for arm in ("L", "P"):
        def f(m, s, arm=arm):
            pred = tslr_c7[s][f"r4_pred_{arm}"]
            m = m & (pred >= 0)
            return 100.0 * rs.balanced_accuracy(y[m], pred[m]) if m.sum() and np.unique(y[m]).size == 2 else np.nan
        tb[arm] = np.mean([per_part(lambda m, s=s: f(m, s), subj, parts) for s in seeds], axis=0)
    lp = B["L"] - B["P"]
    sup = B["C"] - B["LCB"]
    fam = {FAMILY[0]: summ(lp), FAMILY[1]: summ(lp - (tb["L"] - tb["P"])), FAMILY[2]: summ(0.5 * lp - sup)}
    adj = rs.holm({k: v["sign_flip_p"] for k, v in fam.items()}) if hasattr(rs, "holm") else None
    if adj is None:
        keys = sorted(fam, key=lambda k: fam[k]["sign_flip_p"])
        run, adj = 0.0, {}
        for i, k in enumerate(keys):
            run = max(run, min(1.0, (len(keys) - i) * fam[k]["sign_flip_p"]))
            adj[k] = run
    for k in fam:
        fam[k]["holm_p_family"] = adj[k]
    desc = {"absolute_BA_mean": {a: float(np.mean(v)) for a, v in B.items()},
            "PLACEBO_minus_CLEAN_BA": summ(B["P"] - B["C"]), "LEAKED_minus_CLEAN_BA": summ(B["L"] - B["C"]),
            "learning_curve_CLEAN_minus_LCB_BA": summ(sup),
            "exposed_only_LEAKED_minus_PLACEBO_BA": summ(exL - exP), "unexposed_only_LEAKED_minus_PLACEBO_BA": summ(unL - unP),
            "probability_LEAKED_minus_PLACEBO_points": summ(Pp["L"] - Pp["P"]),
            "probability_exposed_only_LEAKED_minus_PLACEBO_points": summ(pexL - pexP),
            "TSLR_C7_same_arms_LEAKED_minus_PLACEBO_BA": summ(tb["L"] - tb["P"]),
            "ratio_support_bound_to_identity_effect_mean": float(np.mean(sup) / np.mean(lp)) if np.mean(lp) != 0 else None}
    return {"family_holm": fam, "descriptive": desc, "n_participants": int(parts.size), "seeds": list(seeds)}
