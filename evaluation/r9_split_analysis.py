"""R9-C analysis (frozen before execution). R4/R8-B aggregation: per participant, class-balanced accuracy (or mean
correct-class probability) pooled over the participant's scored trials of a fold seed, then averaged over seeds;
participant-level exact sign-flip (14 participants), Holm within the family."""

from __future__ import annotations

import numpy as np

from models.vendor_r4 import r4_stats as rs

FAMILY = ("C1_exposure_support_fixed_own_minus_otherhalf_BA",
          "C2_support_half_sameRun_testcopies_otherhalf_minus_placebo_BA",
          "C3_exposure_minus_support_BA")


def summ(d):
    return {k: v for k, v in rs.summarise(d).items() if k not in ("sorted_values_pp_unlabelled", "label")}


def analyse(A, half, dist, seeds, r8b=None):
    """A[seed][f'{p|c}_{L,L1,L2,P}'] over distinct trials (NaN where unscored); half[seed]: 0 unexposed-scored or unscored,
    1 = H1, 2 = H2 (exposed scored). r8b: optional R8-B trial arrays (dict seed -> {'p_L','p_P','c_L','c_P'})."""
    subj, y = dist["subject"], dist["label"]
    parts = np.unique(subj)

    def derived(s, kind):
        a, h = A[s], half[s]
        own = np.where(h == 1, a[f"{kind}_L1"], np.where(h == 2, a[f"{kind}_L2"], np.nan))
        oth = np.where(h == 1, a[f"{kind}_L2"], np.where(h == 2, a[f"{kind}_L1"], np.nan))
        return {"own": own, "other": oth, "L": a[f"{kind}_L"], "P": a[f"{kind}_P"], "L1": a[f"{kind}_L1"], "L2": a[f"{kind}_L2"]}

    def per_part(vals, mask, kind):
        out = []
        for p in parts:
            m = (subj == p) & mask & np.isfinite(vals)
            if kind == "c" and (not m.sum() or np.unique(y[m]).size < 2):
                out.append(np.nan)
                continue
            out.append(100.0 * np.mean([np.mean(vals[m & (y == c)]) for c in (1, 2)]) if m.sum() else np.nan)
        return np.array(out)

    def metric(kind, arm, which):
        res = []
        for s in seeds:
            v = derived(s, kind)[arm]
            scored = np.isfinite(A[s][f"{kind}_L"])
            mask = {"exposed": half[s] > 0, "unexposed": scored & (half[s] == 0), "all": scored}[which]
            res.append(per_part(v, mask, kind))
        return np.nanmean(res, axis=0)

    ex = {a: metric("c", a, "exposed") for a in ("own", "other", "L", "P")}
    exp_, sup = ex["own"] - ex["other"], ex["other"] - ex["P"]
    fam = {FAMILY[0]: summ(exp_), FAMILY[1]: summ(sup), FAMILY[2]: summ(exp_ - sup)}
    adj = rs.holm({k: v["sign_flip_p"] for k, v in fam.items()})
    for k in fam:
        fam[k]["holm_p_family"] = adj[k]
    tot = ex["own"] - ex["P"]
    un = {a: metric("c", a, "unexposed") for a in ("L", "L1", "L2", "P")}
    al = {a: metric("c", a, "all") for a in ("L", "P")}
    pr = {a: metric("p", a, "exposed") for a in ("own", "other", "L", "P")}
    desc = {
        "absolute_BA_exposed": {a: float(np.nanmean(v)) for a, v in ex.items()},
        "total_half_own_minus_placebo_BA_exposed": summ(tot),
        "support_share_of_total_half_ratio_of_means": float(np.nanmean(sup) / np.nanmean(tot)),
        "full_LEAKED_minus_PLACEBO_BA_exposed": summ(ex["L"] - ex["P"]),
        "full_LEAKED_minus_PLACEBO_BA_all_scored": summ(al["L"] - al["P"]),
        "full_minus_total_half_BA_exposed": summ((ex["L"] - ex["P"]) - tot),
        "support_half_unexposed_BA_mean_of_L1_L2_minus_P": summ(0.5 * (un["L1"] + un["L2"]) - un["P"]),
        "full_LEAKED_minus_PLACEBO_BA_unexposed": summ(un["L"] - un["P"]),
        "probability_exposure_own_minus_other_points": summ(pr["own"] - pr["other"]),
        "probability_support_other_minus_placebo_points": summ(pr["other"] - pr["P"]),
        "probability_full_L_minus_P_exposed_points": summ(pr["L"] - pr["P"]),
    }
    if r8b is not None:
        dmax = {a: float(max(np.nanmax(np.abs(A[s][f"p_{a}"] - r8b[s][f"p_{a}"])) for s in seeds)) for a in ("L", "P")}
        agree = {a: float(np.nanmean(np.concatenate([(A[s][f"c_{a}"] == r8b[s][f"c_{a}"])[np.isfinite(A[s][f"c_{a}"])]
                                                     for s in seeds]))) for a in ("L", "P")}
        desc["reproduction_of_R8B_arms"] = {"max_abs_dp": dmax, "decision_agreement": agree}
    return {"family_holm": fam, "descriptive": desc, "n_participants": int(parts.size), "seeds": list(seeds)}
