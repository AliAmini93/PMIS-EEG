"""R8-A analysis (frozen before execution). 9 participants: exact sign-flip; class-balanced participant means."""

from __future__ import annotations

import numpy as np

from evaluation.r6_indep_stats import summary

FAMILY = ("H1_TSLR_d", "H2_EEGNet_d", "H3_EEGNet_minus_TSLR")
LAW_RATIO, LAW_R = (0.8, 1.25), 0.9


def d_of(r, p=""):
    return 0.5 * ((r[p + "pAA"] - r[p + "pAB"]) + (r[p + "pBB"] - r[p + "pBA"]))


def plc(r, p=""):
    return 0.5 * r["sign"] * ((r[p + "pAC"] - r[p + "pAE"]) + (r[p + "pBE"] - r[p + "pBC"]))


def bac(r, p=""):
    c = lambda q: float(r[p + "p" + q] > 0.5)  # noqa: E731
    return 0.5 * ((c("AA") - c("AB")) + (c("BB") - c("BA")))


def S(recs, fn, test=True):
    return summary(np.array([r["subject"] for r in recs]), np.array([r["label"] for r in recs]),
                   np.array([fn(r) for r in recs], dtype=float), test=test)


def holm(p):
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def analyse(recs, aa_max):
    """recs carry TS+LR fields (t_*), its one-step prediction (nt_*) and EEGNet fields (e_*)."""
    fam = {"H1_TSLR_d": S(recs, lambda r: d_of(r, "t_")), "H2_EEGNet_d": S(recs, lambda r: d_of(r, "e_")),
           "H3_EEGNet_minus_TSLR": S(recs, lambda r: d_of(r, "e_") - d_of(r, "t_"))}
    adj = holm({k: v["exact_sign_flip_p"] for k, v in fam.items()})
    for k in fam:
        fam[k]["holm_p_family"] = adj[k]
    ex = np.array([d_of(r, "t_") for r in recs])
    pr = np.array([d_of(r, "nt_") for r in recs])
    law = {"prediction": S(recs, lambda r: d_of(r, "nt_"), test=False), "pair_pearson_r": float(np.corrcoef(ex, pr)[0, 1])}
    law["ratio"] = law["prediction"]["mean_points"] / fam["H1_TSLR_d"]["mean_points"]
    law["met"] = bool(LAW_RATIO[0] <= law["ratio"] <= LAW_RATIO[1] and law["pair_pearson_r"] >= LAW_R)
    desc = {}
    for m, p in (("TSLR", "t_"), ("EEGNet", "e_")):
        desc[f"{m}_placebo"] = S(recs, lambda r, p=p: plc(r, p))
        desc[f"{m}_balanced_accuracy_contrast"] = S(recs, lambda r, p=p: bac(r, p), test=False)
        desc[f"{m}_partner_arm_correct_class_probability"] = S(recs, lambda r, p=p: 0.5 * (r[p + "pAB"] + r[p + "pBA"]), test=False)
        desc[f"{m}_near_le6"] = S([r for r in recs if r["pos_dist"] <= 6], lambda r, p=p: d_of(r, p), test=False)
        desc[f"{m}_far_gt6"] = S([r for r in recs if r["pos_dist"] > 6], lambda r, p=p: d_of(r, p), test=False)
    desc["EEGNet_AA_determinism_max_abs"] = aa_max
    return {"family_holm": fam, "H4_TSLR_influence_law": law, "descriptive": desc,
            "pair_counts": {"pairs": len(recs), "participants": int(np.unique([r["subject"] for r in recs]).size)}}
