"""R6-C4 analysis (frozen before execution)."""

from __future__ import annotations

import numpy as np

from experiments.pmis_r6.c4 import OVERLAP, VARIANTS
from evaluation.r6_mc_stats import summary

SECONDARY = ("S1_sample_identity_increment_d100_minus_d0", "S2_same_episode_no_shared_samples_d0",
             "S3_episode_beyond_precue_d0_minus_dP", "S4_precue_same_trial_dP")


def dv(r, v, p=""):
    return 0.5 * ((r[f"{p}{v}_pAA"] - r[f"{p}{v}_pAB"]) + (r[f"{p}{v}_pBB"] - r[f"{p}{v}_pBA"]))


def S(recs, fn, test=True):
    return summary(np.array([r["subject"] for r in recs]), np.array([r["label"] for r in recs]),
                   np.array([fn(r) for r in recs], dtype=np.float64), test=test)


def holm(p):
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def analyse(recs):
    curve = {v: S(recs, lambda r, v=v: dv(r, v), test=(v == "V0.0")) for v in VARIANTS}
    for v in VARIANTS:
        curve[v]["overlap"] = OVERLAP[v]
    sec = {SECONDARY[0]: S(recs, lambda r: dv(r, "V0.0") - dv(r, "V2.0")),
           SECONDARY[1]: S(recs, lambda r: dv(r, "V2.0")),
           SECONDARY[2]: S(recs, lambda r: dv(r, "V2.0") - dv(r, "P")),
           SECONDARY[3]: S(recs, lambda r: dv(r, "P"))}
    adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in sec.items()})
    for k in sec:
        sec[k]["holm_p_secondary_family"] = adj[k]
    means = [curve[v]["mean_points"] for v in VARIANTS[:5]]
    law = {}
    for v in VARIANTS:
        ex = np.array([dv(r, v) for r in recs])
        pr = np.array([dv(r, v, "nt_") for r in recs])
        s_pr = S(recs, lambda r, v=v: dv(r, v, "nt_"), test=False)
        law[v] = {"prediction_mean_points": s_pr["mean_points"], "exact_mean_points": curve[v]["mean_points"],
                  "pair_pearson_r": float(np.corrcoef(ex, pr)[0, 1])}
    cos = {v: float(np.mean([r[f"cos_{v}"] for r in recs])) for v in VARIANTS}
    return {"primary_curve": curve, "secondary": sec,
            "H2_monotone_in_overlap_100_75_50_25_0": bool(all(x > y for x, y in zip(means, means[1:]))),
            "fraction_of_identity_effect_without_shared_samples_d0_over_d100": means[4] / means[0],
            "influence_law_per_variant": law, "descriptive_mean_tangent_cosine_copy_vs_scored": cos,
            "pair_counts": {"pairs": len(recs), "participants": int(np.unique([r["subject"] for r in recs]).size)}}
