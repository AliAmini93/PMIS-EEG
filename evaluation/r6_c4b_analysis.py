"""R6-C4b analysis (EEGNet overlap decomposition), frozen before execution."""

from __future__ import annotations

import numpy as np

from experiments.pmis_r6.c4b import VARIANTS
from evaluation.r6_mc_stats import summary

S5_MIN_DENOMINATOR = 0.005
SECONDARY = ("S1_sample_identity_increment_d100_minus_d0", "S2_same_episode_no_shared_samples_d0",
             "S3_episode_beyond_precue_d0_minus_dP", "S4_precue_same_trial_dP",
             "S5_EEGNet_minus_TSLR_fraction_without_shared_samples")


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


def analyse(recs, tslr_by_k, aa):
    """tslr_by_k[k][v] = C4 exact TS+LR d_v of pair k (same pairs)."""
    curve = {v: S(recs, lambda r, v=v: dv(r, v), test=(v == "V0.0")) for v in VARIANTS}
    tcurve = {v: S(recs, lambda r, v=v: tslr_by_k[int(r["k"])][v], test=False) for v in VARIANTS}
    # S5: per participant, fraction f = d(V2.0) / d(V0.0) for each model (same pairs); paired MC sign-flip on
    # f_EEGNet - f_TSLR. Pre-registered guard: not computed if any participant's d(V0.0) <= 0.5 points (either model).
    from evaluation.r6_indep_stats import participant_values
    from evaluation.r6_mc_stats import sign_flip_mc
    subj = np.array([r["subject"] for r in recs])
    lab = np.array([r["label"] for r in recs])
    pv = {v: participant_values(subj, lab, np.array([dv(r, v) for r in recs])) for v in ("V0.0", "V2.0")}
    tv = {v: participant_values(subj, lab, np.array([tslr_by_k[int(r["k"])][v] for r in recs])) for v in ("V0.0", "V2.0")}
    keys = sorted(pv["V0.0"])
    den_min = min(min(pv["V0.0"][k] for k in keys), min(tv["V0.0"][k] for k in keys))
    if den_min <= S5_MIN_DENOMINATOR:  # pre-registered guard: ratio undefined / unstable
        s5 = {"n_participants": len(keys), "computable": False, "min_denominator_points": den_min * 100}
    else:
        fe = np.array([pv["V2.0"][k] / pv["V0.0"][k] for k in keys])
        ft = np.array([tv["V2.0"][k] / tv["V0.0"][k] for k in keys])
        diff = fe - ft
        s5 = {"n_participants": len(keys), "computable": True, "median_fraction_EEGNet": float(np.median(fe)),
              "median_fraction_TSLR": float(np.median(ft)), "mean_difference": float(diff.mean()),
              "sign_flip_p_monte_carlo": sign_flip_mc(diff), "n_positive": int((diff > 0).sum()),
              "n_negative": int((diff < 0).sum())}
    sec = {SECONDARY[0]: S(recs, lambda r: dv(r, "V0.0") - dv(r, "V2.0")),
           SECONDARY[1]: S(recs, lambda r: dv(r, "V2.0")),
           SECONDARY[2]: S(recs, lambda r: dv(r, "V2.0") - dv(r, "P")),
           SECONDARY[3]: S(recs, lambda r: dv(r, "P")),
           SECONDARY[4]: s5}
    adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in sec.items() if "sign_flip_p_monte_carlo" in v})
    for k in sec:
        sec[k]["holm_p_secondary_family"] = adj.get(k)
    means = [curve[v]["mean_points"] for v in ("V0.0", "V1.0", "V2.0")]
    return {"primary_curve": curve, "secondary": sec, "TSLR_same_pairs_curve": tcurve,
            "H2_monotone_V0.0_gt_V1.0_gt_V2.0": bool(means[0] > means[1] > means[2]),
            "fraction_without_shared_samples": {"EEGNet": means[2] / means[0],
                                                "TSLR_same_pairs": tcurve["V2.0"]["mean_points"] / tcurve["V0.0"]["mean_points"]},
            "AA_determinism": {"n_checks": len(aa), "max_abs_probability_difference": float(max(aa))},
            "pair_counts": {"pairs": len(recs), "participants": len(keys)}}
