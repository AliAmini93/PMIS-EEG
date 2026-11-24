"""R6-C3a analysis (TS+LR on Lee2019), frozen before execution. Public dict holds aggregates only."""

from __future__ import annotations

import numpy as np

from evaluation.r6_mc_stats import summary

POSITION_CUT = 6
SECONDARY = ("S1_balanced_accuracy_contrast", "S2_placebo_swap", "S3_G3_lbfgs_estimator_identity_swap",
             "S4_position_near_le6", "S5_position_far_gt6")
LAW_RATIO, LAW_R = (0.8, 1.25), 0.9


def d_of(r, p=""):
    return 0.5 * ((r[f"{p}pAA"] - r[f"{p}pAB"]) + (r[f"{p}pBB"] - r[f"{p}pBA"]))


def bac(r):
    return 0.5 * ((r["cAA"] - r["cAB"]) + (r["cBB"] - r["cBA"]))


def plc(r):
    return 0.5 * r["sign"] * ((r["pAC"] - r["pAE"]) + (r["pBE"] - r["pBC"]))


def S(recs, fn, cond=lambda r: True, test=True):
    sel = [r for r in recs if cond(r)]
    return summary(np.array([r["subject"] for r in sel]), np.array([r["label"] for r in sel]),
                   np.array([fn(r) for r in sel], dtype=np.float64), test=test)


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def analyse(recs: list[dict]) -> dict:
    primary = S(recs, d_of)
    sec = {SECONDARY[0]: S(recs, bac), SECONDARY[1]: S(recs, plc), SECONDARY[2]: S(recs, lambda r: d_of(r, "l_")),
           SECONDARY[3]: S(recs, d_of, lambda r: r["pos_dist"] <= POSITION_CUT),
           SECONDARY[4]: S(recs, d_of, lambda r: r["pos_dist"] > POSITION_CUT)}
    adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in sec.items() if "sign_flip_p_monte_carlo" in v})
    for k in sec:
        sec[k]["holm_p_secondary_family"] = adj.get(k)
    d = np.array([d_of(r) for r in recs])
    nt = np.array([d_of(r, "nt_") for r in recs])
    law = {"newton_one_step": S(recs, lambda r: d_of(r, "nt_"), test=False),
           "exact_minus_prediction": S(recs, lambda r: d_of(r) - d_of(r, "nt_"), test=False),
           "pair_pearson_r": float(np.corrcoef(d, nt)[0, 1])}
    law["ratio_of_participant_means_prediction_to_exact"] = law["newton_one_step"]["mean_points"] / primary["mean_points"]
    law["criterion"] = {"ratio_in": list(LAW_RATIO), "pair_r_at_least": LAW_R}
    law["met"] = bool(LAW_RATIO[0] <= law["ratio_of_participant_means_prediction_to_exact"] <= LAW_RATIO[1]
                      and law["pair_pearson_r"] >= LAW_R)
    subj = np.array([r["subject"] for r in recs])
    slopes = []
    for s in np.unique(subj):
        m = subj == s
        x = np.array([r["pos_dist"] for r, k in zip(recs, m) if k], dtype=float)
        if np.ptp(x) > 0:
            slopes.append(float(np.polyfit(x, d[m], 1)[0]) * 100)
    pl = np.array([plc(r) for r in recs])
    part_d = {}
    part_acc = {}
    for s in np.unique(subj):
        part_d[s] = S([r for r in recs if r["subject"] == s], d_of, test=False).get("mean_points")
        part_acc[s] = S([r for r in recs if r["subject"] == s], lambda r: 0.5 * (r["cAB"] + r["cBA"]), test=False).get("mean_points")
    ks = [s for s in part_d if part_d[s] is not None and part_acc[s] is not None]
    desc = {
        "D1_on_target_abs_change": S(recs, lambda r: 0.5 * (abs(r["pAA"] - r["pAB"]) + abs(r["pBB"] - r["pBA"])), test=False),
        "D1_off_target_abs_change_bystanders_same_run": S(recs, lambda r: r["off_run"], test=False),
        "D1_off_target_abs_change_bystanders_same_run_same_class": S(recs, lambda r: r["off_run_same_class"], test=False),
        "D1_placebo_off_target_abs_change_bystanders": S(recs, lambda r: r["plc_off_run"], test=False),
        "D2_context_partner_arm_correct_class_probability": S(recs, lambda r: 0.5 * (r["pAB"] + r["pBA"]), test=False),
        "D2_context_partner_arm_balanced_accuracy": S(recs, lambda r: 0.5 * (r["cAB"] + r["cBA"]), test=False),
        "D3_position_slope_points_per_trial": {"n": len(slopes), "mean": float(np.mean(slopes)),
                                               "sd": float(np.std(slopes, ddof=1))},
        "D4_pair_sd_points": {"primary": float(d.std(ddof=1) * 100), "placebo": float(pl.std(ddof=1) * 100)},
        "D5_participant_corr_d_vs_partner_arm_BA": float(np.corrcoef([part_d[s] for s in ks], [part_acc[s] for s in ks])[0, 1]),
        "D6_leverage": {"mean_self": float(np.mean([0.5 * (r["h_aa"] + r["h_bb"]) for r in recs])),
                        "mean_cross": float(np.mean([r["h_ab"] for r in recs]))},
    }
    return {"primary": primary, "secondary": sec, "H2_influence_law": law, "descriptive": desc,
            "pair_counts": {"pairs": len(recs), "participants": int(np.unique(subj).size)}}
