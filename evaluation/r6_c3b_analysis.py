"""R6-C3b analysis: EEGNet-8,2 PMIS on Lee2019 (injected copies); 54 participants, Monte Carlo sign-flip."""

from __future__ import annotations

import numpy as np
from scipy import stats

from evaluation.r6_indep_stats import participant_values
from evaluation.r6_mc_stats import summary

POSITION_CUT = 6
SECONDARY = ("S1_balanced_accuracy_contrast", "S2_placebo_swap", "S3_EEGNet_minus_TSLR_same_pairs")


def d_of(r, p="p"):
    return 0.5 * ((r[f"{p}AA"] - r[f"{p}AB"]) + (r[f"{p}BB"] - r[f"{p}BA"]))


def plc(r):
    return 0.5 * r["sign"] * ((r["pAC"] - r["pAE"]) + (r["pBE"] - r["pBC"]))


def S(recs, fn, cond=lambda r: True, test=True):
    sel = [r for r in recs if cond(r)]
    return summary(np.array([r["subject"] for r in sel]), np.array([r["label"] for r in sel]),
                   np.array([fn(r) for r in sel], dtype=np.float64), test=test)


def holm(p):
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def analyse(recs, tslr_by_k, aa):
    for r in recs:
        r["d_tslr"] = tslr_by_k[int(r["k"])]
    primary = S(recs, d_of)
    sec = {SECONDARY[0]: S(recs, lambda r: d_of(r, "c")), SECONDARY[1]: S(recs, plc),
           SECONDARY[2]: S(recs, lambda r: d_of(r) - r["d_tslr"])}
    adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in sec.items()})
    for k in sec:
        sec[k]["holm_p_secondary_family"] = adj[k]
    d = np.array([d_of(r) for r in recs])
    pl = np.array([plc(r) for r in recs])
    desc = {
        "D1_TSLR_same_pairs": S(recs, lambda r: r["d_tslr"], test=False),
        "D2_position_near_le6": S(recs, d_of, lambda r: r["pos_dist"] <= POSITION_CUT, test=False),
        "D2_position_far_gt6": S(recs, d_of, lambda r: r["pos_dist"] > POSITION_CUT, test=False),
        "D3_on_target_abs_change": S(recs, lambda r: 0.5 * (abs(r["pAA"] - r["pAB"]) + abs(r["pBB"] - r["pBA"])), test=False),
        "D3_off_target_abs_change_bystanders": S(recs, lambda r: r["off_fold"], lambda r: np.isfinite(r["off_fold"]), test=False),
        "D3_off_target_abs_change_bystanders_same_class": S(recs, lambda r: r["off_session_class"],
                                                            lambda r: np.isfinite(r["off_session_class"]), test=False),
        "D4_context_partner_arm_correct_class_probability": S(recs, lambda r: 0.5 * (r["pAB"] + r["pBA"]), test=False),
        "D4_context_partner_arm_balanced_accuracy": S(recs, lambda r: 0.5 * (r["cAB"] + r["cBA"]), test=False),
        "D5_training_fit_base_rows_accuracy": S(recs, lambda r: r["train_acc_base_A"], test=False),
        "D5_training_fit_extra_row_correct_class_probability": S(
            recs, lambda r: 0.5 * (r["train_p_extra_A"] + r["train_p_extra_B"]), test=False),
        "D6_pair_sd_points": {"primary": float(d.std(ddof=1) * 100), "placebo": float(pl.std(ddof=1) * 100),
                              "pearson_r_with_TSLR_pair_d": float(np.corrcoef(d, [r["d_tslr"] for r in recs])[0, 1])},
        "D7_AA_determinism": {"n_checks": len(aa), "max_abs_probability_difference": float(max(aa))},
    }
    subj = np.array([r["subject"] for r in recs])
    lab = np.array([r["label"] for r in recs])
    pdv = participant_values(subj, lab, d)
    pba = participant_values(subj, lab, np.array([0.5 * (r["cAB"] + r["cBA"]) for r in recs]))
    keys = sorted(pdv)
    xv, yv = np.array([pdv[k] for k in keys]), np.array([pba[k] for k in keys])
    pr, pp = stats.pearsonr(xv, yv)
    sr, sp = stats.spearmanr(xv, yv)
    desc["H3_participant_corr_d_vs_partner_arm_BA"] = {
        "declared_direction": "negative", "n_participants": len(keys), "pearson_r": float(pr), "pearson_p_two_sided": float(pp),
        "spearman_rho": float(sr), "spearman_p_two_sided": float(sp), "direction_met": bool(sr < 0 and pr < 0)}
    return {"primary": primary, "secondary": sec, "descriptive": desc,
            "pair_counts": {"sampled": len(recs), "participants": int(np.unique([r["subject"] for r in recs]).size)}}
