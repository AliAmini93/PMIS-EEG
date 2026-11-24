"""R6-C2 analysis (frozen before execution). Aggregates only in the returned public dict."""

from __future__ import annotations

import numpy as np

from evaluation.r6_indep_stats import summary

POSITION_CUT = 6
SECONDARY = ("S1_balanced_accuracy_contrast", "S2_C1_same_run_minus_different_run_neighbour",
             "S3_C1_identity_minus_neighbour_contrast", "S4_C2_placebo_nontest_swap",
             "S5_EEGNet_minus_TSLR_same_pairs")


def d_of(r, p="p"):
    return 0.5 * ((r[f"{p}AA"] - r[f"{p}AB"]) + (r[f"{p}BB"] - r[f"{p}BA"]))


def n_of(r):
    return 0.5 * ((r["pAB"] - r["pAG"]) + (r["pBA"] - r["pBG"]))


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


def analyse(recs: list[dict], tslr_by_k: dict, aa_max_abs: list[float]) -> dict:
    has = lambda key: (lambda r: key in r)  # noqa: E731
    for r in recs:
        t = tslr_by_k[int(r["k"])]
        r["d_tslr"] = 0.5 * ((t["pAA"] - t["pAB"]) + (t["pBB"] - t["pBA"]))
    primary = S(recs, d_of)
    sec = {
        SECONDARY[0]: S(recs, lambda r: d_of(r, "c")),
        SECONDARY[1]: S(recs, n_of, has("pAG")),
        SECONDARY[2]: S(recs, lambda r: d_of(r) - n_of(r), has("pAG")),
        SECONDARY[3]: S(recs, plc, has("pAC")),
        SECONDARY[4]: S(recs, lambda r: d_of(r) - r["d_tslr"]),
    }
    adj = holm({k: v["exact_sign_flip_p"] for k, v in sec.items() if "exact_sign_flip_p" in v})
    for k in sec:
        sec[k]["holm_p_secondary_family"] = adj.get(k)
    dvals = np.array([d_of(r) for r in recs])
    pl = [r for r in recs if "pAC" in r]
    desc = {
        "D1_TSLR_same_pairs": S(recs, lambda r: r["d_tslr"], test=False),
        "D2_position_near_le6": S(recs, d_of, lambda r: r["pos_dist"] <= POSITION_CUT, test=False),
        "D2_position_far_gt6": S(recs, d_of, lambda r: r["pos_dist"] > POSITION_CUT, test=False),
        "D3_C1_subset_primary": S(recs, d_of, has("pAG"), test=False),
        "D3_C1_identity_vs_different_run_neighbour": S(
            recs, lambda r: 0.5 * ((r["pAA"] - r["pAG"]) + (r["pBB"] - r["pBG"])), has("pAG"), test=False),
        "D4_on_target_abs_change": S(recs, lambda r: 0.5 * (abs(r["pAA"] - r["pAB"]) + abs(r["pBB"] - r["pBA"])), test=False),
        "D4_off_target_abs_change_unexposed_same_fold": S(recs, lambda r: r["off_fold"],
                                                          lambda r: np.isfinite(r["off_fold"]), test=False),
        "D4_off_target_abs_change_unexposed_same_session_class": S(
            recs, lambda r: r["off_session_class"], lambda r: np.isfinite(r["off_session_class"]), test=False),
        "D5_context_partner_arm_correct_class_probability": S(recs, lambda r: 0.5 * (r["pAB"] + r["pBA"]), test=False),
        "D5_context_own_arm_correct_class_probability": S(recs, lambda r: 0.5 * (r["pAA"] + r["pBB"]), test=False),
        "D5_context_partner_arm_balanced_accuracy": S(recs, lambda r: 0.5 * (r["cAB"] + r["cBA"]), test=False),
        "D6_training_fit_base_rows_correct_class_probability": S(recs, lambda r: r["train_p_base_A"], test=False),
        "D6_training_fit_base_rows_accuracy": S(recs, lambda r: r["train_acc_base_A"], test=False),
        "D6_training_fit_extra_row_correct_class_probability": S(
            recs, lambda r: 0.5 * (r["train_p_extra_A"] + r["train_p_extra_B"]), test=False),
        "D7_placebo_dispersion": {
            "n_pairs": len(pl),
            "pair_sd_placebo_points": float(np.std([plc(r) for r in pl], ddof=1) * 100) if len(pl) > 1 else None,
            "pair_sd_primary_same_pairs_points": float(np.std([d_of(r) for r in pl], ddof=1) * 100) if len(pl) > 1 else None},
        "D8_pair_distribution_points": {
            "sd": float(dvals.std(ddof=1) * 100), "quantiles_5_25_50_75_95": [float(q * 100) for q in
                                                                              np.percentile(dvals, [5, 25, 50, 75, 95])],
            "fraction_positive": float(np.mean(dvals > 0)),
            "pearson_r_with_TSLR_pair_d": float(np.corrcoef(dvals, [r["d_tslr"] for r in recs])[0, 1])},
        "D9_AA_determinism": {"n_checks": len(aa_max_abs), "max_abs_probability_difference": float(max(aa_max_abs))},
    }
    return {"primary": primary, "secondary": sec, "descriptive": desc,
            "pair_counts": {"sampled": len(recs), "C1": sum("pAG" in r for r in recs), "C2": len(pl)}}
