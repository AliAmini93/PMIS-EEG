"""R6-C1 analysis: audit (A1-A3) and mechanistic decomposition (M1-M4). Aggregates only in the public dict."""

from __future__ import annotations

import numpy as np

from evaluation.r6_indep_stats import summary

POSITION_CUT = 6


def dval(r: dict, pfx: str) -> float:
    return 0.5 * ((r[f"{pfx}AA"] - r[f"{pfx}AB"]) + (r[f"{pfx}BB"] - r[f"{pfx}BA"]))


def _arr(recs, fn, cond=lambda r: True):
    sel = [r for r in recs if cond(r)]
    return (np.array([r["subject"] for r in sel]), np.array([r["label"] for r in sel]),
            np.array([fn(r) for r in sel], dtype=np.float64))


def S(recs, fn, cond=lambda r: True, test=True):
    return summary(*_arr(recs, fn, cond), test=test)


def recompute_g3(z: dict) -> dict:
    """A1: every G3 public number recomputed from the stored pair scores with independent code."""
    n = z["s42_pAA"].size
    recs = [{k[4:]: float(z[k][i]) for k in z if k.startswith("s42_")} for i in range(n)]
    for r in recs:
        r["subject"], r["label"] = int(r["subject"]), int(r["label"])
    has = lambda key: (lambda r: np.isfinite(r.get(key, np.nan)))  # noqa: E731
    out = {
        "primary": S(recs, lambda r: dval(r, "p")),
        "S1": S(recs, lambda r: dval(r, "c")),
        "S2": S(recs, lambda r: 0.5 * ((r["pAB"] - r["pAG"]) + (r["pBA"] - r["pBG"])), has("pAG")),
        "S3": S(recs, lambda r: dval(r, "p") - 0.5 * ((r["pAB"] - r["pAG"]) + (r["pBA"] - r["pBG"])), has("pAG")),
        "S4": S(recs, lambda r: 0.5 * r["sign"] * ((r["pAC"] - r["pAE"]) + (r["pBE"] - r["pBC"])), has("pAC")),
        "S5": S(recs, lambda r: dval(r, "m")),
        "S6": S(recs, lambda r: dval(r, "p"), lambda r: r["pos_dist"] <= POSITION_CUT),
        "S7": S(recs, lambda r: dval(r, "p"), lambda r: r["pos_dist"] > POSITION_CUT),
    }
    seeds = {}
    for s in (42, 43, 44, 45, 46):
        m = z[f"s{s}_pAA"].size
        rr = [{"subject": int(z[f"s{s}_subject"][i]), "label": int(z[f"s{s}_label"][i]),
               **{q: float(z[f"s{s}_{q}"][i]) for q in ("pAA", "pAB", "pBA", "pBB")}} for i in range(m)]
        seeds[str(s)] = S(rr, lambda r: dval(r, "p"), test=False)["mean_points"]
    out["D7_seed_means"] = seeds
    return out, recs


def compare_g3(indep: dict, g3_public: dict) -> dict:
    res = g3_public["results"]
    key = {"S1": "S1_balanced_accuracy_contrast", "S2": "S2_C1_same_run_minus_different_run_neighbour",
           "S3": "S3_C1_identity_minus_neighbour_contrast", "S4": "S4_C2_placebo_nontest_swap",
           "S5": "S5_C4_multiset_base_identity_swap", "S6": "S6_position_near_le6", "S7": "S7_position_far_gt6"}
    diffs = {}
    ref = {"primary": res["primary"], **{k: res["secondary"][v] for k, v in key.items()}}
    for k, g in ref.items():
        mine = indep[k]
        diffs[k] = {"abs_diff_mean": abs(mine["mean_points"] - g["mean_points"]),
                    "abs_diff_ci": max(abs(a - b) for a, b in zip(mine["ci95_points"], g["ci95_points"])),
                    "abs_diff_sign_flip_p": abs(mine["exact_sign_flip_p"] - g["exact_sign_flip_p"]),
                    "n_participants_equal": mine["n_participants"] == g["n_participants"],
                    "n_pairs_equal": mine["n_pairs"] == g["n_pairs"]}
    seed_g3 = {s: v["mean_points"] for s, v in res["descriptive"]["D7_fold_seed_sensitivity_primary"]["per_seed"].items()}
    diffs["D7"] = {"max_abs_diff_seed_means": max(abs(indep["D7_seed_means"][s] - seed_g3[s]) for s in seed_g3)}
    worst = max(max(v.values()) if isinstance(v, dict) else 0 for v in diffs.values()
                if all(isinstance(x, (int, float, bool)) for x in v.values()))
    return {"per_quantity": diffs, "all_counts_equal": all(v.get("n_participants_equal", True) and v.get("n_pairs_equal", True)
                                                           for v in diffs.values()),
            "max_abs_numeric_diff": float(worst)}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def analyse(recs: list[dict], stored: list[dict]) -> dict:
    by_k = {int(r["k"]): r for r in stored}
    # A2 reproduction of G3 arms by re-running the frozen fit code
    diffs = []
    n_bitwise = 0
    for r in recs:
        s = by_k[int(r["k"])]
        for q in ("pAA", "pAB", "pBA", "pBB"):
            diffs.append(abs(r[f"e1_{q}"] - s[q]))
            n_bitwise += int(r[f"e1_{q}"] == s[q])
    a2 = {"values_compared": len(diffs), "bitwise_equal": n_bitwise, "max_abs_diff": float(max(diffs))}
    # A3 independent implementation (exact optimum) vs stored G3 arms (lbfgs at tol 1e-4)
    groups = {"primary": ("pAA", "pAB", "pBA", "pBB"), "multiset": ("mAA", "mAB", "mBA", "mBB"),
              "neighbour": ("pAG", "pBG"), "placebo": ("pAC", "pBC", "pAE", "pBE")}
    aud = [r for r in recs if r.get("audit") == 1.0]
    a3 = {"pairs_with_all_arm_types_audited": len(aud), "per_arm_type": {}}
    for g, qs in groups.items():
        pfx = "x1_" if g == "primary" else "i_"
        src = recs if g == "primary" else aud
        dd = np.array([abs(r[f"{pfx}{q}"] - by_k[int(r["k"])][q]) for r in src for q in qs if f"{pfx}{q}" in r])
        a3["per_arm_type"][g] = {"values": int(dd.size), "max_abs_diff_probability": float(dd.max()) if dd.size else None,
                                 "median_abs_diff_probability": float(np.median(dd)) if dd.size else None}
    dd = np.array([abs(dval(r, "x1_p") - dval(by_k[int(r["k"])], "p")) for r in recs])
    a3["pair_d_abs_diff_points"] = {"max": float(dd.max() * 100), "median": float(np.median(dd) * 100)}
    a3["karcher_iter_max"] = int(max(r["x1_karcher_iter_max"] for r in recs))
    a3["newton_iter_max"] = int(max(r["x1_newton_iter_max"] for r in recs))
    # M1 pathway decomposition (k = 1)
    d = {name: np.array([dval(r, pfx) for r in recs]) for name, pfx in
         (("exact", "x1_p"), ("g3_lbfgs_tol1e-4", "e1_p"), ("fixed_reference", "f_p"), ("reference_only", "r_p"))}
    d["influence_function"] = np.array([0.5 * ((r["if_pAA_k1"] - r["if_pAB_k1"]) + (r["if_pBB_k1"] - r["if_pBA_k1"]))
                                        for r in recs])
    d["newton_one_step"] = np.array([0.5 * ((r["nt_pAA_k1"] - r["nt_pAB_k1"]) + (r["nt_pBB_k1"] - r["nt_pBA_k1"]))
                                     for r in recs])
    subj = np.array([r["subject"] for r in recs])
    lab = np.array([r["label"] for r in recs])
    m1 = {}
    for name, v in d.items():
        m1[name] = summary(subj, lab, v)
        if name != "exact":
            m1[name]["pair_pearson_r_with_exact"] = pearson(v, d["exact"])
            m1[name]["pair_mean_abs_diff_from_exact_points"] = float(np.mean(np.abs(v - d["exact"])) * 100)
            m1[name]["ratio_of_participant_means_to_exact"] = m1[name]["mean_points"] / m1["exact"]["mean_points"]
            slope = float(np.sum(v * d["exact"]) / np.sum(v * v))
            m1[name]["pair_calibration_slope_exact_on_prediction_through_origin"] = slope
    m1["fixed_plus_reference_minus_exact"] = summary(subj, lab, d["fixed_reference"] + d["reference_only"] - d["exact"],
                                                     test=False)
    # M2 dose response
    m2 = {"k1": m1["exact"]}
    for k in (2, 4, 8):
        v = np.array([dval(r, f"e{k}_p") for r in recs])
        m2[f"k{k}"] = summary(subj, lab, v)
        m2[f"k{k}"]["ratio_to_k1"] = m2[f"k{k}"]["mean_points"] / m1["exact"]["mean_points"]
    for k in (1, 2, 4, 8):
        for nm in ("if", "nt"):
            v = np.array([0.5 * ((r[f"{nm}_pAA_k{k}"] - r[f"{nm}_pAB_k{k}"]) + (r[f"{nm}_pBB_k{k}"] - r[f"{nm}_pBA_k{k}"]))
                          for r in recs])
            m2[f"pred_{nm}_k{k}"] = summary(subj, lab, v, test=False)
            ex = d["exact"] if k == 1 else np.array([dval(r, f"e{k}_p") for r in recs])
            m2[f"pred_{nm}_k{k}"]["pair_pearson_r_with_exact"] = pearson(v, ex)
    # M3 multiset base: prediction vs stored G3 S5 arms
    v = np.array([0.5 * ((r["m_nt_pAA"] - r["m_nt_pAB"]) + (r["m_nt_pBB"] - r["m_nt_pBA"])) for r in recs])
    ex = np.array([dval(by_k[int(r["k"])], "m") for r in recs])
    m3 = {"stored_exact_G3_S5": summary(subj, lab, ex, test=False), "newton_one_step": summary(subj, lab, v, test=False),
          "pair_pearson_r": pearson(v, ex)}
    vi = np.array([0.5 * ((r["m_if_pAA"] - r["m_if_pAB"]) + (r["m_if_pBB"] - r["m_if_pBA"])) for r in recs])
    m3["influence_function"] = summary(subj, lab, vi, test=False)
    m3["influence_function"]["pair_pearson_r"] = pearson(vi, ex)
    # M4 leverage
    haa = np.array([r["h_aa"] for r in recs])
    hbb = np.array([r["h_bb"] for r in recs])
    hab = np.array([r["h_ab"] for r in recs])
    p_feats = None
    m4 = {"mean_self_leverage": float(np.mean(0.5 * (haa + hbb))), "mean_cross_leverage": float(np.mean(hab)),
          "median_cross_over_self": float(np.median(2 * hab / (haa + hbb))),
          "self_leverage_by_participant_mean_range": [float(min(np.mean(0.5 * (haa + hbb)[subj == s]) for s in np.unique(subj))),
                                                      float(max(np.mean(0.5 * (haa + hbb)[subj == s]) for s in np.unique(subj)))]}
    # position strata of the decomposition (descriptive)
    pdist = np.array([r["pos_dist"] for r in recs])
    strata = {}
    for nm, msk in (("near_le6", pdist <= POSITION_CUT), ("far_gt6", pdist > POSITION_CUT)):
        strata[nm] = {"exact": summary(subj[msk], lab[msk], d["exact"][msk], test=False).get("mean_points"),
                      "newton_one_step": summary(subj[msk], lab[msk], d["newton_one_step"][msk], test=False).get("mean_points"),
                      "mean_cross_leverage": float(np.mean(hab[msk])) if msk.any() else None}
    del p_feats
    return {"A2_reproduction_of_G3_arms": a2, "A3_independent_implementation": a3,
            "M1_pathway_decomposition_k1": m1, "M2_dose_response": m2, "M3_multiset_base": m3,
            "M4_leverage": m4, "M5_position_strata": strata}
