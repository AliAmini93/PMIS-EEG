"""R7-C1b analysis: per-arm agreement with G3 and recomputation of every G3 seed-42 public number."""

from __future__ import annotations

import numpy as np

from evaluation import r6_c1_analysis as c1_analysis

K1_GRAD, K2_ARM, K3_POINTS, K3_BA_POINTS = 1e-8, 0.01, 0.05, 0.5
MAP = (("A_pa", "pAA"), ("A_pb", "pBA"), ("B_pa", "pAB"), ("B_pb", "pBB"), ("mA_pa", "mAA"), ("mA_pb", "mBA"),
       ("mB_pa", "mAB"), ("mB_pb", "mBB"), ("G_pa", "pAG"), ("G_pb", "pBG"), ("C_pa", "pAC"), ("C_pb", "pBC"),
       ("E_pa", "pAE"), ("E_pb", "pBE"))


def analyse(recs: list[dict], stored: dict, g3_public: dict, indep_g3keys: list[dict]) -> dict:
    """recs: independent records (with subject, label, pos_dist, sign); stored: G3 npz dict; indep_g3keys: mapped."""
    grads = np.array([v for r in recs for k, v in r.items() if k.endswith("_J")])
    k1 = {"arms": int(grads.size), "max_grad_norm": float(grads.max()), "n_above_1e-8": int((grads > K1_GRAD).sum()),
          "median_grad_norm": float(np.median(grads)), "pass": bool((grads <= K1_GRAD).all())}
    diffs = {}
    for ik, gk in MAP:
        d = np.array([abs(r[ik] - float(stored[f"s42_{gk}"][int(r["k"])])) for r in recs if ik in r])
        diffs[gk] = {"n": int(d.size), "max": float(d.max()), "median": float(np.median(d)), "p99": float(np.percentile(d, 99))}
    allmax = max(v["max"] for v in diffs.values())
    k2 = {"per_score": diffs, "max_abs_diff_probability": allmax, "pass": bool(allmax <= K2_ARM)}
    # recompute G3 public numbers from the independent arms, with G3's own (frozen R6 independent) aggregation path
    keys = set().union(*[set(r) for r in indep_g3keys])
    z = {f"s42_{k}": np.array([r.get(k, np.nan) for r in indep_g3keys], dtype=float) for k in keys}
    for k in ("subject", "label", "pos_dist", "sign"):
        z[f"s42_{k}"] = np.array([r[k] for r in indep_g3keys], dtype=float)
    for s in (43, 44, 45, 46):  # seed means are not re-audited (descriptive D7); reuse stored to satisfy the reader
        for q in ("pAA", "pAB", "pBA", "pBB", "subject", "label"):
            z[f"s{s}_{q}"] = stored[f"s{s}_{q}"]
    rec_vals, _ = c1_analysis.recompute_g3(z)
    res = g3_public["results"]
    key = {"S1": "S1_balanced_accuracy_contrast", "S2": "S2_C1_same_run_minus_different_run_neighbour",
           "S3": "S3_C1_identity_minus_neighbour_contrast", "S4": "S4_C2_placebo_nontest_swap",
           "S5": "S5_C4_multiset_base_identity_swap", "S6": "S6_position_near_le6", "S7": "S7_position_far_gt6"}
    ref = {"primary": res["primary"], **{k: res["secondary"][v] for k, v in key.items()}}
    k3 = {}
    ok = True
    for k, g in ref.items():
        mine = rec_vals[k]
        if "mean_points" not in mine or "mean_points" not in g:  # < 2 participants (synthetic fixtures only)
            k3[k] = {"computable": False, "pass": ("mean_points" not in mine) and ("mean_points" not in g)}
            ok &= k3[k]["pass"]
            continue
        dm = abs(mine["mean_points"] - g["mean_points"])
        dci = max(abs(a - b) for a, b in zip(mine["ci95_points"], g["ci95_points"]))
        same_signs = mine["n_positive"] == g["n_participants_positive"] and mine["n_negative"] == g["n_participants_negative"]
        tol = K3_BA_POINTS if k == "S1" else K3_POINTS
        k3[k] = {"g3_mean": g["mean_points"], "independent_mean": mine["mean_points"], "abs_diff_mean": dm,
                 "abs_diff_ci": dci, "independent_signs": [mine["n_positive"], mine["n_negative"]],
                 "g3_signs": [g["n_participants_positive"], g["n_participants_negative"]],
                 "independent_sign_flip_p": mine["exact_sign_flip_p"], "g3_sign_flip_p": g["exact_sign_flip_p"],
                 "tolerance_points": tol, "pass": bool(dm <= tol and dci <= tol and same_signs)}
        ok &= k3[k]["pass"]
    return {"K1_convergence": k1, "K2_per_arm_agreement": k2, "K3_g3_public_numbers": {"per_quantity": k3, "pass": bool(ok)},
            "all_pass": bool(k1["pass"] and k2["pass"] and ok)}
