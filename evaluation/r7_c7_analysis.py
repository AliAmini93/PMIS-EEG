"""R7-C7 analysis (frozen before execution). Participant-level aggregates only in the public dict."""

from __future__ import annotations

import numpy as np

from models.vendor_r4 import r4_stats as rs

SPLITS = {1: "S1", 2: "S2"}
SEEDS = (42, 43, 44, 45, 46)
LAW_RATIO, LAW_R = (0.8, 1.25), 0.9
PRIVATE_KEYS = ("sorted_values_pp_unlabelled", "label")  # participant-level values stay private; no SESOI label


def summ(d):
    """R4's summarise (exact numbers) without participant-level values or R4's SESOI relevance label."""
    return {k: v for k, v in rs.summarise(d).items() if k not in PRIVATE_KEYS}


def assemble(results, n):
    """-> arrays[(split, seed)][key] over all distinct trials (NaN / -1 where not scored)."""
    A = {}
    for r in results:
        key = (r["split"], r["seed"])
        if key not in A:
            A[key] = {k: np.full(n, np.nan) for k in ("r4_p_L", "r4_p_P", "r4_p_C", "opt_p_L", "opt_p_P", "law_p_L", "mult_in_L")}
            A[key].update({k: np.full(n, -1, dtype=np.int8) for k in ("r4_pred_L", "r4_pred_P", "r4_pred_C")})
        for f in r["folds"]:
            T = f["T"]
            for k in A[key]:
                if np.any(~np.isnan(A[key][k][T]) if A[key][k].dtype != np.int8 else A[key][k][T] != -1):
                    raise AssertionError("trial scored twice")
                A[key][k][T] = f[k]
    return A


def _cb_mean(v, y, m):
    """Class-balanced mean of v over mask m (mean over the two classes of per-class means)."""
    vals = [np.nanmean(v[m & (y == c)]) for c in (1, 2) if np.any(m & (y == c))]
    return float(np.mean(vals)) if len(vals) == 2 else np.nan


def per_participant(fn, subj, participants):
    return np.array([fn(subj == p) for p in participants])


def analyse(A, dist, man, r4_pred, r4_public):
    subj, y = dist["subject"], dist["label"]
    participants = np.unique(subj)
    out = {}
    for split, name in SPLITS.items():
        o = {}
        # R1: retrospective reproduction of R4 (hard labels bitwise; published BA contrasts)
        eq = {arm: int(sum(np.sum((A[(split, s)][f"r4_pred_{arm}"] == r4_pred[f"{name}_{s}_{arm}"])
                                  & (r4_pred[f"{name}_{s}_{arm}"] >= 0)) for s in SEEDS)) for arm in ("L", "P", "C")}
        tot = int(sum(np.sum(r4_pred[f"{name}_{s}_L"] >= 0) for s in SEEDS))
        ba = lambda s, arm, mask=None: per_participant(  # noqa: E731
            lambda m: 100.0 * rs.balanced_accuracy(y[m if mask is None else m & mask], A[(split, s)][f"r4_pred_{arm}"][m if mask is None else m & mask])
            if (m if mask is None else m & mask).sum() and np.unique(y[m if mask is None else m & mask]).size == 2 else np.nan,
            subj, participants)
        B = {arm: np.mean([ba(s, arm) for s in SEEDS], axis=0) for arm in ("L", "P", "C")}
        exL = np.nanmean([ba(s, "L", man[f"exposed_{name}_{s}"]) for s in SEEDS], axis=0)
        exP = np.nanmean([ba(s, "P", man[f"exposed_{name}_{s}"]) for s in SEEDS], axis=0)
        unL = np.nanmean([ba(s, "L", ~man[f"exposed_{name}_{s}"]) for s in SEEDS], axis=0)
        unP = np.nanmean([ba(s, "P", ~man[f"exposed_{name}_{s}"]) for s in SEEDS], axis=0)
        mine = {"primary_LEAKED_minus_PLACEBO": summ(B["L"] - B["P"]), "PLACEBO_minus_CLEAN": summ(B["P"] - B["C"]),
                "LEAKED_minus_CLEAN": summ(B["L"] - B["C"]),
                "exposed_only_LEAKED_minus_PLACEBO": summ(exL - exP),
                "unexposed_only_LEAKED_minus_PLACEBO": summ(unL - unP)}
        pub = r4_public["splits"][name]
        o["R1_retrospective_reproduction_of_R4"] = {
            "hard_label_equal": eq, "scored_trials_x_seeds": tot,
            "all_hard_labels_equal": bool(all(v == tot for v in eq.values())),
            "ba_contrast_mean_pp_ours_vs_R4": {k: [mine[k]["mean_pp"], pub[k]["mean_pp"]] for k in mine},
            "max_abs_diff_mean_pp": float(max(abs(mine[k]["mean_pp"] - pub[k]["mean_pp"]) for k in mine))}
        # probability-scale inflation (class-balanced within participant, mean over seeds)
        def dp(kL, kP, mask_fn=None):
            vals = []
            for s in SEEDS:
                d = A[(split, s)][kL] - A[(split, s)][kP]
                e = man[f"exposed_{name}_{s}"]
                vals.append(per_participant(lambda m: _cb_mean(d, y, m if mask_fn is None else m & mask_fn(e)), subj, participants))
            return np.nanmean(vals, axis=0)
        P_all = {"r4_lbfgs": dp("r4_p_L", "r4_p_P"), "exact_optimum": dp("opt_p_L", "opt_p_P"), "law_prediction": dp("law_p_L", "opt_p_P")}
        P_exp = {"exact_optimum": dp("opt_p_L", "opt_p_P", lambda e: e), "law_prediction": dp("law_p_L", "opt_p_P", lambda e: e)}
        P_une = {"exact_optimum": dp("opt_p_L", "opt_p_P", lambda e: ~e), "law_prediction": dp("law_p_L", "opt_p_P", lambda e: ~e)}
        o["probability_inflation_LEAKED_minus_PLACEBO_points"] = {
            "all_scored": {k: summ(100 * v) for k, v in P_all.items()},
            "exposed_only": {k: summ(100 * v) for k, v in P_exp.items()},
            "unexposed_only": {k: summ(100 * v) for k, v in P_une.items()}}
        # law vs exact at participant level
        ex, lw = P_all["exact_optimum"], P_all["law_prediction"]
        ratio = float(np.mean(lw) / np.mean(ex))
        r = float(np.corrcoef(ex, lw)[0, 1])
        trial_r = []
        for s in SEEDS:
            dE = A[(split, s)]["opt_p_L"] - A[(split, s)]["opt_p_P"]
            dL = A[(split, s)]["law_p_L"] - A[(split, s)]["opt_p_P"]
            m = np.isfinite(dE) & np.isfinite(dL)
            trial_r.append(float(np.corrcoef(dE[m], dL[m])[0, 1]))
        o["L1_law_benchmark_level"] = {"participant_mean_ratio_prediction_to_exact": ratio, "participant_level_r": r,
                                       "trial_level_r_by_seed": trial_r, "criterion": {"ratio_in": list(LAW_RATIO), "r_at_least": LAW_R},
                                       "met": bool(LAW_RATIO[0] <= ratio <= LAW_RATIO[1] and r >= LAW_R)}
        # accounting identity on instance means (seed 42 .. 46 pooled): share of the inflation carried by exposed trials
        f_exp, share = [], []
        for s in SEEDS:
            d = A[(split, s)]["opt_p_L"] - A[(split, s)]["opt_p_P"]
            e = man[f"exposed_{name}_{s}"]
            m = np.isfinite(d)
            f_exp.append(float(np.mean(e[m])))
            share.append(float(np.sum(d[m & e]) / np.sum(d[m])))
        o["accounting"] = {"fraction_of_scored_trials_exposed": float(np.mean(f_exp)),
                           "share_of_total_probability_gain_on_exposed_trials": float(np.mean(share))}
        # dose: exposed trials' gain by multiplicity in LEAKED (exact optimum; instance means pooled over seeds)
        dose = {}
        for lo, hi, lab in ((1, 1, "m=1"), (2, 2, "m=2"), (3, 4, "m=3-4"), (5, 8, "m=5-8"), (9, 10**6, "m>=9")):
            vals, n = [], 0
            for s in SEEDS:
                d = A[(split, s)]["opt_p_L"] - A[(split, s)]["opt_p_P"]
                mm = A[(split, s)]["mult_in_L"]
                sel = np.isfinite(d) & (mm >= lo) & (mm <= hi)
                vals.append(d[sel])
                n += int(sel.sum())
            v = np.concatenate(vals)
            dose[lab] = {"n_trial_seeds": n, "mean_gain_points": float(np.mean(v) * 100) if v.size else None}
        o["exposed_gain_by_multiplicity_in_LEAKED"] = dose
        out[name] = o
    return out
