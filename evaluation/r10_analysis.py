"""R10-A frozen analysis (declared before any ShallowConvNet or replicate fit).

Input per replicate: c[model][arm][seed] and p[model][arm][seed], arrays over the 23,382 distinct trials (every trial
scored once per fold seed), arms L, L1, L2, P, C; exposed[seed] (bool) and half[seed] (0 unexposed, 1 = H1, 2 = H2).
Unit: participant (14). BA per participant = class-balanced accuracy over that seed's scored trials, mean over seeds.

Declared families (exact two-sided sign-flip over 2^14 patterns; t95 intervals; Holm within family):
  H1 (Holm over 3)  D_AB(L | P) = G_AB(L) - G_AB(P), all scored trials, for the pairs in PAIRS
  SEC (Holm over 3) D_AB(L | C), used only for the CLEAN -> LEAKED reversal classification
  H4 (Holm over 6)  on exposed scored trials, dE_AB = E_A - E_B (E = OWN - OTHER, exposure with half support fixed) and
                    dS_AB = S_A - S_B (S = OTHER - P, half same-run support without the own copy)
Everything else is descriptive. SESOI: 2 BA points (vendored R4 / Paper A) for labels and the selection-regret rule only.
"""

from __future__ import annotations

import numpy as np

from evaluation import r10_ranking as rk

MODELS = ("TSLR", "EEGNet", "Shallow")
PAIRS = (("EEGNet", "TSLR"), ("Shallow", "TSLR"), ("Shallow", "EEGNet"))
BENCH_ARMS = ("C", "P", "L1", "L2", "L")
ALPHA = 0.05


def _pk(A, B):
    return f"{A}_vs_{B}"


def _f1(correct, y, mask):
    """Macro-F1 (%) for a binary task from 0/1 correctness and labels (prediction = label if correct, else the other)."""
    m = mask & np.isfinite(correct)
    if not m.any() or np.unique(y[m]).size < 2:
        return np.nan
    yt = y[m]
    pred = np.where(correct[m] == 1.0, yt, 3 - yt)
    f = []
    for k in (1, 2):
        tp = np.sum((pred == k) & (yt == k))
        fp = np.sum((pred == k) & (yt != k))
        fn = np.sum((pred != k) & (yt == k))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return 100.0 * float(np.mean(f))


def tables(c, p, dist, exposed, half, seeds, models=MODELS):
    subj, y = dist["subject"], dist["label"]
    parts = np.unique(subj)
    BA = {"all": {}, "exposed": {}, "unexposed": {}}
    PR, F1 = {}, {}
    for m in models:
        for sc in BA:
            BA[sc][m] = {}
            for a in c[m]:
                per = []
                for s in seeds:
                    mask = {"all": None, "exposed": exposed[s], "unexposed": ~exposed[s]}[sc]
                    per.append(rk.ba_participants(c[m][a][s], y, subj, parts, mask))
                BA[sc][m][a] = rk.seed_mean(per)
        for name, (a1, a2) in (("OWN", ("L1", "L2")), ("OTHER", ("L2", "L1"))):
            per = []
            for s in seeds:
                v = np.where(half[s] == 1, c[m][a1][s], np.where(half[s] == 2, c[m][a2][s], np.nan))
                per.append(rk.ba_participants(v, y, subj, parts, half[s] > 0))
            BA["exposed"][m][name] = rk.seed_mean(per)
        PR[m], F1[m] = {}, {}
        for a in ("C", "P", "L"):
            pp, ff = [], []
            for s in seeds:
                pp.append(np.array([100.0 * np.mean([np.mean(p[m][a][s][(subj == q) & (y == k)]) for k in (1, 2)]) for q in parts]))
                ff.append(np.array([_f1(c[m][a][s], y, subj == q) for q in parts]))
            PR[m][a], F1[m][a] = rk.seed_mean(pp), rk.seed_mean(ff)
    return BA, PR, F1


def _did(T, A, B, a2, a1):
    return (T[A][a2] - T[B][a2]) - (T[A][a1] - T[B][a1])


def analyse_replicate(c, p, dist, exposed, half, seeds, n_boot, boot_seed, models=MODELS, pairs=PAIRS):
    BA, PR, F1 = tables(c, p, dist, exposed, half, seeds, models)
    B = BA["all"]
    out = {"absolute_BA": {}, "order": {}, "top1": {}, "participant_winners": {}}
    for a in BENCH_ARMS:
        mm = rk.cohort_means(B, a, models)
        out["absolute_BA"][a] = {m: {"mean": round(mm[m], 3), "ci95": [round(v + 50.0, 3) for v in rk.summ(B[m][a] - 50.0)["ci95_pp"]]}
                                 for m in models}
        out["order"][a], out["top1"][a] = rk.order(mm), rk.order(mm)[0]
        out["participant_winners"][a] = rk.winners(B, a, models)
    out["validity_CLEAN_BA_ci95_lower_above_50"] = {m: bool(out["absolute_BA"]["C"][m]["ci95"][0] > 50.0) for m in models}
    out["gaps"] = {_pk(A, Bm): {a: rk.summ(B[A][a] - B[Bm][a], sesoi_label=True) for a in BENCH_ARMS} for A, Bm in pairs}

    # H1 and SEC families
    h1 = {_pk(A, Bm): rk.summ(_did(B, A, Bm, "L", "P"), sesoi_label=True) for A, Bm in pairs}
    adj = rk.holm({k: v["sign_flip_p"] for k, v in h1.items()})
    for k in h1:
        h1[k]["holm_p_family"] = adj[k]
    sec = {_pk(A, Bm): rk.summ(_did(B, A, Bm, "L", "C"), sesoi_label=True) for A, Bm in pairs}
    adj = rk.holm({k: v["sign_flip_p"] for k, v in sec.items()})
    for k in sec:
        sec[k]["holm_p_family"] = adj[k]
    out["H1_family_D_L_minus_P"] = h1
    out["SEC_family_D_L_minus_C"] = sec
    desc = {}
    for A, Bm in pairs:
        k = _pk(A, Bm)
        desc[k] = {"D_P_minus_C": rk.summ(_did(B, A, Bm, "P", "C"), sesoi_label=True),
                   "D_L1_minus_P": rk.summ(_did(B, A, Bm, "L1", "P"), sesoi_label=True),
                   "D_L2_minus_P": rk.summ(_did(B, A, Bm, "L2", "P"), sesoi_label=True),
                   "D_L_minus_P_exposed": rk.summ(_did(BA["exposed"], A, Bm, "L", "P"), sesoi_label=True),
                   "D_L_minus_P_unexposed": rk.summ(_did(BA["unexposed"], A, Bm, "L", "P"), sesoi_label=True),
                   "D_L_minus_P_probability_points": rk.summ(_did(PR, A, Bm, "L", "P")),
                   "D_L_minus_P_macroF1": rk.summ(_did(F1, A, Bm, "L", "P"))}
    out["descriptive_did"] = desc
    out["inflation_L_minus_P_by_model"] = {m: rk.summ(B[m]["L"] - B[m]["P"], sesoi_label=True) for m in models}
    out["placebo_P_minus_C_by_model"] = {m: rk.summ(B[m]["P"] - B[m]["C"], sesoi_label=True) for m in models}
    out["L_minus_C_by_model"] = {m: rk.summ(B[m]["L"] - B[m]["C"], sesoi_label=True) for m in models}
    out["exposed_unexposed_L_minus_P_by_model"] = {m: {"exposed": rk.summ(BA["exposed"][m]["L"] - BA["exposed"][m]["P"]),
                                                       "unexposed": rk.summ(BA["unexposed"][m]["L"] - BA["unexposed"][m]["P"])}
                                                   for m in models}

    # H2 ranking reversal
    h2 = {}
    for A, Bm in pairs:
        k = _pk(A, Bm)
        g = out["gaps"][k]
        h2[k] = {"P_to_L": rk.reversal_class(g["P"], g["L"], h1[k], h1[k]["holm_p_family"], ALPHA),
                 "C_to_L": rk.reversal_class(g["C"], g["L"], sec[k], sec[k]["holm_p_family"], ALPHA),
                 "C_to_P": rk.reversal_class(g["C"], g["P"], desc[k]["D_P_minus_C"], None, ALPHA),
                 "participants_P_to_L": rk.participant_order_changes(B, A, Bm, "P", "L"),
                 "participants_C_to_L": rk.participant_order_changes(B, A, Bm, "C", "L")}
    top = {}
    for ref in ("P", "C"):
        t_ref, t_l = out["top1"][ref], out["top1"]["L"]
        changed = t_ref != t_l
        cls = None
        if changed:
            k = _pk(t_l, t_ref) if (t_l, t_ref) in pairs else _pk(t_ref, t_l)
            cls = h2[k][f"{ref}_to_L"]
        top[f"{ref}_to_L"] = {"top1_ref": t_ref, "top1_L": t_l, "changed": bool(changed), "pair_classification": cls,
                              "supported": bool(changed and cls == "SUPPORTED_REVERSAL")}
    out["H2_reversal"] = h2
    out["H2_top1"] = top
    out["bootstrap_rankings"] = rk.bootstrap_rankings(B, list(BENCH_ARMS), models, n_boot, boot_seed, pairs)
    out["kendall_distance"] = {f"{a1}|{a2}": rk.kendall_distance(out["order"][a1], out["order"][a2])
                               for a1, a2 in (("C", "P"), ("P", "L"), ("C", "L"), ("P", "L1"), ("P", "L2"))}

    # H3 model selection (within-benchmark counterfactual)
    sel = {}
    for s_, e_ in (("L", "P"), ("L", "C"), ("L1", "P"), ("L2", "P")):
        r = rk.selection(B, s_, e_, models)
        r["bootstrap"] = rk.bootstrap_selection(B, s_, e_, models, n_boot, boot_seed + 1)
        r["regret_ge_sesoi_2pp"] = bool(r["regret_cohort_mean_pp"] >= rk.SESOI_BA)
        r["label"] = "within-benchmark counterfactual; not an estimate of generalisation or deployment performance"
        sel[f"select_{s_}_evaluate_{e_}"] = r
    out["H3_selection"] = sel

    # H4 attribution (exposed scored trials; stitched arms are trial-conditional, not deployable models)
    X = BA["exposed"]
    comp = {m: {"E_exposure_OWN_minus_OTHER": X[m]["OWN"] - X[m]["OTHER"], "S_support_OTHER_minus_P": X[m]["OTHER"] - X[m]["P"],
                "W_support_with_own_L_minus_OWN": X[m]["L"] - X[m]["OWN"], "D_L_minus_P": X[m]["L"] - X[m]["P"]} for m in models}
    out["H4_per_model_components_exposed"] = {m: {k: rk.summ(v) for k, v in comp[m].items()} for m in models}
    h4 = {}
    for A, Bm in pairs:
        h4[f"dE_{_pk(A, Bm)}"] = rk.summ(comp[A]["E_exposure_OWN_minus_OTHER"] - comp[Bm]["E_exposure_OWN_minus_OTHER"], sesoi_label=True)
        h4[f"dS_{_pk(A, Bm)}"] = rk.summ(comp[A]["S_support_OTHER_minus_P"] - comp[Bm]["S_support_OTHER_minus_P"], sesoi_label=True)
    adj = rk.holm({k: v["sign_flip_p"] for k, v in h4.items()})
    for k in h4:
        h4[k]["holm_p_family"] = adj[k]
    out["H4_family"] = h4
    attr = {}
    for A, Bm in pairs:
        k = _pk(A, Bm)
        dE = comp[A]["E_exposure_OWN_minus_OTHER"] - comp[Bm]["E_exposure_OWN_minus_OTHER"]
        dS = comp[A]["S_support_OTHER_minus_P"] - comp[Bm]["S_support_OTHER_minus_P"]
        dW = comp[A]["W_support_with_own_L_minus_OWN"] - comp[Bm]["W_support_with_own_L_minus_OWN"]
        dD = comp[A]["D_L_minus_P"] - comp[Bm]["D_L_minus_P"]
        mD = float(np.mean(dD))
        shares = {"exposure": float(np.mean(dE)) / mD, "support_without_own": float(np.mean(dS)) / mD,
                  "support_with_own": float(np.mean(dW)) / mD} if mD != 0 else None
        e_ok = h4[f"dE_{k}"]["holm_p_family"] < ALPHA and np.sign(np.mean(dE)) == np.sign(mD)
        s_ok = h4[f"dS_{k}"]["holm_p_family"] < ALPHA and np.sign(np.mean(dS)) == np.sign(mD)
        if not h1[k]["holm_p_family"] < ALPHA:
            cls = "NO_SUPPORTED_DISTORTION"
        elif e_ok and shares is not None and shares["exposure"] > 0.5:
            cls = "EXPOSURE_DOMINANT_AT_HALF_DENSITY"
        elif not e_ok:
            cls = "NOT_ATTRIBUTABLE_TO_EXPOSURE"
        else:
            cls = "MIXED"
        o = {a: (A if np.mean(X[A][a]) > np.mean(X[Bm][a]) else Bm) for a in ("P", "OTHER", "OWN", "L")}
        if o["P"] == o["L"]:
            rcls = "NO_REVERSAL_ON_EXPOSED"
        elif o["P"] != o["OTHER"]:
            rcls = "REVERSAL_AT_SUPPORT_STEP"
        elif o["OTHER"] != o["OWN"]:
            rcls = "REVERSAL_AT_OWN_COPY_STEP"
        else:
            rcls = "REVERSAL_ONLY_AT_FULL_DENSITY"
        attr[k] = {"D_exposed_mean": mD, "path_shares_ratio_of_means": shares, "dW_with_own_support": rk.summ(dW),
                   "exposure_supported": bool(e_ok), "support_supported": bool(s_ok), "distortion_class": cls,
                   "leader_on_exposed_by_arm": o, "reversal_step_on_exposed": rcls}
    out["H4_attribution"] = attr
    out["stitched_orders_exposed"] = {a: rk.order(rk.cohort_means(X, a, models)) for a in ("C", "P", "OTHER", "OWN", "L")}

    # per fold seed (fold seed and training seed vary jointly)
    subj, y = dist["subject"], dist["label"]
    parts = np.unique(subj)
    ps = {}
    for s in seeds:
        Bs = {m: {a: rk.ba_participants(c[m][a][s], y, subj, parts) for a in ("C", "P", "L")} for m in models}
        ps[s] = {"order": {a: rk.order(rk.cohort_means(Bs, a, models)) for a in ("C", "P", "L")},
                 "D_L_minus_P": {_pk(A, Bm): {"mean": float(np.mean(_did(Bs, A, Bm, "L", "P"))),
                                              "n_positive": int(np.sum(_did(Bs, A, Bm, "L", "P") > 0))} for A, Bm in pairs}}
    out["per_fold_seed"] = ps

    # leave-one-participant-out robustness of top-1 and of the sign of D (descriptive)
    n = parts.size
    loo = {"top1_changes": {a: 0 for a in ("C", "P", "L")}, "D_sign_changes": {_pk(A, Bm): 0 for A, Bm in pairs}}
    for i in range(n):
        keep = np.arange(n) != i
        Bi = {m: {a: B[m][a][keep] for a in ("C", "P", "L")} for m in models}
        for a in ("C", "P", "L"):
            loo["top1_changes"][a] += int(rk.order(rk.cohort_means(Bi, a, models))[0] != out["top1"][a])
        for A, Bm in pairs:
            loo["D_sign_changes"][_pk(A, Bm)] += int(np.sign(np.mean(_did(Bi, A, Bm, "L", "P"))) != np.sign(h1[_pk(A, Bm)]["mean_pp"]))
    out["leave_one_participant_out"] = loo
    return out, BA


def robustness(r1: dict, r2: dict, pairs=PAIRS) -> dict:
    """Declared training-seed robustness criteria: identical top-1 per arm (C, P, L); identical sign and Holm decision of
    H1 per pair; identical H2 classification (P -> L) per pair; identical H3 winner (L -> P) and regret-vs-SESOI status."""
    out = {"top1_same": {a: r1["top1"][a] == r2["top1"][a] for a in ("C", "P", "L")},
           "H1_same_sign_and_decision": {}, "H2_P_to_L_same": {}, "H3_same": None}
    for A, Bm in pairs:
        k = _pk(A, Bm)
        a, b = r1["H1_family_D_L_minus_P"][k], r2["H1_family_D_L_minus_P"][k]
        out["H1_same_sign_and_decision"][k] = bool(np.sign(a["mean_pp"]) == np.sign(b["mean_pp"])
                                                   and (a["holm_p_family"] < ALPHA) == (b["holm_p_family"] < ALPHA))
        out["H2_P_to_L_same"][k] = r1["H2_reversal"][k]["P_to_L"] == r2["H2_reversal"][k]["P_to_L"]
    s1, s2 = r1["H3_selection"]["select_L_evaluate_P"], r2["H3_selection"]["select_L_evaluate_P"]
    out["H3_same"] = bool(s1["winner_select_arm"] == s2["winner_select_arm"] and s1["winner_eval_arm"] == s2["winner_eval_arm"]
                          and s1["regret_ge_sesoi_2pp"] == s2["regret_ge_sesoi_2pp"])
    out["all_criteria_met"] = bool(all(out["top1_same"].values()) and all(out["H1_same_sign_and_decision"].values())
                                   and all(out["H2_P_to_L_same"].values()) and out["H3_same"])
    return out


def training_seed_variability(BA1, BA2, models=("EEGNet", "Shallow")) -> dict:
    """Participant-level R1 - R2 differences of BA (all scored trials) per deep model and arm (descriptive)."""
    return {m: {a: rk.summ(BA1["all"][m][a] - BA2["all"][m][a]) | {"sd_participant_diff": float(np.std(BA1["all"][m][a] - BA2["all"][m][a], ddof=1))}
                for a in ("C", "P", "L")} for m in models}


def averaged_over_training_seeds(BA1, BA2, models=MODELS, pairs=PAIRS) -> dict:
    """Descriptive: BA averaged over the two training-seed replicates (TS+LR is deterministic, so unchanged)."""
    Bav = {m: {a: 0.5 * (BA1["all"][m][a] + BA2["all"][m][a]) for a in ("C", "P", "L")} for m in models}
    return {"order": {a: rk.order(rk.cohort_means(Bav, a, models)) for a in ("C", "P", "L")},
            "mean_BA": {a: rk.cohort_means(Bav, a, models) for a in ("C", "P", "L")},
            "D_L_minus_P": {_pk(A, Bm): rk.summ(_did(Bav, A, Bm, "L", "P")) for A, Bm in pairs}}
