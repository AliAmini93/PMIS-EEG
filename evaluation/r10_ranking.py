"""R10-A participant-level ranking statistics (shared by the retrospective G0 audit and the frozen G4 analysis).

Aggregation (R4 / R8-B / R9-C, unchanged): for participant p, model m, arm a and fold seed s, balanced accuracy is
the mean of the two class recalls over p's scored trials of seed s (pooled over folds); the participant value is the
mean over seeds. The participant is the only independent unit; folds, seeds and trials are never treated as
independent observations.

Notation: BA[m][a] is a (n_participants,) array in percentage points.
  gap  G_AB(a)        = BA[A][a] - BA[B][a]
  DiD  D_AB(a2 | a1)  = G_AB(a2) - G_AB(a1)          (primary: a2 = LEAKED, a1 = PLACEBO)
Inference: exact two-sided sign-flip over 2^n patterns (n <= 20) and a t95 interval (vendored R4 statistics);
the R4/Paper A SESOI label (2 BA points) is attached where declared.
Ranking uncertainty: participant-cluster bootstrap (participants resampled with replacement, arms and models kept
paired within a participant).
"""

from __future__ import annotations

import itertools

import numpy as np

from models.vendor_r4 import r4_stats as rs

SESOI_BA = rs.SESOI  # 2.0 BA points (Paper A Section 2.3, vendored R4); used for labels only


def ba_participants(correct, y, subj, parts, mask=None) -> np.ndarray:
    """Class-balanced accuracy (%) per participant over trials in mask with finite correctness; NaN if a class is
    absent. correct: 0/1 (float, NaN = unscored)."""
    ok = np.isfinite(correct) if mask is None else (np.isfinite(correct) & mask)
    out = np.full(parts.size, np.nan)
    for i, p in enumerate(parts):
        m = ok & (subj == p)
        if not m.any():
            continue
        cls = [np.mean(correct[m & (y == c)]) for c in (1, 2) if (m & (y == c)).any()]
        if len(cls) == 2:
            out[i] = 100.0 * float(np.mean(cls))
    return out


def seed_mean(per_seed: list[np.ndarray]) -> np.ndarray:
    a = np.vstack(per_seed)
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=0) if np.isnan(a).any() else a.mean(axis=0)


def summ(d, sesoi_label: bool = False) -> dict:
    out = {k: v for k, v in rs.summarise(d).items() if k not in ("sorted_values_pp_unlabelled", "label")}
    if sesoi_label:
        dd = np.asarray(d, dtype=np.float64)
        out["sesoi_2pp_label"] = rs.label(dd[~np.isnan(dd)])
    return out


def holm(pvals: dict) -> dict:
    return rs.holm(pvals)


def cohort_means(BA: dict, arm: str, models) -> dict:
    return {m: float(np.mean(BA[m][arm])) for m in models}


def order(means: dict) -> list:
    """Models by decreasing cohort mean; exact ties broken by the declared model order (stable sort)."""
    return [m for m, _ in sorted(means.items(), key=lambda kv: -kv[1])]


def kendall_distance(o1: list, o2: list) -> int:
    pos2 = {m: i for i, m in enumerate(o2)}
    return int(sum(1 for a, b in itertools.combinations(o1, 2) if pos2[a] > pos2[b]))


def winners(BA: dict, arm: str, models) -> dict:
    """Per-participant winner counts (strictly highest BA); ties reported separately."""
    M = np.vstack([BA[m][arm] for m in models])
    best = M.max(axis=0)
    n_best = (M == best).sum(axis=0)
    out = {m: int(((M[i] == best) & (n_best == 1)).sum()) for i, m in enumerate(models)}
    out["ties"] = int((n_best > 1).sum())
    return out


def participant_order_changes(BA: dict, A: str, B: str, arm1: str, arm2: str) -> dict:
    g1, g2 = BA[A][arm1] - BA[B][arm1], BA[A][arm2] - BA[B][arm2]
    return {"A_ahead_in_arm1": int((g1 > 0).sum()), "A_ahead_in_arm2": int((g2 > 0).sum()),
            "B_ahead_to_A_ahead": int(((g1 < 0) & (g2 > 0)).sum()), "A_ahead_to_B_ahead": int(((g1 > 0) & (g2 < 0)).sum()),
            "ties_either": int(((g1 == 0) | (g2 == 0)).sum())}


def bootstrap_rankings(BA: dict, arms, models, n_boot: int, seed: int, pairs=()) -> dict:
    """Participant-cluster bootstrap of cohort rankings. Returns, per arm, P(model ranked first) and, per pair, P(A ahead
    of B); per pair and arm pair (arm1, arm2), P(the order of A and B differs between the arms) and the bootstrap
    percentile interval of the DiD."""
    rng = np.random.default_rng(seed)
    n = next(iter(BA[models[0]].values())).size
    idx = rng.integers(0, n, size=(n_boot, n))
    mean = {m: {a: BA[m][a][idx].mean(axis=1) for a in arms} for m in models}
    out = {"n_boot": int(n_boot), "seed": int(seed), "p_first": {}, "p_A_ahead": {}, "p_order_differs": {}, "did_pct95": {}}
    for a in arms:
        M = np.vstack([mean[m][a] for m in models])
        first = np.argmax(M, axis=0)
        out["p_first"][a] = {m: float(np.mean(first == i)) for i, m in enumerate(models)}
    for (A, B) in pairs:
        key = f"{A}_vs_{B}"
        out["p_A_ahead"][key] = {a: float(np.mean(mean[A][a] > mean[B][a])) for a in arms}
        out["p_order_differs"][key] = {}
        out["did_pct95"][key] = {}
        for a1, a2 in itertools.combinations(arms, 2):
            s1, s2 = mean[A][a1] > mean[B][a1], mean[A][a2] > mean[B][a2]
            out["p_order_differs"][key][f"{a1}|{a2}"] = float(np.mean(s1 != s2))
            d = (mean[A][a2] - mean[B][a2]) - (mean[A][a1] - mean[B][a1])
            out["did_pct95"][key][f"{a2}-{a1}"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def reversal_class(gap_ref: dict, gap_cont: dict, did: dict, did_holm_p: float | None, alpha: float = 0.05) -> str:
    """Declared classification of the order of a model pair between a reference arm and a contaminated arm.
    gap_*: summaries of G_AB (mean_pp, ci95_pp); did: summary of D_AB; did_holm_p: Holm-adjusted p of D_AB (or None).
      NO_REVERSAL                  cohort means keep their order
      DESCRIPTIVE_REVERSAL         cohort means change order, D_AB not supported (Holm p >= alpha or not tested)
      SUPPORTED_CROSSING           order changes and D_AB is supported, but at least one arm's gap CI includes 0
      SUPPORTED_REVERSAL           order changes, D_AB supported, and both gaps' 95 % CIs exclude 0 in opposite
                                   directions (the order is resolved in each arm and opposite)."""
    s_ref, s_cont = np.sign(gap_ref["mean_pp"]), np.sign(gap_cont["mean_pp"])
    if s_ref == s_cont or s_ref == 0 or s_cont == 0:
        return "NO_REVERSAL"
    if did_holm_p is None or did_holm_p >= alpha:
        return "DESCRIPTIVE_REVERSAL"
    lo_r, hi_r = gap_ref["ci95_pp"]
    lo_c, hi_c = gap_cont["ci95_pp"]
    resolved_ref = lo_r > 0 or hi_r < 0
    resolved_cont = lo_c > 0 or hi_c < 0
    return "SUPPORTED_REVERSAL" if (resolved_ref and resolved_cont) else "SUPPORTED_CROSSING"


def selection(BA: dict, select_arm: str, eval_arm: str, models) -> dict:
    """Within-benchmark counterfactual: the model a researcher would pick from select_arm's cohort means, scored on
    eval_arm. Regret = eval-arm cohort mean of eval_arm's own winner minus that of select_arm's winner (>= 0)."""
    ms, me = cohort_means(BA, select_arm, models), cohort_means(BA, eval_arm, models)
    w_sel, w_eval = order(ms)[0], order(me)[0]
    reg = BA[w_eval][eval_arm] - BA[w_sel][eval_arm]
    return {"winner_select_arm": w_sel, "winner_eval_arm": w_eval, "same_winner": bool(w_sel == w_eval),
            "regret_cohort_mean_pp": float(me[w_eval] - me[w_sel]),
            "regret_participant_summary": summ(reg) if w_sel != w_eval else None,
            "reported_margin_select_arm_pp": float(ms[order(ms)[0]] - ms[order(ms)[1]]) if len(models) > 1 else None,
            "margin_of_select_winner_in_eval_arm_pp": float(me[w_sel] - max(v for k, v in me.items() if k != w_sel))
            if len(models) > 1 else None}


def bootstrap_selection(BA: dict, select_arm: str, eval_arm: str, models, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = BA[models[0]][select_arm].size
    idx = rng.integers(0, n, size=(n_boot, n))
    S = np.vstack([BA[m][select_arm][idx].mean(axis=1) for m in models])
    E = np.vstack([BA[m][eval_arm][idx].mean(axis=1) for m in models])
    ws, we = np.argmax(S, axis=0), np.argmax(E, axis=0)
    regret = E[we, np.arange(n_boot)] - E[ws, np.arange(n_boot)]
    return {"n_boot": int(n_boot), "seed": int(seed), "p_winner_differs": float(np.mean(ws != we)),
            "regret_mean_pp": float(regret.mean()), "regret_pct95": [float(np.percentile(regret, 2.5)),
                                                                   float(np.percentile(regret, 97.5))],
            "p_regret_ge_sesoi": float(np.mean(regret >= SESOI_BA))}
