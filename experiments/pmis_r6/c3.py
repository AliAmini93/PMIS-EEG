"""R6-C3: independent-cohort replication on Lee2019/OpenBMI (injected exact copies). TS+LR runner + EEGNet plan.

TS+LR per pair (a, b) and placebo (c, e), every arm refitted from scratch:
  x_*  exact-optimum arms: R4 multiplicity-weighted Karcher reference of the arm + damped-Newton L2-logistic
       (the estimator whose G3/C1 equivalence is audited), scored on the whole held-out run
  l_*  the frozen R4/G3 arm fit (sklearn lbfgs at default tol), scored on a and b (comparability with G3)
  nt_* parameter-free one-Newton-step prediction from the base model (the C1 law, tested prospectively)
EEGNet: plan builder in the C2 format (outcome-blind third of the pairs per task; arms A, B, C, E and one A/A per
task; at most M_MAX arms per GPU group), executed with the unmodified C2 functions.
"""

from __future__ import annotations

import hashlib
import math
import os
import resource
import time

import numpy as np

from evaluation.g2_estimands import correct_class_probability
from models.tslr import arm_extra_copy, arm_multiset, fit_arm
from models.vendor_r4 import r4_models as rm

from models.indep_tslr import logreg_newton
from models.influence import BaseInfluence

G: dict = {}
EEG_SAMPLE_STREAM, EEG_PAIR_STREAM, EEG_GROUP_STREAM = 7411, 7412, 7413
EEG_M_MAX = 48


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def opt_pc(u, cnt, idx):
    covs, y = G["covs"], G["y"]
    pos = int(np.max(y))
    ref = rm.riemann_mean_w(covs[u], cnt)
    th, it = logreg_newton(rm.tangent_space_b(covs[u], ref), (y[u] == pos).astype(float), cnt.astype(float))
    f = rm.tangent_space_b(covs[idx], ref)
    pp = 1.0 / (1.0 + np.exp(-(f @ th[:-1] + th[-1])))
    return np.where(y[idx] == pos, pp, 1.0 - pp), it


def lbfgs_pc(u, cnt, idx):
    covs, y = G["covs"], G["y"]
    m = fit_arm(covs[u], y[u], cnt)
    return correct_class_probability(m.clf.predict_proba(rm.tangent_space_b(covs[idx], m.ref)), m.clf.classes_, y[idx])


def run_task_tslr(task: dict) -> dict:
    covs, y = G["covs"], G["y"]
    t0 = time.perf_counter()
    T = task["T"]
    pos = {int(t): i for i, t in enumerate(T.tolist())}
    base = task["base_canon"]
    ones = np.ones(base.size, dtype=np.int64)
    pl = int(np.max(y))
    ref_c = rm.riemann_mean_w(covs[base], ones)
    inf = BaseInfluence(rm.tangent_space_b(covs[base], ref_c), (y[base] == pl).astype(float), ones)
    a_ids = np.array([p["a"] for p in task["pairs"]])
    b_ids = np.array([p["b"] for p in task["pairs"]])
    q = inf.predict_pair(rm.tangent_space_b(covs[a_ids], ref_c), (y[a_ids] == pl).astype(float),
                         rm.tangent_space_b(covs[b_ids], ref_c), (y[b_ids] == pl).astype(float), 1)
    um = task["unexposed"]
    recs, it_max, n_fits = [], 0, 0
    for j, pr in enumerate(task["pairs"]):
        a, b = pr["a"], pr["b"]
        ia, ib = pos[a], pos[b]
        ab = np.array([a, b])
        pA, i1 = opt_pc(*arm_multiset(base, ones, a), T)
        pB, i2 = opt_pc(*arm_multiset(base, ones, b), T)
        pC, i3 = opt_pc(*arm_extra_copy(base, ones, pr["c"]), T)
        pE, i4 = opt_pc(*arm_extra_copy(base, ones, pr["e"]), T)
        lA = lbfgs_pc(*arm_multiset(base, ones, a), ab)
        lB = lbfgs_pc(*arm_multiset(base, ones, b), ab)
        n_fits += 6
        it_max = max(it_max, i1, i2, i3, i4)
        sc = um & (task["T_label"] == pr["label"])
        r = {"k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"], "sign": pr["sign"],
             "pAA": pA[ia], "pAB": pB[ia], "pBA": pA[ib], "pBB": pB[ib],
             "cAA": float(pA[ia] > 0.5), "cAB": float(pB[ia] > 0.5), "cBA": float(pA[ib] > 0.5), "cBB": float(pB[ib] > 0.5),
             "pAC": pC[ia], "pBC": pC[ib], "pAE": pE[ia], "pBE": pE[ib],
             "l_pAA": lA[0], "l_pBA": lA[1], "l_pAB": lB[0], "l_pBB": lB[1],
             "off_run": float(np.mean(np.abs(pA[um] - pB[um]))),
             "off_run_same_class": float(np.mean(np.abs(pA[sc] - pB[sc]))),
             "plc_off_run": float(np.mean(np.abs(pC[um] - pE[um]))),
             "h_aa": q["h_aa"][j], "h_bb": q["h_bb"][j], "h_ab": q["h_ab"][j]}
        for key in ("pAA", "pAB", "pBA", "pBB"):
            r[f"nt_{key}"] = q[f"nt_{key}"][j]
        recs.append(r)
    return {"subject": task["subject"], "fold": task["fold"], "records": recs, "fits": n_fits,
            "newton_iter_max": it_max, "seconds": time.perf_counter() - t0, "pid": os.getpid(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def build_eegnet_plan(tasks: list[dict], fraction_den: int = 3, m_max: int = EEG_M_MAX) -> dict:
    groups, n_pairs = [], 0
    for task in tasks:
        subj, fold = int(task["subject"]), int(task["fold"])
        n = len(task["pairs"])
        rng = np.random.default_rng([EEG_SAMPLE_STREAM, subj, fold])
        sel = sorted(int(i) for i in rng.choice(n, math.ceil(n / fraction_den), replace=False))
        units = [(task["pairs"][j], [("A", task["pairs"][j]["a"]), ("B", task["pairs"][j]["b"]),
                                     ("C", task["pairs"][j]["c"]), ("E", task["pairs"][j]["e"])]) for j in sel]
        gi, cur = 0, []

        def flush(cur, gi):
            arms, meta = [], []
            for p_idx, (pr, a_list) in enumerate(cur):
                meta.append({"k": int(pr["k"]), "seed": _seed(EEG_PAIR_STREAM, subj, fold, pr["k"]), "a": int(pr["a"]),
                             "b": int(pr["b"]), "label": int(pr["label"]), "session": int(pr["session"]),
                             "pos_dist": int(pr["pos_dist"]), "g": -1, "c": int(pr["c"]), "e": int(pr["e"]),
                             "sign": int(pr["sign"])})
                arms += [{"pair": p_idx, "arm": nm, "extra": int(x)} for nm, x in a_list]
            if gi == 0:
                arms.append({"pair": 0, "arm": "AA", "extra": int(cur[0][0]["a"])})
            groups.append({"subject": subj, "fold": fold, "group": gi, "group_seed": _seed(EEG_GROUP_STREAM, subj, fold, gi),
                           "pairs": meta, "arms": arms, "n_base": int(task["base_canon"].size)})

        for u in units:
            size = sum(len(x[1]) for x in cur) + (1 if gi == 0 else 0)
            if cur and size + len(u[1]) > m_max:
                flush(cur, gi)
                gi, cur = gi + 1, []
            cur.append(u)
        if cur:
            flush(cur, gi)
        n_pairs += len(units)
    return {"groups": groups, "n_pairs": n_pairs, "n_arms": sum(len(g["arms"]) for g in groups)}


def eegnet_commitment(plan: dict, tasks: list[dict]) -> str:
    h = hashlib.sha256()
    tmap = {(int(t["subject"]), int(t["fold"])): t for t in tasks}
    for g in plan["groups"]:
        t = tmap[(g["subject"], g["fold"])]
        h.update(np.ascontiguousarray(t["base_canon"]).tobytes())
        h.update(np.ascontiguousarray(t["T"]).tobytes())
        h.update(repr((g["subject"], g["fold"], g["group"], g["group_seed"], g["pairs"], g["arms"])).encode())
    return h.hexdigest()


def tslr_commitment(tasks: list[dict]) -> str:
    h = hashlib.sha256()
    for t in tasks:
        for arr in (t["T"], t["base_canon"], t["unexposed"].astype(np.int8)):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr((t["subject"], t["fold"], t["pairs"])).encode())
    return h.hexdigest()
