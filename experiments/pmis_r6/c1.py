"""R6-C1: independent audit of G3 and mechanistic decomposition of the exact TS+LR PMIS contrast.

Exploratory by declaration: the data (HS-Stroke, seed 42, 8,080 pairs) were examined in G3.
Per (participant, fold) task of the frozen G3 plan, for every pair (a, b):
  E1        frozen G3 arm fit (R4 TSLRw, sklearn lbfgs at its default tol=1e-4), k = 1, scored on the whole
            fold: must reproduce G3's stored values bit for bit
  X1        the same arms solved to the EXACT optimum by the independent implementation (independent
            Karcher mean + damped Newton L2-logistic): quantifies G3's solver-tolerance error
  E2/E4/E8  exact-optimum per-arm refit (R4 Karcher + Newton LR) with k copies of a' (resp. b')   [dose]
  F         exact-optimum LR with the reference FIXED at the base Karcher mean M_C       [classifier pathway]
  R         Karcher reference of C+a' with the exact-optimum LR trained on base rows only [reference pathway]
  IF/NT     parameter-free first-order / one-Newton-step predictions from the base model (k = 1, 2, 4, 8;
            canonical and multiset base)
and, for one outcome-blind pair per task, the independent implementation (`indep_tslr`) of every G3 arm.
Frozen G3 modules are imported unchanged.
"""

from __future__ import annotations

import os
import resource
import time

import numpy as np
from evaluation.g2_estimands import correct_class_probability
from models.tslr import arm_extra_copy, arm_multiset, fit_arm
from models.vendor_r4 import r4_models as rm

from models.indep_tslr import IndepTSLR, logreg_newton
from models.influence import BaseInfluence

DOSES = (2, 4, 8)
IF_DOSES = (1, 2, 4, 8)
AUDIT_STREAM = 7201
G: dict = {}


def arm_k(base_ids, base_mult, new_id, k):
    u, cnt = arm_multiset(base_ids, base_mult, new_id)
    cnt = cnt.copy()
    cnt[np.flatnonzero(u == int(new_id))[0]] = int(k)
    return u, cnt


def _pc(model, idx):
    covs, y = G["covs"], G["y"]
    feats = rm.tangent_space_b(covs[idx], model.ref)
    return correct_class_probability(model.clf.predict_proba(feats), model.clf.classes_, y[idx])


def exact(u, cnt, idx):
    m = fit_arm(G["covs"][u], G["y"][u], cnt)
    return m, _pc(m, idx)


def _newton_pc(feats_tr, y_tr, w_tr, feats_sc, y_sc):
    pos = int(np.max(G["y"]))
    th, it = logreg_newton(feats_tr, (y_tr == pos).astype(float), np.asarray(w_tr, dtype=float))
    pp = 1.0 / (1.0 + np.exp(-(feats_sc @ th[:-1] + th[-1])))
    return np.where(y_sc == pos, pp, 1.0 - pp), it


def opt_arm(u, cnt, idx):
    """R4 Karcher reference of the arm + exact-optimum (Newton) LR on the arm's rows (weights = multiplicity)."""
    covs, y = G["covs"], G["y"]
    ref = rm.riemann_mean_w(covs[u], cnt)
    return _newton_pc(rm.tangent_space_b(covs[u], ref), y[u], cnt, rm.tangent_space_b(covs[idx], ref), y[idx])[0]


def fixed_ref(ref, base_ids, new_id, idx):
    covs, y = G["covs"], G["y"]
    ids = np.concatenate([base_ids, [new_id]])
    return _newton_pc(rm.tangent_space_b(covs[ids], ref), y[ids], np.ones(ids.size),
                      rm.tangent_space_b(covs[idx], ref), y[idx])[0]


def ref_only(base_ids, new_id, idx):
    covs, y = G["covs"], G["y"]
    u, cnt = arm_multiset(base_ids, np.ones(base_ids.size, dtype=np.int64), new_id)
    ref = rm.riemann_mean_w(covs[u], cnt)
    return _newton_pc(rm.tangent_space_b(covs[base_ids], ref), y[base_ids], np.ones(base_ids.size),
                      rm.tangent_space_b(covs[idx], ref), y[idx])[0]


def indep(u, cnt, idx):
    m = IndepTSLR().fit(G["covs"][u], G["y"][u], cnt)
    return m.p_correct(G["covs"][idx], G["y"][idx]), m.karcher_iter, m.newton_iter


def audit_index(task) -> int:
    rng = np.random.default_rng([AUDIT_STREAM, int(task["seed"]), int(task["subject"]), int(task["fold"])])
    return int(rng.integers(len(task["pairs"])))


def run_task(task: dict) -> dict:
    covs, y = G["covs"], G["y"]
    t0 = time.perf_counter()
    T = task["T"]
    pos = {int(t): i for i, t in enumerate(T.tolist())}
    base = task["base_canon"]
    ones = np.ones(base.size, dtype=np.int64)
    pos_label = int(np.max(y))
    # base models (reference fixed) for the classifier-pathway refit and the influence predictions
    ref_c = rm.riemann_mean_w(covs[base], ones)
    f_c = rm.tangent_space_b(covs[base], ref_c)
    inf_c = BaseInfluence(f_c, (y[base] == pos_label).astype(float), ones)
    ref_m = rm.riemann_mean_w(covs[base], task["base_multi_cnt"])
    f_m = rm.tangent_space_b(covs[base], ref_m)
    inf_m = BaseInfluence(f_m, (y[base] == pos_label).astype(float), task["base_multi_cnt"])
    a_ids = np.array([p["a"] for p in task["pairs"]], dtype=np.int64)
    b_ids = np.array([p["b"] for p in task["pairs"]], dtype=np.int64)
    ya, yb = (y[a_ids] == pos_label).astype(float), (y[b_ids] == pos_label).astype(float)
    pred = {}
    for k in IF_DOSES:
        q = inf_c.predict_pair(rm.tangent_space_b(covs[a_ids], ref_c), ya, rm.tangent_space_b(covs[b_ids], ref_c), yb, k)
        for key, v in q.items():
            if key.startswith(("h_", "pA0", "pB0")):
                if k == 1:
                    pred[key] = v
                continue
            pred[f"{key}_k{k}"] = v
    qm = inf_m.predict_pair(rm.tangent_space_b(covs[a_ids], ref_m), ya, rm.tangent_space_b(covs[b_ids], ref_m), yb, 1)
    for key, v in qm.items():
        pred[f"m_{key}"] = v
    j_audit = audit_index(task)
    recs = []
    n_fits = 0
    for j, pr in enumerate(task["pairs"]):
        a, b = pr["a"], pr["b"]
        ab = np.array([a, b], dtype=np.int64)
        rec = {"k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"]}
        # E1 scored on the whole fold T, exactly as G3
        _, pa = exact(*arm_multiset(base, ones, a), T)
        _, pb = exact(*arm_multiset(base, ones, b), T)
        ia, ib = pos[a], pos[b]
        rec.update({"e1_pAA": pa[ia], "e1_pBA": pa[ib], "e1_pAB": pb[ia], "e1_pBB": pb[ib]})
        n_fits += 2
        xa, ka, na = indep(*arm_multiset(base, ones, a), ab)
        xb, kb, nb = indep(*arm_multiset(base, ones, b), ab)
        rec.update({"x1_pAA": xa[0], "x1_pBA": xa[1], "x1_pAB": xb[0], "x1_pBB": xb[1],
                    "x1_karcher_iter_max": max(ka, kb), "x1_newton_iter_max": max(na, nb)})
        n_fits += 2
        for k in DOSES:
            qa = opt_arm(*arm_k(base, ones, a, k), ab)
            qb = opt_arm(*arm_k(base, ones, b, k), ab)
            rec.update({f"e{k}_pAA": qa[0], f"e{k}_pBA": qa[1], f"e{k}_pAB": qb[0], f"e{k}_pBB": qb[1]})
            n_fits += 2
        fa, fb = fixed_ref(ref_c, base, a, ab), fixed_ref(ref_c, base, b, ab)
        rec.update({"f_pAA": fa[0], "f_pBA": fa[1], "f_pAB": fb[0], "f_pBB": fb[1]})
        ra, rb = ref_only(base, a, ab), ref_only(base, b, ab)
        rec.update({"r_pAA": ra[0], "r_pBA": ra[1], "r_pAB": rb[0], "r_pBB": rb[1]})
        n_fits += 4
        for key, v in pred.items():
            rec[key] = float(v[j])
        if j == j_audit:  # independent implementation of every G3 arm of this pair
            rec["audit"] = 1.0
            ma, _, _ = indep(*arm_multiset(base, task["base_multi_cnt"], a), ab)
            mb, _, _ = indep(*arm_multiset(base, task["base_multi_cnt"], b), ab)
            rec.update({"i_mAA": ma[0], "i_mBA": ma[1], "i_mAB": mb[0], "i_mBB": mb[1]})
            if pr["g"] >= 0:
                ga, _, _ = indep(*arm_multiset(base, ones, pr["g"]), ab)
                rec.update({"i_pAG": ga[0], "i_pBG": ga[1]})
            if pr["c"] >= 0:
                ca, _, _ = indep(*arm_extra_copy(base, ones, pr["c"]), ab)
                ea, _, _ = indep(*arm_extra_copy(base, ones, pr["e"]), ab)
                rec.update({"i_pAC": ca[0], "i_pBC": ca[1], "i_pAE": ea[0], "i_pBE": ea[1]})
        recs.append(rec)
    return {"seed": task["seed"], "subject": task["subject"], "fold": task["fold"], "records": recs,
            "fits": n_fits, "seconds": time.perf_counter() - t0, "pid": os.getpid(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "newton_iter_base": [inf_c.newton_iter, inf_m.newton_iter]}
