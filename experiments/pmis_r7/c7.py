"""R7-C7: from trial-level exact-copy effects to benchmark-level inflation (TS+LR, HS-Stroke, R4 arms; CPU).

R4 arms (G2 manifest, frozen): LEAKED L = C + E + X (the producers' training multiset: canonical non-test trials C,
their cumulative copies E, and the copies X of exposed TEST trials), PLACEBO P = C + E + Z (Z: session/class-matched
copies of non-test trials replacing X; rows fixed), CLEAN = C. For every (split, seed, unit, fold):
  r4_*   frozen R4 TSLRw fit (vendored, sklearn lbfgs) of L, P, C: hard labels (must equal R4's stored predictions)
         and correct-class probabilities on the scored set T
  opt_*  exact-optimum fits of P and L (R4 Karcher reference of the arm + damped-Newton L2-logistic)
  law    parameter-free prediction of L from the exact P model WITHOUT fitting L: P's reference and features,
         one joint Newton step on the L objective from theta_P, i.e. every distinct row's multiplicity changes from
         its P count to its L count (X rows added, Z rows removed); Sherman-free dense solve
R4's aggregate outcomes are known: every comparison with them is a RETROSPECTIVE CONSISTENCY CHECK.
"""

from __future__ import annotations

import os
import resource
import time

import numpy as np

from evaluation.g2_estimands import correct_class_probability
from models.vendor_r4 import r4_models as rm
from models.indep_tslr import _sig, logreg_newton

ARMS = ("T", "L", "C", "P", "LCB", "TEST_INST")
G: dict = {}


def _get(sel, arm):
    r = sel[sel[:, 4] == ARMS.index(arm)][0]
    return G["rows"][r[5]:r[6]]


def _r4(tr, T):
    covs, y = G["covs"], G["label"]
    u, cnt = np.unique(tr, return_counts=True)
    classes = np.unique(y[u])
    if classes.size < 2:  # R4's pre-specified degenerate rule: predict the only training class
        k = int(classes[0])
        return np.full(T.size, k, dtype=np.int8), (y[T] == k).astype(float), True
    m = rm.TSLRw().fit(covs[u], y[u], cnt)
    f = rm.tangent_space_b(covs[T], m.ref)
    pred = np.asarray(m.clf.predict(f)).astype(np.int8)
    return pred, correct_class_probability(m.clf.predict_proba(f), m.clf.classes_, y[T]), False


def _opt(u, cnt):
    covs, y = G["covs"], G["label"]
    ref = rm.riemann_mean_w(covs[u], cnt)
    th, it = logreg_newton(rm.tangent_space_b(covs[u], ref), (y[u] == 2).astype(float), cnt.astype(float))
    return ref, th, it


def _pc(ref, th, T):
    y = G["label"]
    pp = _sig(rm.tangent_space_b(G["covs"][T], ref) @ th[:-1] + th[-1])
    return np.where(y[T] == 2, pp, 1.0 - pp)


def law_predict(refP, thP, idsP, cntP, idsL, cntL, T):
    """One joint Newton step on the L objective from the exact P optimum, in P's tangent space."""
    covs, y = G["covs"], G["label"]
    ids = np.union1d(idsP, idsL)
    cP = np.zeros(ids.size)
    cL = np.zeros(ids.size)
    cP[np.searchsorted(ids, idsP)] = cntP
    cL[np.searchsorted(ids, idsL)] = cntL
    x = np.hstack([rm.tangent_space_b(covs[ids], refP), np.ones((ids.size, 1))])
    y01 = (y[ids] == 2).astype(float)
    p = _sig(x @ thP)
    reg = np.ones(x.shape[1])
    reg[-1] = 0.0
    grad = x.T @ (cL * (p - y01)) + reg * thP          # gradient of the L objective at theta_P
    H = (x * (cL * p * (1 - p))[:, None]).T @ x + np.diag(reg)  # Hessian of the L objective at theta_P
    thL = thP - np.linalg.solve(H, grad)
    return _pc(refP, thL, T)


def run_task(task):
    split, seed, unit = task
    t = G["table"]
    sel_all = t[(t[:, 0] == split) & (t[:, 1] == seed) & (t[:, 2] == unit)]
    t0 = time.perf_counter()
    out, n_single, n_law_skipped, newton_max = [], 0, 0, 0
    for f in np.unique(sel_all[:, 3]):
        sel = sel_all[sel_all[:, 3] == f]
        T = _get(sel, "T")
        res = {"fold": int(f), "T": T}
        for arm in ("L", "P", "C"):
            pred, pc, single = _r4(_get(sel, arm), T)
            res[f"r4_pred_{arm}"], res[f"r4_p_{arm}"] = pred, pc
            n_single += int(single)
        uP, cP = np.unique(_get(sel, "P"), return_counts=True)
        uL, cL = np.unique(_get(sel, "L"), return_counts=True)
        pos = np.searchsorted(uL, T)
        hit = (pos < uL.size) & (uL[np.minimum(pos, uL.size - 1)] == T)
        res["mult_in_L"] = np.where(hit, cL[np.minimum(pos, uL.size - 1)], 0)  # copies of each scored trial in LEAKED
        if np.unique(G["label"][uP]).size < 2 or np.unique(G["label"][uL]).size < 2:
            n_law_skipped += 1
            for k in ("opt_p_L", "opt_p_P", "law_p_L"):
                res[k] = np.full(T.size, np.nan)
        else:
            refP, thP, i1 = _opt(uP, cP)
            refL, thL, i2 = _opt(uL, cL)
            newton_max = max(newton_max, i1, i2)
            res["opt_p_P"], res["opt_p_L"] = _pc(refP, thP, T), _pc(refL, thL, T)
            res["law_p_L"] = law_predict(refP, thP, uP, cP, uL, cL, T)
        out.append(res)
    return {"split": split, "seed": seed, "unit": unit, "folds": out, "single_class_r4_fits": n_single,
            "law_skipped_folds": n_law_skipped, "newton_iter_max": newton_max, "seconds": time.perf_counter() - t0,
            "pid": os.getpid(), "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
