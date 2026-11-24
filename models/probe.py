"""R8-E1: frozen pretrained encoder (CBraMod) + exact linear probe, identity swap on the full C3 Lee2019 design.

Embedding of a trial: CBraMod backbone (TUEG-pretrained checkpoint, eval mode, proj_out removed) on its (20, 4, 200)
input in units of 100 uV, averaged over channels and patches (the official 'avgpooling_patch_reps' read-out) -> 200-d.
The encoder is frozen and per-trial (no batch statistics), so a copy's embedding is bit-identical to its trial's.
Probe per arm (exact refit): standardisation fitted on the arm's rows (row-multiplicity weighted, as a pipeline scaler
would be) + L2 logistic regression C = 1 at the exact optimum (damped Newton, gradient max-norm < 1e-10).
Arms per C3 task (base = the other 300 trials of the participant): A / B = base + a' / b' for each of the 40 pairs and
C / E = base + extra copy of the pair's placebo trials c / e; all arms scored on the held-out run's 100 trials.
"""

from __future__ import annotations

import numpy as np

from models.indep_tslr import _sig, logreg_newton


def fit_score(F, y01, rows, T):
    u, cnt = np.unique(rows, return_counts=True)
    w = cnt.astype(np.float64)
    mu = (w[:, None] * F[u]).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (F[u] - mu) ** 2).sum(0) / w.sum()) + 1e-12
    Z = (F[u] - mu) / sd
    yy = y01[u].astype(np.float64)
    th, it = logreg_newton(Z, yy, w, C=1.0, tol=1e-10, maxiter=100)
    xt = np.hstack([Z, np.ones((Z.shape[0], 1))])
    g = float(np.max(np.abs(xt.T @ (w * (_sig(xt @ th) - yy)) + np.r_[np.ones(Z.shape[1]), 0.0] * th)))
    # v1.1: the backtracking line search can stagnate at the floating-point floor just above tol (observed 5e-10..8e-8);
    # such a fit is accepted only if max|gradient| <= 1e-6 (strongly convex objective, Hessian >= I), and counted
    if it >= 100 and g > 1e-6:
        raise RuntimeError(f"probe Newton did not converge (max|grad| {g:.2e})")
    p1 = _sig(((F[T] - mu) / sd) @ th[:-1] + th[-1])
    return np.where(y01[T] == 1, p1, 1.0 - p1), it, g


def run_task(t, F, y01):
    base = np.asarray(t["base_canon"], dtype=np.int64)
    T = np.asarray(t["T"], dtype=np.int64)
    pos = {int(x): i for i, x in enumerate(T)}
    um = np.asarray(t["unexposed"], dtype=bool)
    recs, its, gs = [], [], []
    for pr in t["pairs"]:
        P = {}
        for arm, extra in (("A", pr["a"]), ("B", pr["b"]), ("C", pr["c"]), ("E", pr["e"])):
            P[arm], it, g = fit_score(F, y01, np.r_[base, extra], T)
            its.append(it)
            gs.append(g)
        ia, ib = pos[pr["a"]], pos[pr["b"]]
        recs.append({"subject": int(t["subject"]), "fold": int(t["fold"]), "k": int(pr["k"]), "label": int(pr["label"]),
                     "pos_dist": int(pr["pos_dist"]), "sign": int(pr["sign"]),
                     "pAA": P["A"][ia], "pBA": P["A"][ib], "pAB": P["B"][ia], "pBB": P["B"][ib],
                     "pAC": P["C"][ia], "pBC": P["C"][ib], "pAE": P["E"][ia], "pBE": P["E"][ib],
                     "off": float(np.mean(np.abs(P["A"][um] - P["B"][um])))})
    return recs, {"newton_max_iter": max(its), "n_fits": len(its), "n_stagnated": int(sum(i >= 100 for i in its)),
                  "max_abs_grad": max(gs)}
