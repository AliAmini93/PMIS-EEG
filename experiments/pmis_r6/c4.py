"""R6-C4: does the exact-copy advantage need shared samples? Overlap / episode / pre-cue decomposition (TS+LR).

All features are window-local 2-s covariances of Lee2019 trials at six positions (scripts/r6_c3_build_lee2019.py).
Variant w of trial i sits at index w * N + i of the stacked array (w = 0 is the scored/base window V0.0).
For each C3 pair (a, b) and each variant v: arms C + v(a) and C + v(b) (exact-optimum per-arm refit: per-arm Karcher
reference + damped-Newton LR); base C and scored trials use V0.0. The PMIS contrast per variant is
    d_v = 1/2 [(p(a | C + v(a)) - p(a | C + v(b))) + (p(b | C + v(b)) - p(b | C + v(a)))].
A parameter-free one-Newton-step prediction from the base model is computed for every variant (cross-leverage).
"""

from __future__ import annotations

import os
import resource
import time

import numpy as np

from models.tslr import arm_multiset
from models.vendor_r4 import r4_models as rm

from experiments.pmis_r6 import c3
from models.indep_tslr import _sig
from models.influence import BaseInfluence

VARIANTS = ("V0.0", "V0.5", "V1.0", "V1.5", "V2.0", "P")
OVERLAP = {"V0.0": 1.0, "V0.5": 0.75, "V1.0": 0.5, "V1.5": 0.25, "V2.0": 0.0, "P": 0.0}
N_TRIALS = 21600
G = c3.G  # shared: G["covs"] = stacked (6 * N, 20, 20), G["y"] = tiled labels


def stack(win_covs: np.ndarray, y: np.ndarray):
    n, w = win_covs.shape[:2]
    assert n == N_TRIALS and w == len(VARIANTS)
    return np.ascontiguousarray(np.swapaxes(win_covs, 0, 1).reshape(w * n, *win_covs.shape[2:])), np.tile(y, w)


def nt_predict(inf: BaseInfluence, f_s: np.ndarray, y_s01: np.ndarray, f_z: np.ndarray, y_z01: np.ndarray) -> np.ndarray:
    """Correct-class probability of scored rows f_s after adding row z (one Newton step), vectorised over rows."""
    xs = np.hstack([f_s, np.ones((f_s.shape[0], 1))])
    xz = np.hstack([f_z, np.ones((f_z.shape[0], 1))])
    uz = xz @ inf.Hinv
    h_sz = np.einsum("ij,ij->i", uz, xs)
    h_zz = np.einsum("ij,ij->i", uz, xz)
    pz = _sig(xz @ inf.theta)
    eta = xs @ inf.theta - (pz - y_z01) * h_sz / (1.0 + pz * (1 - pz) * h_zz)
    return _sig((2 * y_s01 - 1) * eta)


def run_task(task: dict) -> dict:
    covs, y = G["covs"], G["y"]
    t0 = time.perf_counter()
    base = task["base_canon"]
    ones = np.ones(base.size, dtype=np.int64)
    pl = int(np.max(y))
    ref = rm.riemann_mean_w(covs[base], ones)
    inf = BaseInfluence(rm.tangent_space_b(covs[base], ref), (y[base] == pl).astype(float), ones)
    recs, n_fits, it_max = [], 0, 0
    for pr in task["pairs"]:
        a, b = pr["a"], pr["b"]
        ab = np.array([a, b])
        r = {"k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"]}
        fs = rm.tangent_space_b(covs[ab], ref)
        ys = (y[ab] == pl).astype(float)
        for w, v in enumerate(VARIANTS):
            za, zb = w * N_TRIALS + a, w * N_TRIALS + b
            pA, i1 = c3.opt_pc(*arm_multiset(base, ones, za), ab)
            pB, i2 = c3.opt_pc(*arm_multiset(base, ones, zb), ab)
            n_fits += 2
            it_max = max(it_max, i1, i2)
            r.update({f"{v}_pAA": pA[0], f"{v}_pBA": pA[1], f"{v}_pAB": pB[0], f"{v}_pBB": pB[1]})
            fz = rm.tangent_space_b(covs[[za, zb]], ref)
            qa = nt_predict(inf, fs, ys, np.repeat(fz[:1], 2, 0), np.repeat(ys[:1], 2))
            qb = nt_predict(inf, fs, ys, np.repeat(fz[1:], 2, 0), np.repeat(ys[1:], 2))
            r.update({f"nt_{v}_pAA": qa[0], f"nt_{v}_pBA": qa[1], f"nt_{v}_pAB": qb[0], f"nt_{v}_pBB": qb[1]})
            # tangent-space similarity of the copy to its own scored window (descriptive)
            r[f"cos_{v}"] = float(0.5 * sum(np.dot(fz[i], fs[i]) / (np.linalg.norm(fz[i]) * np.linalg.norm(fs[i]))
                                            for i in range(2)))
        recs.append(r)
    return {"subject": task["subject"], "fold": task["fold"], "records": recs, "fits": n_fits, "newton_iter_max": it_max,
            "seconds": time.perf_counter() - t0, "pid": os.getpid(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
