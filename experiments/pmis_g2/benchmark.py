"""Synthetic CPU/RAM benchmark of the exact PMIS TS+LR fit (NO real EEG, NO real features).

Synthetic 11 x 11 SPD covariances are produced like the R4 pipeline: an 11-channel, 1,024-sample
(4 s at 256 Hz) Gaussian segment is mixed by a fixed random "montage" with class-dependent source
variance and per-trial gain jitter, then shrunk by the vendored R4 `oas_cov`. Each PMIS pair is fitted
with the exact R4 mechanics (vendored `TSLRw` via `pmis_g2.tslr.fit_arm`): a fresh multiplicity-weighted
Karcher reference and a fresh LogisticRegression(C=1.0, max_iter=2000) per arm, then P(correct class)
is predicted for the fold's whole scored set. Only timing, memory and convergence diagnostics are
reported. Synthetic class separation exists only to vary optimiser work; no accuracy is computed.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass

import numpy as np

from models.tslr import arm_multiset, fit_arm, predict_correct_proba
from models.vendor_r4 import r4_models as rm

P = 11
N_SAMPLES = 1024


@dataclass
class Fold:
    covs: np.ndarray
    y: np.ndarray
    base: np.ndarray
    mult: np.ndarray
    test: np.ndarray
    pairs: list
    n_distinct: int
    n_rows: int


def make_fold(n_distinct: int, n_rows: int, n_test: int, separation: float, seed: int) -> Fold:
    """Synthetic fold: base of n_distinct trials expanded to n_rows by copy multiplicities (cap 8),
    and a scored set of n_test trials; pairs are same-class scored trials."""
    rng = np.random.default_rng(seed)
    n = n_distinct + n_test
    y = np.where(np.arange(n) % 2 == 0, 1, 2)
    q, _ = np.linalg.qr(rng.standard_normal((P, P)))
    mix = q * np.exp(rng.uniform(np.log(1.0), np.log(30.0), P))
    covs = np.empty((n, P, P))
    for i in range(n):
        g = np.exp(rng.normal(0.0, 0.2, P))
        g[[0, 1] if y[i] == 2 else [2, 3]] *= 1.0 + separation
        covs[i] = rm.oas_cov((rng.standard_normal((N_SAMPLES, P)) * g) @ mix.T)
    mult = np.ones(n_distinct, dtype=np.int64)
    extra = n_rows - n_distinct
    if extra < 0 or extra > 7 * n_distinct:
        raise ValueError("impossible multiplicity target")
    while extra:
        cand = np.flatnonzero(mult < 8)
        k = min(extra, cand.size)
        mult[rng.choice(cand, size=k, replace=False)] += 1
        extra -= k
    test = np.arange(n_distinct, n)
    same = {lab: [t for t in test.tolist() if y[t] == lab] for lab in (1, 2)}
    pairs = [(same[lab][i], same[lab][i + 1]) for lab in (1, 2) for i in range(0, len(same[lab]) - 1, 2)]
    order = np.random.default_rng(seed + 1).permutation(len(pairs))
    return Fold(covs, y, np.arange(n_distinct), mult, test, [pairs[i] for i in order],
                n_distinct, int(mult.sum()))


def fit_and_score(fold: Fold, alias: int):
    """One exact PMIS arm: C + [alias'] -> fresh reference + fresh LR -> P(correct) on the scored set."""
    w0, c0 = time.perf_counter(), time.process_time()
    u, cnt = arm_multiset(fold.base, fold.mult, alias)
    model = fit_arm(fold.covs[u], fold.y[u], cnt)
    p = predict_correct_proba(model, fold.covs[fold.test], fold.y[fold.test])
    return model, p, time.perf_counter() - w0, time.process_time() - c0


def run_pair(fold: Fold, a: int, b: int) -> dict:
    ma, pa, wa, ca = fit_and_score(fold, a)
    mb, pb, wb, cb = fit_and_score(fold, b)
    rebuilt = (ma.ref is not mb.ref) and not np.array_equal(ma.ref, mb.ref) and (ma.clf is not mb.clf)
    return {"wall": [wa, wb], "cpu": [ca, cb], "lr_iter": [int(ma.clf.n_iter_[0]), int(mb.clf.n_iter_[0])],
            "reference_rebuilt": bool(rebuilt), "finite": bool(np.isfinite(pa).all() and np.isfinite(pb).all())}


def karcher_iterations(covs: np.ndarray, w: np.ndarray) -> int:
    """Count Karcher iterations of the vendored weighted mean by counting its sqrtm calls (diagnostic)."""
    calls = [0]
    orig = rm.sqrtm

    def counting(c):
        calls[0] += 1
        return orig(c)
    rm.sqrtm = counting
    try:
        rm.riemann_mean_w(covs, w)
    finally:
        rm.sqrtm = orig
    return calls[0]


def verification_and_components(fold: Fold, a: int, b: int) -> dict:
    """Exactness checks for one pair plus a component-timed replica of TSLRw.fit (verified bit-equal)."""
    from sklearn.linear_model import LogisticRegression
    ua, ca = arm_multiset(fold.base, fold.mult, a)
    ub, cb = arm_multiset(fold.base, fold.mult, b)
    ma = fit_arm(fold.covs[ua], fold.y[ua], ca)
    mb = fit_arm(fold.covs[ub], fold.y[ub], cb)
    t0 = time.perf_counter()
    ref = rm.riemann_mean_w(fold.covs[ua], ca)
    t1 = time.perf_counter()
    feats = rm.tangent_space_b(fold.covs[ua], ref)
    t2 = time.perf_counter()
    rep = np.repeat(np.arange(ua.size), ca)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(feats[rep], fold.y[ua][rep])
    t3 = time.perf_counter()
    p = clf.predict_proba(rm.tangent_space_b(fold.covs[fold.test], ref))
    t4 = time.perf_counter()
    replica_equal = bool(np.array_equal(ref, ma.ref) and np.array_equal(clf.coef_, ma.clf.coef_))
    base_ref = rm.riemann_mean_w(fold.covs[fold.base], fold.mult)
    # labelled shortcut (DIFFERENT ESTIMATOR, never used): keep the base reference, refit only the LR
    sf = rm.tangent_space_b(fold.covs[ua], base_ref)
    s_clf = LogisticRegression(C=1.0, max_iter=2000).fit(sf[rep], fold.y[ua][rep])
    s_p = s_clf.predict_proba(rm.tangent_space_b(fold.covs[fold.test], base_ref))
    exact_p = predict_correct_proba(ma, fold.covs[fold.test], fold.y[fold.test])
    idx = np.searchsorted(s_clf.classes_, fold.y[fold.test])
    short_p = s_p[np.arange(fold.test.size), idx]
    return {
        "replica_bit_equal_to_vendored_TSLRw_fit": replica_equal,
        "reference_equals_fresh_weighted_karcher_mean_arm_A": bool(np.array_equal(
            ma.ref, rm.riemann_mean_w(fold.covs[ua], ca))),
        "reference_equals_fresh_weighted_karcher_mean_arm_B": bool(np.array_equal(
            mb.ref, rm.riemann_mean_w(fold.covs[ub], cb))),
        "reference_A_differs_from_B_max_abs": float(np.abs(ma.ref - mb.ref).max()),
        "reference_A_differs_from_base_only_max_abs": float(np.abs(ma.ref - base_ref).max()),
        "shortcut_fixed_base_reference_DIFFERENT_ESTIMATOR": {
            "used": False,
            "max_abs_tangent_feature_difference": float(np.abs(sf - feats).max()),
            "max_abs_correct_class_probability_difference": float(np.abs(short_p - exact_p).max()),
        },
        "component_seconds": {"karcher_reference": t1 - t0, "tangent_map_train": t2 - t1,
                              "logistic_regression_fit": t3 - t2, "predict_scored_set": t4 - t3},
        "karcher_iterations": karcher_iterations(fold.covs[ua], ca),
        "probability_rows_sum_to_one": bool(np.allclose(p.sum(1), 1.0)),
    }


def stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max())}


def max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ------------------------------------------------------------------------------ pool worker helpers
_W: dict = {}


def worker_init(cfg: dict, nice: int) -> None:
    if nice:
        os.nice(nice)
    _W["fold"] = make_fold(**cfg)


def worker_task(j: int) -> dict:
    fold = _W["fold"]
    a, b = fold.pairs[j % len(fold.pairs)]
    out = run_pair(fold, a, b)
    out["pid"] = os.getpid()
    out["max_rss_mb"] = max_rss_mb()
    return out


def iteration_cap_stress(fold: Fold, alias: int, repeats: int = 3) -> dict:
    """TIMING UPPER BOUND ONLY (not the estimator): force the vendored Karcher mean to its maxiter=50 cap
    (tol=0) and the LR towards max_iter=2000 (tol=0), to bound per-fit cost if real EEG features converge
    far more slowly than synthetic ones. Never used for any estimate."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    u, cnt = arm_multiset(fold.base, fold.mult, alias)
    covs, y = fold.covs[u], fold.y[u]
    kt, lt, lit = [], [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ref = rm.riemann_mean_w(covs, cnt, tol=0.0)
        kt.append(time.perf_counter() - t0)
        feats = rm.tangent_space_b(covs, ref)
        rep = np.repeat(np.arange(u.size), cnt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            t1 = time.perf_counter()
            clf = LogisticRegression(C=1.0, max_iter=2000, tol=0.0).fit(feats[rep], y[rep])
            lt.append(time.perf_counter() - t1)
        lit.append(int(clf.n_iter_[0]))
    per_lr_iter = float(np.median(lt)) / max(1, int(np.median(lit)))
    return {"label": "TIMING_UPPER_BOUND_NOT_THE_ESTIMATOR",
            "karcher_50_iterations_s": float(np.median(kt)),
            "lr_tol0_s": float(np.median(lt)), "lr_tol0_iterations": int(np.median(lit)),
            "lr_seconds_per_iteration": per_lr_iter,
            "lr_2000_iterations_extrapolated_s": per_lr_iter * 2000}
