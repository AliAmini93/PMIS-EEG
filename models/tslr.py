"""PMIS arm fitting with the frozen R4 TS+LR (vendored `r4_models.TSLRw`, unmodified).

Fit mechanics are exactly R4 G3_tslr.run_task: the arm's training multiset is reduced to distinct rows +
integer multiplicities, the affine-invariant Riemannian reference is the multiplicity-weighted Karcher
mean recomputed FROM SCRATCH for every arm, the tangent features use that arm's own reference, and a new
LogisticRegression(C=1.0, max_iter=2000) is fitted on the expanded rows. The only addition is reading
P(correct class) from the fitted classifier (R4 used hard labels only).

Any shortcut that reuses the base reference, warm-starts the classifier or updates features
incrementally is a DIFFERENT ESTIMATOR and must be labelled as such; none is implemented here.
"""

from __future__ import annotations

import numpy as np

from evaluation.g2_estimands import correct_class_probability
from models.vendor_r4 import r4_models as rm


def arm_multiset(base_distinct: np.ndarray, base_mult: np.ndarray, alias_distinct: int):
    """Distinct ids + multiplicities of C + [alias]. The alias identity must be new (not in C)."""
    base_distinct = np.asarray(base_distinct, dtype=np.int64)
    base_mult = np.asarray(base_mult, dtype=np.int64)
    if base_distinct.size != np.unique(base_distinct).size:
        raise ValueError("base distinct ids repeat")
    if (base_mult < 1).any():
        raise ValueError("multiplicity < 1")
    if int(alias_distinct) in set(base_distinct.tolist()):
        raise ValueError("alias identity already in base (would not add support)")
    u = np.concatenate([base_distinct, [int(alias_distinct)]])
    cnt = np.concatenate([base_mult, [1]])
    order = np.argsort(u, kind="stable")
    return u[order], cnt[order]


def arm_extra_copy(base_distinct: np.ndarray, base_mult: np.ndarray, copy_of: int):
    """Distinct ids + multiplicities of C + [c'] where c is ALREADY a base identity (placebo arm):
    support stays |C|, the multiplicity of c rises by one."""
    base_distinct = np.asarray(base_distinct, dtype=np.int64)
    cnt = np.asarray(base_mult, dtype=np.int64).copy()
    hit = np.flatnonzero(base_distinct == int(copy_of))
    if hit.size != 1:
        raise ValueError("placebo copy must duplicate exactly one base identity")
    cnt[hit[0]] += 1
    order = np.argsort(base_distinct, kind="stable")
    return base_distinct[order], cnt[order]


def fit_arm(covs_distinct: np.ndarray, y_distinct: np.ndarray, mult: np.ndarray) -> rm.TSLRw:
    """Exact R4 fit: fresh reference + fresh classifier for this arm."""
    if np.unique(np.asarray(y_distinct)).size < 2:
        raise ValueError("single-class arm: PMIS arms must contain both classes")
    return rm.TSLRw().fit(covs_distinct, y_distinct, mult)


def predict_correct_proba(model: rm.TSLRw, covs: np.ndarray, y: np.ndarray) -> np.ndarray:
    feats = rm.tangent_space_b(covs, model.ref)
    return correct_class_probability(model.clf.predict_proba(feats), model.clf.classes_, y)
