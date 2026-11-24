"""Independent re-implementation of the TS+LR arm fit (R6-C1 audit).

Written from the mathematical definition, without R4/pyriemann/sklearn code:
  * reference: weighted affine-invariant Karcher mean, log-Euclidean initialisation, unit-step Riemannian
    gradient iteration to ||J||_F < 1e-12 (R4 uses an arithmetic-mean start and an adaptive step);
  * tangent map: S = logm(M^-1/2 C M^-1/2); features = diagonal then sqrt(2) x strict upper triangle
    (column-major; a different order from R4, irrelevant for L2-regularised LR);
  * classifier: L2-regularised logistic regression, objective sum_i w_i * logloss_i + ||beta||^2 / (2C),
    intercept unpenalised (the sklearn lbfgs objective), solved by damped Newton to ||grad||_inf < 1e-10.
Multiplicities enter as sample weights (equivalent to expanded rows).
"""

from __future__ import annotations

import numpy as np


def _eig_apply(c: np.ndarray, f) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    return np.einsum("...ij,...j,...kj->...ik", v, f(w), v)


def karcher_mean(covs: np.ndarray, weights: np.ndarray, tol: float = 1e-12, maxiter: int = 500) -> tuple[np.ndarray, int]:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    m = _eig_apply(np.einsum("n,nij->ij", w, _eig_apply(covs, np.log)), np.exp)  # log-Euclidean start
    for it in range(1, maxiter + 1):
        mh = _eig_apply(m, np.sqrt)
        mih = _eig_apply(m, lambda x: 1.0 / np.sqrt(x))
        j = np.einsum("n,nij->ij", w, _eig_apply(mih @ covs @ mih, np.log))
        m = mh @ _eig_apply(0.5 * (j + j.T), np.exp) @ mh
        m = 0.5 * (m + m.T)
        if np.linalg.norm(j, "fro") < tol:
            return m, it
    return m, maxiter


def tangent(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mih = _eig_apply(ref, lambda x: 1.0 / np.sqrt(x))
    s = _eig_apply(mih @ covs @ mih, np.log)
    n = ref.shape[0]
    iu = [(i, j) for j in range(n) for i in range(j)]  # strict upper, column-major
    diag = s[..., np.arange(n), np.arange(n)]
    off = np.stack([s[..., i, j] for i, j in iu], axis=-1) * np.sqrt(2.0)
    return np.concatenate([diag, off], axis=-1)


def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def logreg_newton(x: np.ndarray, y01: np.ndarray, sw: np.ndarray, C: float = 1.0, tol: float = 1e-10,
                  maxiter: int = 100) -> tuple[np.ndarray, int]:
    """theta = [beta, intercept]; returns (theta, iterations)."""
    xt = np.hstack([x, np.ones((x.shape[0], 1))])
    reg = np.full(xt.shape[1], 1.0 / C)
    reg[-1] = 0.0
    th = np.zeros(xt.shape[1])

    def obj(t):
        z = xt @ t
        return float(np.sum(sw * (np.logaddexp(0.0, z) - y01 * z)) + 0.5 * np.sum(reg * t * t))

    f = obj(th)
    for it in range(1, maxiter + 1):
        p = _sig(xt @ th)
        g = xt.T @ (sw * (p - y01)) + reg * th
        if np.max(np.abs(g)) < tol:
            return th, it
        h = (xt * (sw * p * (1 - p))[:, None]).T @ xt + np.diag(reg)
        step = np.linalg.solve(h, g)
        s = 1.0
        while True:
            cand = th - s * step
            fc = obj(cand)
            if fc <= f - 1e-4 * s * float(g @ step) or s < 1e-10:
                break
            s *= 0.5
        th, f = cand, fc
    return th, maxiter


class IndepTSLR:
    """Fit on distinct rows + multiplicities; classes are the two label values, positive = larger label."""

    def fit(self, covs, y, mult, C: float = 1.0):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("needs two classes")
        self.ref, self.karcher_iter = karcher_mean(covs, mult)
        self.theta, self.newton_iter = logreg_newton(tangent(covs, self.ref), (y == self.classes_[1]).astype(float),
                                                     np.asarray(mult, dtype=float), C=C)
        return self

    def p_positive(self, covs):
        f = tangent(covs, self.ref)
        return _sig(f @ self.theta[:-1] + self.theta[-1])

    def p_correct(self, covs, y):
        pp = self.p_positive(covs)
        return np.where(np.asarray(y) == self.classes_[1], pp, 1.0 - pp)
