"""Independent, provably descending Karcher-mean solver for the R7-C1b audit.

Replaces the defective unit-step iteration of pmis_r6.indep_tslr (which diverged on 63 % of HS-Stroke arms; see
reports/R6_C1_REPORT.md section 1.1, preserved unchanged). Written from the definition, without R4 code:

  objective   f(M) = sum_i w_i || log(M^-1/2 C_i M^-1/2) ||_F^2          (weights normalised to sum 1)
  direction   J(M) = sum_i w_i log(M^-1/2 C_i M^-1/2); along the geodesic M(t) = M^1/2 exp(t J) M^1/2 the
              directional derivative of f at t = 0 is -2 ||J||_F^2 (descent direction)
  step        Armijo backtracking: start t = 1, halve until f(M(t)) <= f(M) - 2 c t ||J||_F^2, c = 1e-4; once the
              predicted decrease is below float64 resolution of f, accept the largest halving that lowers ||J||
  init        weighted log-Euclidean mean
  stop        ||J||_F <= tol (default 1e-10), stagnation (no accepted step down to t = 1e-12) or maxiter; the final
              ||J|| is returned and audited
f is geodesically convex on the SPD manifold under the affine-invariant metric, so gradient descent with Armijo steps
converges to the unique Karcher mean.
"""

from __future__ import annotations

import numpy as np

from models.indep_tslr import _eig_apply, logreg_newton, tangent, _sig


def _logs(m, covs):
    mih = _eig_apply(m, lambda x: 1.0 / np.sqrt(x))
    return _eig_apply(mih @ covs @ mih, np.log)


def karcher_ls(covs: np.ndarray, weights: np.ndarray, tol: float = 1e-10, maxiter: int = 2000, c: float = 1e-4):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    m = _eig_apply(np.einsum("n,nij->ij", w, _eig_apply(covs, np.log)), np.exp)
    L = _logs(m, covs)
    f = float(np.einsum("n,nij,nij->", w, L, L))
    j = np.einsum("n,nij->ij", w, L)
    gn = float(np.linalg.norm(j, "fro"))
    it, backtracks, stalled = 0, 0, False
    while gn > tol and it < maxiter:
        it += 1
        mh = _eig_apply(m, np.sqrt)
        t = 1.0
        while True:
            cand = mh @ _eig_apply(0.5 * (j + j.T) * t, np.exp) @ mh
            cand = 0.5 * (cand + cand.T)
            Lc = _logs(cand, covs)
            fc = float(np.einsum("n,nij,nij->", w, Lc, Lc))
            resolvable = 2.0 * t * gn * gn > 1e-10 * max(1.0, abs(f))  # Armijo test meaningful at float64 precision?
            if resolvable and fc <= f - 2.0 * c * t * gn * gn:
                break
            if not resolvable:  # near convergence: require a decrease of the gradient norm instead
                jc = float(np.linalg.norm(np.einsum("n,nij->ij", w, Lc), "fro"))
                if jc < gn:
                    break
            if t < 1e-12:
                break
            t *= 0.5
            backtracks += 1
        if t < 1e-12:  # stagnation at float64 resolution: keep the current iterate and stop
            stalled = True
            break
        m, L, f = cand, Lc, fc
        j = np.einsum("n,nij->ij", w, L)
        gn = float(np.linalg.norm(j, "fro"))
    return m, {"iterations": it, "grad_norm": gn, "objective": f, "backtracks": backtracks, "stalled": stalled}


class IndepTSLR2:
    """Independent TS+LR arm fit: Karcher-LS reference + damped-Newton exact L2-logistic (C = 1, unpenalised intercept)."""

    def fit(self, covs, y, mult):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("needs two classes")
        self.ref, self.karcher = karcher_ls(covs, mult)
        self.theta, self.newton_iter = logreg_newton(tangent(covs, self.ref), (y == self.classes_[1]).astype(float),
                                                     np.asarray(mult, dtype=float))
        return self

    def p_correct(self, covs, y):
        f = tangent(covs, self.ref)
        pp = _sig(f @ self.theta[:-1] + self.theta[-1])
        return np.where(np.asarray(y) == self.classes_[1], pp, 1.0 - pp)
