"""Parameter-free first-order predictions of the PMIS contrast for TS+LR (R6-C1).

Base model: reference M_C and tangent features of the base multiset (fixed), L2-logistic optimum theta_C
(objective sum_i m_i logloss_i + ||beta||^2/(2C), intercept unpenalised), Hessian
H = sum_i m_i p_i (1-p_i) x_i x_i^T + diag(1/C, ..., 1/C, 0) with x = [tangent features, 1].
Adding row z with multiplicity k changes the logit of any x by
    IF:        d_eta(x) = -k r_z h_xz                         (classical influence function)
    Newton-1:  d_eta(x) = -k r_z h_xz / (1 + k p_z(1-p_z) h_zz) (one exact Newton step, Sherman-Morrison)
with r_z = p_z - y_z (y in {0,1}), h_xz = x^T H^-1 z. Neither uses any arm refit, so a match with the exact
arm refits is a genuine (parameter-free) prediction. The reference is held at M_C, so the comparison with the
exact fixed-reference refit isolates the linearisation error, and the comparison with the exact per-arm refit
adds the Riemannian-reference pathway.
"""

from __future__ import annotations

import numpy as np

from models.indep_tslr import _sig, logreg_newton


class BaseInfluence:
    def __init__(self, base_feats: np.ndarray, base_y01: np.ndarray, base_mult: np.ndarray, C: float = 1.0):
        self.theta, self.newton_iter = logreg_newton(base_feats, base_y01, base_mult.astype(float), C=C)
        xt = np.hstack([base_feats, np.ones((base_feats.shape[0], 1))])
        p = _sig(xt @ self.theta)
        reg = np.full(xt.shape[1], 1.0 / C)
        reg[-1] = 0.0
        self.H = (xt * (base_mult * p * (1 - p))[:, None]).T @ xt + np.diag(reg)
        self.Hinv = np.linalg.inv(self.H)

    @staticmethod
    def _aug(f):
        return np.hstack([f, np.ones((f.shape[0], 1))])

    def predict_pair(self, fa: np.ndarray, ya01: np.ndarray, fb: np.ndarray, yb01: np.ndarray, k: int = 1) -> dict:
        """Vectorised over pairs: fa, fb (n_pairs, p); y in {0,1}. Returns correct-class probabilities
        pXY = P(correct of X | base + k copies of Y) for both approximations, and leverages."""
        xa, xb = self._aug(fa), self._aug(fb)
        ua, ub = xa @ self.Hinv, xb @ self.Hinv
        h_aa = np.einsum("ij,ij->i", ua, xa)
        h_bb = np.einsum("ij,ij->i", ub, xb)
        h_ab = np.einsum("ij,ij->i", ua, xb)
        eta_a, eta_b = xa @ self.theta, xb @ self.theta
        p_a, p_b = _sig(eta_a), _sig(eta_b)
        r_a, r_b = p_a - ya01, p_b - yb01
        s_a, s_b = 2 * ya01 - 1, 2 * yb01 - 1  # +1 if the correct class is the positive one
        out = {"h_aa": h_aa, "h_bb": h_bb, "h_ab": h_ab, "pA0": _sig(s_a * eta_a), "pB0": _sig(s_b * eta_b)}
        for name, damp_a, damp_b in (("if", 1.0, 1.0),
                                     ("nt", 1.0 + k * p_a * (1 - p_a) * h_aa, 1.0 + k * p_b * (1 - p_b) * h_bb)):
            # add a (z = a): logits of a and b move
            de_a_by_a = -k * r_a * h_aa / damp_a
            de_b_by_a = -k * r_a * h_ab / damp_a
            de_a_by_b = -k * r_b * h_ab / damp_b
            de_b_by_b = -k * r_b * h_bb / damp_b
            out[f"{name}_pAA"] = _sig(s_a * (eta_a + de_a_by_a))
            out[f"{name}_pBA"] = _sig(s_b * (eta_b + de_b_by_a))
            out[f"{name}_pAB"] = _sig(s_a * (eta_a + de_a_by_b))
            out[f"{name}_pBB"] = _sig(s_b * (eta_b + de_b_by_b))
        return out
