"""R8-A model fits on the Shin2022 tasks: TS+LR exact optimum (+ one-step law), EEGNet-8,2 at 250 Hz with CRN."""

from __future__ import annotations

import numpy as np

from models.vendor_r4 import r4_models as rm
from models.vendor_r4.r4_models import make_eegnet
from models.indep_tslr import _sig, logreg_newton
from models.influence import BaseInfluence
from models.deep_group import EEGNetG

from experiments.pmis_r8.shin import arm_rows

KERN_250 = 125  # EEGNet-8,2 temporal kernel = fs / 2 at 250 Hz (Paper A / R3 setting)


class EEGNet250(EEGNetG):
    def __init__(self, M, C, T):
        super().__init__(M, C, T, kern=KERN_250)

    @staticmethod
    def make_single(C, T):
        return make_eegnet(C, T, kern=KERN_250)


def _pc(covs, y, ref, th, idx):
    pp = _sig(rm.tangent_space_b(covs[idx], ref) @ th[:-1] + th[-1])
    return np.where(y[idx] == 2, pp, 1.0 - pp)


def tslr_task(task, covs, y):
    """Exact-optimum arm fits (R4 Karcher of the arm + damped Newton) scored on T; one-step prediction for A-arms."""
    rows, extras = arm_rows(task)
    T = task["T"]
    P = np.zeros((len(extras), T.size))
    it_max = 0
    for j in range(len(extras)):
        u, cnt = np.unique(rows[j], return_counts=True)
        ref = rm.riemann_mean_w(covs[u], cnt)
        th, it = logreg_newton(rm.tangent_space_b(covs[u], ref), (y[u] == 2).astype(float), cnt.astype(float))
        P[j] = _pc(covs, y, ref, th, T)
        it_max = max(it_max, it)
    base = task["base"]
    ref_c = rm.riemann_mean_w(covs[base], np.ones(base.size))
    inf = BaseInfluence(rm.tangent_space_b(covs[base], ref_c), (y[base] == 2).astype(float), np.ones(base.size))
    fT = rm.tangent_space_b(covs[T], ref_c)
    xT = np.hstack([fT, np.ones((T.size, 1))])
    yT = (y[T] == 2).astype(float)
    U = xT @ inf.Hinv
    pz = _sig(xT @ inf.theta)
    NT = np.zeros((T.size, T.size))  # NT[z, s]: predicted correct-class p of scored s after adding copy of z
    for zi in range(T.size):
        h_sz = U[zi] @ xT.T
        h_zz = U[zi] @ xT[zi]
        eta = xT @ inf.theta - (pz[zi] - yT[zi]) * h_sz / (1.0 + pz[zi] * (1 - pz[zi]) * h_zz)
        NT[zi] = _sig((2 * yT - 1) * eta)
    return P, NT, it_max


def eegnet_task(task, X, y01, device, seeds):
    """EEGNet-8,2 (100 epochs, batch 64, Adam 1e-3) on every arm, all arms of a participant sharing init, row order and
    dropout masks within each seed (CRN), plus one A/A duplicate of the first arm. Returns (P mean over seeds, AA max)."""
    from models.deep_group import train_generic
    rows, extras = arm_rows(task)
    rows = np.vstack([rows, rows[:1]])
    T = task["T"]
    acc, aa = [], 0.0
    for s in seeds:
        out = train_generic(EEGNet250, X, y01, rows, np.zeros(rows.shape[0], dtype=np.int64), [int(s)], int(s) + 1,
                            device, score_idx=T, val_idx=None)  # this protocol reports only the final epoch
        p = out["p_final"]
        if not np.all(np.isfinite(p)):
            raise FloatingPointError("non-finite EEGNet output")
        aa = max(aa, float(np.max(np.abs(p[-1] - p[0]))))
        acc.append(p[:-1])
    return np.mean(acc, axis=0), aa
