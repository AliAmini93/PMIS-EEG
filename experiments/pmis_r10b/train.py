"""R10-B model execution: EEGNet-8,2 (grouped CRN training, C2/C5 recipe) and TS+LR (exact optimum, C5 recipe).

EEGNet: vendored R4 EEGNet-8,2 via pmis_r6.eegnet_group.train_group (kern 64 at 128 Hz, Adam 1e-3, batch 64,
100 epochs, final epoch, no validation, no early stopping). Input: window-local 4-40 Hz 128-Hz windows with the C5
per-window channel z-score (a fixed per-window transform; nothing is learned from test data). Every arm of a group
shares the initialisation, the per-epoch row-position permutation and the dropout masks (CRN).
TS+LR: per arm, the R4 multiplicity-weighted Karcher reference of the arm's rows, tangent map at that reference,
damped-Newton L2 logistic regression (C = 1) - the C3/C5 exact-optimum estimator (pmis_r6.c3.opt_pc), refitted
from scratch for every arm. Scored rows: the 100 anchors (offset 0) of the held-out run; C0: every test window.
"""

from __future__ import annotations

import numpy as np

from models.vendor_r4 import r4_models as rm
from models.indep_tslr import logreg_newton

from experiments.pmis_r10b import design as ds
from experiments.pmis_r10b import geometry as geo


class Local:
    """Participant-local feature store: local id = vi * 400 + position of the trial among the participant's trials."""

    def __init__(self, trials: np.ndarray):
        self.trials = np.asarray(trials, dtype=np.int64)
        self.n = self.trials.size
        self.pos = {int(t): i for i, t in enumerate(self.trials)}

    def ids(self, global_rows) -> np.ndarray:
        g = np.asarray(global_rows, dtype=np.int64)
        v, t = g // ds.N_TRIALS, g % ds.N_TRIALS
        return v * self.n + np.array([self.pos[int(x)] for x in t], dtype=np.int64)


def gather_local(arrays: dict, trials: np.ndarray) -> np.ndarray:
    """arrays: source -> array (N, n_variants_in_source, ...). Returns (11 * n_trials, ...) in OFFSETS order."""
    parts = []
    for o in geo.OFFSETS:
        src, j = geo.VARIANTS[o]
        parts.append(np.asarray(arrays[src][trials, j]))
    return np.concatenate(parts)


# ------------------------------------------------------------------------------------------------ EEGNet
def eegnet_groups(f: dict) -> list[tuple[str, tuple[str, ...], dict]]:
    """[(group tag, arm names, {arm: global rows})] in the frozen execution order."""
    out = [("B", ds.TRACK_B_ORDER, ds.track_b_arms(f))]
    for s in geo.STRIDES:
        ta = ds.track_a_arms(f, s)
        for gi, g in enumerate(ds.track_a_groups(s)):
            out.append((f"A{s}g{gi}", g, {a: ta[a] for a in g}))
    return out


def run_fold_eegnet(f: dict, loc: Local, X, y01, device) -> dict:
    import torch

    from models import eegnet_group as eg
    anchors = loc.ids(ds.row(0, f["T"]))
    yT = y01[torch.as_tensor(anchors, device=device)]
    out = {}
    for tag, names, arms in eegnet_groups(f):
        rows = np.stack([loc.ids(arms[a]) for a in names])
        pair_of = np.zeros(len(names), dtype=np.int64)
        m = eg.train_group(X, y01, rows, pair_of, [f["pair_seed"]], f["group_seed"], device)
        pr = eg.predict_proba(m, X, anchors)  # (M, 100, 2)
        p = torch.gather(pr, 2, yT[None, :, None].expand(pr.shape[0], -1, 1))[..., 0].double().cpu().numpy()
        c = (pr.argmax(-1) == yT[None]).double().cpu().numpy()
        for j, a in enumerate(names):
            key = a if tag == "B" else f"A{tag[1:tag.index('g')]}_{a}"
            out[f"p_{key}"], out[f"c_{key}"] = p[j], c[j]
        if "AA" in names:
            out[f"aa_{tag}"] = np.array(float((pr[names.index("AA")] - pr[names.index("B1_0" if tag == "B" else "A1")]).abs().max()))
        del m, pr
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {k}")
    return out


def run_c0_eegnet(sp: dict, loc: Local, X, y01, device) -> dict:
    import torch

    from models import eegnet_group as eg
    tr = ds.c0_train_rows(sp)
    rows = np.stack([loc.ids(r) for _, _, r in tr])
    m = eg.train_group(X, y01, rows, np.zeros(len(tr), dtype=np.int64), [sp["pair_seed"]], sp["group_seed"], device)
    out = {}
    for j, (scheme, k, _) in enumerate(tr):
        te = sp[scheme][k]
        ids = loc.ids(te)
        pr = eg.predict_proba(m, X, ids)[j]
        yt = y01[torch.as_tensor(ids, device=device)]
        out[f"{scheme}{k}_rows"] = te
        out[f"{scheme}{k}_p"] = torch.gather(pr, 1, yt[:, None])[:, 0].double().cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------------ TS+LR
def tslr_fit_predict(covs: np.ndarray, y: np.ndarray, train_ids: np.ndarray, test_ids: np.ndarray) -> np.ndarray:
    """Exact-optimum TS+LR (as pmis_r6.c3.opt_pc with unit multiplicities); correct-class probability on test_ids."""
    pos = 2
    cnt = np.ones(train_ids.size, dtype=np.float64)
    ref = rm.riemann_mean_w(covs[train_ids], cnt)
    th, _ = logreg_newton(rm.tangent_space_b(covs[train_ids], ref), (y[train_ids] == pos).astype(float), cnt)
    f = rm.tangent_space_b(covs[test_ids], ref)
    pp = 1.0 / (1.0 + np.exp(-(f @ th[:-1] + th[-1])))
    return np.where(y[test_ids] == pos, pp, 1.0 - pp)


def run_fold_tslr(f: dict, loc: Local, covs: np.ndarray, y: np.ndarray) -> dict:
    anchors = loc.ids(ds.row(0, f["T"]))
    out = {}
    for tag, names, arms in eegnet_groups(f):
        for a in names:
            if a == "AA":
                continue  # TS+LR is deterministic: the A/A check recomputes B1_0 / A1 below
            key = a if tag == "B" else f"A{tag[1:tag.index('g')]}_{a}"
            p = tslr_fit_predict(covs, y, loc.ids(arms[a]), anchors)
            out[f"p_{key}"], out[f"c_{key}"] = p, (p > 0.5).astype(np.float64)
    tb = ds.track_b_arms(f)
    out["aa_B"] = np.array(float(np.max(np.abs(tslr_fit_predict(covs, y, loc.ids(tb["AA"]), anchors) - out["p_B1_0"]))))
    return out


def run_c0_tslr(sp: dict, loc: Local, covs: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for scheme, k, trn in ds.c0_train_rows(sp):
        te = sp[scheme][k]
        out[f"{scheme}{k}_rows"] = te
        out[f"{scheme}{k}_p"] = tslr_fit_predict(covs, y, loc.ids(trn), loc.ids(te))
    return out
