"""R8-B: end-to-end EEGNet-8,2 benchmark inflation on R4's HS-Stroke arms (split S2, fold seeds 42-44).

R4 frozen manifest arms per (seed, participant, fold): T (scored distinct trials), LEAKED = C + E + X, PLACEBO = C + E + Z
(rows equal to LEAKED; Z = session/class-matched copies of non-test trials replacing X; LEAKED has k_f more DISTINCT trials,
the test twins), CLEAN = C, LCB = C minus k_f class-matched distinct trials (learning-curve bound).
EEGNet-8,2 exactly as R4/C2 (kern 64 at 128 Hz, 100 epochs, batch 64, Adam 1e-3, max-norm; per-trial channel z-score).
LEAKED and PLACEBO are trained as one CRN pair (same initialisation, same per-epoch permutation of row POSITIONS so that
X_i and Z_i occupy the same slot, same dropout masks); CLEAN and LCB are trained separately with the same initialisation
seed. Every arm is scored on T: correct-class probability and decision.
"""

from __future__ import annotations

import hashlib

import numpy as np

ARMS = ("T", "L", "C", "P", "LCB", "TEST_INST")


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def fold_arms(table, rows, split, seed, unit, fold):
    sel = table[(table[:, 0] == split) & (table[:, 1] == seed) & (table[:, 2] == unit) & (table[:, 3] == fold)]
    get = lambda arm: rows[sel[sel[:, 4] == ARMS.index(arm)][0][5]:sel[sel[:, 4] == ARMS.index(arm)][0][6]]  # noqa: E731
    return {a: np.asarray(get(a), dtype=np.int64) for a in ("T", "L", "P", "C", "LCB")}


def tasks(table, seeds, split=2):
    return sorted({(int(r[1]), int(r[2]), int(r[3])) for r in table if r[0] == split and int(r[1]) in seeds})


def commitment(table, rows, task_list) -> str:
    h = hashlib.sha256()
    for (s, u, f) in task_list:
        for a, v in fold_arms(table, rows, 2, s, u, f).items():
            h.update(a.encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def check_arms(arms, exposed):
    """Fail-closed invariants (distinct-trial indexing; copies share their trial's index):
    L and P have equal rows; C contains no scored trial; P contains no scored trial; the scored trials present in L are
    exactly the exposed scored trials; LCB is a sub-multiset of C with fewer distinct trials."""
    T, L, P, C, LCB = (arms[k] for k in ("T", "L", "P", "C", "LCB"))
    if L.size != P.size:
        raise AssertionError("LEAKED and PLACEBO rows differ")
    if np.intersect1d(C, T).size or np.intersect1d(P, T).size or np.intersect1d(LCB, T).size:
        raise AssertionError("scored trial in CLEAN / PLACEBO / LCB")
    if not np.array_equal(np.intersect1d(L, T), np.sort(T[exposed[T]])):
        raise AssertionError("scored trials in LEAKED differ from the exposed scored trials")
    if not (np.setdiff1d(LCB, C).size == 0 and np.unique(LCB).size < np.unique(C).size):
        raise AssertionError("LCB is not a reduced CLEAN")
    return {"k_extra_distinct_L_vs_P": int(np.unique(L).size - np.unique(P).size),
            "k_removed_LCB": int(np.unique(C).size - np.unique(LCB).size)}


def run_fold(arms, X, y01, device, seed_tuple, aa=False):
    """Returns correct-class probabilities and decisions on T for L, P, C, LCB (+ A/A max |dp| when aa)."""
    import torch

    from models import eegnet_group as eg
    s = _seed(8301, *seed_tuple)
    T = arms["T"]
    yT = y01[torch.as_tensor(T, device=device)]
    out = {}
    lp = np.stack([arms["L"], arms["P"]] + ([arms["L"]] if aa else []))
    m = eg.train_group(X, y01, lp, np.zeros(lp.shape[0], dtype=np.int64), [s], _seed(8302, *seed_tuple), device)
    pr = eg.predict_proba(m, X, T)
    if aa:
        out["aa_max_abs"] = np.array(float((pr[2] - pr[0]).abs().max()))
    for j, a in enumerate(("L", "P")):
        out[f"p_{a}"] = torch.gather(pr[j], 1, yT[:, None])[:, 0].double().cpu().numpy()
        out[f"c_{a}"] = (pr[j].argmax(-1) == yT).double().cpu().numpy()
    for a in ("C", "LCB"):
        m = eg.train_group(X, y01, arms[a][None], np.array([0]), [s], _seed(8302, *seed_tuple), device)
        pr = eg.predict_proba(m, X, T)
        out[f"p_{a}"] = torch.gather(pr[0], 1, yT[:, None])[:, 0].double().cpu().numpy()
        out[f"c_{a}"] = (pr[0].argmax(-1) == yT).double().cpu().numpy()
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {k}")
    return out
