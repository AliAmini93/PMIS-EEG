"""R11-B model execution on the R7-C6 Lee2019 input and recipes (no new model, no tuning).

Deep (EEGNet-8,2 = pmis_r7.deep_group.EEGNetG; ShallowConvNet = pmis_r7.deep_group.GroupShallow): the R7-C6 recipe
through pmis_r7.deep_group.train_generic: Adam 1e-3, batch 64, 100 epochs; read-outs from the same run: the final
epoch and the epoch of minimum cross-entropy on C6's 60 validation trials (early stopping by selection). Input: C3's
20-channel 4-40 Hz 128-Hz 4-s MI windows with the per-trial channel z-score. Two CRN groups per fold (all arms of a
group share initialisation, per-epoch row-position permutation and dropout masks; both groups share the
initialisation seed): group A = P, L20, L40, L40c, L60, L100, AA (340 rows each), group B = C, AAC (240 rows).
TS+LR: the exact-optimum estimator of R6-C3a / R10-B (multiplicity-weighted Karcher reference, tangent space,
damped-Newton L2 logistic regression C = 1) on C3's 8-30 Hz OAS covariances; deterministic; A/A = recomputation.
Scored: the fold's 100 held-out-run trials in every arm (correct-class probability)."""

from __future__ import annotations

import numpy as np

from experiments.pmis_r11b import design as ds

TSLR_ARMS = ("C", "P", "L20", "L40", "L40c", "L60", "L100")


def run_fold_deep(cls, f: dict, X, y01, rep: str, device) -> dict:
    from models.deep_group import train_generic
    s_init, s_drop = ds.seeds(f, rep)
    out = {}
    for names in (ds.GROUP_A, ds.GROUP_B):
        rows = np.stack([f["arms"][a] for a in names]).astype(np.int64)
        res = train_generic(cls, X, y01, rows, np.zeros(len(names), dtype=np.int64), [s_init], s_drop, device,
                            score_idx=f["T"], val_idx=f["val"])
        for j, a in enumerate(names):
            out[f"final_p_{a}"] = res["p_final"][j]
            out[f"es_p_{a}"] = res["p_es"][j]
            out[f"es_epoch_{a}"] = np.array(int(res["es_epoch"][j]))
        dup, ref = (names.index("AA"), names.index("L100")) if "AA" in names else (names.index("AAC"), names.index("C"))
        tag = "A" if "AA" in names else "C"
        out[f"aa_{tag}"] = np.array(float(max(np.max(np.abs(res["p_final"][dup] - res["p_final"][ref])),
                                              np.max(np.abs(res["p_es"][dup] - res["p_es"][ref])),
                                              abs(int(res["es_epoch"][dup]) - int(res["es_epoch"][ref])))))
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {k}")
    return out


def run_fold_tslr(f: dict, covs: np.ndarray, y: np.ndarray) -> dict:
    from experiments.pmis_r10b.train import tslr_fit_predict
    out = {}
    for a in TSLR_ARMS:
        out[f"exact_p_{a}"] = tslr_fit_predict(covs, y, f["arms"][a], f["T"])
    out["aa_T"] = np.array(float(np.max(np.abs(tslr_fit_predict(covs, y, f["arms"]["AA"], f["T"]) - out["exact_p_L100"]))))
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {k}")
    return out
