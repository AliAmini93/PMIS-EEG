"""R12-X execution units (frozen recipe; used by scripts/r12x_run.py and by the synthetic tests).

Deep unit (participant, stage, seed k): one CRN group through pmis_r7.deep_group.train_generic (Adam 1e-3, batch 64,
100 epochs; per-arm shared initialisation, row order and dropout masks), arms = selection (S_P, S_L, AA_L; scored
on B; ES on A_val) or final (F, AA_F; scored on C; ES on pool_val). Read-outs: final epoch and min-validation-loss
epoch. A/A arms must reproduce their reference bit-for-bit, or the run stops.
TS+LR unit (participant, stage): exact optimum (pmis_r10b.train.tslr_fit_predict), A/A = recomputation.
Amendment 1: deep units run inside one machine-wide GPU lease (the external GPU scheduler) with availability waits and usage
metering (pmis_r12x.guards); there is no time limit.
"""

from __future__ import annotations

import numpy as np

from experiments.pmis_r12x import design as ds


class AAStop(RuntimeError):
    pass


def unit_rows(d: dict, stage: str):
    if stage == "selection":
        return [d["arms"][a] for a in ds.SEL_ARMS], d["B"], d["A_val"], ("S_L", "AA_L")
    return [d["arms"][a] for a in ds.FIN_ARMS], d["C"], d["pool_val"], ("F", "AA_F")


def run_deep_unit(cls, d: dict, stage: str, k: int, X, y01, device, epochs: int = 100) -> dict:
    from models.deep_group import train_generic
    rows, score, val, (ref, dup) = unit_rows(d, stage)
    names = ds.SEL_ARMS if stage == "selection" else ds.FIN_ARMS
    s_init, s_drop = ds.seeds(d["participant"], stage, k)
    R = np.stack(rows).astype(np.int64)
    res = train_generic(cls, X, y01, R, np.zeros(len(rows), dtype=np.int64), [s_init], s_drop, device,
                        score_idx=score, val_idx=val, epochs=epochs)
    out = {}
    for j, a in enumerate(names):
        out[f"final_p_{a}"] = np.asarray(res["p_final"][j])
        out[f"es_p_{a}"] = np.asarray(res["p_es"][j])
        out[f"es_epoch_{a}"] = np.array(int(res["es_epoch"][j]))
    i_ref, i_dup = names.index(ref), names.index(dup)
    aa = max(float(np.max(np.abs(out[f"final_p_{dup}"] - out[f"final_p_{ref}"]))),
             float(np.max(np.abs(out[f"es_p_{dup}"] - out[f"es_p_{ref}"]))),
             abs(int(res["es_epoch"][i_dup]) - int(res["es_epoch"][i_ref])))
    if aa != 0.0:
        raise AAStop(f"A/A not exact ({aa:.3e}) participant {d['participant']} {stage} seed {k}")
    for kk, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {kk}")
    return out


def run_tslr_unit(d: dict, stage: str, covs: np.ndarray, y12: np.ndarray) -> dict:
    from experiments.pmis_r10b.train import tslr_fit_predict
    rows, score, _val, (ref, dup) = unit_rows(d, stage)
    names = ds.SEL_ARMS if stage == "selection" else ds.FIN_ARMS
    out = {f"exact_p_{a}": tslr_fit_predict(covs, y12, r, score) for a, r in zip(names, rows)}
    if not np.array_equal(out[f"exact_p_{ref}"], out[f"exact_p_{dup}"]):
        raise AAStop(f"TS+LR A/A not exact participant {d['participant']} {stage}")
    return out
