"""R10-A training parts on R4's HS-Stroke S2 arms (fold seeds 42-44), final-epoch read-out (the R4 / R8-B / R9-C recipe).

Arms per (seed, participant, fold): R4 frozen LEAKED (L), PLACEBO (P; rows = L), CLEAN (C), and R9-C's split-half arms
L1 / L2 (identical stratified outcome-blind halves, default_rng([9201, seed, participant, fold])).

Parts (each over all 210 folds):
  SH1  ShallowConvNet (R7-C6 GroupShallow), primary training seeds: pair seed _seed(8301, s, u, f), group seed
       _seed(8302, s, u, f) (the derivation R8-B / R9-C used for EEGNet)
  SH2  ShallowConvNet, training-seed replicate: _seed(10301, s, u, f), _seed(10302, s, u, f)
  EG2  EEGNet-8,2 (R6 GroupEEGNet, exactly R8-B / R9-C), training-seed replicate: _seed(10301, ...), _seed(10302, ...)
Per fold and part: one CRN group (L, L1, L2, P, L') and one CRN group (C, C'), L' and C' being A/A duplicates in EVERY
fold (must be bit-identical). Within a group all arms share initialisation, row-position permutation and dropout masks.
The group size is fixed (5 and 2) in every fold, because ShallowConvNet's grouped kernels are not bit-invariant to the
group size (synthetic pilot: |dp| ~ 1e-4), while A/A within a group is exact.
"""

from __future__ import annotations

import hashlib

import numpy as np

from experiments.pmis_r8 import bench
from experiments.pmis_r9 import split_bench

PARTS = {"SH1": {"model": "Shallow", "pair_salt": 8301, "group_salt": 8302, "role": "primary training seeds"},
         "SH2": {"model": "Shallow", "pair_salt": 10301, "group_salt": 10302, "role": "training-seed replicate"},
         "EG2": {"model": "EEGNet", "pair_salt": 10301, "group_salt": 10302, "role": "training-seed replicate"}}
ARMS = ("L", "L1", "L2", "P", "C")


def fold_inputs(table, rows, dist, exposed, s, u, f):
    arms = bench.fold_arms(table, rows, 2, s, u, f)
    h1, h2 = split_bench.split_halves(arms, dist, (s, u, f))
    L1, L2 = split_bench.make_split_arms(arms, h1, h2)
    split_bench.check_split(arms, L1, L2, h1, h2, exposed)
    return arms, L1, L2, h1, h2


def seeds_of(part, seed_tuple):
    c = PARTS[part]
    return bench._seed(c["pair_salt"], *seed_tuple), bench._seed(c["group_salt"], *seed_tuple)


def commitment(table, rows, dist, man, task_list, parts=tuple(PARTS)) -> str:
    h = hashlib.sha256()
    h.update(repr(sorted((p, PARTS[p]["model"], PARTS[p]["pair_salt"], PARTS[p]["group_salt"]) for p in parts)).encode())
    for (s, u, f) in task_list:
        arms, L1, L2, h1, h2 = fold_inputs(table, rows, dist, man[f"exposed_S2_{s}"], s, u, f)
        for name, v in (("T", arms["T"]), ("L", arms["L"]), ("P", arms["P"]), ("C", arms["C"]), ("L1", L1), ("L2", L2),
                        ("H1", h1), ("H2", h2)):
            h.update(name.encode())
            h.update(np.ascontiguousarray(v).tobytes())
        for p in parts:
            h.update(repr((p, *seeds_of(p, (s, u, f)))).encode())
    return h.hexdigest()


def run_fold(part, arms, L1, L2, X, y01, device, seed_tuple):
    """Returns p_/c_ (correct-class probability, 0/1 decision) on T for L, L1, L2, P, C and the two A/A max |dp|."""
    import torch

    from models import eegnet_group as eg
    from models.deep_group import GroupShallow

    from models import hsstroke_deep as deep
    model = PARTS[part]["model"]
    T = arms["T"]
    out = {}
    ps, gs = seeds_of(part, seed_tuple)  # CLEAN uses the same initialisation and group seed as the main group (R8-B)
    for key, grp in (("main", np.stack([arms["L"], L1, L2, arms["P"], arms["L"]])), ("clean", np.stack([arms["C"], arms["C"]]))):
        pair_of = np.zeros(grp.shape[0], dtype=np.int64)
        if model == "Shallow":
            m = deep.train_group_final(GroupShallow, X, y01, grp, pair_of, [ps], gs, device)
            p, c, pr = deep.correct_class(m, X, y01, T)
        else:
            m = eg.train_group(X, y01, grp, pair_of, [ps], gs, device)
            pr = eg.predict_proba(m, X, T)
            yT = y01[torch.as_tensor(T, device=device)]
            p = torch.gather(pr, 2, yT[None, :, None].expand(pr.shape[0], -1, 1))[..., 0].double().cpu().numpy()
            c = (pr.argmax(-1) == yT[None]).double().cpu().numpy()
        aa = float((pr[-1] - pr[0]).abs().max())
        names = ("L", "L1", "L2", "P") if key == "main" else ("C",)
        for j, a in enumerate(names):
            out[f"p_{a}"], out[f"c_{a}"] = p[j], c[j]
        out[f"aa_{key}_max_abs"] = np.array(aa)
        del m, pr
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {k}")
    return out
