"""R6-C2: PMIS identity swap for EEGNet-8,2 on HS-Stroke (GPU), with neighbour, placebo and A/A controls.

Plan (metadata only): the frozen G3 plan (commitment-checked), seed-42 tasks; in each (participant, fold) task an
outcome-blind simple random third of the pairs (rng [7301, 42, participant, fold]). Every sampled pair gets arms
A = C + a', B = C + b' and, where G3 defined them, G = C + g' (C1 neighbour) and C/E = C + c'/e' (C2 placebo).
All arms of a pair share initialisation, row order and dropout masks (common random numbers). The first group
of every task carries one A/A duplicate arm, which must reproduce its twin exactly (determinism check).
Each arm is a full R4 EEGNet-8,2 fit (100 epochs, batch 64, Adam 1e-3) on base C plus one extra row,
evaluated on the fold's whole test set T and on its own training rows.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

SAMPLE_STREAM, PAIR_STREAM, GROUP_STREAM = 7301, 7302, 7303
SAMPLE_FRACTION_DEN = 3
M_MAX = 96


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def sample_pairs(task: dict) -> list[int]:
    n = len(task["pairs"])
    m = math.ceil(n / SAMPLE_FRACTION_DEN)
    rng = np.random.default_rng([SAMPLE_STREAM, int(task["seed"]), int(task["subject"]), int(task["fold"])])
    return sorted(int(i) for i in rng.choice(n, m, replace=False))


def build_plan(g3_tasks: list[dict]) -> dict:
    groups = []
    n_pairs = n_arms = 0
    for task in g3_tasks:
        subj, fold = int(task["subject"]), int(task["fold"])
        base = np.asarray(task["base_canon"], dtype=np.int64)
        units = []  # (pair entry, list of (arm_name, extra_row))
        for j in sample_pairs(task):
            pr = task["pairs"][j]
            arms = [("A", pr["a"]), ("B", pr["b"])]
            if pr["g"] >= 0:
                arms.append(("G", pr["g"]))
            if pr["c"] >= 0:
                arms += [("C", pr["c"]), ("E", pr["e"])]
            units.append((pr, arms))
        cur, gi = [], 0

        def flush(cur, gi, aa):
            arms, pair_meta = [], []
            for p_idx, (pr, a_list) in enumerate(cur):
                pair_meta.append({"k": int(pr["k"]), "seed": _seed(PAIR_STREAM, 42, subj, fold, pr["k"]),
                                  "a": int(pr["a"]), "b": int(pr["b"]), "label": int(pr["label"]),
                                  "session": int(pr["session"]), "pos_dist": int(pr["pos_dist"]),
                                  "g": int(pr["g"]), "c": int(pr["c"]), "e": int(pr["e"]), "sign": int(pr["sign"])})
                for name, extra in a_list:
                    arms.append({"pair": p_idx, "arm": name, "extra": int(extra)})
            if aa:
                arms.append({"pair": 0, "arm": "AA", "extra": int(cur[0][0]["a"])})
            groups.append({"subject": subj, "fold": fold, "group": gi, "group_seed": _seed(GROUP_STREAM, 42, subj, fold, gi),
                           "pairs": pair_meta, "arms": arms, "n_base": int(base.size)})

        for u in units:
            size = sum(len(x[1]) for x in cur) + (1 if gi == 0 else 0)  # group 0 also carries the A/A arm
            if cur and size + len(u[1]) > M_MAX:
                flush(cur, gi, gi == 0)
                gi, cur = gi + 1, []
            cur.append(u)
        if cur:
            flush(cur, gi, gi == 0)
        n_pairs += len(units)
    for g in groups:
        n_arms += len(g["arms"])
    return {"groups": groups, "n_pairs": n_pairs, "n_arms": n_arms}


def plan_commitment(plan: dict, g3_tasks: list[dict]) -> str:
    h = hashlib.sha256()
    tmap = {(int(t["subject"]), int(t["fold"])): t for t in g3_tasks}
    for g in plan["groups"]:
        t = tmap[(g["subject"], g["fold"])]
        h.update(np.ascontiguousarray(t["base_canon"]).tobytes())
        h.update(np.ascontiguousarray(t["T"]).tobytes())
        h.update(repr((g["subject"], g["fold"], g["group"], g["group_seed"], g["pairs"], g["arms"])).encode())
    return h.hexdigest()


def arm_rows(base: np.ndarray, arms: list[dict]) -> np.ndarray:
    return np.stack([np.concatenate([base, [a["extra"]]]) for a in arms]).astype(np.int64)


# ------------------------------------------------------------------------------------------------ execute
def run_group(group: dict, task: dict, X, y01, device, epochs: int = 100) -> dict:
    """Train every arm of one group; return correct-class probabilities on T and training-fit summaries."""
    import torch

    from models import eegnet_group as eg
    base = np.asarray(task["base_canon"], dtype=np.int64)
    rows = arm_rows(base, group["arms"])
    pair_of = np.array([a["pair"] for a in group["arms"]], dtype=np.int64)
    seeds = [p["seed"] for p in group["pairs"]]
    model = eg.train_group(X, y01, rows, pair_of, seeds, group["group_seed"], device, epochs=epochs)
    T = np.asarray(task["T"], dtype=np.int64)
    yT = y01[torch.as_tensor(T, device=device)]
    pT = eg.predict_proba(model, X, T)  # (M, |T|, 2)
    pc = torch.gather(pT, 2, yT[None, :, None].expand(pT.shape[0], -1, 1))[..., 0]
    own = eg.predict_own(model, X, rows)  # (M, n+1, 2)
    yo = y01[torch.as_tensor(rows, device=device)]
    po = torch.gather(own, 2, yo[..., None])[..., 0]
    acc_own = (own.argmax(-1) == yo).float()
    out = {"p_T": pc.double().cpu().numpy(), "pred_T": pT.argmax(-1).cpu().numpy().astype(np.int8),
           "train_p_base_mean": po[:, :-1].mean(1).double().cpu().numpy(),
           "train_acc_base": acc_own[:, :-1].mean(1).double().cpu().numpy(),
           "train_p_extra": po[:, -1].double().cpu().numpy()}
    if not np.all(np.isfinite(out["p_T"])):
        raise FloatingPointError("non-finite EEGNet output")
    return out


# ------------------------------------------------------------------------------------------------ analyse
def pair_records(plan: dict, tasks: dict, results: dict, y: np.ndarray) -> list[dict]:
    """results[(subject, fold, group)] -> run_group output. Returns one record per sampled pair."""
    recs = []
    aa = []
    for g in plan["groups"]:
        key = (g["subject"], g["fold"], g["group"])
        res = results[key]
        task = tasks[(g["subject"], g["fold"])]
        T = np.asarray(task["T"], dtype=np.int64)
        pos = {int(t): i for i, t in enumerate(T.tolist())}
        um = np.asarray(task["unexposed"], dtype=bool)
        by = {}
        for m, a in enumerate(g["arms"]):
            by[(a["pair"], a["arm"])] = m
        if (0, "AA") in by:
            aa.append(float(np.max(np.abs(res["p_T"][by[(0, "AA")]] - res["p_T"][by[(0, "A")]]))))
        for p_idx, pr in enumerate(g["pairs"]):
            ia, ib = pos[pr["a"]], pos[pr["b"]]
            mA, mB = by[(p_idx, "A")], by[(p_idx, "B")]
            pA, pB = res["p_T"][mA], res["p_T"][mB]
            corr = lambda m, i: float(res["pred_T"][m, i] == y[T[i]] - 1)  # noqa: E731
            sc = um & (np.asarray(task["T_session"]) == pr["session"]) & (np.asarray(task["T_label"]) == pr["label"])
            r = {"subject": g["subject"], "fold": g["fold"], "k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"],
                 "pAA": pA[ia], "pAB": pB[ia], "pBA": pA[ib], "pBB": pB[ib],
                 "cAA": corr(mA, ia), "cAB": corr(mB, ia), "cBA": corr(mA, ib), "cBB": corr(mB, ib),
                 "off_fold": float(np.mean(np.abs(pA[um] - pB[um]))) if um.any() else np.nan,
                 "off_session_class": float(np.mean(np.abs(pA[sc] - pB[sc]))) if sc.any() else np.nan,
                 "train_p_base_A": res["train_p_base_mean"][mA], "train_acc_base_A": res["train_acc_base"][mA],
                 "train_p_extra_A": res["train_p_extra"][mA], "train_p_extra_B": res["train_p_extra"][mB]}
            if (p_idx, "G") in by:
                pG = res["p_T"][by[(p_idx, "G")]]
                r.update({"pAG": pG[ia], "pBG": pG[ib]})
            if (p_idx, "C") in by:
                pC, pE = res["p_T"][by[(p_idx, "C")]], res["p_T"][by[(p_idx, "E")]]
                r.update({"pAC": pC[ia], "pBC": pC[ib], "pAE": pE[ia], "pBE": pE[ib], "sign": pr["sign"]})
            recs.append(r)
    return recs, aa
