"""R6-C4b: overlap / same-episode / pre-cue decomposition of the exact-copy advantage for EEGNet-8,2 (Lee2019).

Input: window-local 2-s EEGNet windows (scripts/r6_c3_build_lee2019.py; 20 channels x 256 at 128 Hz). Four of the six
C4 variants are used (cost bound): V0.0 (100 % overlap), V1.0 (50 %), V2.0 (0 %, same MI episode), P (pre-cue).
Stacked index of variant j (in VARIANTS order) of trial i = j * N + i. Base and scored rows use V0.0.
Pairs: the C3 pairs; per task an outcome-blind random quarter (rng [7421, participant, run]); per pair and variant
arms C + v(a) / C + v(b) (8 arms per pair), one A/A per task, <= 48 arms per GPU group; CRN within pair (C2 mechanics,
unmodified pmis_r6.c2.run_group / eegnet_group).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

VARIANTS = ("V0.0", "V1.0", "V2.0", "P")
C4_INDEX = {"V0.0": 0, "V0.5": 1, "V1.0": 2, "V1.5": 3, "V2.0": 4, "P": 5}  # position in the C4 build array
N_TRIALS = 21600
SAMPLE_STREAM, PAIR_STREAM, GROUP_STREAM = 7421, 7422, 7423
M_MAX = 48


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def stack(win: np.ndarray) -> np.ndarray:
    """win: (N, 6, C, T) from the C4 build -> (len(VARIANTS) * N, C, T), variant-major."""
    assert win.shape[0] == N_TRIALS and win.shape[1] == 6
    return np.concatenate([np.asarray(win[:, C4_INDEX[v]]) for v in VARIANTS])


def build_plan(tasks: list[dict], fraction_den: int = 4, m_max: int = M_MAX) -> dict:
    groups, n_pairs = [], 0
    for task in tasks:
        subj, fold = int(task["subject"]), int(task["fold"])
        n = len(task["pairs"])
        rng = np.random.default_rng([SAMPLE_STREAM, subj, fold])
        sel = sorted(int(i) for i in rng.choice(n, math.ceil(n / fraction_den), replace=False))
        units = []
        for j in sel:
            pr = task["pairs"][j]
            arms = []
            for vi, v in enumerate(VARIANTS):
                arms += [(f"{v}:A", vi * N_TRIALS + pr["a"]), (f"{v}:B", vi * N_TRIALS + pr["b"])]
            units.append((pr, arms))
        gi, cur = 0, []

        def flush(cur, gi):
            arms, meta = [], []
            for p_idx, (pr, a_list) in enumerate(cur):
                meta.append({"k": int(pr["k"]), "seed": _seed(PAIR_STREAM, subj, fold, pr["k"]), "a": int(pr["a"]),
                             "b": int(pr["b"]), "label": int(pr["label"]), "session": int(pr["session"]),
                             "pos_dist": int(pr["pos_dist"])})
                arms += [{"pair": p_idx, "arm": nm, "extra": int(x)} for nm, x in a_list]
            if gi == 0:
                arms.append({"pair": 0, "arm": "AA", "extra": int(cur[0][0]["a"])})
            groups.append({"subject": subj, "fold": fold, "group": gi, "group_seed": _seed(GROUP_STREAM, subj, fold, gi),
                           "pairs": meta, "arms": arms, "n_base": int(task["base_canon"].size)})

        for u in units:
            size = sum(len(x[1]) for x in cur) + (1 if gi == 0 else 0)
            if cur and size + len(u[1]) > m_max:
                flush(cur, gi)
                gi, cur = gi + 1, []
            cur.append(u)
        if cur:
            flush(cur, gi)
        n_pairs += len(units)
    return {"groups": groups, "n_pairs": n_pairs, "n_arms": sum(len(g["arms"]) for g in groups)}


def commitment(plan: dict, tasks: list[dict]) -> str:
    h = hashlib.sha256()
    tmap = {(int(t["subject"]), int(t["fold"])): t for t in tasks}
    for g in plan["groups"]:
        t = tmap[(g["subject"], g["fold"])]
        h.update(np.ascontiguousarray(t["base_canon"]).tobytes())
        h.update(np.ascontiguousarray(t["T"]).tobytes())
        h.update(repr((g["subject"], g["fold"], g["group"], g["group_seed"], g["pairs"], g["arms"])).encode())
    return h.hexdigest()


def pair_records(plan: dict, tasks: dict, results: dict) -> tuple[list[dict], list[float]]:
    recs, aa = [], []
    for g in plan["groups"]:
        res = results[(g["subject"], g["fold"], g["group"])]
        T = np.asarray(tasks[(g["subject"], g["fold"])]["T"], dtype=np.int64)
        pos = {int(t): i for i, t in enumerate(T.tolist())}
        by = {(a["pair"], a["arm"]): m for m, a in enumerate(g["arms"])}
        if (0, "AA") in by:
            aa.append(float(np.max(np.abs(res["p_T"][by[(0, "AA")]] - res["p_T"][by[(0, "V0.0:A")]]))))
        for p_idx, pr in enumerate(g["pairs"]):
            ia, ib = pos[pr["a"]], pos[pr["b"]]
            r = {"subject": g["subject"], "fold": g["fold"], "k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"]}
            for v in VARIANTS:
                pA, pB = res["p_T"][by[(p_idx, f"{v}:A")]], res["p_T"][by[(p_idx, f"{v}:B")]]
                r.update({f"{v}_pAA": pA[ia], f"{v}_pBA": pA[ib], f"{v}_pAB": pB[ia], f"{v}_pBB": pB[ib]})
            recs.append(r)
    return recs, aa
