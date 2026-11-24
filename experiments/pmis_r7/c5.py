"""R7-C5: fine temporal-shift sensitivity of the exact-copy advantage (Lee2019, 2-s windows; TS+LR and EEGNet).

Copies of the scored trial's window (base and scored rows: V0.0 = [0, 2) s after the cue), shifted by
d = 0 (identical), 62.5, 125, 250, 500, 1000 ms (sample overlap 100, 96.9, 93.75, 87.5, 75, 50 %); window-local
filtering throughout. Sources: V0.0 / V0.5 / V1.0 from the frozen C4 build (data/r6_c4), the three fine shifts from the
C5 build (data/r7_c5). Stacked index of variant j of trial i = j * N + i.
Pairs: the C3 pairs; per task an outcome-blind 5 of 40 (rng [7701, participant, run]); per pair and variant arms
C + v(a) / C + v(b). TS+LR: exact-optimum per-arm refit (c3.opt_pc). EEGNet: C2 mechanics (unmodified
pmis_r6.c2.run_group), one A/A per task, <= 48 arms per GPU group.
"""

from __future__ import annotations

import hashlib
import os
import resource
import time

import numpy as np

from models.tslr import arm_multiset

from experiments.pmis_r6 import c3

VARIANTS = (("D0", "c4", 0), ("D62.5", "c5", 0), ("D125", "c5", 1), ("D250", "c5", 2), ("D500", "c4", 1), ("D1000", "c4", 2))
SHIFT_MS = {"D0": 0.0, "D62.5": 62.5, "D125": 125.0, "D250": 250.0, "D500": 500.0, "D1000": 1000.0}
N_TRIALS = 21600
SAMPLE_STREAM, PAIR_STREAM, GROUP_STREAM = 7701, 7702, 7703
N_PAIRS, M_MAX = 5, 48
G = c3.G


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def stack(c4_arr, c5_arr) -> np.ndarray:
    src = {"c4": c4_arr, "c5": c5_arr}
    return np.concatenate([np.asarray(src[s][:, i]) for _, s, i in VARIANTS])


def sample(task) -> list[int]:
    rng = np.random.default_rng([SAMPLE_STREAM, int(task["subject"]), int(task["fold"])])
    return sorted(int(i) for i in rng.choice(len(task["pairs"]), N_PAIRS, replace=False))


def commitment(tasks, plan=None) -> str:
    h = hashlib.sha256()
    for t in tasks:
        h.update(repr((t["subject"], t["fold"], sample(t), [t["pairs"][j] for j in sample(t)])).encode())
        h.update(np.ascontiguousarray(t["T"]).tobytes())
        h.update(np.ascontiguousarray(t["base_canon"]).tobytes())
    if plan is not None:
        for g in plan["groups"]:
            h.update(repr((g["subject"], g["fold"], g["group"], g["group_seed"], g["pairs"], g["arms"])).encode())
    return h.hexdigest()


def run_task_tslr(task: dict) -> dict:
    t0 = time.perf_counter()
    base = task["base_canon"]
    ones = np.ones(base.size, dtype=np.int64)
    recs, n, it_max = [], 0, 0
    for j in sample(task):
        pr = task["pairs"][j]
        a, b = pr["a"], pr["b"]
        ab = np.array([a, b])
        r = {"k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"]}
        for vi, (v, _, _) in enumerate(VARIANTS):
            pA, i1 = c3.opt_pc(*arm_multiset(base, ones, vi * N_TRIALS + a), ab)
            pB, i2 = c3.opt_pc(*arm_multiset(base, ones, vi * N_TRIALS + b), ab)
            r.update({f"{v}_pAA": pA[0], f"{v}_pBA": pA[1], f"{v}_pAB": pB[0], f"{v}_pBB": pB[1]})
            n, it_max = n + 2, max(it_max, i1, i2)
        recs.append(r)
    return {"subject": task["subject"], "fold": task["fold"], "records": recs, "fits": n, "newton_iter_max": it_max,
            "seconds": time.perf_counter() - t0, "pid": os.getpid(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def build_eegnet_plan(tasks: list[dict]) -> dict:
    groups = []
    for task in tasks:
        subj, fold = int(task["subject"]), int(task["fold"])
        units = []
        for j in sample(task):
            pr = task["pairs"][j]
            arms = []
            for vi, (v, _, _) in enumerate(VARIANTS):
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
            if cur and size + len(u[1]) > M_MAX:
                flush(cur, gi)
                gi, cur = gi + 1, []
            cur.append(u)
        if cur:
            flush(cur, gi)
    return {"groups": groups, "n_pairs": sum(len(g["pairs"]) for g in groups), "n_arms": sum(len(g["arms"]) for g in groups)}


def eegnet_pair_records(plan, tasks, results):
    recs, aa = [], []
    for g in plan["groups"]:
        res = results[(g["subject"], g["fold"], g["group"])]
        T = np.asarray(tasks[(g["subject"], g["fold"])]["T"], dtype=np.int64)
        pos = {int(t): i for i, t in enumerate(T.tolist())}
        by = {(a["pair"], a["arm"]): m for m, a in enumerate(g["arms"])}
        if (0, "AA") in by:
            aa.append(float(np.max(np.abs(res["p_T"][by[(0, "AA")]] - res["p_T"][by[(0, "D0:A")]]))))
        for p_idx, pr in enumerate(g["pairs"]):
            ia, ib = pos[pr["a"]], pos[pr["b"]]
            r = {"subject": g["subject"], "fold": g["fold"], "k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"]}
            for v, _, _ in VARIANTS:
                pA, pB = res["p_T"][by[(p_idx, f"{v}:A")]], res["p_T"][by[(p_idx, f"{v}:B")]]
                r.update({f"{v}_pAA": pA[ia], f"{v}_pBA": pA[ib], f"{v}_pAB": pB[ia], f"{v}_pBB": pB[ib]})
            recs.append(r)
    return recs, aa
