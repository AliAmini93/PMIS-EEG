"""R8-D / R8-E2 plan: a fixed, metadata-only subset of the R7-C6 Lee2019 groups (sequential trainers are ~20x slower per
arm than the grouped EEGNet/Shallow trainers, so the full C6 design is not affordable).

From every C6 group (participant x held-out run; C6's 5 outcome-blind pairs in C6 order) keep ONE pair: the first whose
class is 1 + (run mod 2), else the first pair (so each participant contributes both classes whenever C6 sampled them).
Arms: A = train + a', B = train + b', C / E = train + extra copy of the placebo trials c / e, and an A/A duplicate of A.
Train / validation sets, pair seeds and group seeds are C6's, so every R8-D/E pair has EEGNet-8,2 and ShallowConvNet
records in C6 (same arms, same trials) and a TS+LR record in C3a.
"""

from __future__ import annotations

import hashlib

import numpy as np


def subset_plan(c6_plan: dict) -> dict:
    groups = []
    for g in c6_plan["groups"]:
        want = 1 + int(g["fold"]) % 2
        j = next((i for i, p in enumerate(g["pairs"]) if int(p["label"]) == want), 0)
        p = g["pairs"][j]
        arms = [{"pair": 0, "arm": a, "extra": int(p[x])} for a, x in (("A", "a"), ("B", "b"), ("C", "c"), ("E", "e"))]
        arms.append({"pair": 0, "arm": "AA", "extra": int(p["a"])})
        groups.append({**{k: g[k] for k in ("subject", "fold", "group", "group_seed", "train", "val", "T", "unexposed")},
                       "pairs": [p], "arms": arms, "c6_pair_index": j})
    return {"groups": groups, "n_pairs": len(groups), "n_arms": sum(len(g["arms"]) for g in groups)}


def commitment(plan: dict) -> str:
    h = hashlib.sha256()
    for g in plan["groups"]:
        h.update(repr((g["subject"], g["fold"], g["group_seed"], g["pairs"], g["arms"], g["train"], g["val"], g["T"])).encode())
    return h.hexdigest()


def shard(groups: list, k: int, n: int) -> list:
    return [g for i, g in enumerate(groups) if i % n == k]
