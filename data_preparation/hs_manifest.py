#!/usr/bin/env python3
"""R4-G2c: replay the producers' 5-fold splits on the HISTORICAL v1 enumeration, enumerate the observed
cross-boundary physical-trial twins, and build the frozen three-arm (+ learning-curve control) manifest.

Splits (both producer-derived; seeds 42..46, 42 = producer seed):
  S1 "CrossTrialSplit" (original producer procedure): per (subject, session); unit = session-level
     instance trial_id = file offset + index (SourceDataset.build_trial_offsets); KFold(5, shuffle=True,
     random_state=seed) over sorted(str(trial_id)) exactly as the producer code sorts string IDs.
  S2 "KFoldPerSubjectCrossTrial" (v1 Code.zip, duplicate era): per subject pooled over sessions; unit =
     run-file stem; KFold(5, shuffle=True, random_state=seed) over the stems in SORTED order (the
     producer's list(set()) order is hash-seed dependent and unrecoverable; sorted is the declared replay).
Scored population: every distinct physical trial, via its canonical (first-appearance) instance; each is
scored exactly once per (split, seed). For fold f:
  T_f = distinct trials whose canonical instance is in fold f
  X_f = training instances whose distinct trial is in T_f       (observed crossing copies = test twins)
  C_f = canonical training instances of the other trials        (one row per distinct training trial)
  E_f = remaining training instances (copies of C_f trials)
  LEAKED = C+E+X (the producer's actual training multiset), CLEAN = C, PLACEBO = C+E+Z,
  Z_f = |X_f| class-matched clones of C_f trials, matched on (session, acquisition run) where possible,
        else (session), else (subject); seeded;
  LCB_f = C_f minus k_f = |distinct(X_f)| class/session(/run)-matched distinct trials (learning-curve
        control for LEAKED's extra distinct trials; secondary, not an arm of the Paper A design).
Assertions before any fit: rows(L) = rows(P), class counts equal, distinct(L) - distinct(P) = k_f,
exposure audit (hash) = 0 for CLEAN, PLACEBO, LCB and = #exposed for LEAKED; global hash uniqueness.
Private manifest -> --private-out; aggregate-only JSON -> --output.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

SEEDS = (42, 43, 44, 45, 46)
SPLITS = {"S1": 1, "S2": 2}
ARMS = ("T", "L", "C", "P", "LCB", "TEST_INST")


def fold_of_instances(split: str, seed: int, inst: dict, stems: np.ndarray) -> np.ndarray:
    fold = np.full(inst["label"].size, -1, dtype=np.int8)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    if split == "S1":
        for ses in np.unique(inst["session"]):
            idx = np.flatnonzero(inst["session"] == ses)
            ids = inst["b_trial_id"][idx]
            keys = sorted(str(int(t)) for t in ids)
            pos = {k: j for j, k in enumerate(keys)}
            fk = np.empty(len(keys), dtype=np.int8)
            for f, (_, te) in enumerate(kf.split(keys)):
                fk[te] = f
            fold[idx] = [fk[pos[str(int(t))]] for t in ids]
    else:
        for subj in np.unique(inst["subject"]):
            idx = np.flatnonzero(inst["subject"] == subj)
            units = sorted(set(stems[idx].tolist()))
            pos = {u: j for j, u in enumerate(units)}
            fu = np.empty(len(units), dtype=np.int8)
            for f, (_, te) in enumerate(kf.split(units)):
                fu[te] = f
            fold[idx] = [fu[pos[u]] for u in stems[idx]]
    if (fold < 0).any():
        raise RuntimeError("unassigned instance")
    return fold


def _pools(dists, dist, levels):
    out = []
    for lev in levels:
        g = collections.defaultdict(list)
        for d in dists:
            g[tuple(int(dist[c][d]) for c in lev)].append(int(d))
        out.append(g)
    return out


LEVELS = (("session", "origin_run", "label"), ("session", "label"), ("subject", "label"))


def draw(targets, pools, dist, rng, replace: bool, fallback: collections.Counter, allow_unmatched=False):
    """For each target distinct trial pick one pool trial matching it at the finest available level.
    allow_unmatched (LCB only): if every class-matched candidate is used up, take any remaining pool trial
    and count it at level 3 ('class-unmatched'); the PLACEBO draw never allows this."""
    chosen, used = [], set()
    everything = sorted({c for g in pools[-1].values() for c in g})
    for t in targets:
        for lev_i, (lev, pool) in enumerate(zip(LEVELS, pools)):
            cand = pool.get(tuple(int(dist[c][t]) for c in lev), [])
            if not replace:
                cand = [c for c in cand if c not in used]
            if cand:
                pick = cand[int(rng.integers(len(cand)))]
                chosen.append(pick)
                used.add(pick)
                fallback[lev_i] += 1
                break
        else:
            rest = [c for c in everything if c not in used] if allow_unmatched else []
            if not rest:
                raise RuntimeError("no class-matched candidate for placebo/LCB")
            pick = rest[int(rng.integers(len(rest)))]
            chosen.append(pick)
            used.add(pick)
            fallback[3] += 1
    return chosen


def build_unit(split, seed, unit_i, fold, members, inst, dist, hashes, stats):
    """members: instance indices of this split unit. Returns list of (fold, {arm: array})."""
    out = []
    canon_fold = {}
    for i in members:
        if inst["canonical"][i]:
            canon_fold[int(inst["distinct"][i])] = int(fold[i])
    for f in range(5):
        tr = members[fold[members] != f]
        te = members[fold[members] == f]
        T = np.array(sorted(d for d, cf in canon_fold.items() if cf == f), dtype=np.int32)
        d_tr = inst["distinct"][tr]
        is_x = np.array([canon_fold[int(d)] == f for d in d_tr], dtype=bool)
        is_c = inst["canonical"][tr] & ~is_x
        X, C, E = d_tr[is_x], d_tr[is_c], d_tr[~is_x & ~is_c]
        rng = np.random.default_rng([seed, SPLITS[split], unit_i, f])
        pools = _pools(np.unique(C), dist, LEVELS)
        Z = np.array(draw(X.tolist(), pools, dist, rng, True, stats["z_fallback"]), dtype=np.int32)
        kx = np.unique(X)
        R = set(draw(kx.tolist(), pools, dist, rng, False, stats["lcb_fallback"], allow_unmatched=True))
        LCB = np.array([c for c in C.tolist() if c not in R], dtype=np.int32)
        L = np.concatenate([C, E, X]).astype(np.int32)
        P = np.concatenate([C, E, Z]).astype(np.int32)
        C = C.astype(np.int32)
        # ---- assertions (before any fit)
        if L.size != P.size:
            raise AssertionError("rows(L) != rows(P)")
        if not np.array_equal(np.bincount(dist["label"][L], minlength=3), np.bincount(dist["label"][P], minlength=3)):
            raise AssertionError("class counts differ L vs P")
        if np.unique(L).size - np.unique(P).size != kx.size:
            raise AssertionError("distinct(L) - distinct(P) != k_f")
        hT = set(hashes[T].tolist())
        for name, arm in (("C", C), ("P", P), ("LCB", LCB)):
            if hT & set(hashes[arm].tolist()):
                raise AssertionError(f"exposure in {name}")
        n_exp = len(hT & set(hashes[L].tolist()))
        if n_exp != kx.size:
            raise AssertionError("LEAKED exposure != #exposed")
        if np.unique(C).size != C.size:
            raise AssertionError("CLEAN not one row per distinct trial")
        stats["folds"] += 1
        stats["scored"] += T.size
        stats["exposed"] += kx.size
        stats["rows_L"] += L.size
        stats["rows_C"] += C.size
        stats["rows_LCB"] += LCB.size
        stats["X_rows"] += X.size
        if kx.size:
            stats["exposed_folds"] += 1
            stats["extra_distinct_sum"] += kx.size
            stats["extra_distinct_pct_sum"] += 100.0 * kx.size / np.unique(P).size
        out.append((f, {"T": T, "L": L, "C": C, "P": P, "LCB": LCB,
                        "TEST_INST": inst["distinct"][te].astype(np.int32)}))
    return out


def build_manifest(priv: Path) -> dict:
    inst = dict(np.load(priv / "G2_instances.npz"))
    dist = dict(np.load(priv / "G2_distinct.npz"))
    meta = json.loads((priv / "G2_sessions_PRIVATE.json").read_text())
    hashes = dist["hash"]
    if np.unique(hashes).size != hashes.size:
        raise AssertionError("a physical trial hash occurs in more than one session/participant")
    sess = meta["sessions"]
    stems = np.array([f"{sess[s][0]}_{sess[s][1]}_task-motorimagery_run-{int(r):02d}_eeg"
                      for s, r in zip(inst["session"], inst["file_run"])])
    rows, table = [], []
    pos = 0
    agg = {"stage": "R4-G2c", "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "n_distinct_scored_per_split_seed": int(dist["label"].size),
           "global_hash_unique": True, "splits": {}}
    exposed_flags = {}
    for split in SPLITS:
        agg["splits"][split] = {}
        for seed in SEEDS:
            fold = fold_of_instances(split, seed, inst, stems)
            stats = collections.Counter()
            stats["z_fallback"], stats["lcb_fallback"] = collections.Counter(), collections.Counter()
            unit_key = inst["session"] if split == "S1" else inst["subject"]
            exp = np.zeros(dist["label"].size, dtype=bool)
            for unit_i in np.unique(unit_key):
                members = np.flatnonzero(unit_key == unit_i)
                for f, arms in build_unit(split, seed, int(unit_i), fold, members, inst, dist, hashes, stats):
                    expo = np.intersect1d(arms["T"], arms["L"])
                    exp[expo] = True
                    for arm in ARMS:
                        v = arms[arm]
                        table.append((SPLITS[split], seed, int(unit_i), f, ARMS.index(arm), pos, pos + v.size))
                        rows.append(v)
                        pos += v.size
            exposed_flags[f"{split}_{seed}"] = exp
            if stats["scored"] != dist["label"].size:
                raise AssertionError("not every distinct trial scored exactly once")
            per_p = [100.0 * exp[dist["subject"] == s].mean() for s in np.unique(dist["subject"])]
            agg["splits"][split][str(seed)] = {
                "folds": stats["folds"], "scored": stats["scored"], "exposed": stats["exposed"],
                "exposed_pct": round(100.0 * stats["exposed"] / stats["scored"], 2),
                "participants_with_exposed_scored_trials": int(sum(p > 0 for p in per_p)),
                "participant_exposed_pct": {"min": round(min(per_p), 2), "median": round(float(np.median(per_p)), 2),
                                            "max": round(max(per_p), 2)},
                "mean_train_rows_LEAKED_PLACEBO": round(stats["rows_L"] / stats["folds"], 1),
                "mean_train_rows_CLEAN": round(stats["rows_C"] / stats["folds"], 1),
                "mean_train_rows_LCB": round(stats["rows_LCB"] / stats["folds"], 1),
                "mean_X_rows": round(stats["X_rows"] / stats["folds"], 1),
                "exposed_folds": stats["exposed_folds"],
                "mean_extra_distinct_per_exposed_fold": round(stats["extra_distinct_sum"] / max(stats["exposed_folds"], 1), 2),
                "mean_extra_distinct_pct_of_placebo": round(stats["extra_distinct_pct_sum"] / max(stats["exposed_folds"], 1), 2),
                "placebo_match_level_counts": {LEVELS_NAMES[k]: v for k, v in sorted(stats["z_fallback"].items())},
                "lcb_match_level_counts": {LEVELS_NAMES[k]: v for k, v in sorted(stats["lcb_fallback"].items())},
            }
            print(split, seed, json.dumps(agg["splits"][split][str(seed)]), flush=True)
    np.savez(priv / "G2_manifest.npz", rows=np.concatenate(rows).astype(np.int32),
             table=np.asarray(table, dtype=np.int64), **{f"exposed_{k}": v for k, v in exposed_flags.items()})
    agg["assertions_passed"] = ["rows(LEAKED)=rows(PLACEBO) every fold", "class counts LEAKED=PLACEBO every fold",
                                "distinct(LEAKED)-distinct(PLACEBO)=k_f every fold",
                                "exposure audit 0 in CLEAN, PLACEBO, LCB; = #exposed in LEAKED",
                                "every distinct trial scored exactly once per split x seed",
                                "global SHA-256 uniqueness of distinct trials across sessions/participants"]
    agg["crossing_observed"] = {s: all(v["exposed"] > 0 for v in agg["splits"][s].values()) for s in SPLITS}
    return agg


LEVELS_NAMES = {0: "session+acq_run+class", 1: "session+class", 2: "subject+class", 3: "class-unmatched (LCB only)"}
