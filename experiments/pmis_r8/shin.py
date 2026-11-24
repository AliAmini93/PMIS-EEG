"""R8-A: PMIS identity swap on Shin2022 with NATURAL duplicates (byte-identical BW30 / NT0 run aliases).

Read-only inputs (paths in git-ignored data/local_paths.json; SHA-256 pinned in the protocol):
  r2_shin_windows_v1.npz           Paper A R2 window cache: one row per trial of each DISTINCT recording (recording
                                   SHA-256 'sha', trial index 'block', 'subject', 'label', 'b_main' eligibility), 0.96-s
                                   first-feedback window FB_EARLY: OAS covariance 8-30 Hz (16 x 16) and 4-40 Hz EEG (16 x 240)
  S2_CUE_TASK_MASK_REGISTRY.json   110 paths -> 101 recording SHA-256; a recording listed under two paths is an alias
Design per affected participant (a recording with two paths): test set T = the duplicated recording's eligible trials
(canonical copies); base C = eligible trials of every OTHER distinct recording of the participant (both paths of the
duplicated recording excluded); arm for test trial t = C + t' where t' is its alias copy (bit-identical features: same
recording bytes, same deterministic pipeline). Placebo arms = C + extra copy of a base trial (support |C|), class matched.
Every same-class pair (a, b) in T is scored from the single-copy arms: d = 1/2[(p(a|C+a') - p(a|C+b')) + (p(b|C+b') - p(b|C+a'))].
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json

import numpy as np

PLACEBO_STREAM, PLACEBO_PAIR_STREAM = 8101, 8102


def alias_recordings(registry_rows) -> dict:
    by = collections.defaultdict(list)
    for r in registry_rows:
        by[r["annex_sha256"]].append(r)
    return {v[0]["subject"]: s for s, v in by.items() if len(v) > 1}


def build_tasks(z: dict, registry_rows) -> list[dict]:
    dup = alias_recordings(registry_rows)
    sha, sub, lab, main, blk = (np.asarray(z[k]) for k in ("sha", "subject", "label", "b_main", "block"))
    tasks = []
    for s in sorted(dup):
        m = sub == s
        T = np.flatnonzero(m & main & (sha == dup[s]))
        base = np.flatnonzero(m & main & (sha != dup[s]))
        T = T[np.argsort(blk[T], kind="stable")]
        rng = np.random.default_rng([PLACEBO_STREAM, int(s[1:])])
        plc = []
        for c in (1, 2):  # as many placebo arms per class as test trials of that class
            pool = base[lab[base] == c]
            plc += sorted(int(x) for x in rng.choice(pool, int(np.sum(lab[T] == c)), replace=False))
        pairs = []
        for a, b in itertools.combinations(T.tolist(), 2):
            if lab[a] != lab[b]:
                continue
            prng = np.random.default_rng([PLACEBO_PAIR_STREAM, int(s[1:]), a, b])
            pool = [x for x in plc if lab[x] == lab[a]]
            c_, e_ = (int(x) for x in prng.choice(pool, 2, replace=False))
            pairs.append({"a": int(a), "b": int(b), "label": int(lab[a]), "pos_dist": int(abs(blk[a] - blk[b])),
                          "c": c_, "e": e_, "sign": int(prng.choice([-1, 1]))})
        tasks.append({"subject": s, "T": T.astype(np.int64), "base": np.sort(base).astype(np.int64),
                      "placebo_arms": np.array(plc, dtype=np.int64), "pairs": pairs})
    return tasks


def commitment(tasks) -> str:
    h = hashlib.sha256()
    for t in tasks:
        h.update(repr((t["subject"], t["T"].tolist(), t["base"].tolist(), t["placebo_arms"].tolist(), t["pairs"])).encode())
    return h.hexdigest()


def arm_rows(task) -> tuple[np.ndarray, list]:
    """Training rows of every arm (A-arms for each test trial, then placebo arms); all arms have |C| + 1 rows."""
    base = task["base"]
    extras = [("T", int(t)) for t in task["T"]] + [("P", int(c)) for c in task["placebo_arms"]]
    rows = np.stack([np.concatenate([base, [x]]) for _, x in extras]).astype(np.int64)
    return rows, extras


def pair_records(task, P: np.ndarray, extras: list, y: np.ndarray) -> list[dict]:
    """P: (n_arms, |T|) correct-class probabilities on T (arm order as arm_rows). Returns one record per pair."""
    pos = {int(t): i for i, t in enumerate(task["T"].tolist())}
    arm = {(k, x): j for j, (k, x) in enumerate(extras)}
    recs = []
    for pr in task["pairs"]:
        a, b = pr["a"], pr["b"]
        ia, ib = pos[a], pos[b]
        A, B = P[arm[("T", a)]], P[arm[("T", b)]]
        Cc, Ee = P[arm[("P", pr["c"])]], P[arm[("P", pr["e"])]]
        recs.append({"subject": task["subject"], "label": pr["label"], "pos_dist": pr["pos_dist"], "sign": pr["sign"],
                     "pAA": A[ia], "pAB": B[ia], "pBA": A[ib], "pBB": B[ib],
                     "pAC": Cc[ia], "pBC": Cc[ib], "pAE": Ee[ia], "pBE": Ee[ib]})
    return recs
