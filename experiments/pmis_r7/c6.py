"""R7-C6: deep-decoder robustness of the exact-copy effect on Lee2019 (EEGNet early-stopping read-out; ShallowConvNet).

Per C3 task (participant x held-out run): an outcome-blind eighth of the C3 pairs (5 of 40; rng [7601, participant,
run]); a fixed validation set of 30 + 30 base trials per class (rng [7602, participant, run]) drawn from base trials that
are not the placebo trials of any sampled pair, held out of training in EVERY arm; training base = the other 240.
Arms per pair: A = base + a', B = base + b', C / E = base + extra copy of c / e (placebo); one A/A duplicate per task.
Both models are trained 100 epochs (batch 64, Adam 1e-3, CRN within pair); read-outs: the final epoch and each
model's minimum-validation-loss epoch (early stopping by selection, same run).
"""

from __future__ import annotations

import hashlib

import numpy as np

SAMPLE_STREAM, VAL_STREAM, PAIR_STREAM, GROUP_STREAM = 7601, 7602, 7603, 7604
N_PAIRS, N_VAL_PER_CLASS = 5, 30
MODELS = ("EEGNet", "Shallow")


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def build_plan(tasks: list[dict], labels: np.ndarray) -> dict:
    groups = []
    for t in tasks:
        subj, fold = int(t["subject"]), int(t["fold"])
        rng = np.random.default_rng([SAMPLE_STREAM, subj, fold])
        sel = sorted(int(i) for i in rng.choice(len(t["pairs"]), N_PAIRS, replace=False))
        prs = [t["pairs"][j] for j in sel]
        reserved = {x for p in prs for x in (p["c"], p["e"])}
        vrng = np.random.default_rng([VAL_STREAM, subj, fold])
        base = np.asarray(t["base_canon"], dtype=np.int64)
        val = []
        for c in (1, 2):
            pool = np.array([b for b in base if labels[b] == c and int(b) not in reserved])
            val += sorted(int(v) for v in vrng.choice(pool, N_VAL_PER_CLASS, replace=False))
        val = np.array(sorted(val), dtype=np.int64)
        train = np.setdiff1d(base, val)
        assert reserved <= set(train.tolist())
        arms, meta = [], []
        for p_idx, pr in enumerate(prs):
            meta.append({"k": int(pr["k"]), "seed": _seed(PAIR_STREAM, subj, fold, pr["k"]), "a": int(pr["a"]), "b": int(pr["b"]),
                         "label": int(pr["label"]), "session": int(pr["session"]), "pos_dist": int(pr["pos_dist"]),
                         "c": int(pr["c"]), "e": int(pr["e"]), "sign": int(pr["sign"])})
            arms += [{"pair": p_idx, "arm": "A", "extra": int(pr["a"])}, {"pair": p_idx, "arm": "B", "extra": int(pr["b"])},
                     {"pair": p_idx, "arm": "C", "extra": int(pr["c"])}, {"pair": p_idx, "arm": "E", "extra": int(pr["e"])}]
        arms.append({"pair": 0, "arm": "AA", "extra": int(prs[0]["a"])})
        groups.append({"subject": subj, "fold": fold, "group": 0, "group_seed": _seed(GROUP_STREAM, subj, fold),
                       "pairs": meta, "arms": arms, "train": train.tolist(), "val": val.tolist(), "T": t["T"].tolist(),
                       "unexposed": t["unexposed"].tolist()})
    return {"groups": groups, "n_pairs": sum(len(g["pairs"]) for g in groups), "n_arms": sum(len(g["arms"]) for g in groups)}


def commitment(plan: dict) -> str:
    h = hashlib.sha256()
    for g in plan["groups"]:
        h.update(repr((g["subject"], g["fold"], g["group_seed"], g["pairs"], g["arms"], g["train"], g["val"], g["T"])).encode())
    return h.hexdigest()


def run_group(cls, g: dict, X, y01, device) -> dict:
    from models.deep_group import train_generic
    train = np.asarray(g["train"], dtype=np.int64)
    rows = np.stack([np.concatenate([train, [a["extra"]]]) for a in g["arms"]]).astype(np.int64)
    pair_of = np.array([a["pair"] for a in g["arms"]], dtype=np.int64)
    out = train_generic(cls, X, y01, rows, pair_of, [p["seed"] for p in g["pairs"]], g["group_seed"], device,
                        score_idx=np.asarray(g["T"], dtype=np.int64), val_idx=np.asarray(g["val"], dtype=np.int64))
    for k in ("p_final", "p_es"):
        if not np.all(np.isfinite(out[k])):
            raise FloatingPointError("non-finite output")
    return out


def pair_records(plan: dict, results: dict, model: str) -> tuple[list[dict], list[float]]:
    recs, aa = [], []
    for g in plan["groups"]:
        res = results[(model, g["subject"], g["fold"])]
        T = g["T"]
        pos = {int(t): i for i, t in enumerate(T)}
        um = np.asarray(g["unexposed"], dtype=bool)
        by = {(a["pair"], a["arm"]): m for m, a in enumerate(g["arms"])}
        aa.append(float(max(np.max(np.abs(res[k][by[(0, "AA")]] - res[k][by[(0, "A")]])) for k in ("p_final", "p_es"))))
        for p_idx, pr in enumerate(g["pairs"]):
            ia, ib = pos[pr["a"]], pos[pr["b"]]
            r = {"subject": g["subject"], "fold": g["fold"], "k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"],
                 "sign": pr["sign"]}
            for ro in ("final", "es"):
                P = res[f"p_{ro}"]
                pA, pB, pC, pE = (P[by[(p_idx, x)]] for x in ("A", "B", "C", "E"))
                r.update({f"{ro}_pAA": pA[ia], f"{ro}_pBA": pA[ib], f"{ro}_pAB": pB[ia], f"{ro}_pBB": pB[ib],
                          f"{ro}_pAC": pC[ia], f"{ro}_pBC": pC[ib], f"{ro}_pAE": pE[ia], f"{ro}_pBE": pE[ib],
                          f"{ro}_off": float(np.mean(np.abs(pA[um] - pB[um])))})
            r.update({"es_epoch_A": int(res["es_epoch"][by[(p_idx, "A")]]), "es_epoch_B": int(res["es_epoch"][by[(p_idx, "B")]])})
            recs.append(r)
    return recs, aa
