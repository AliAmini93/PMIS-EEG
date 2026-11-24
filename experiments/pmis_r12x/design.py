"""R12-X A/B/C partitions, arms and seeds (metadata only: run labels, trial positions, class labels).

Per participant: A = R1-R4 (160 trials) -> A_train 128 / A_val 32 (class-stratified); B = R5 (40, 20/20);
C = R6 (40, 20/20), fixed once, never B, never in any fitted or selection arm; final pool = R1-R5 (200) ->
pool_train 160 / pool_val 40 (class-stratified).
Selection arms (scored on all B trials): S_P = A_train + 40 class-matched copies of A_train rows (placebo);
S_L = A_train + one exact copy of every B trial (delta = 1.0, controlled stress test); AA_L = recomputation of S_L.
Final arms (scored once on C): F = pool_train; AA_F = recomputation of F. Validation rows (A_val, pool_val) are
used only by the early-stopping read-out and never appear in any arm's rows.
All draws: numpy default_rng([stream, participant, class]) over trial indices sorted by (run, position).
"""

from __future__ import annotations

import hashlib

import numpy as np

STREAMS = {"a_split": 12101, "placebo": 12102, "final_split": 12103, "init": 12301, "drop": 12302}
K_SEEDS = 5
RUNS_A, RUN_B, RUN_C = ("R1", "R2", "R3", "R4"), "R5", "R6"
N_A_VAL_PER_CLASS, N_POOL_VAL_PER_CLASS, N_PLACEBO_PER_CLASS = 16, 20, 20
LABELS = (1, 2)
SEL_ARMS, FIN_ARMS = ("S_P", "S_L", "AA_L"), ("F", "AA_F")


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def seeds(participant: int, stage: str, k: int) -> tuple[int, int]:
    st = {"selection": 1, "final": 2}[stage]
    return _seed(STREAMS["init"], participant, st, k), _seed(STREAMS["drop"], participant, st, k)


def _split(ids, lab, n_val_per_class, stream, participant):
    val = []
    for c in LABELS:
        cand = ids[lab[ids] == c]
        val.append(np.random.default_rng([stream, participant, c]).permutation(cand)[:n_val_per_class])
    val = np.sort(np.concatenate(val))
    return np.setdiff1d(ids, val), val


def build_participant(participant: int, run: np.ndarray, pos: np.ndarray, lab: np.ndarray, idx: np.ndarray) -> dict:
    """run (str array), pos, lab (1/2), idx (global row ids) for ONE participant's trials."""
    order = np.lexsort((pos, run))
    run, pos, lab, idx = run[order], pos[order], lab[order], idx[order]
    L = dict(zip(idx.tolist(), lab.tolist()))
    labg = np.zeros(int(idx.max()) + 1, dtype=np.int64)
    labg[idx] = lab

    def ids(rs):
        return np.sort(idx[np.isin(run, rs)])

    A, B, C, pool = ids(RUNS_A), ids([RUN_B]), ids([RUN_C]), ids(list(RUNS_A) + [RUN_B])
    for part, n in ((B, 40), (C, 40)):
        assert part.size == n and all(sum(L[i] == c for i in part.tolist()) == n // 2 for c in LABELS)
    assert A.size == 160
    a_train, a_val = _split(A, labg, N_A_VAL_PER_CLASS, STREAMS["a_split"], participant)
    p_train, p_val = _split(pool, labg, N_POOL_VAL_PER_CLASS, STREAMS["final_split"], participant)
    plc = np.concatenate([np.random.default_rng([STREAMS["placebo"], participant, c]).permutation(
        a_train[labg[a_train] == c])[:N_PLACEBO_PER_CLASS] for c in LABELS])
    arms = {"S_P": np.concatenate([a_train, np.sort(plc)]), "S_L": np.concatenate([a_train, B])}
    arms["AA_L"] = arms["S_L"].copy()
    arms["F"] = p_train.copy()
    arms["AA_F"] = arms["F"].copy()
    d = {"participant": participant, "A": A, "A_train": a_train, "A_val": a_val, "B": B, "C": C, "pool": pool,
         "pool_train": p_train, "pool_val": p_val, "placebo_rows": np.sort(plc), "arms": arms}
    check(d, labg)
    return d


def check(d: dict, labg: np.ndarray) -> None:
    """Firewall invariants (fail closed)."""
    C, B = set(d["C"].tolist()), set(d["B"].tolist())
    for name, rows in d["arms"].items():
        assert not (set(rows.tolist()) & C), f"C trial in arm {name}"
    for name in ("A_train", "A_val", "pool_train", "pool_val", "placebo_rows"):
        assert not (set(d[name].tolist()) & C), f"C trial in {name}"
    assert not (set(d["A_train"].tolist()) & B) and not (set(d["A_val"].tolist()) & B)
    a_val = set(d["A_val"].tolist())
    assert not (a_val & (set(d["arms"]["S_P"].tolist()) | set(d["arms"]["S_L"].tolist()))), "A_val in a training arm"
    assert not (set(d["pool_val"].tolist()) & set(d["arms"]["F"].tolist()))
    assert set(d["placebo_rows"].tolist()) <= set(d["A_train"].tolist())
    sp, sl = d["arms"]["S_P"], d["arms"]["S_L"]
    assert sp.size == sl.size == 168
    for c in LABELS:
        assert np.sum(labg[sp] == c) == np.sum(labg[sl] == c) == 84
    assert d["arms"]["F"].size == 160 and d["pool_val"].size == 40
    assert np.array_equal(d["arms"]["AA_L"], sl) and np.array_equal(d["arms"]["AA_F"], d["arms"]["F"])


def commitment(designs: list[dict]) -> str:
    h = hashlib.sha256()
    for d in designs:
        h.update(repr(d["participant"]).encode())
        for k in ("A_train", "A_val", "B", "C", "pool_train", "pool_val", "placebo_rows"):
            h.update(np.ascontiguousarray(d[k], dtype=np.int64).tobytes())
        for k in SEL_ARMS + FIN_ARMS:
            h.update(np.ascontiguousarray(d["arms"][k], dtype=np.int64).tobytes())
    return h.hexdigest()
