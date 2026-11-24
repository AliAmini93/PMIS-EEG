"""R11-B design: fixed-rows exact-copy dose arms on the R7-C6 Lee2019 benchmark folds (metadata only).

Folds: the 216 R7-C6 tasks (54 participants x leave-one-run-out). Per fold, with C6's own sets:
  T     = the 100 held-out-run trials (scored in every arm; 50 per class),
  train = C6's 240 training base trials (120 per class), val = C6's 60 validation trials (never in any arm).
Selections (numpy default_rng on metadata only, nested, class-stratified):
  dose order of T per class c:   rng([DOSE_STREAM, s, r, c]).permutation(T_c)  -> S_delta = first n_delta per class,
                                 n_delta = 10, 20, 30, 50 for delta = 0.2, 0.4, 0.6, 1.0;
  split complement for 0.4:      H2 = positions 20..39 per class (disjoint from S_0.4 = H1);
  base-copy order per class c:   rng([BASE_STREAM, s, r, c]).permutation(train_c) -> B_k = first k / 2 per class.
Arms (every arm except CLEAN has exactly 340 rows = 240 + 100 extra rows, 50 extra per class):
  C     = train                                   (240 rows; CLEAN)
  P     = train + B_100                           (PLACEBO: 100 extra copies of training-base trials)
  L<d>  = train + S_d + B_{100 (1 - d)}           (LEAKED at dose d: copies of 100 d scored trials replace base copies)
  L40c  = train + H2 + B_60                       (the split-half complement of L40; crossover for the decomposition)
  AA    = duplicate of L100 (group A) / of C (group B): A/A determinism, must be bit-identical.
Every L<d> differs from P only in which 100 d extra rows it holds: scored-trial copies instead of base copies.
"""

from __future__ import annotations

import hashlib

import numpy as np

DOSES = (0.2, 0.4, 0.6, 1.0)
N_PER_CLASS = {0.2: 10, 0.4: 20, 0.6: 30, 1.0: 50}
DOSE_TAG = {0.2: "L20", 0.4: "L40", 0.6: "L60", 1.0: "L100"}
EXTRA_PER_CLASS = 50
N_TRAIN, N_VAL, N_T = 240, 60, 100
DOSE_STREAM, BASE_STREAM = 11201, 11202
SEED_STREAMS = {"R1": (11301, 11302), "R2": (11311, 11312)}   # (initialisation / row-order seed, dropout seed)
GROUP_A = ("P", "L20", "L40", "L40c", "L60", "L100", "AA")
GROUP_B = ("C", "AAC")
LEAK_ARMS = ("L20", "L40", "L40c", "L60", "L100")
LABELS = (1, 2)


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def folds_from_c6(c6_plan: dict, labels: np.ndarray) -> list[dict]:
    """One fold per C6 group, in C6 order, with the R11-B selections and arms."""
    lab = np.asarray(labels)
    out = []
    for g in c6_plan["groups"]:
        s, r = int(g["subject"]), int(g["fold"])
        T = np.asarray(g["T"], dtype=np.int64)
        train = np.asarray(g["train"], dtype=np.int64)
        val = np.asarray(g["val"], dtype=np.int64)
        order_T, order_B = {}, {}
        for c in LABELS:
            order_T[c] = np.random.default_rng([DOSE_STREAM, s, r, c]).permutation(T[lab[T] == c])
            order_B[c] = np.random.default_rng([BASE_STREAM, s, r, c]).permutation(train[lab[train] == c])
        S = {d: np.sort(np.concatenate([order_T[c][:N_PER_CLASS[d]] for c in LABELS])) for d in DOSES}
        H2 = np.sort(np.concatenate([order_T[c][20:40] for c in LABELS]))
        B = {k: np.sort(np.concatenate([order_B[c][:k // 2] for c in LABELS])) for k in (0, 40, 60, 80, 100)}
        arms = {"C": train, "P": np.concatenate([train, B[100]])}
        for d in DOSES:
            arms[DOSE_TAG[d]] = np.concatenate([train, S[d], B[int(round(100 * (1 - d)))]])
        arms["L40c"] = np.concatenate([train, H2, B[60]])
        arms["AA"] = arms["L100"].copy()
        arms["AAC"] = arms["C"].copy()
        out.append({"subject": s, "fold": r, "T": T, "train": train, "val": val, "S": S, "H1": S[0.4], "H2": H2,
                    "B": B, "arms": arms})
    return out


def seeds(fold: dict, rep: str) -> tuple[int, int]:
    a, b = SEED_STREAMS[rep]
    return _seed(a, fold["subject"], fold["fold"]), _seed(b, fold["subject"], fold["fold"])


def check_fold(f: dict, labels: np.ndarray) -> None:
    """Fail-closed invariants (raise AssertionError)."""
    lab = np.asarray(labels)
    T, train, val = f["T"], f["train"], f["val"]
    assert T.size == N_T and train.size == N_TRAIN and val.size == N_VAL
    assert np.intersect1d(T, train).size == 0 and np.intersect1d(T, val).size == 0 and np.intersect1d(train, val).size == 0
    for c in LABELS:
        assert int(np.sum(lab[T] == c)) == 50 and int(np.sum(lab[train] == c)) == 120
    prev = np.array([], dtype=np.int64)
    for d in DOSES:
        Sd = f["S"][d]
        assert Sd.size == 2 * N_PER_CLASS[d] and np.all(np.isin(Sd, T)) and np.all(np.isin(prev, Sd))
        for c in LABELS:
            assert int(np.sum(lab[Sd] == c)) == N_PER_CLASS[d]
        prev = Sd
    assert np.intersect1d(f["H1"], f["H2"]).size == 0 and f["H2"].size == 40
    for k, Bk in f["B"].items():
        assert Bk.size == k and np.all(np.isin(Bk, train))
    assert np.all(np.isin(f["B"][60], f["B"][100])) and np.all(np.isin(f["B"][40], f["B"][60]))
    A = f["arms"]
    assert A["C"].size == N_TRAIN
    extra_ref = None
    for a in GROUP_A:
        rows = A[a]
        assert rows.size == N_TRAIN + 100, a
        assert np.all(rows[:N_TRAIN] == train), a
        extra = rows[N_TRAIN:]
        for c in LABELS:
            assert int(np.sum(lab[extra] == c)) == EXTRA_PER_CLASS, a
        assert np.intersect1d(rows, val).size == 0, a
        if a == "P":
            assert np.intersect1d(extra, T).size == 0
            extra_ref = extra
    for d in DOSES:
        extra = A[DOSE_TAG[d]][N_TRAIN:]
        held = extra[np.isin(extra, T)]
        assert np.array_equal(np.sort(held), f["S"][d])
        base_part = extra[~np.isin(extra, T)]
        assert np.all(np.isin(base_part, extra_ref)) and base_part.size == int(round(100 * (1 - d)))
    ex = A["L40c"][N_TRAIN:]
    assert np.array_equal(np.sort(ex[np.isin(ex, T)]), f["H2"])
    assert np.array_equal(A["AA"], A["L100"]) and np.array_equal(A["AAC"], A["C"])


def commitment(folds: list[dict]) -> str:
    h = hashlib.sha256()
    for f in folds:
        h.update(repr((f["subject"], f["fold"])).encode())
        for k in ("T", "train", "val", "H1", "H2"):
            h.update(np.ascontiguousarray(f[k]).tobytes())
        for d in DOSES:
            h.update(np.ascontiguousarray(f["S"][d]).tobytes())
        for a in GROUP_A + GROUP_B:
            h.update(a.encode())
            h.update(np.ascontiguousarray(f["arms"][a]).tobytes())
        for rep in SEED_STREAMS:
            h.update(repr(seeds(f, rep)).encode())
    return h.hexdigest()
