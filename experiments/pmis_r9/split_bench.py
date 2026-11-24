"""R9-C: cross-fitted split-half decomposition of EEGNet benchmark inflation on R4's HS-Stroke arms.

R8-B separated exposure from support only through a learning-curve bound (CLEAN - LCB), which assumes diminishing returns
and that the k removed non-test trials are as useful as the k extra distinct test-trial copies in LEAKED. This design
replaces that assumption with randomisation.

Per (seed, participant, fold) the exposed scored trials (those whose copies sit in LEAKED) are split into halves H1 / H2,
stratified by (session, origin run, class, number of copies in LEAKED), with an outcome-blind seeded RNG. Two arms are
built from LEAKED by replacing, position by position, the copies of one half with PLACEBO's rows at the same positions
(PLACEBO is LEAKED with every test copy replaced by a session/class-matched non-test copy):
  L1 = LEAKED with H2's copies replaced  (exposes H1 only)
  L2 = LEAKED with H1's copies replaced  (exposes H2 only)
All four arms (L, L1, L2, P) have identical rows and are trained as one CRN group with R8-B's seeds. For an exposed trial t
in H1: own-exposed arm = L1, other-half arm = L2 (and vice versa for H2). Both arms hold the same number of extra distinct
same-run test trials (|H1| ~ |H2|) of exchangeable provenance, so:
  exposure (support fixed)            = own-exposed - other-half
  support of half the test copies     = other-half - PLACEBO
are identified by randomisation, at half density, without a learning-curve assumption.
"""

from __future__ import annotations

import hashlib

import numpy as np

from experiments.pmis_r8 import bench

SPLIT_SALT = 9201


def exposed_distinct(arms):
    L, T = arms["L"], arms["T"]
    x = L[np.isin(L, T)]
    ex, counts = np.unique(x, return_counts=True)
    return ex, counts


def split_halves(arms, dist, seed_tuple):
    """Stratified outcome-blind split of the exposed scored trials into (H1, H2). Strata: session, origin run, class,
    multiplicity in LEAKED. Within a stratum members are permuted and alternately assigned; the half receiving the
    extra member of an odd stratum is a fair coin."""
    ex, counts = exposed_distinct(arms)
    rng = np.random.default_rng([SPLIT_SALT, *[int(v) for v in seed_tuple]])
    key = np.stack([dist["session"][ex].astype(np.int64), dist["origin_run"][ex].astype(np.int64),
                    dist["label"][ex].astype(np.int64), counts.astype(np.int64)], axis=1)
    order = np.lexsort(key.T[::-1])
    ex, key = ex[order], key[order]
    h1, h2 = [], []
    starts = np.flatnonzero(np.r_[True, np.any(key[1:] != key[:-1], axis=1)])
    ends = np.r_[starts[1:], key.shape[0]]
    for a, b in zip(starts, ends):
        members = ex[a:b][rng.permutation(b - a)]
        first = int(rng.integers(2))
        for j, m in enumerate(members):
            (h1 if (j + first) % 2 == 0 else h2).append(int(m))
    return np.sort(np.array(h1, dtype=np.int64)), np.sort(np.array(h2, dtype=np.int64))


def make_split_arms(arms, h1, h2):
    L, P = arms["L"], arms["P"]
    L1, L2 = L.copy(), L.copy()
    in_h2, in_h1 = np.isin(L, h2), np.isin(L, h1)
    L1[in_h2] = P[in_h2]
    L2[in_h1] = P[in_h1]
    return L1, L2


def check_split(arms, L1, L2, h1, h2, exposed):
    """Fail-closed invariants."""
    T, L, P = arms["T"], arms["L"], arms["P"]
    bench.check_arms(arms, exposed)
    if not (L1.size == L2.size == L.size == P.size):
        raise AssertionError("split arms differ in rows")
    if np.intersect1d(h1, h2).size or not np.array_equal(np.union1d(h1, h2), np.unique(L[np.isin(L, T)])):
        raise AssertionError("halves do not partition the exposed scored trials")
    if not (np.array_equal(np.unique(L1[np.isin(L1, T)]), h1) and np.array_equal(np.unique(L2[np.isin(L2, T)]), h2)):
        raise AssertionError("a split arm exposes the wrong half")
    nonx = ~np.isin(L, T)
    if not (np.array_equal(L1[nonx], L[nonx]) and np.array_equal(L2[nonx], L[nonx]) and np.array_equal(P[nonx], L[nonx])):
        raise AssertionError("non-test rows differ between arms")
    if np.intersect1d(P, T).size:
        raise AssertionError("scored trial in PLACEBO")
    d1, d2 = np.unique(L1).size - np.unique(P).size, np.unique(L2).size - np.unique(P).size
    if not (d1 == h1.size and d2 == h2.size):
        raise AssertionError("extra distinct support of a split arm differs from its half size")
    return {"n_h1": int(h1.size), "n_h2": int(h2.size), "rows": int(L.size),
            "copy_rows_h1": int(np.isin(L, h1).sum()), "copy_rows_h2": int(np.isin(L, h2).sum())}


def commitment(table, rows, dist, task_list) -> str:
    h = hashlib.sha256()
    for (s, u, f) in task_list:
        arms = bench.fold_arms(table, rows, 2, s, u, f)
        h1, h2 = split_halves(arms, dist, (s, u, f))
        L1, L2 = make_split_arms(arms, h1, h2)
        for name, v in (("T", arms["T"]), ("L", arms["L"]), ("P", arms["P"]), ("L1", L1), ("L2", L2), ("H1", h1), ("H2", h2)):
            h.update(name.encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


ARMS = ("L", "L1", "L2", "P")


def run_fold(arms, L1, L2, X, y01, device, seed_tuple, aa=False):
    """One CRN group (L, L1, L2, P [, L again for A/A]) with R8-B's pair and group seeds; scored on T."""
    import torch

    from models import eegnet_group as eg
    T = arms["T"]
    yT = y01[torch.as_tensor(T, device=device)]
    grp = [arms["L"], L1, L2, arms["P"]] + ([arms["L"]] if aa else [])
    m = eg.train_group(X, y01, np.stack(grp), np.zeros(len(grp), dtype=np.int64), [bench._seed(8301, *seed_tuple)],
                       bench._seed(8302, *seed_tuple), device)
    pr = eg.predict_proba(m, X, T)
    out = {}
    if aa:
        out["aa_max_abs"] = np.array(float((pr[4] - pr[0]).abs().max()))
    for j, a in enumerate(ARMS):
        out[f"p_{a}"] = torch.gather(pr[j], 1, yT[:, None])[:, 0].double().cpu().numpy()
        out[f"c_{a}"] = (pr[j].argmax(-1) == yT).double().cpu().numpy()
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {k}")
    return out
