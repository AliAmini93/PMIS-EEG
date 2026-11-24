"""R9-C2 synthetic tests: A/A ordering identical to R9-C, deterministic frozen R4 TSLRw arm fits that equal a direct fit,
and an end-to-end synthetic unit through split, fits and assembly (CPU)."""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from evaluation.g2_estimands import correct_class_probability  # noqa: E402
from models.vendor_r4 import r4_models as rm  # noqa: E402
from experiments.pmis_r7 import c7  # noqa: E402
from experiments.pmis_r8 import bench  # noqa: E402
from experiments.pmis_r9 import split_bench, split_tslr  # noqa: E402


def synth(rng, n=200, ch=4):
    C = np.arange(100)
    T = np.arange(100, 140)
    exposed_ids = np.arange(100, 130)
    E = rng.choice(C, 60)
    X = np.concatenate([exposed_ids, exposed_ids[:12]])
    Z = rng.choice(C, X.size)
    arms = {"T": T, "L": np.concatenate([C, E, X]), "P": np.concatenate([C, E, Z]), "C": C, "LCB": C[:-10]}
    exposed = np.zeros(n, dtype=bool)
    exposed[exposed_ids] = True
    y = np.tile([1, 2], n // 2)
    covs = np.empty((n, ch, ch))
    for i in range(n):
        a = rng.standard_normal((ch, 3 * ch))
        a[0] *= 1.0 + 0.6 * (y[i] == 2)  # class signal in the first channel's power
        covs[i] = a @ a.T / (3 * ch) + 1e-3 * np.eye(ch)
    dist = {"session": np.repeat(np.arange(20), 10), "origin_run": np.tile(np.repeat([0, 1], 5), 20), "label": y,
            "subject": np.zeros(n, dtype=np.int64)}
    return arms, exposed, dist, covs, y


def test_unit_tasks_aa_order():
    tl = [(s, u, f) for s in (42, 43, 44) for u in range(14) for f in range(5)]
    ut = split_tslr.unit_tasks(tl)
    assert len(ut) == 42 and all(len(t[2]) == 5 for t in ut)
    flags = {(s, u, f): aa for (s, u, folds) in ut for (f, aa) in folds}
    assert [i for i, k in enumerate(tl) if flags[k]] == list(range(0, len(tl), 10))


def test_fit_arm_deterministic_and_equals_direct_fit():
    rng = np.random.default_rng(3)
    arms, exposed, dist, covs, y = synth(rng)
    c7.G.update({"covs": covs, "label": y.astype(np.int64)})
    T = arms["T"]
    p1, c1, s1 = split_tslr.fit_arm(arms["L"], T)
    p2, c2, s2 = split_tslr.fit_arm(arms["L"].copy(), T)
    assert np.array_equal(p1, p2) and np.array_equal(c1, c2) and not s1 and not s2
    u, cnt = np.unique(arms["L"], return_counts=True)
    m = rm.TSLRw().fit(covs[u], y[u], cnt)
    f = rm.tangent_space_b(covs[T], m.ref)
    pd = correct_class_probability(m.clf.predict_proba(f), m.clf.classes_, y[T])
    assert np.array_equal(p1, pd)
    assert np.array_equal(c1, (np.asarray(m.clf.predict(f)) == y[T]).astype(float))


def test_run_unit_end_to_end(monkeypatch):
    rng = np.random.default_rng(4)
    arms, exposed, dist, covs, y = synth(rng)
    c7.G.update({"covs": covs, "label": y.astype(np.int64), "rows": None, "table": None, "dist": dist,
                 "exposed": {42: exposed}})
    monkeypatch.setattr(bench, "fold_arms", lambda table, rows, split, seed, unit, fold: arms)
    r = split_tslr.run_unit((42, 0, ((0, True),)))
    fr = r["folds"][0]
    assert fr["aa_max_abs"] == 0.0
    h1, h2 = split_bench.split_halves(arms, dist, (42, 0, 0))
    assert np.array_equal(fr["H1"], h1) and np.array_equal(fr["H2"], h2)  # identical halves to R9-C's construction
    for a in split_tslr.ARMS:
        assert np.all(np.isfinite(fr[f"p_{a}"])) and set(np.unique(fr[f"c_{a}"])) <= {0.0, 1.0}
    A, half, aa = split_tslr.assemble([r], 200, (42,))
    assert aa == [0.0] and set(np.unique(half[42][arms["T"]])) <= {0, 1, 2}
    assert np.all(np.isfinite(A[42]["p_L1"][arms["T"]])) and np.all(np.isnan(A[42]["p_L1"][:100]))
