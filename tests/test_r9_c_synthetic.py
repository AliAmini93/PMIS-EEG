"""R9-C synthetic tests: split invariants, planted decomposition recovery, CRN group-size independence (GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from evaluation import r9_split_analysis as split_analysis
from experiments.pmis_r9 import split_bench  # noqa: E402


def synth_arms(rng):
    C = np.arange(100)
    T = np.arange(100, 140)
    exposed_ids = np.arange(100, 130)
    E = rng.choice(C, 60)
    X = np.concatenate([exposed_ids, exposed_ids[:12]])  # multiplicity 1 or 2
    Z = rng.choice(C, X.size)
    L = np.concatenate([C, E, X])
    P = np.concatenate([C, E, Z])
    LCB = C[:-10]
    exposed = np.zeros(200, dtype=bool)
    exposed[exposed_ids] = True
    dist = {"session": np.repeat(np.arange(20), 10), "origin_run": np.tile(np.repeat([0, 1], 5), 20),
            "label": np.tile([1, 2], 100), "subject": np.zeros(200, dtype=np.int64)}
    return {"T": T, "L": L, "P": P, "C": C, "LCB": LCB}, exposed, dist


def test_split_invariants():
    rng = np.random.default_rng(0)
    arms, exposed, dist = synth_arms(rng)
    h1, h2 = split_bench.split_halves(arms, dist, (42, 1, 0))
    L1, L2 = split_bench.make_split_arms(arms, h1, h2)
    st = split_bench.check_split(arms, L1, L2, h1, h2, exposed)
    assert st["n_h1"] + st["n_h2"] == 30 and abs(st["n_h1"] - st["n_h2"]) <= 6
    assert np.array_equal(np.unique(L1[np.isin(L1, arms["T"])]), h1)
    h1b, h2b = split_bench.split_halves(arms, dist, (42, 1, 0))
    assert np.array_equal(h1, h1b) and np.array_equal(h2, h2b)  # deterministic
    bad = L1.copy()
    bad[-1] = h2[0]
    with pytest.raises(AssertionError):
        split_bench.check_split(arms, bad, L2, h1, h2, exposed)


def test_planted_decomposition():
    rng = np.random.default_rng(1)
    n_part, per = 14, 400
    n = n_part * per
    subj = np.repeat(np.arange(n_part), per)
    y = np.tile([1, 2], n // 2)
    dist = {"subject": subj, "label": y}
    seeds = (42, 43, 44)
    A, half = {}, {}
    for s in seeds:
        h = rng.integers(0, 3, n)  # 0 unexposed, 1 H1, 2 H2
        own = rng.random(n) < 0.90
        oth = rng.random(n) < 0.80
        plc = rng.random(n) < 0.70
        un_sup = rng.random(n) < 0.72
        c = {"L1": np.where(h == 1, own, np.where(h == 2, oth, un_sup)).astype(float),
             "L2": np.where(h == 2, own, np.where(h == 1, oth, un_sup)).astype(float),
             "L": np.where(h > 0, own, un_sup).astype(float), "P": plc.astype(float)}
        A[s] = {**{f"c_{k}": v for k, v in c.items()}, **{f"p_{k}": v for k, v in c.items()}}
        half[s] = h
    res = split_analysis.analyse(A, half, dist, seeds)
    fam = res["family_holm"]
    assert abs(fam[split_analysis.FAMILY[0]]["mean_pp"] - 10) < 2.5
    assert abs(fam[split_analysis.FAMILY[1]]["mean_pp"] - 10) < 2.5
    assert abs(fam[split_analysis.FAMILY[2]]["mean_pp"]) < 3.5
    assert abs(res["descriptive"]["support_half_unexposed_BA_mean_of_L1_L2_minus_P"]["mean_pp"] - 2) < 2.5


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="GPU")
def test_group_size_independence_gpu():
    """Masks and permutations are drawn per CRN pair, so L and P trained in a 4-arm group should equal a 2-arm group."""
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from models import eegnet_group as eg
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 8, 128, generator=g).to(dev)
    y = torch.as_tensor(np.tile([0, 1], 150), device=dev)
    rng = np.random.default_rng(0)
    L, P = rng.choice(300, 128), rng.choice(300, 128)
    L1, L2 = L.copy(), P.copy()
    m2 = eg.train_group(X, y, np.stack([L, P]), np.zeros(2, dtype=np.int64), [11], 12, dev, epochs=2)
    m4 = eg.train_group(X, y, np.stack([L, L1, L2, P]), np.zeros(4, dtype=np.int64), [11], 12, dev, epochs=2)
    idx = np.arange(64)
    p2, p4 = eg.predict_proba(m2, X, idx), eg.predict_proba(m4, X, idx)
    d_L = float((p2[0] - p4[0]).abs().max())
    d_P = float((p2[1] - p4[3]).abs().max())
    print("group-size difference L, P:", d_L, d_P)
    assert float((p4[0] - p4[1]).abs().max()) == 0.0  # identical rows within the group -> identical outputs
    assert d_L < 1e-3 and d_P < 1e-3
