"""R10-B synthetic tests: window geometry, frozen curves and predictors, split / arm invariants, TS+LR estimator
identity with C5, EEGNet A/A + CRN, analysis decision rules on planted outcomes, end-to-end analysis on synthetic data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from evaluation import r10b_analysis as an  # noqa: E402
from experiments.pmis_r10b import design as ds  # noqa: E402
from experiments.pmis_r10b import geometry as geo  # noqa: E402
from experiments.pmis_r10b import train as tr  # noqa: E402


def synth_meta(n_subj=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subj + 1):
        for r in range(4):
            lab = rng.permutation(np.repeat([1, 2], 50))
            for p in range(100):
                rows.append((s, 1 + r // 2, r, p, lab[p], 10_000 + 13_000 * p))
    a = np.array(rows, dtype=np.int64)
    return {k: a[:, i] for i, k in enumerate(("subject", "session", "run", "pos", "label", "onset"))}


# ------------------------------------------------------------------------------------------------ geometry
def test_overlap_geometry():
    assert geo.shared_samples(0) == 2000 and geo.shared_samples(62) == 1938 and geo.shared_samples(2000) == 0
    assert geo.overlap_fraction(500) == 0.75 and geo.overlap_fraction(2500) == 0.0
    assert geo.stride_offsets(500) == (0, 500, 1000, 1500, 2000) and len(geo.stride_offsets(250)) == 9
    assert set(o for s in geo.STRIDES for o in geo.stride_offsets(s)) | set(geo.LAGS_B) == set(geo.OFFSETS)


def test_frozen_curves_equal_published_aggregates():
    c = geo.frozen_curves(ROOT)
    # These two forecast inputs are taken from the recorded single-copy curves.
    assert c["EEGNet"][250] == 0.38140176150961896
    assert c["TSLR"][1000] == 0.4315517387795375
    assert abs(c["EEGNet"][2000] - 0.037946029860345225) < 1e-15 and abs(c["TSLR"][2000] - 0.0942916858648344) < 1e-15
    assert all(c[m][0] == 1.0 for m in c)


def test_predictors_and_baselines():
    lin = {lag: geo.overlap_fraction(lag) for lag in (0, 62, 125, 250, 500, 1000, 2000)}
    for s in geo.STRIDES:  # a curve equal to the overlap fraction reproduces the OVL baseline exactly
        assert abs(geo.track_a_prediction(lin, s, "additive") - geo.baseline_a("OVL", s)) < 1e-12
    k = {0: 1.0, 62: 0.5, 125: 0.5, 250: 0.5, 500: 0.5, 1000: 0.5, 2000: 0.5}
    assert abs(geo.track_a_prediction(k, 1000, "additive") - 0.5) < 1e-12  # 2 windows x 0.5 -> 1 / 2
    assert abs(geo.track_a_prediction(k, 1000, "saturating") - 0.75) < 1e-12
    assert geo.track_a_prediction(k, 500, "nearest") == 0.5
    assert geo.baseline_a("CNT", 500) == 0.8 and geo.baseline_a("BIN", 250) == 7 / 8 and geo.baseline_b("BIN", 2000) == 0.0
    with pytest.raises(ValueError):
        geo.interp(k, 2500)


# ------------------------------------------------------------------------------------------------ design
def test_folds_and_arm_invariants():
    meta = synth_meta()
    fl = ds.folds(meta)
    assert len(fl) == 24
    for f in fl:
        ds.check_fold(f, meta)
    assert ds.commitment(fl, meta) == ds.commitment(ds.folds(meta), meta)  # deterministic


def test_invariant_violations_detected():
    meta = synth_meta(2)
    f = ds.folds(meta)[0]
    bad = dict(f)
    bad["H2"] = np.concatenate([f["H2"][1:], f["H1"][:1]])  # a trial in both halves
    with pytest.raises(ds.DesignError):
        ds.check_fold(bad, meta)
    bad = dict(f)
    bad["Z"] = f["T"][:50]  # placebo from the held-out run
    with pytest.raises(ds.DesignError):
        ds.check_fold(bad, meta)


def test_arm_geometry_records():
    meta = synth_meta(2)
    lab = np.asarray(meta["label"])
    f = ds.folds(meta)[0]
    ta = ds.track_a_arms(f, 500)
    g = ds.arm_geometry(ta["G"], f["T"], f["T"], lab)
    assert g["anchors_exposed"] == 0 and g["distinct_heldout_run_trials"] == 0 and g["rows"] == 1500
    l_ = ds.arm_geometry(ta["L"], f["T"], f["T"], lab)
    assert l_["own_windows_per_exposed_anchor"] == 4 and l_["overlapping_own_windows_per_exposed_anchor"] == 3
    assert l_["min_lag_ms_exposed"] == [500] and l_["shared_samples_nearest_exposed"] == [1500]
    x = ds.arm_geometry(ta["X1"], f["H1"], f["T"], lab)
    assert x["min_lag_ms_exposed"] == [0] and x["anchors_exposed"] == 50
    b = ds.arm_geometry(ds.track_b_arms(f)["B2_2000"], f["H2"], f["T"], lab)
    assert b["overlapping_own_windows_per_exposed_anchor"] == 0 and b["min_lag_ms_exposed"] == [2000]


def test_split_label_invariance():
    """Halves / arms depend only on metadata and frozen streams: relabelling outcomes cannot change them, and a
    label permutation that keeps class counts keeps every invariant."""
    meta = synth_meta(3, seed=1)
    fl = ds.folds(meta)
    m2 = dict(meta)
    m2["onset"] = meta["onset"] + 7  # a non-design field
    assert ds.commitment(fl, meta) == ds.commitment(ds.folds(m2), m2)
    m3 = synth_meta(3, seed=2)  # different labels, same structure
    for f in ds.folds(m3):
        ds.check_fold(f, m3)


def test_c0_splits_partition():
    meta = synth_meta(2)
    sp = ds.c0_splits(meta, 1)
    for sch in ("rw", "tg"):
        allte = np.concatenate(sp[sch])
        assert allte.size == 2000 and np.unique(allte).size == 2000
        assert all(te.size == 400 for te in sp[sch])
    t_fold = {}
    for k, te in enumerate(sp["tg"]):
        for t in ds.trial_of(te):
            t_fold.setdefault(int(t), set()).add(k)
    assert all(len(v) == 1 for v in t_fold.values())  # trial-grouped: no trial spans folds
    spread = {}
    for k, te in enumerate(sp["rw"]):
        for t in ds.trial_of(te):
            spread.setdefault(int(t), set()).add(k)
    assert np.mean([len(v) > 1 for v in spread.values()]) > 0.9  # random windows: most trials span folds
    assert all(len(r) == 1600 for _, _, r in ds.c0_train_rows(sp))


# ------------------------------------------------------------------------------------------------ models
def _spd(n, rng, d=20):
    a = rng.standard_normal((n, d, 40))
    return np.einsum("nij,nkj->nik", a, a) / 40 + 0.1 * np.eye(d)


def test_tslr_equals_c5_estimator():
    from experiments.pmis_r6 import c3
    rng = np.random.default_rng(3)
    covs = _spd(120, rng)
    y = np.repeat([1, 2], 60)
    covs[y == 2] *= 1.3
    train_ids, test_ids = np.arange(0, 100), np.arange(100, 120)
    c3.G["covs"], c3.G["y"] = covs, y
    ref, _ = c3.opt_pc(train_ids, np.ones(100, dtype=np.int64), test_ids)
    mine = tr.tslr_fit_predict(covs, y, train_ids, test_ids)
    assert np.array_equal(ref, mine)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="GPU-only")
def test_eegnet_aa_and_crn_gpu():
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from models import eegnet_group as eg
    torch.use_deterministic_algorithms(True)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    X = torch.randn(200, 20, 256, generator=g).to(dev)
    y = (torch.arange(200) % 2).to(dev)
    base = np.arange(0, 150)
    rows = np.stack([np.r_[base, 150:175], np.r_[base, 175:200], np.r_[base, 150:175]])
    m = eg.train_group(X, y, rows, np.zeros(3, dtype=np.int64), [11], 12, dev, epochs=3)
    p = eg.predict_proba(m, X, np.arange(150, 200))
    assert float((p[2] - p[0]).abs().max()) == 0.0  # A/A exact
    assert float((p[1] - p[0]).abs().max()) > 0.0  # arms differ only in their extra rows


def test_run_fold_bookkeeping_with_stub(monkeypatch):
    import torch
    from models import eegnet_group as eg
    meta = synth_meta(1)
    f = ds.folds(meta)[0]
    trials = np.flatnonzero(np.asarray(meta["subject"]) == 1)
    loc = tr.Local(trials)
    X = torch.zeros((len(geo.OFFSETS) * 400, 2, 4))
    y01 = torch.as_tensor(np.tile(np.asarray(meta["label"])[trials] - 1, len(geo.OFFSETS)))

    def fake_train(X, y, rows, pair_of, seeds, gs, dev, **k):
        return rows

    def fake_pred(m, X, idx):
        M = m.shape[0]
        return torch.full((M, len(idx), 2), 0.5) + torch.arange(M)[:, None, None] * 0.0

    monkeypatch.setattr(eg, "train_group", fake_train)
    monkeypatch.setattr(eg, "predict_proba", fake_pred)
    out = tr.run_fold_eegnet(f, loc, X, y01, "cpu")
    for lag in geo.LAGS_B:
        assert f"p_B1_{lag}" in out and f"p_B2_{lag}" in out
    for s in geo.STRIDES:
        for a in ("G", "L", "A1", "A2", "X1", "X2"):
            assert out[f"p_A{s}_{a}"].shape == (100,)
    assert float(out["aa_B"]) == 0.0 and float(out["aa_A1000g0"]) == 0.0


# ------------------------------------------------------------------------------------------------ analysis
def _planted(curve, n=40, e0=10.0, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    den = e0 + rng.standard_normal(n)
    num = {lag: curve[lag] * den + noise * rng.standard_normal(n) for lag in geo.LAGS_B_NONZERO}
    return num, den


def test_prediction_test_recovers_planted_c5_curve():
    curve = geo.frozen_curves(ROOT)["EEGNet"]
    pred = geo.track_b_prediction(curve)
    base = {b: {lag: geo.baseline_b(b, lag) for lag in geo.LAGS_B_NONZERO} for b in geo.BASELINES_B}
    t = an.prediction_test(*_planted(curve), pred, base, geo.LAGS_B_NONZERO)
    an.holm(t["comparisons"])
    assert an.classify(t) == "SUPPORTED_STRONG"


def test_prediction_test_detects_overlap_world():
    curve = geo.frozen_curves(ROOT)["EEGNet"]
    ovl = {lag: geo.overlap_fraction(lag) for lag in [0, *geo.LAGS_B_NONZERO]}
    pred = geo.track_b_prediction(curve)
    base = {b: {lag: geo.baseline_b(b, lag) for lag in geo.LAGS_B_NONZERO} for b in geo.BASELINES_B}
    t = an.prediction_test(*_planted(ovl, seed=1), pred, base, geo.LAGS_B_NONZERO)
    an.holm(t["comparisons"])
    assert an.classify(t) == "CONTRADICTED"


def test_guard_fallback_and_signflip():
    num, den = _planted(geo.frozen_curves(ROOT)["TSLR"], seed=2)
    den[0] = 0.1
    pred = geo.track_b_prediction(geo.frozen_curves(ROOT)["TSLR"])
    base = {"EXACT": {lag: 0.0 for lag in geo.LAGS_B_NONZERO}}
    t = an.prediction_test(num, den, pred, base, geo.LAGS_B_NONZERO)
    assert not t["guard_passed"] and t["scale"] == "residual fallback"
    rng = np.random.default_rng(5)
    assert an.signflip_p(rng.standard_normal(30), B=20000) > 0.01
    assert an.signflip_p(np.ones(30), B=20000) < 1e-3
    fam = an.holm({"a": {"signflip_p": 0.01}, "b": {"signflip_p": 0.04}})
    assert abs(fam["a"]["holm_p"] - 0.02) < 1e-12 and abs(fam["b"]["holm_p"] - 0.04) < 1e-12


def test_analyse_all_end_to_end_planted():
    meta = synth_meta(12, seed=4)
    fl = ds.folds(meta)
    curves = geo.frozen_curves(ROOT)
    rng = np.random.default_rng(9)
    scores = {}
    for m in geo.MODELS:
        R = curves[m]
        fo = []
        for f in fl:
            h1 = np.isin(f["T"], f["H1"])
            base = 0.6 + 0.02 * rng.standard_normal(100)
            d = {}
            for lag in geo.LAGS_B:
                eff = 0.12 * R[lag]
                d[f"p_B1_{lag}"] = base + eff * h1
                d[f"p_B2_{lag}"] = base + eff * ~h1
            d["p_Z"] = base.copy()
            d["p_AA"] = d["p_B1_0"].copy()
            for s in geo.STRIDES:
                qa = geo.track_a_prediction(R, s)
                d[f"p_A{s}_G"], d[f"p_A{s}_L"] = base.copy(), base + 0.05
                d[f"p_A{s}_X1"], d[f"p_A{s}_X2"] = base + 0.2 * h1, base + 0.2 * ~h1
                d[f"p_A{s}_A1"], d[f"p_A{s}_A2"] = base + 0.2 * qa * h1, base + 0.2 * qa * ~h1
            for kk in list(d):
                d["c_" + kk[2:]] = (d[kk] > 0.6).astype(float)
            d["aa_B"] = np.array(0.0)
            fo.append(d)
        scores[m] = fo
    res = an.analyse_all(scores, fl, meta, curves, None)
    assert res["families"]["H3_trackB"]["classification"] == {"EEGNet": "SUPPORTED_STRONG", "TSLR": "SUPPORTED_STRONG"}
    assert res["families"]["H3A_trackA"]["classification"]["EEGNet"] in ("SUPPORTED_STRONG", "SUPPORTED_RELATIVE")
    assert res["verdict"] == "PREDICTIVE_TRANSFER_SUPPORTED"
    e0 = res["descriptive"]["EEGNet"]["TrackB"][0]["E"]["p"]["mean"]
    assert abs(e0 - 12.0) < 1e-9
