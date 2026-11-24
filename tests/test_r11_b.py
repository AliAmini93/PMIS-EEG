"""R11-B synthetic tests: dose design invariants on a synthetic C6-like plan, frozen predictions from the committed G0
aggregate, decision rules on planted outcomes (prediction test, classification, reversal, crossing, ceiling), and an
end-to-end analysis on synthetic scores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from evaluation import r11b_analysis as an  # noqa: E402
from experiments.pmis_r11b import design as ds  # noqa: E402
from experiments.pmis_r11b import predict as pr  # noqa: E402


def synth_plan(n_subj=4, seed=0):
    """A C6-like plan: per participant 4 runs x 100 trials; T = held-out run, 60 val + 240 train from the rest."""
    rng = np.random.default_rng(seed)
    labels, groups, subj = [], [], []
    for s in range(1, n_subj + 1):
        base_id = len(labels)
        for r in range(4):
            labels += list(rng.permutation(np.repeat([1, 2], 50)))
            subj += [s] * 100
        lab = np.array(labels)
        ids = np.arange(base_id, base_id + 400)
        for r in range(4):
            T = ids[r * 100:(r + 1) * 100]
            rest = np.setdiff1d(ids, T)
            val = np.concatenate([rng.choice(rest[lab[rest] == c], 30, replace=False) for c in (1, 2)])
            train = np.setdiff1d(rest, val)
            groups.append({"subject": s, "fold": r, "T": T.tolist(), "train": train.tolist(), "val": sorted(val.tolist())})
    return {"groups": groups}, np.array(labels), np.array(subj)


def test_design_invariants_and_nesting():
    plan, lab, _ = synth_plan()
    fl = ds.folds_from_c6(plan, lab)
    for f in fl:
        ds.check_fold(f, lab)
        assert all(f["arms"][a].size == 340 for a in ds.GROUP_A) and f["arms"]["C"].size == 240
        # L_delta and P differ in exactly 100 delta extra rows
        for d in ds.DOSES:
            diff = np.setdiff1d(f["arms"][ds.DOSE_TAG[d]], f["arms"]["P"])
            assert diff.size == int(round(100 * d)) and np.all(np.isin(diff, f["T"]))
    assert ds.commitment(fl) == ds.commitment(ds.folds_from_c6(plan, lab))


def test_check_fold_catches_validation_leak():
    plan, lab, _ = synth_plan(n_subj=1)
    f = ds.folds_from_c6(plan, lab)[0]
    f["arms"]["P"] = f["arms"]["P"].copy()
    f["arms"]["P"][-1] = f["val"][0]
    with pytest.raises(AssertionError):
        ds.check_fold(f, lab)


def test_frozen_predictions_in_release_protocol():
    p = pr.protocol_predictions()
    se = p["pairs"]["Shallow-EEGNet/es"]
    assert abs(se["dc"] - 6.996) < 1e-3 and abs(se["gap_prior"] + 3.091) < 1e-3 and se["crossing_test_eligible"]
    assert abs(se["D_pred"]["1.0"] - se["dc"]) < 1e-12 and se["discriminating_vs_CEILING"]
    assert not p["pairs"]["Shallow-EEGNet/final"]["crossing_test_eligible"]
    assert not p["pairs"]["EEGNet-TSLR/es"]["discriminating_vs_ZERO"]
    assert p["pairs"]["Shallow-TSLR/final"]["crossing_test_eligible"]


def test_reversal_and_crossing_rules():
    s = lambda m, lo, hi: {"mean": m, "ci95": [lo, hi]}  # noqa: E731
    assert an.reversal_class(s(-2, -3, -1), s(3, 2, 4), True) == "SUPPORTED_REVERSAL"
    assert an.reversal_class(s(-2, -3, 0.5), s(3, 2, 4), True) == "SUPPORTED_CROSSING"
    assert an.reversal_class(s(-2, -3, -1), s(3, 2, 4), False) == "DESCRIPTIVE_REVERSAL"
    assert an.reversal_class(s(-2, -3, -1), s(-1, -2, 0), True) == "NO_REVERSAL"
    n = 30
    rng = np.random.default_rng(1)
    G = {k: v + 0.01 * rng.standard_normal(n) for k, v in (("P", -3.0), ("L20", -1.6), ("L40", -0.2), ("L60", 1.2), ("L100", 4.0))}
    x = an.crossing(G, [0.3, 0.6], n)
    assert x["class"] == "CROSSING_AS_PREDICTED" and abs(x["delta_star_observed"] - (0.4 + 0.2 * 0.2 / 1.4)) < 1e-2
    G2 = {k: np.full(n, v) for k, v in (("P", 1.0), ("L20", 2.0), ("L40", 3.0), ("L60", 4.0), ("L100", 5.0))}
    assert an.crossing(G2, [0.3, 0.6], n)["class"] == "NOT_EVALUABLE_PREMISE"
    G3 = {k: np.full(n, v) for k, v in (("P", -5.0), ("L20", -4.0), ("L40", -3.0), ("L60", -2.0), ("L100", -1.0))}
    assert an.crossing(G3, [0.3, 0.6], n)["class"] == "NO_CROSSING_BY_1.0"


def test_ceiling_correct():
    p = np.array([0.2, 0.7, 0.4, 0.9])
    e = np.array([True, False, True, False])
    assert np.array_equal(an.ceiling_correct(p, e), [1.0, 1.0, 1.0, 1.0])
    assert np.array_equal(an.ceiling_correct(p, ~e), [0.0, 1.0, 0.0, 1.0])


def test_prediction_test_classifies_planted_outcomes():
    n = 54
    rng = np.random.default_rng(3)
    pred = {d: 7.0 * d for d in ds.DOSES}
    good = {d: pred[d] + 0.3 * rng.standard_normal(n) for d in ds.DOSES}
    t = an.prediction_test(good, pred, {"ZERO": {d: 0.0 for d in ds.DOSES}}, {"ZERO": True})
    an.holm(t["comparisons"])
    assert an.classify(t) == "SUPPORTED_STRONG"
    zero = {d: 0.3 * rng.standard_normal(n) for d in ds.DOSES}
    t = an.prediction_test(zero, pred, {"ZERO": {d: 0.0 for d in ds.DOSES}}, {"ZERO": True})
    an.holm(t["comparisons"])
    assert an.classify(t) == "CONTRADICTED"
    half = {d: 3.5 * d + 0.3 * rng.standard_normal(n) for d in ds.DOSES}
    t = an.prediction_test(half, pred, {"ZERO": {d: 0.0 for d in ds.DOSES}}, {"ZERO": True})
    an.holm(t["comparisons"])
    assert an.classify(t) in ("NOT_SUPPORTED", "PARTIAL", "SUPPORTED_RELATIVE")


def test_end_to_end_synthetic():
    plan, lab, subj = synth_plan(n_subj=6, seed=2)
    fl = ds.folds_from_c6(plan, lab)
    tr = an.Trials(fl, lab, subj)
    preds = pr.protocol_predictions()
    rng = np.random.default_rng(5)
    base = {m: 0.55 + 0.1 * rng.random(tr.T.size) for m in pr.MODELS}
    S = {}
    for m in pr.MODELS:
        ros = ("exact",) if m == "TSLR" else ("final", "es")
        S[m] = {}
        for ro in ros:
            arms = {"C": base[m].copy(), "P": base[m].copy()}
            for d in ds.DOSES:
                v = base[m].copy()
                v[tr.exposed[d]] = 0.95
                arms[ds.DOSE_TAG[d]] = v
            v = base[m].copy()
            v[tr.h2] = 0.95
            arms["L40c"] = v
            S[m][ro] = arms
    out = an.analyse_replicate(S, tr, preds)
    assert set(out["classification"]) == {f"{a}-{b}/{ro}" for a, b in pr.PAIRS for ro in pr.READOUTS}
    assert out["verdict"]["verdict"] in an.RANK
    assert an.combine(out, out)["seed_robust"]
