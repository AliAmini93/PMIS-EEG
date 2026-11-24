"""R12-X design / analysis / ENFORCED-guard tests on synthetic data only (CPU; no EEG, no GPU)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from evaluation import r12x_analysis as an  # noqa: E402
from experiments.pmis_r12x import design as ds  # noqa: E402
from experiments.pmis_r12x import execute as ex  # noqa: E402
from experiments.pmis_r12x import guards as gu  # noqa: E402


def synth_meta(n_part=3, seed=0):
    rng = np.random.default_rng(seed)
    P, R, Q, L = [], [], [], []
    for p in range(1, n_part + 1):
        for r in ("R1", "R2", "R3", "R4", "R5", "R6"):
            lab = rng.permutation([1] * 20 + [2] * 20)
            P += [p] * 40
            R += [r] * 40
            Q += list(range(40))
            L += lab.tolist()
    return {"participant": np.array(P), "run": np.array(R), "pos": np.array(Q), "label": np.array(L)}


def build(meta):
    out = []
    for p in np.unique(meta["participant"]):
        m = meta["participant"] == p
        out.append(ds.build_participant(int(p), meta["run"][m], meta["pos"][m], meta["label"][m], np.flatnonzero(m)))
    return out


def test_design_firewall_and_commitment():
    meta = synth_meta()
    D = build(meta)
    for d in D:
        C = set(d["C"].tolist())
        assert all(not (set(v.tolist()) & C) for v in d["arms"].values())
        assert set(meta["run"][d["C"]]) == {"R6"} and set(meta["run"][d["B"]]) == {"R5"}
    assert ds.commitment(D) == ds.commitment(build(synth_meta()))
    assert ds.seeds(1, "selection", 0) != ds.seeds(1, "final", 0)


def test_design_rejects_c_leak():
    d = build(synth_meta(1))[0]
    lab = synth_meta(1)["label"]
    d["arms"]["F"] = np.concatenate([d["arms"]["F"][:-1], d["C"][:1]])
    with pytest.raises(AssertionError):
        ds.check(d, lab)


def _mats(n, flip=True, gap=2.0, reverse=False, seed=1):
    rng = np.random.default_rng(seed)
    BBP = 70 + rng.normal(0, 3, (n, 3))
    BBP[:, 0] += 3                         # clean benchmark prefers TS+LR
    BBL = BBP.copy()
    if flip:
        BBL[:, 2] += 12                    # contamination inflates ShallowConvNet
    BC = 70 + rng.normal(0, 4, (n, 3))
    BC[:, 0] += (-gap if reverse else gap)
    return BBP, BBL, BC


def test_analysis_classes():
    n = 86
    blk = np.arange(n) % 2
    r = an.analyse(*_mats(n), blk, n_boot=400)
    assert r["E1"]["class"] == "FLIP_AND_REGRET" and r["E1"]["flip_fraction"] == 1.0
    assert abs(r["E1"]["regret"] - r["E2"]["regret"]) < 1e-9
    r0 = an.analyse(*_mats(n, flip=False), blk, n_boot=400)
    assert r0["E1"]["class"] == "NO_FLIP" and r0["E1"]["regret"] == 0.0 and r0["E1"]["mass_at_zero"] == 1.0
    rr = an.analyse(*_mats(n, reverse=True, gap=4.0), blk, n_boot=400)
    assert rr["E1"]["class"] == "REVERSE_REGRET"


def test_loo_excludes_own_participant():
    BBP, BBL, BC = _mats(10, flip=False)
    BBL[0, 2] = 1e6                        # one participant alone would flip the cohort choice
    e1, e2 = an.e1(BBP, BBL, BC), an.e2(BBP, BBL, BC)
    assert e2["flip"] and e1["w_contam"][0] == "TSLR"   # participant 0's own data cannot drive its own selection


def test_wait_safe_has_no_timeout_and_waits_for_memory():
    seq = [{"MemAvailable": 1 << 30, "psi_full_avg10": 0.0}] * 25 + [{"MemAvailable": 16 << 30, "psi_full_avg10": 0.0}]
    calls = []
    st = lambda: seq[min(len(calls), len(seq) - 1)]  # noqa: E731

    def sleep(s):
        calls.append(s)
    gu.wait_safe(None, 0, sleep=sleep, state=st, poll_s=45.0)
    assert len(calls) == 25 and set(calls) == {45.0}   # waited through 25 unsafe polls, then proceeded


def test_wait_safe_respects_memory_pressure():
    seq = [{"MemAvailable": 16 << 30, "psi_full_avg10": 30.0}] * 3 + [{"MemAvailable": 16 << 30, "psi_full_avg10": 0.5}]
    calls = []
    gu.wait_safe(None, 0, sleep=lambda s: calls.append(s), state=lambda: seq[min(len(calls), 3)])
    assert len(calls) == 3


def test_oom_retry_then_genuine_stop():
    class Boom(Exception):
        pass
    n = {"calls": 0, "recover": 0}

    def fn():
        n["calls"] += 1
        if n["calls"] < 3:
            raise Boom()
        return "ok"
    assert gu.with_oom_retry(fn, lambda e: isinstance(e, Boom), lambda: n.__setitem__("recover", n["recover"] + 1)) == "ok"
    assert n == {"calls": 3, "recover": 2}

    def always():
        raise Boom()
    with pytest.raises(gu.ResourceStop):
        gu.with_oom_retry(always, lambda e: isinstance(e, Boom), lambda: None)
    with pytest.raises(ValueError):                    # non-OOM errors are never retried
        gu.with_oom_retry(lambda: (_ for _ in ()).throw(ValueError("x")), lambda e: isinstance(e, Boom), lambda: None)


def test_unit_meter_records_without_stopping():
    with gu.UnitMeter() as m:
        time.sleep(0.05)
    assert m.rec["wall_s"] >= 0.05 and m.rec["host_rss_bytes"] > 0


torch = pytest.importorskip("torch")


def test_aa_gate_fails_closed(monkeypatch):
    import models.deep_group as dg
    d = build(synth_meta(1))[0]

    def fake(cls, X, y, rows, pair_of, seeds, gseed, dev, score_idx, val_idx, epochs):
        p = np.full((rows.shape[0], len(score_idx)), 0.7)
        if fake.perturb:
            p[2] += 1e-9                    # AA_L differs from S_L by 1e-9
        return {"p_final": p, "p_es": p.copy(), "es_epoch": np.zeros(rows.shape[0], int)}
    monkeypatch.setattr(dg, "train_generic", fake)
    fake.perturb = False
    assert set(ex.run_deep_unit(None, d, "selection", 0, None, None, None)) >= {"final_p_S_L", "final_p_AA_L"}
    fake.perturb = True
    with pytest.raises(ex.AAStop):
        ex.run_deep_unit(None, d, "selection", 0, None, None, None)


def test_real_training_cpu_runs_to_completion_without_time_limit():
    """A real (CPU, synthetic) ShallowConvNet group unit runs its full recipe; outputs are finite (A/A exactness is a
    GPU-determinism property; on CPU the gate may fail closed, which is tested separately)."""
    from models import deep_group as dg
    meta = synth_meta(1)
    d = build(meta)[0]
    X = torch.randn(240, 20, 160, generator=torch.Generator().manual_seed(0))
    y01 = torch.as_tensor(meta["label"] - 1)
    try:
        out = ex.run_deep_unit(dg.GroupShallow, d, "final", 0, X, y01, torch.device("cpu"), epochs=2)
        assert all(np.all(np.isfinite(v)) for v in out.values())
    except ex.AAStop:
        pass
