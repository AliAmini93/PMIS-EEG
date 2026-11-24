"""Dry run of the persistent worker's control flow on CPU with synthetic features (no EEG, no GPU):
real TS+LR units on synthetic SPD covariances; deep stage with a deterministic stub trainer and a
CPU device. Checks unit completeness, checkpoint resumability (second pass runs nothing) and usage logging."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")




def _synthetic(run, tmp, parts=(2, 61)):
    rng = np.random.default_rng(3)
    P, R, Q, L = [], [], [], []
    for p in parts:
        for r in ("R1", "R2", "R3", "R4", "R5", "R6"):
            P += [p] * 40
            R += [r] * 40
            Q += list(range(40))
            L += rng.permutation([1] * 20 + [2] * 20).tolist()
    f = tmp / "features"
    f.mkdir(parents=True)
    np.savez(f / "meta.npz", participant=np.array(P), run=np.array(R), pos=np.array(Q), label=np.array(L))
    n = len(P)
    sha = {}
    for w in run.TSLR_WINDOWS:
        c = 3 if w.startswith("eog") else 20
        A = rng.standard_normal((n, c, 2 * c))
        np.save(f / f"cov_{w}.npy", A @ A.transpose(0, 2, 1) / (2 * c) + np.eye(c) * 0.1)
    for w, (c, t) in run.WINDOW_DEEP.items():
        np.save(f / f"eeg_{w}.npy", rng.standard_normal((n, c, t)).astype(np.float32))
    for p in f.iterdir():
        sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return sha




def test_worker_stages_dry_run(tmp_path, monkeypatch):
    from experiments import r12x_run as run

    monkeypatch.setattr(run, "PRIVATE", tmp_path)
    monkeypatch.setattr(run, "FEAT", tmp_path / "features")
    monkeypatch.setattr(run, "PUBLIC", tmp_path / "results.json")
    sha = _synthetic(run, tmp_path)
    meta = run.load_meta()
    D = run.designs(meta)
    commit = run.ds.commitment(D)
    proto = {"feature_sha256": sha, "design_commitment": commit}
    y12 = meta["label"].astype(np.int64)
    logs = []
    assert run.tslr_stage(proto, D, y12, commit, logs.append) == len(D) * 2 * len(run.TSLR_WINDOWS)
    assert run.tslr_stage(proto, D, y12, commit, logs.append) == 0               # resumable: nothing re-run

    import models.deep_group as dg

    def fake_train(cls, X, y, rows, pair_of, seeds, gseed, dev, score_idx, val_idx, epochs):
        g = np.random.default_rng([int(seeds[0]), int(gseed), rows.shape[0], X.shape[2]])
        p = np.tile(g.random(len(score_idx)), (rows.shape[0], 1))                  # A/A arms identical
        return {"p_final": p, "p_es": p.copy(), "es_epoch": np.full(rows.shape[0], 7)}
    monkeypatch.setattr(dg, "train_generic", fake_train)
    monkeypatch.setattr(run.gpucoord, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(run.gu, "wait_safe", lambda *a, **k: 0.0)

    class _Meter:
        def __init__(self, t):
            self.rec = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.rec = {"wall_s": 0.01, "host_rss_bytes": 1, "gpu_peak_allocated_bytes": 0, "gpu_peak_reserved_bytes": 0}
            return False
    monkeypatch.setattr(run.gu, "UnitMeter", _Meter)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    n_deep = len(run.DEEP) * len(run.WINDOW_DEEP) * len(D) * 2 * run.ds.K_SEEDS
    assert run.deep_stage(proto, D, y12, commit, logs.append) == n_deep
    assert run.deep_stage(proto, D, y12, commit, logs.append) == 0                # resumable
    usage = (tmp_path / "unit_usage.jsonl").read_text().splitlines()
    assert len(usage) == n_deep and json.loads(usage[0])["model"] == "EEGNet"
    files = list((tmp_path / "units").rglob("u_*.npz"))
    assert len(files) == n_deep + len(D) * 2 * len(run.TSLR_WINDOWS)
