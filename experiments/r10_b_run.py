#!/usr/bin/env python3
"""Lee2019 fixed-density and realistic sliding-window transfer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity
from experiments.gpu_resources import select_device

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from evaluation import r10b_analysis as analysis
from experiments.pmis_r10b import design, geometry, train  # noqa: E402

PROTOCOL_ID = "R10-B-SLIDING-WINDOW-C5-PREDICTION-LEE2019-v1"
PROTOCOL = CONFIG_ROOT / "R10_B_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r10_b"
PUBLIC = ROOT / "results" / "R10_B_results.json"
FEATURES = {"c4": ("data/r6_c4/lee2019_win_covs_8_30.npy", "data/r6_c4/lee2019_win_eegnet_4_40_128hz.npy"),
            "c5": ("data/r7_c5/lee2019_shift_covs_8_30.npy", "data/r7_c5/lee2019_shift_eegnet_4_40_128hz.npy"),
            "b": ("data/r10_b/lee2019_r10b_covs_8_30.npy", "data/r10_b/lee2019_r10b_eegnet_4_40_128hz.npy")}
META = "data/r6_c3/lee2019_meta.npz"
GPU_CAP_GB, GPU_MIN_FREE_GB, MAX_WAIT_S, OOM_RETRIES = 6.0, 6.0, 6 * 3600, 7
TSLR_WORKERS = 4


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg, name="R10_B_run.log"):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / name, "a") as fh:
        fh.write(line + "\n")


def load_meta():
    z = np.load(ROOT / META)
    return {k: z[k] for k in ("subject", "session", "run", "pos", "label", "onset")}


def plan_summary(meta):
    fl = design.folds(meta)
    for f in fl:
        design.check_fold(f, meta)
    lab = np.asarray(meta["label"])
    geo_rec = {}
    f0 = fl[0]
    tb = design.track_b_arms(f0)
    for a in ("B1_0", "B1_62", "B1_250", "B1_1000", "B1_2000", "Z"):
        geo_rec[f"TrackB_{a}"] = design.arm_geometry(tb[a], f0["H1"], f0["T"], lab)
    for s in geometry.STRIDES:
        ta = design.track_a_arms(f0, s)
        for a in ("G", "L", "A1", "X1"):
            geo_rec[f"TrackA_{s}_{a}"] = design.arm_geometry(ta[a], f0["H1"] if a in ("A1", "X1") else f0["T"], f0["T"], lab)
    n_rows_b = 16 * 350
    n_rows_a = sum(sum(design.track_a_arms(f0, s)[a].size for g in design.track_a_groups(s) for a in g) for s in geometry.STRIDES)
    return fl, {"folds": len(fl), "participants": int(np.unique(meta["subject"]).size),
                "commitment": design.commitment(fl, meta),
                "geometry_first_fold_H1_anchors_or_all_anchors": geo_rec,
                "eegnet_row_arms_per_fold": {"trackB": n_rows_b, "trackA": n_rows_a},
                "c0_fits_per_participant": 10, "c0_train_rows": 1600}


PRE_REG = {
    "question": "Does the frozen R7-C5 (plus C4/C4b at 2 s) single-copy temporal-response curve predict the relative "
                "own-trial exposure effects of sliding-window training sets at benchmark level?",
    "evidence_status": "Lee2019 participants and recordings are those of C3/C4/C5 (not an independent cohort). Track A / "
                       "Track B / C0 arms were never fitted. This is a prospective prediction of new pipeline endpoints on "
                       "known participants, not a replication on independent participants.",
    "primary_scale": "correct-class probability points (C5's scale) for H2-H4; BA for H1; BA reported for everything",
    "H1": "Track A realistic contamination: L_s - G_s (BA) > 0 for each stride and model (Holm over 6). Context only; "
          "L and G differ in rows and held-out-run support (declared).",
    "H2": "Track B matched overlap effect: E(lag) - E(2000) for lag in 62..1000 and E(2000) (same trial, no shared "
          "sample) per model (p scale; Holm over 12). OWN vs OTHER match rows, class composition, held-out-run "
          "provenance, number of distinct held-out source trials and window offset; they do not match content.",
    "H3_trackB_primary": "Per model, participant retention rho(lag) = E(lag)/E(0) vs the frozen curve R(lag) (functional-free: "
                         "one window per exposed anchor). Loss = sum over 6 lags of squared error; test D = L(baseline) - "
                         "L(C5) > 0 vs OVL (shared-sample fraction), BIN (any overlap = copy) and EXACT (only copies leak); "
                         "Holm over 6. Accuracy: 90 % CI of the cohort error within +/-0.10 at every lag (TOST, alpha 0.05).",
    "H3A_trackA_secondary": "Per model, q(s) = E_A(s)/E_X(s) vs the frozen PRIMARY additive functional sum R/(1+sum R) "
                            "(first-order influence: effects of added rows sum) over lags s, 2s, ..., 2000; baselines OVL, "
                            "CNT (count / density), BIN, EXACT in the same functional; Holm over 8. Nearest-lag and "
                            "saturating functionals are reported descriptively and never used for a decision.",
    "discriminability_rule": "a comparison counts toward a classification only if prediction and baseline differ by >= 0.05 "
                             "at some condition (computed from frozen predictions only).",
    "classification": "per model and track: CONTRADICTED if a discriminating baseline is significantly better; "
                      "SUPPORTED_STRONG if C5 beats every discriminating baseline (Holm) and is within +/-0.10 everywhere; "
                      "SUPPORTED_RELATIVE if it beats every discriminating baseline only; PARTIAL if some; else NOT_SUPPORTED.",
    "overall": "PREDICTIVE_TRANSFER_SUPPORTED if Track B EEGNet SUPPORTED_STRONG, Track B TS+LR at least PARTIAL and Track A "
               "EEGNet at least SUPPORTED_RELATIVE; PARTIAL_TRANSFER if Track B EEGNet at least SUPPORTED_RELATIVE; "
               "else NO_PREDICTIVE_TRANSFER.",
    "H4": "Curve swap: each model's own C5 curve predicts its Track B retention better than the other model's curve (2 tests); "
          "EEGNet - TS+LR retention at 62..1000 ms (5 tests; C5 predicts negative). Holm over 7.",
    "C0": "Conventional contextual benchmark (stride 500): random window-level vs trial-grouped 5-fold CV per participant; "
          "window-level, trial-aggregated and anchor-only BA; descriptive.",
    "guards": "retention ratios only if every participant's denominator exceeds 0.5 points (else declared fallback).",
    "inference": "54 participants; t95; two-sided MC sign-flip 1e6 (seed 10399); Holm within families.",
    "forbidden_after_any_fit": "changing a lag, stride, arm, functional, baseline, threshold, family, scale or participant set.",
}


def cmd_plan_only():
    meta = load_meta()
    _, ps = plan_summary(meta)
    print(json.dumps(ps, indent=1, default=float))


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    return proto


def load_features(proto, kind):
    """kind 'eeg' or 'cov': SHA-256 verified before load; returns {source: memmap}."""
    out = {}
    for src, (cov, eeg) in FEATURES.items():
        p = eeg if kind == "eeg" else cov
        g3._stop(reference_matches(sha(ROOT / p), proto["data_sha256"][p]), f"{p} differs from its frozen SHA-256 (checked before load)")
        out[src] = np.load(ROOT / p, mmap_mode="r")
    return out


def wait_gpu(torch, tag):
    t0, last = time.time(), 0.0
    while True:
        torch.cuda.empty_cache()
        free = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30
        st = g3.resource_state()
        if free >= GPU_MIN_FREE_GB and st["mem_available_mb"] >= 4096 and not (st["psi_full_avg10"] > 5.0):
            return
        g3._stop(time.time() - t0 < MAX_WAIT_S, "GPU/host unavailable for 6 h (not preempting)", "STOPPED_RESOURCE")
        if time.time() - last > 600:
            log(f"{tag} waiting for resources: gpu free {free:.1f} GB", f"R10_B_{tag}.log")
            last = time.time()
        time.sleep(60)


def participant_units(meta, fl, k, n):
    parts = sorted({f["subject"] for f in fl})
    mine = [u for i, u in enumerate(parts) if i % n == k]
    return mine, {u: [f for f in fl if f["subject"] == u] for u in mine}


def _eeg_local(feats, trials):
    x = train.gather_local(feats, trials)
    return (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)  # C5 per-window channel z-score


def cmd_execute_eegnet(k, n):
    tag = f"EEGNet_shard{k}of{n}"
    try:
        import torch
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R10_B_{tag}.log")
        os.nice(10)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dev = select_device()
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / torch.cuda.mem_get_info()[1])
        meta = load_meta()
        g3._stop(reference_matches(sha(ROOT / META), proto["data_sha256"][META]), "metadata differ")
        fl, ps = plan_summary(meta)
        g3._stop(reference_matches(ps["commitment"], proto["plan"]["commitment"]), "plan commitment differs")
        feats = load_features(proto, "eeg")
        lab = np.asarray(meta["label"])
        fdir = PRIVATE / "folds" / "EEGNet"
        fdir.mkdir(parents=True, exist_ok=True)
        mine, by = participant_units(meta, fl, k, n)
        t0 = time.time()
        for j, u in enumerate(mine):
            trials = np.flatnonzero(np.asarray(meta["subject"]) == u)
            loc = train.Local(trials)
            X = torch.as_tensor(_eeg_local(feats, trials), device=dev)
            y01 = torch.as_tensor(np.tile(lab[trials] - 1, len(geometry.OFFSETS)).astype(np.int64), device=dev)
            jobs = [("fold", f) for f in by[u]] + [("c0", None)]
            for kind, f in jobs:
                fp = fdir / (f"f_{u}_{f['run']}.npz" if kind == "fold" else f"c0_{u}.npz")
                if fp.exists() and str(np.load(fp)["commitment"]) == ps["commitment"]:
                    continue
                res = None
                for attempt in range(OOM_RETRIES):
                    wait_gpu(torch, tag)
                    try:
                        res = train.run_fold_eegnet(f, loc, X, y01, dev) if kind == "fold" else \
                            train.run_c0_eegnet(design.c0_splits(meta, u), loc, X, y01, dev)
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        log(f"{tag} {fp.name}: out of memory (attempt {attempt + 1}); identical recompute", f"R10_B_{tag}.log")
                        time.sleep(120)
                g3._stop(res is not None, "out of memory 7 times", "STOPPED_RESOURCE")
                for key in [x for x in res if x.startswith("aa_")]:
                    g3._stop(float(res[key]) == 0.0, f"A/A not exact ({key} {float(res[key]):.2e}, {fp.name})")
                np.savez(fp, commitment=np.array(ps["commitment"]), **res)
            del X, y01
            log(f"{tag} participant {j + 1}/{len(mine)} (subject {u}) {time.time() - t0:.0f} s", f"R10_B_{tag}.log")
        log(f"{tag} DONE", f"R10_B_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R10_B_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R10_B_{tag}.log")
        return 2


_G: dict = {}


def _tslr_unit(args):
    u, folds_u, commit = args
    meta, feats = _G["meta"], _G["feats"]
    trials = np.flatnonzero(np.asarray(meta["subject"]) == u)
    loc = train.Local(trials)
    covs = train.gather_local(feats, trials).astype(np.float64)
    y = np.tile(np.asarray(meta["label"])[trials], len(geometry.OFFSETS))
    fdir = PRIVATE / "folds" / "TSLR"
    for f in folds_u:
        fp = fdir / f"f_{u}_{f['run']}.npz"
        if fp.exists() and str(np.load(fp)["commitment"]) == commit:
            continue
        res = train.run_fold_tslr(f, loc, covs, y)
        if float(res["aa_B"]) != 0.0:
            raise g3.G3Stop("BLOCKED", f"TS+LR A/A not exact ({fp.name})")
        np.savez(fp, commitment=np.array(commit), **res)
    fp = fdir / f"c0_{u}.npz"
    if not (fp.exists() and str(np.load(fp)["commitment"]) == commit):
        np.savez(fp, commitment=np.array(commit), **train.run_c0_tslr(design.c0_splits(meta, u), loc, covs, y))
    return u


def cmd_execute_tslr(k, n):
    tag = f"TSLR_shard{k}of{n}"
    try:
        import multiprocessing as mp
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R10_B_{tag}.log")
        os.nice(10)
        meta = load_meta()
        fl, ps = plan_summary(meta)
        g3._stop(reference_matches(ps["commitment"], proto["plan"]["commitment"]), "plan commitment differs")
        _G["meta"], _G["feats"] = meta, load_features(proto, "cov")
        (PRIVATE / "folds" / "TSLR").mkdir(parents=True, exist_ok=True)
        mine, by = participant_units(meta, fl, k, n)
        t0 = time.time()
        with mp.get_context("fork").Pool(TSLR_WORKERS) as pool:
            for j, u in enumerate(pool.imap_unordered(_tslr_unit, [(u, by[u], ps["commitment"]) for u in mine])):
                log(f"{tag} participant {j + 1}/{len(mine)} (subject {u}) {time.time() - t0:.0f} s", f"R10_B_{tag}.log")
        log(f"{tag} DONE", f"R10_B_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R10_B_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R10_B_{tag}.log")
        return 2


def cmd_analyse():
    out = {"stage": "R10-B", "protocol_id": PROTOCOL_ID}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        meta = load_meta()
        fl, ps = plan_summary(meta)
        g3._stop(reference_matches(ps["commitment"], proto["plan"]["commitment"]), "plan commitment differs")
        scores, c0, book = {}, {}, {}
        for m in geometry.MODELS:
            fdir = PRIVATE / "folds" / m
            fo, miss = [], 0
            for f in fl:
                fp = fdir / f"f_{f['subject']}_{f['run']}.npz"
                if not fp.exists() or str(np.load(fp)["commitment"]) != ps["commitment"]:
                    miss += 1
                    continue
                fo.append(dict(np.load(fp)))
            cz, cmiss = {}, 0
            for u in sorted({f["subject"] for f in fl}):
                fp = fdir / f"c0_{u}.npz"
                if not fp.exists() or str(np.load(fp)["commitment"]) != ps["commitment"]:
                    cmiss += 1
                    continue
                cz[u] = dict(np.load(fp))
            aa = [float(v) for z in fo for kk, v in z.items() if kk.startswith("aa_")]
            book[m] = {"folds_missing": miss, "c0_missing": cmiss, "aa_checks": len(aa), "aa_max_abs": max(aa) if aa else None}
            g3._stop(miss == 0 and cmiss == 0, f"{m}: incomplete ({miss} folds, {cmiss} C0 units missing): no analysis")
            g3._stop(max(aa) == 0.0, f"{m}: A/A not exact")
            scores[m], c0[m] = fo, cz
        out["bookkeeping"] = book
        curves = {m: {int(k): v for k, v in proto["frozen_curves"][m].items()} for m in geometry.MODELS}
        g3._stop(curves == geometry.frozen_curves(ROOT), "frozen curves differ from their sources")
        out["results"] = analysis.analyse_all(scores, fl, meta, curves, c0)
        out["status"] = "COMPLETED"
        out["plan"] = {k: ps[k] for k in ("folds", "participants", "commitment")}
        code = 0
    except g3.G3Stop as e:
        out.update({"status": e.status, "stop_message": e.message})
        code = 2
    except Exception as e:
        out.update({"status": "BLOCKED", "stop_message": f"{type(e).__name__}: {e}"})
        log("analyse STOP BLOCKED: " + traceback.format_exc())
        code = 2
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(out["status"])
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mx = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute", "--analyse"):
        mx.add_argument(flag, action="store_true")
    ap.add_argument("--model", default="EEGNet", choices=geometry.MODELS)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.plan_only:
        sys.exit(cmd_plan_only())
    if a.execute:
        kk, nn = (int(v) for v in a.shard.split("/"))
        sys.exit(cmd_execute_eegnet(kk, nn) if a.model == "EEGNet" else cmd_execute_tslr(kk, nn))
    if a.analyse:
        sys.exit(cmd_analyse())
