#!/usr/bin/env python3
"""Lee2019 controlled duplicate doses and frozen ranking-forecast analysis."""

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
from experiments.pmis_r6 import lee2019  # noqa: E402
from experiments.pmis_r7 import c6  # noqa: E402
from evaluation import r11b_analysis as analysis
from experiments.pmis_r11b import design, predict, train  # noqa: E402

PROTOCOL_ID = "R11-B-LEE2019-DOSE-RANKING-PREDICTION-v1"
PROTOCOL = CONFIG_ROOT / "R11_LEE_RANKING_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r11_b"
PUBLIC = ROOT / "results" / "R11_B_results.json"
META = "data/r6_c3/lee2019_meta.npz"
EEG = "data/r6_c3/lee2019_eegnet_4_40_128hz.npy"
COV = "data/r6_c3/lee2019_covs_8_30.npy"
GPU_CAP_GB, GPU_MIN_FREE_GB, MAX_WAIT_S, OOM_RETRIES = 6.0, 6.0, 6 * 3600, 7
TSLR_WORKERS = 4
DEEP = ("EEGNet", "Shallow")
REPS = ("R1", "R2")
GPU_BUDGET_H = 6.0   # hard cap on summed GPU process-hours for all deep parts (frozen)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg, name="R11_B_run.log"):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / name, "a") as fh:
        fh.write(line + "\n")


def load_meta():
    z = np.load(ROOT / META)
    return {k: z[k] for k in ("subject", "session", "run", "pos", "label")}


def build(meta):
    lab = np.asarray(meta["label"]).astype(np.int64)
    tasks = lee2019.build_tasks(meta)
    c6p = c6.build_plan(tasks, lab)
    c6_commit = c6.commitment(c6p)
    historical = load_protocol(CONFIG_ROOT / "R7_C6_PROTOCOL.json")["plan"]["commitment"]
    g3._stop(reference_matches(c6_commit, historical), "C6 plan commitment differs from the frozen R7-C6 value")
    fl = design.folds_from_c6(c6p, lab)
    for f in fl:
        design.check_fold(f, lab)
    return fl, c6_commit


def plan_summary(meta):
    fl, c6_commit = build(meta)
    lab = np.asarray(meta["label"])
    f0 = fl[0]
    return fl, {"folds": len(fl), "participants": int(np.unique(meta["subject"]).size),
                "commitment": design.commitment(fl), "c6_commitment": c6_commit,
                "doses": list(design.DOSES), "n_per_class": {str(k): v for k, v in design.N_PER_CLASS.items()},
                "arm_rows": {a: int(f0["arms"][a].size) for a in design.GROUP_A + design.GROUP_B},
                "arm_class_counts_first_fold": {a: [int(np.sum(lab[f0["arms"][a]] == c)) for c in design.LABELS]
                                                for a in design.GROUP_A + design.GROUP_B},
                "deep_fits_per_fold_per_model": len(design.GROUP_A) + len(design.GROUP_B),
                "tslr_fits_per_fold": len(train.TSLR_ARMS) + 1}


PRE_REG = {
    "question": "Does a frozen, quantitative single-copy sensitivity prediction (the exposed-trial BA contrast of one "
                "added exact copy, measured in R7-C6 / R6-C3a) anticipate the benchmark-level distortion of pairwise "
                "model gaps, the rank-crossing dose and model-selection regret when a controlled fraction of scored test "
                "trials has exact copies in training?",
    "evidence_status": "Lee2019 participants, folds, training / validation sets and the single-copy contrasts are those of "
                       "R6-C3 / R7-C6 (54 known participants; not an independent cohort). No R11-B arm (CLEAN, PLACEBO, LEAKED "
                       "at any dose, L40c) was ever fitted. The near-clean gaps are effectively known in advance (G0); the "
                       "prospective endpoints are the dose arms. Not an external replication; not independent of the "
                       "prediction source.",
    "dose": "delta = fraction of the 100 scored held-out-run trials whose exact copies are in training: 0.2, 0.4, 0.6, 1.0 "
            "(10, 20, 30, 50 per class; nested, class-stratified, outcome-blind). Fixed rows: every non-CLEAN arm holds 240 "
            "base + 100 extra rows (50 per class); L_delta's extra rows are 100 delta scored-trial copies plus 100 (1 - delta) "
            "base copies; PLACEBO's are 100 base copies (the delta = 0 point).",
    "primary_pair": "ShallowConvNet - EEGNet (identical tasks, base, pairs and recipe as R7-C6). Read-outs: final epoch "
                    "and minimum-validation-loss epoch from the same runs. TS+LR pairs are secondary (their single-copy "
                    "prior was measured on a 300-row base).",
    "H1_primary": "Distortion prediction D_pred(delta) = delta * dc (frozen). Loss = sum over 4 doses of squared participant "
                  "error. Test L(competitor) - L(prediction) > 0 against ZERO (D = 0) and CEILING (every exposed trial "
                  "correct for both models, from the observed PLACEBO arm), for final and ES: Holm over 4 (all discriminating "
                  "a priori by the 2-point rule). Accuracy: 90 % CI of the cohort-mean error inside +/- 2 BA at every dose.",
    "H1_secondary": "The same for EEGNet - TS+LR and ShallowConvNet - TS+LR at both read-outs; only comparisons "
                    "discriminating a priori (|prediction - competitor| >= 2 at some dose) count; Holm over those.",
    "classification": "per pair and read-out: CONTRADICTED if a discriminating competitor is significantly better; "
                      "SUPPORTED_STRONG if the prediction beats every discriminating competitor (Holm) and is within "
                      "+/- 2 BA at every dose; SUPPORTED_RELATIVE if it beats every discriminating competitor only; PARTIAL "
                      "if some; else NOT_SUPPORTED.",
    "H2_crossing": "For pairs with a frozen eligible crossing prediction (prior gap CI below 0, dc > 0, bootstrap interval "
                   "of delta* inside (0, 1]): primary ShallowConvNet - EEGNet (ES), delta* = 0.442 [0.297, 0.644]; secondary "
                   "ShallowConvNet - TS+LR (final), 0.301 [0.109, 0.442]. delta*_obs = first zero crossing of the cohort-mean "
                   "leaked-arm gap over (P, 0.2, 0.4, 0.6, 1.0), linear between points. Classes: CROSSING_AS_PREDICTED "
                   "(inside the frozen interval), CROSSING_EARLIER / LATER_THAN_PREDICTED, NO_CROSSING_BY_1.0, "
                   "NOT_EVALUABLE_PREMISE (the gap under PLACEBO is not negative).",
    "reversal": "At each dose, the R10-A rule on P -> L_delta with D tested per dose (Holm over the 4 doses of the pair and "
                "read-out): NO_REVERSAL / DESCRIPTIVE_REVERSAL / SUPPORTED_CROSSING / SUPPORTED_REVERSAL.",
    "selection": "Roster {TS+LR, EEGNet, ShallowConvNet} per read-out: choose by the LEAKED cohort mean at each dose; evaluate "
                 "on the matched PLACEBO arm (primary) and on CLEAN. Regret relevant if >= 2 BA. A top-1 change counts as "
                 "supported only if its pair is SUPPORTED_REVERSAL at that dose. Within-benchmark counterfactual only.",
    "decomposition": "delta = 0.4 split halves (H1 = S_0.4, H2 disjoint, class-stratified): E = OWN - OTHER (exposure at fixed "
                     "support of 40 same-run copies) vs the single-copy c; S = OTHER - P (support). Stitched counterfactuals, "
                     "descriptive.",
    "overall": "On R1: CONTRADICTED if the primary pair is CONTRADICTED at either read-out; SUPPORTED if it is "
               "SUPPORTED_STRONG at both read-outs and the ES crossing is CROSSING_AS_PREDICTED; PARTIAL if it is "
               "SUPPORTED_STRONG / RELATIVE at either read-out or the crossing is as predicted; else NOT_SUPPORTED. The final "
               "verdict is the lower of R1 and R2 (deep models re-trained with independent initialisation / order / dropout "
               "seeds; TS+LR is deterministic), order SUPPORTED > PARTIAL > NOT_SUPPORTED > CONTRADICTED.",
    "inference": "54 participants; t95; two-sided MC sign-flip 1e6 (seed 11399); participant bootstrap 20,000 (seed 11010); "
                 "Holm within families; SESOI 2 BA points (vendored R4 / Paper A).",
    "forbidden_after_any_fit": "changing a dose, arm, seed, read-out, model, prediction, competitor, threshold, family or "
                               "participant set; adding a model.",
}


def cmd_plan_only():
    meta = load_meta()
    _, ps = plan_summary(meta)
    print(json.dumps(ps, indent=1, default=float))


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    return proto


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
            log(f"{tag} waiting for resources: gpu free {free:.1f} GB", f"R11_B_{tag}.log")
            last = time.time()
        time.sleep(60)


def fold_path(model, rep, f):
    return PRIVATE / "folds" / f"{model}_{rep}" / f"f_{f['subject']}_{f['fold']}.npz"


def done(fp, commit):
    if not fp.exists():
        return False
    with np.load(fp) as z:
        return str(z["commitment"]) == commit


def cmd_execute_deep(model, rep, k, n):
    tag = f"{model}_{rep}_shard{k}of{n}"
    try:
        import torch
        from models import deep_group as dg
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R11_B_{tag}.log")
        os.nice(10)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dev = select_device()
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / torch.cuda.mem_get_info()[1])
        meta = load_meta()
        g3._stop(reference_matches(sha(ROOT / META), proto["data_sha256"][META]), "metadata differ")
        fl, ps = plan_summary(meta)
        commit = ps["commitment"]
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        g3._stop(reference_matches(sha(ROOT / EEG), proto["data_sha256"][EEG]), "EEG features differ (checked before load)")
        x = np.load(ROOT / EEG, allow_pickle=False)
        g3._stop(x.dtype == np.float32 and x.shape == (21600, 20, 512) and bool(np.all(np.isfinite(x))), "features")
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
        X = torch.as_tensor(x, device=dev)
        del x
        y01 = torch.as_tensor(np.asarray(meta["label"]).astype(np.int64) - 1, device=dev)
        cls = {"EEGNet": dg.EEGNetG, "Shallow": dg.GroupShallow}[model]
        mine = [f for i, f in enumerate(fl) if i % n == k]
        (PRIVATE / "folds" / f"{model}_{rep}").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for j, f in enumerate(mine):
            fp = fold_path(model, rep, f)
            if done(fp, commit):
                continue
            res = None
            for attempt in range(OOM_RETRIES):
                wait_gpu(torch, tag)
                try:
                    res = train.run_fold_deep(cls, f, X, y01, rep, dev)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"{tag} {fp.name}: out of memory (attempt {attempt + 1}); identical recompute", f"R11_B_{tag}.log")
                    time.sleep(120)
            g3._stop(res is not None, "out of memory 7 times", "STOPPED_RESOURCE")
            for key in ("aa_A", "aa_C"):
                g3._stop(float(res[key]) == 0.0, f"A/A not exact ({key} {float(res[key]):.2e}, {fp.name})")
            np.savez(fp, commitment=np.array(commit), **res)
            if (j + 1) % 6 == 0 or j + 1 == len(mine):
                log(f"{tag} fold {j + 1}/{len(mine)} {time.time() - t0:.0f} s", f"R11_B_{tag}.log")
        log(f"{tag} DONE {time.time() - t0:.0f} s", f"R11_B_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R11_B_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R11_B_{tag}.log")
        return 2


_G: dict = {}


def _tslr_fold(args):
    f, commit = args
    fp = fold_path("TSLR", "R1", f)
    if done(fp, commit):
        return f["subject"]
    res = train.run_fold_tslr(f, _G["covs"], _G["y"])
    if float(res["aa_T"]) != 0.0:
        raise g3.G3Stop("BLOCKED", f"TS+LR A/A not exact ({fp.name})")
    np.savez(fp, commitment=np.array(commit), **res)
    return f["subject"]


def cmd_execute_tslr(k, n):
    tag = f"TSLR_shard{k}of{n}"
    try:
        import multiprocessing as mp
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R11_B_{tag}.log")
        os.nice(10)
        meta = load_meta()
        fl, ps = plan_summary(meta)
        commit = ps["commitment"]
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        g3._stop(reference_matches(sha(ROOT / COV), proto["data_sha256"][COV]), "covariances differ (checked before load)")
        _G["covs"] = np.load(ROOT / COV, allow_pickle=False).astype(np.float64)
        _G["y"] = np.asarray(meta["label"]).astype(np.int64)
        (PRIVATE / "folds" / "TSLR_R1").mkdir(parents=True, exist_ok=True)
        mine = [f for i, f in enumerate(fl) if i % n == k]
        t0 = time.time()
        with mp.get_context("fork").Pool(TSLR_WORKERS) as pool:
            for j, _ in enumerate(pool.imap_unordered(_tslr_fold, [(f, commit) for f in mine])):
                if (j + 1) % 24 == 0 or j + 1 == len(mine):
                    log(f"{tag} fold {j + 1}/{len(mine)} {time.time() - t0:.0f} s", f"R11_B_{tag}.log")
        log(f"{tag} DONE {time.time() - t0:.0f} s", f"R11_B_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R11_B_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R11_B_{tag}.log")
        return 2


def load_scores(fl, model, rep, commit):
    """model -> read-out -> arm -> concatenated p over folds (frozen order); also A/A and ES-epoch bookkeeping."""
    files, miss = [], 0
    for f in fl:
        fp = fold_path(model, rep, f)
        if not done(fp, commit):
            miss += 1
            continue
        with np.load(fp) as z:
            files.append({k: z[k] for k in z.files})
    if miss:
        return None, {"folds_missing": miss}
    ros = ("exact",) if model == "TSLR" else ("final", "es")
    arms = train.TSLR_ARMS if model == "TSLR" else tuple(a for a in design.GROUP_A + design.GROUP_B if a not in ("AA", "AAC"))
    S = {ro: {a: np.concatenate([z[f"{ro}_p_{a}"] for z in files]) for a in arms} for ro in ros}
    aa = [float(z[k]) for z in files for k in z if k.startswith("aa_")]
    book = {"folds_missing": 0, "aa_checks": len(aa), "aa_max_abs": max(aa)}
    if model != "TSLR":
        book["es_epoch_median"] = {a: float(np.median([int(z[f"es_epoch_{a}"]) for z in files])) for a in arms}
        book["es_epoch_fraction_at_100"] = {a: float(np.mean([int(z[f"es_epoch_{a}"]) == 100 for z in files])) for a in arms}
    return S, book


def cmd_analyse():
    out = {"stage": "R11-B", "protocol_id": PROTOCOL_ID}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        meta = load_meta()
        fl, ps = plan_summary(meta)
        commit = ps["commitment"]
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        preds = proto["frozen_predictions"]
        tr = analysis.Trials(fl, meta["label"], meta["subject"])
        g3._stop(tr.T.size == 21600 and np.unique(tr.T).size == 21600, "every trial must be scored exactly once")
        book, reps = {}, {}
        St, bt = load_scores(fl, "TSLR", "R1", commit)
        book["TSLR"] = bt
        g3._stop(St is not None, "TS+LR incomplete: no analysis")
        g3._stop(bt["aa_max_abs"] == 0.0, "TS+LR A/A not exact")
        for rep in REPS:
            S = {"TSLR": St}
            ok = True
            for m in DEEP:
                Sm, bm = load_scores(fl, m, rep, commit)
                book[f"{m}_{rep}"] = bm
                if Sm is None:
                    ok = False
                    continue
                g3._stop(bm["aa_max_abs"] == 0.0, f"{m} {rep}: A/A not exact")
                S[m] = Sm
            if rep == "R1":
                g3._stop(ok, "R1 incomplete: no analysis (partial results are never analysed)")
            if ok:
                reps[rep] = analysis.analyse_replicate(S, tr, preds)
        out["bookkeeping"] = book
        out["results"] = reps
        out["combined"] = analysis.combine(reps["R1"], reps.get("R2"))
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
    print(out["status"], out.get("combined", {}).get("final_verdict"))
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mx = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute", "--analyse"):
        mx.add_argument(flag, action="store_true")
    ap.add_argument("--model", default="EEGNet", choices=DEEP + ("TSLR",))
    ap.add_argument("--rep", default="R1", choices=REPS)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.plan_only:
        sys.exit(cmd_plan_only())
    if a.execute:
        kk, nn = (int(v) for v in a.shard.split("/"))
        sys.exit(cmd_execute_tslr(kk, nn) if a.model == "TSLR" else cmd_execute_deep(a.model, a.rep, kk, nn))
    if a.analyse:
        sys.exit(cmd_analyse())
