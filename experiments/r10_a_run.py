#!/usr/bin/env python3
"""HS-Stroke natural-duplicate decoder ranking and apparent choice gap."""

from __future__ import annotations

import argparse
import datetime as dt
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
    os.environ[_v] = "2"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches, verify_recorded_input, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity
from experiments.gpu_resources import select_device

ROOT = RUN_ROOT
from data_preparation import hs_metadata as ad  # noqa: E402
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r8 import bench  # noqa: E402
from experiments.pmis_r9 import split_bench  # noqa: E402
from evaluation import r10_analysis as analysis
from experiments.pmis_r10 import bench10  # noqa: E402

PROTOCOL_ID = "R10-A-RANKING-DISTORTION-HSSTROKE-v1"
PROTOCOL = CONFIG_ROOT / "R10_A_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r10_a"
PUBLIC = ROOT / "results" / "R10_A_results.json"
SEEDS = (42, 43, 44)
EEG = "G2_eegnet_4_40_128hz.npy"
EXISTING = {"R7_C7_trial_arrays_PRIVATE.npz": ROOT / "data/r7_c7/R7_C7_trial_arrays_PRIVATE.npz",
            "R9_C2_trial_arrays_PRIVATE.npz": ROOT / "data/r9_c2/R9_C2_trial_arrays_PRIVATE.npz",
            "R8_B_trial_arrays_PRIVATE.npz": ROOT / "data/r8_b/R8_B_trial_arrays_PRIVATE.npz",
            "R9_C_trial_arrays_PRIVATE.npz": ROOT / "data/r9_c/R9_C_trial_arrays_PRIVATE.npz"}
GPU_CAP_GB, GPU_MIN_FREE_GB, MAX_WAIT_S = 6.0, 6.0, 6 * 3600
N_BOOT, BOOT_SEED = 20000, 10_010


def log(msg, name="R10_A_run.log"):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / name, "a") as fh:
        fh.write(line + "\n")


def load():
    priv = ad.local_paths(ROOT)["r4_private"]
    g3._stop(ad.frozen_check(priv)["all_match"], "R4 metadata differ from frozen hashes")
    _, dist, _, man, _ = ad.load_metadata(priv)
    return priv, dist, man, bench.tasks(man["table"], SEEDS)


def plan_summary():
    _, dist, man, tl = load()
    split_c = split_bench.commitment(man["table"], man["rows"], dist, tl)
    rows = {a: [] for a in ("L", "C", "T")}
    for (s, u, f) in tl:
        arms = bench.fold_arms(man["table"], man["rows"], 2, s, u, f)
        for a in rows:
            rows[a].append(int(arms[a].size))
    historical_split = load_protocol(CONFIG_ROOT / "R9_C_PROTOCOL.json")["plan"]["commitment"]
    return {"folds": len(tl), "parts": {p: {k: v for k, v in c.items()} for p, c in bench10.PARTS.items()},
            "commitment": bench10.commitment(man["table"], man["rows"], dist, man, tl),
            "r9c_split_commitment": split_c, "r9c_split_commitment_matches_frozen_R9C": (split_c == historical_split if historical_split is not None else None),
            "rows_sum": {a: int(sum(v)) for a, v in rows.items()}, "rows_median": {a: float(np.median(v)) for a, v in rows.items()}}


def cmd_plan_only():
    print(json.dumps(plan_summary(), indent=2))


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
            log(f"{tag} waiting for resources: gpu free {free:.1f} GB", f"R10_A_{tag}.log")
            last = time.time()
        time.sleep(60)


def cmd_execute(part, k, n):
    tag = f"{part}_shard{k}of{n}"
    try:
        import torch
        g3._stop(part in bench10.PARTS, "unknown part")
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R10_A_{tag}.log")
        os.nice(10)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dev = select_device()
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / torch.cuda.mem_get_info()[1])
        priv, dist, man, tl = load()
        commit = bench10.commitment(man["table"], man["rows"], dist, man, tl)
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        g3._stop(reference_matches(pv.sha256_file(priv / EEG), pv.historical_sha256("G2_eegnet_4_40_128hz.npy")), "EEG feature SHA-256 differs (checked before load)")
        x = np.load(priv / EEG, allow_pickle=False)
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
        X = torch.as_tensor(x, device=dev)
        del x
        y01 = torch.as_tensor(dist["label"].astype(np.int64) - 1, device=dev)
        fdir = PRIVATE / "folds" / part
        fdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        mine = [(i, t) for i, t in enumerate(tl) if i % n == k]
        for j, (i, (s, u, f)) in enumerate(mine):
            fp = fdir / f"f_{s}_{u}_{f}.npz"
            if fp.exists() and str(np.load(fp)["commitment"]) == commit:
                continue
            arms, L1, L2, h1, h2 = bench10.fold_inputs(man["table"], man["rows"], dist, man[f"exposed_S2_{s}"], s, u, f)
            res = None
            for attempt in range(7):
                wait_gpu(torch, tag)
                try:
                    res = bench10.run_fold(part, arms, L1, L2, X, y01, dev, (s, u, f))
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"{tag} fold {i}: out of memory (attempt {attempt + 1}); identical recompute", f"R10_A_{tag}.log")
                    time.sleep(120)
            g3._stop(res is not None, "out of memory 7 times", "STOPPED_RESOURCE")
            for key in ("aa_main_max_abs", "aa_clean_max_abs"):
                g3._stop(float(res[key]) == 0.0, f"A/A not exact ({key} {float(res[key]):.2e}, fold {i})")
            np.savez(fp, commitment=np.array(commit), T=arms["T"], H1=h1, H2=h2, **res)
            if (j + 1) % 5 == 0 or j + 1 == len(mine):
                log(f"{tag} fold {j + 1}/{len(mine)} {time.time() - t0:.0f} s", f"R10_A_{tag}.log")
        log(f"{tag} DONE", f"R10_A_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R10_A_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R10_A_{tag}.log")
        return 2


def assemble(part, commit, tl, n):
    A = {f"{q}_{a}": {s: np.full(n, np.nan) for s in SEEDS} for q in ("p", "c") for a in bench10.ARMS}
    half = {s: np.zeros(n, dtype=np.int64) for s in SEEDS}
    aa, missing = [], 0
    for (s, u, f) in tl:
        fp = PRIVATE / "folds" / part / f"f_{s}_{u}_{f}.npz"
        if not fp.exists() or str(np.load(fp)["commitment"]) != commit:
            missing += 1
            continue
        z = np.load(fp)
        T = z["T"]
        for key in A:
            g3._stop(bool(np.all(np.isnan(A[key][s][T]))), "trial scored twice")
            A[key][s][T] = z[key]
        half[s][z["H1"]] = 1
        half[s][z["H2"]] = 2
        aa += [float(z["aa_main_max_abs"]), float(z["aa_clean_max_abs"])]
    return A, half, aa, missing


def cmd_analyse():
    out = {"stage": "R10-A", "protocol_id": PROTOCOL_ID}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        for k, v in EXISTING.items():
            g3._stop(verify_recorded_input(v, proto["data_sha256"][k]), f"{k} changed (checked before load)")
        _, dist, man, tl = load()
        commit = bench10.commitment(man["table"], man["rows"], dist, man, tl)
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        y = dist["label"]
        n = int(y.size)
        exposed = {s: man[f"exposed_S2_{s}"].astype(bool) for s in SEEDS}
        Z = {k: np.load(v, allow_pickle=False) for k, v in EXISTING.items()}
        c7, c2, r8b, r9c = (Z[k] for k in EXISTING)
        half_existing = {s: c2[f"{s}_half"] for s in SEEDS}
        tslr_c = {a: {s: (c7[f"S2_{s}_r4_pred_{a}"].astype(np.int64) == y).astype(np.float64) for s in SEEDS} for a in ("L", "P", "C")}
        tslr_p = {a: {s: c7[f"S2_{s}_r4_p_{a}"] for s in SEEDS} for a in ("L", "P", "C")}
        for a in ("L1", "L2"):
            tslr_c[a] = {s: c2[f"{s}_c_{a}"] for s in SEEDS}
            tslr_p[a] = {s: c2[f"{s}_p_{a}"] for s in SEEDS}
        eeg1_c = {a: {s: r8b[f"{s}_c_{a}"] for s in SEEDS} for a in ("L", "P", "C")}
        eeg1_p = {a: {s: r8b[f"{s}_p_{a}"] for s in SEEDS} for a in ("L", "P", "C")}
        for a in ("L1", "L2"):
            eeg1_c[a] = {s: r9c[f"{s}_c_{a}"] for s in SEEDS}
            eeg1_p[a] = {s: r9c[f"{s}_p_{a}"] for s in SEEDS}
        new, book = {}, {}
        for part in bench10.PARTS:
            A, half, aa, missing = assemble(part, commit, tl, n)
            book[part] = {"folds_missing": missing, "aa_checks": len(aa), "aa_max_abs": max(aa) if aa else None}
            if missing == 0:
                g3._stop(max(aa) == 0.0, f"{part}: A/A not exact")
                for s in SEEDS:
                    g3._stop(np.array_equal(half[s], half_existing[s]), f"{part}: halves differ from R9-C/R9-C2")
                    for key in A:
                        g3._stop(bool(np.all(np.isfinite(A[key][s]))), f"{part}: unscored trials")
                new[part] = A
        out["bookkeeping"] = book
        g3._stop("SH1" in new, "primary part SH1 incomplete: no analysis")
        np.savez(PRIVATE / "R10_A_trial_arrays_PRIVATE.npz",
                 **{f"{part}_{key}_{s}": v for part, A in new.items() for key, d in A.items() for s, v in d.items()})

        def unpack(A):
            return ({a: A[f"c_{a}"] for a in bench10.ARMS}, {a: A[f"p_{a}"] for a in bench10.ARMS})
        sh1_c, sh1_p = unpack(new["SH1"])
        r1, BA1 = analysis.analyse_replicate({"TSLR": tslr_c, "EEGNet": eeg1_c, "Shallow": sh1_c},
                                             {"TSLR": tslr_p, "EEGNet": eeg1_p, "Shallow": sh1_p},
                                             dist, exposed, half_existing, SEEDS, N_BOOT, BOOT_SEED)
        g3._stop(round(r1["H1_family_D_L_minus_P"]["EEGNet_vs_TSLR"]["mean_pp"], 3) == 7.141, "known EEGNet-TS+LR D not reproduced")
        res = {"R1_primary": r1}
        if "SH2" in new and "EG2" in new:
            sh2_c, sh2_p = unpack(new["SH2"])
            eg2_c, eg2_p = unpack(new["EG2"])
            r2, BA2 = analysis.analyse_replicate({"TSLR": tslr_c, "EEGNet": eg2_c, "Shallow": sh2_c},
                                                 {"TSLR": tslr_p, "EEGNet": eg2_p, "Shallow": sh2_p},
                                                 dist, exposed, half_existing, SEEDS, N_BOOT, BOOT_SEED)
            res.update({"R2_training_seed_replicate": r2, "robustness_R1_vs_R2": analysis.robustness(r1, r2),
                        "training_seed_variability": analysis.training_seed_variability(BA1, BA2),
                        "averaged_over_training_seeds": analysis.averaged_over_training_seeds(BA1, BA2)})
            out["status"] = "COMPLETED"
        else:
            res["R2_training_seed_replicate"] = "NOT_COMPLETED"
            out["status"] = "COMPLETED_PRIMARY_ONLY"
        out.update({"results": res, "plan": {k: proto["plan"][k] for k in ("folds", "commitment", "r9c_split_commitment")}})
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
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute", "--analyse"):
        m.add_argument(flag, action="store_true")
    ap.add_argument("--part", default="SH1")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.plan_only:
        sys.exit(cmd_plan_only())
    if a.execute:
        kk, nn = (int(v) for v in a.shard.split("/"))
        sys.exit(cmd_execute(a.part, kk, nn))
    sys.exit(cmd_analyse())
