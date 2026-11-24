#!/usr/bin/env python3
"""HS-Stroke randomized split halves with grouped EEGNet refits."""

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
from evaluation import r9_split_analysis as split_analysis
from experiments.pmis_r9 import split_bench  # noqa: E402

PROTOCOL_ID = "R9-C-SPLITHALF-SUPPORT-EXPOSURE-HSSTROKE-v1"
PROTOCOL = CONFIG_ROOT / "R9_C_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r9_c"
PUBLIC = ROOT / "results" / "R9_C_results.json"
SEEDS = (42, 43, 44)
EEG = "G2_eegnet_4_40_128hz.npy"
R8B = ROOT / "data" / "r8_b" / "R8_B_trial_arrays_PRIVATE.npz"
GPU_CAP_GB, GPU_MIN_FREE_GB, MAX_WAIT_S = 6.0, 6.0, 6 * 3600


def log(msg, name="R9_C_run.log"):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / name, "a") as fh:
        fh.write(line + "\n")


def load():
    priv = ad.local_paths(ROOT)["r4_private"]
    g3._stop(ad.frozen_check(priv)["all_match"], "R4 metadata differ from frozen hashes")
    inst, dist, meta, man, _ = ad.load_metadata(priv)
    return priv, dist, man, bench.tasks(man["table"], SEEDS)


def plan_summary():
    _, dist, man, tl = load()
    n1, n2, imb, rows = [], [], [], []
    for (s, u, f) in tl:
        arms = bench.fold_arms(man["table"], man["rows"], 2, s, u, f)
        h1, h2 = split_bench.split_halves(arms, dist, (s, u, f))
        L1, L2 = split_bench.make_split_arms(arms, h1, h2)
        st = split_bench.check_split(arms, L1, L2, h1, h2, man[f"exposed_S2_{s}"])
        n1.append(st["n_h1"]); n2.append(st["n_h2"]); imb.append(abs(st["copy_rows_h1"] - st["copy_rows_h2"])); rows.append(st["rows"])
    return {"folds": len(tl), "commitment": split_bench.commitment(man["table"], man["rows"], dist, tl),
            "half_size_median": float(np.median(n1 + n2)), "half_size_min": int(min(n1 + n2)),
            "max_abs_half_size_difference": int(max(abs(a - b) for a, b in zip(n1, n2))),
            "copy_row_imbalance_median": float(np.median(imb)), "copy_row_imbalance_max": int(max(imb)), "rows_median": float(np.median(rows))}


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
            log(f"{tag} waiting for resources: gpu free {free:.1f} GB", f"R9_C_{tag}.log")
            last = time.time()
        time.sleep(60)


def cmd_execute(k, n):
    tag = f"shard{k}of{n}"
    try:
        import torch
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R9_C_{tag}.log")
        os.nice(10)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dev = select_device()
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / torch.cuda.mem_get_info()[1])
        priv, dist, man, tl = load()
        commit = split_bench.commitment(man["table"], man["rows"], dist, tl)
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        g3._stop(reference_matches(pv.sha256_file(priv / EEG), pv.historical_sha256("G2_eegnet_4_40_128hz.npy")), "EEGNet feature SHA-256 differs (checked before load)")
        x = np.load(priv / EEG, allow_pickle=False)
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
        X = torch.as_tensor(x, device=dev)
        del x
        y01 = torch.as_tensor(dist["label"].astype(np.int64) - 1, device=dev)
        fdir = PRIVATE / "folds"
        fdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        mine = [(i, t) for i, t in enumerate(tl) if i % n == k]
        for j, (i, (s, u, f)) in enumerate(mine):
            fp = fdir / f"f_{s}_{u}_{f}.npz"
            if fp.exists() and str(np.load(fp)["commitment"]) == commit:
                continue
            arms = bench.fold_arms(man["table"], man["rows"], 2, s, u, f)
            h1, h2 = split_bench.split_halves(arms, dist, (s, u, f))
            L1, L2 = split_bench.make_split_arms(arms, h1, h2)
            split_bench.check_split(arms, L1, L2, h1, h2, man[f"exposed_S2_{s}"])
            res = None
            for attempt in range(7):
                wait_gpu(torch, tag)
                try:
                    res = split_bench.run_fold(arms, L1, L2, X, y01, dev, (s, u, f), aa=(i % 10 == 0))
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"{tag} fold {i}: out of memory (attempt {attempt + 1}); identical recompute", f"R9_C_{tag}.log")
                    time.sleep(120)
            g3._stop(res is not None, "out of memory 7 times", "STOPPED_RESOURCE")
            if "aa_max_abs" in res:
                g3._stop(float(res["aa_max_abs"]) == 0.0, f"A/A not exact ({float(res['aa_max_abs']):.2e})")
            np.savez(fp, commitment=np.array(commit), T=arms["T"], H1=h1, H2=h2, **res)
            if (j + 1) % 5 == 0 or j + 1 == len(mine):
                log(f"{tag} fold {j + 1}/{len(mine)} {time.time() - t0:.0f} s", f"R9_C_{tag}.log")
        log(f"{tag} DONE", f"R9_C_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R9_C_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R9_C_{tag}.log")
        return 2


def cmd_analyse():
    out = {"stage": "R9-C", "protocol_id": PROTOCOL_ID}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        g3._stop(verify_recorded_input(R8B, proto["data_sha256"]["R8_B_trial_arrays_PRIVATE.npz"]), "R8-B arrays changed")
        _, dist, man, tl = load()
        commit = split_bench.commitment(man["table"], man["rows"], dist, tl)
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        n = int(dist["label"].size)
        A = {s: {f"{q}_{a}": np.full(n, np.nan) for q in ("p", "c") for a in split_bench.ARMS} for s in SEEDS}
        half = {s: np.zeros(n, dtype=np.int64) for s in SEEDS}
        aa, missing = [], 0
        for (s, u, f) in tl:
            fp = PRIVATE / "folds" / f"f_{s}_{u}_{f}.npz"
            if not fp.exists() or str(np.load(fp)["commitment"]) != commit:
                missing += 1
                continue
            z = np.load(fp)
            T = z["T"]
            for key in A[s]:
                g3._stop(bool(np.all(np.isnan(A[s][key][T]))), "trial scored twice")
                A[s][key][T] = z[key]
            half[s][z["H1"]] = 1
            half[s][z["H2"]] = 2
            if "aa_max_abs" in z.files:
                aa.append(float(z["aa_max_abs"]))
        g3._stop(missing == 0, f"{missing} folds missing")
        g3._stop(bool(aa) and max(aa) == 0.0, "A/A determinism check failed")
        zb = np.load(R8B)
        r8b = {s: {f"{q}_{a}": zb[f"{s}_{q}_{a}"] for q in ("p", "c") for a in ("L", "P")} for s in SEEDS}
        np.savez(PRIVATE / "R9_C_trial_arrays_PRIVATE.npz", **{f"{s}_{k}": v for s, d in A.items() for k, v in d.items()},
                 **{f"{s}_half": half[s] for s in SEEDS})
        out.update({"status": "COMPLETED", "results": split_analysis.analyse(A, half, dist, SEEDS, r8b), "plan": proto["plan"],
                    "bookkeeping": {"folds": len(tl), "aa_checks": len(aa), "aa_max_abs": max(aa)}})
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
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.plan_only:
        print(json.dumps(plan_summary()))
        sys.exit(0)
    if a.execute:
        kk, nn = (int(v) for v in a.shard.split("/"))
        sys.exit(cmd_execute(kk, nn))
    sys.exit(cmd_analyse())
