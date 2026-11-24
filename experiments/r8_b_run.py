#!/usr/bin/env python3
"""HS-Stroke natural benchmark arms with EEGNet."""

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
from experiments.pmis_r8 import bench
from evaluation import r8_bench_analysis as bench_analysis  # noqa: E402

PROTOCOL_ID = "R8-B-EEGNET-BENCHMARK-INFLATION-HSSTROKE-v1"
PROTOCOL = CONFIG_ROOT / "R8_B_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r8_b"
PUBLIC = ROOT / "results" / "R8_B_results.json"
SEEDS = (42, 43, 44)
EEG = "G2_eegnet_4_40_128hz.npy"
C7 = ROOT / "data" / "r7_c7" / "R7_C7_trial_arrays_PRIVATE.npz"
GPU_MIN_FREE_GB, MAX_WAIT_S = 4.0, 6 * 3600


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R8_B_run.log", "a") as fh:
        fh.write(line + "\n")


def load():
    paths = ad.local_paths(ROOT)
    priv = paths["r4_private"]
    g3._stop(ad.frozen_check(priv)["all_match"], "R4 metadata differ from frozen hashes")
    inst, dist, meta, man, _ = ad.load_metadata(priv)
    tl = bench.tasks(man["table"], SEEDS)
    return priv, dist, man, tl


def cmd_plan_only():
    _, dist, man, tl = load()
    print(json.dumps({"folds": len(tl), "commitment": bench.commitment(man["table"], man["rows"], tl)}))


def wait_gpu(torch):
    t0, last = time.time(), 0.0
    while True:
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()  # free to us, counting our own cache
        st = g3.resource_state()
        if free / 2**30 >= GPU_MIN_FREE_GB and st["mem_available_mb"] >= 4096 and not (st["psi_full_avg10"] > 5.0):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise g3.G3Stop("STOPPED_RESOURCE", "GPU/host unavailable for 6 h (not preempting)")
        if time.time() - last > 600:
            log(f"waiting for resources: gpu free {free / 2**30:.1f} GB")
            last = time.time()
        time.sleep(60)


def cmd_execute():
    out = {"stage": "R8-B", "protocol_id": PROTOCOL_ID}
    try:
        import torch
        proto = load_protocol(PROTOCOL)
        g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
        g3._stop(verify_recorded_input(C7, proto["data_sha256"]["R7_C7_trial_arrays_PRIVATE.npz"]), "C7 arrays changed")
        out["port_manifest_sha256"] = port_identity()
        log(f"PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dev = select_device()
        _, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(4.0 * 2**30 / total)
        priv, dist, man, tl = load()
        commit = bench.commitment(man["table"], man["rows"], tl)
        g3._stop(reference_matches(commit, proto["plan"]["commitment"]), "plan commitment differs")
        g3._stop(reference_matches(pv.sha256_file(priv / EEG), pv.historical_sha256("G2_eegnet_4_40_128hz.npy")), "EEGNet feature SHA-256 differs (checked before load)")
        x = np.load(priv / EEG, allow_pickle=False)
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
        X = torch.as_tensor(x, device=dev)
        del x
        y01 = torch.as_tensor(dist["label"].astype(np.int64) - 1, device=dev)
        fdir = PRIVATE / "folds"
        fdir.mkdir(parents=True, exist_ok=True)
        n = int(dist["label"].size)
        A = {s: {f"{k}_{a}": np.full(n, np.nan) for k in ("p", "c") for a in ("L", "P", "C", "LCB")} for s in SEEDS}
        t0, meta_f = time.time(), []
        for i, (s, u, f) in enumerate(tl):
            arms = bench.fold_arms(man["table"], man["rows"], 2, s, u, f)
            fp = fdir / f"f_{s}_{u}_{f}.npz"
            if fp.exists() and str(np.load(fp)["commitment"]) == commit:
                res = {k: v for k, v in np.load(fp).items() if k != "commitment"}
            else:
                ks = bench.check_arms(arms, man[f"exposed_S2_{s}"])
                g3._stop(ks["k_extra_distinct_L_vs_P"] >= 0 and ks["k_removed_LCB"] > 0, "support bookkeeping")
                res = None
                for attempt in range(7):
                    wait_gpu(torch)
                    try:
                        res = bench.run_fold(arms, X, y01, dev, (s, u, f), aa=(i % 10 == 0))
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        log(f"fold {i}: out of memory (attempt {attempt + 1}); identical recompute")
                        time.sleep(120)
                g3._stop(res is not None, "out of memory 7 times", "STOPPED_RESOURCE")
                if "aa_max_abs" in res:
                    g3._stop(float(res["aa_max_abs"]) == 0.0, f"A/A not exact ({float(res['aa_max_abs']):.2e})")
                res["k_extra"] = np.array(ks["k_extra_distinct_L_vs_P"])
                res["k_removed"] = np.array(ks["k_removed_LCB"])
                np.savez(fp, commitment=np.array(commit), **res)
            meta_f.append({k: float(res[k]) for k in ("k_extra", "k_removed", "aa_max_abs") if k in res})
            res = {k: v for k, v in res.items() if k.startswith(("p_", "c_"))}
            for k, v in res.items():
                g3._stop(bool(np.all(np.isnan(A[s][k][arms["T"]]))), "trial scored twice")
                A[s][k][arms["T"]] = v
            if (i + 1) % 10 == 0 or i + 1 == len(tl):
                log(f"fold {i + 1}/{len(tl)} {time.time() - t0:.0f} s")
        z = np.load(C7)
        tslr = {s: {f"r4_pred_{a}": z[f"S2_{s}_r4_pred_{a}"] for a in ("L", "P")} for s in SEEDS}
        exposed = {s: man[f"exposed_S2_{s}"] for s in SEEDS}
        np.savez(PRIVATE / "R8_B_trial_arrays_PRIVATE.npz", **{f"{s}_{k}": v for s, d in A.items() for k, v in d.items()})
        out.update({"status": "COMPLETED", "results": bench_analysis.analyse(A, dist, exposed, SEEDS, tslr), "plan": proto["plan"],
                    "runtime": {"wall_s": time.time() - t0},
                    "bookkeeping": {"folds": len(meta_f),
                                    "k_extra_distinct_L_vs_P_median": float(np.median([m["k_extra"] for m in meta_f])),
                                    "k_removed_LCB_median": float(np.median([m["k_removed"] for m in meta_f])),
                                    "k_extra_equals_k_removed_all": bool(all(m["k_extra"] == m["k_removed"] for m in meta_f)),
                                    "aa_checks": int(sum("aa_max_abs" in m for m in meta_f)),
                                    "aa_max_abs": float(max([m["aa_max_abs"] for m in meta_f if "aa_max_abs" in m], default=np.nan))}})
        code = 0
    except g3.G3Stop as e:
        out.update({"status": e.status, "stop_message": e.message})
        log(f"STOP {e.status}: {e.message}")
        code = 2
    except Exception as e:
        out.update({"status": "BLOCKED", "stop_message": f"{type(e).__name__}: {e}"})
        log("STOP BLOCKED: " + traceback.format_exc())
        code = 2
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps(out, indent=2, default=float) + "\n")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute"):
        m.add_argument(flag, action="store_true")
    a = ap.parse_args()
    sys.exit(cmd_plan_only() if a.plan_only else cmd_execute())
