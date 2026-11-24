#!/usr/bin/env python3
"""Lee2019 validation-split paired EEGNet and ShallowConvNet comparison."""

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
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches, verify_recorded_input, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity
from experiments.gpu_resources import select_device

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r6 import lee2019  # noqa: E402
from experiments.pmis_r7 import c6
from evaluation import r7_c6_analysis as c6_analysis  # noqa: E402

PROTOCOL_ID = "R7-C6-DEEP-ROBUSTNESS-LEE2019-v1"
PROTOCOL = CONFIG_ROOT / "R7_C6_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r7_c6"
C3 = ROOT / "data" / "r6_c3"
PUBLIC = ROOT / "results" / "R7_C6_results.json"
EEG = "lee2019_eegnet_4_40_128hz.npy"
TSLR = "R6_C3a_pair_records_PRIVATE.npz"
GPU_CAP_GB, GPU_MIN_FREE_GB, HOST_MIN_AVAILABLE_MB, MAX_WAIT_S, OOM_RETRIES = 7.5, 10.0, 4096, 6 * 3600, 6


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R7_C6_run.log", "a") as fh:
        fh.write(line + "\n")


def build():
    z = np.load(C3 / "lee2019_meta.npz")
    meta = {k: z[k] for k in ("subject", "session", "run", "pos", "label")}
    tasks = lee2019.build_tasks(meta)
    plan = c6.build_plan(tasks, meta["label"].astype(np.int64))
    plan["commitment"] = c6.commitment(plan)
    return meta, tasks, plan


def cmd_plan_only():
    _, _, plan = build()
    print(json.dumps({"commitment": plan["commitment"], "n_pairs": plan["n_pairs"], "n_arms_per_model": plan["n_arms"],
                      "n_groups_per_model": len(plan["groups"])}))


def verify():
    import torch
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    g3._stop(torch.__version__ == proto["environment"]["torch"] and torch.cuda.is_available(), "torch/CUDA")
    return proto


def wait_for_resources(torch):
    t0, last = time.time(), 0.0
    while True:
        free, _ = torch.cuda.mem_get_info()
        st = g3.resource_state()
        if free / 2**30 >= GPU_MIN_FREE_GB and st["mem_available_mb"] >= HOST_MIN_AVAILABLE_MB and not (st["psi_full_avg10"] > 5.0):
            return {"gpu_free_gb": free / 2**30, **st}
        if time.time() - t0 > MAX_WAIT_S:
            raise g3.G3Stop("STOPPED_RESOURCE", "GPU/host resources unavailable for 6 h (not preempting)")
        if time.time() - last > 600:
            log(f"waiting for resources: gpu free {free / 2**30:.1f} GB, host avail {st['mem_available_mb']} MB")
            last = time.time()
        time.sleep(60)


def cmd_execute():
    out = {"stage": "R7-C6", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        import torch
        from models import deep_group as dg
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        out["protocol_sha256"] = pv.sha256_file(PROTOCOL)
        log(f"PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dev = select_device()
        _, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / total)
        for f in (EEG, "lee2019_meta.npz", TSLR):
            g3._stop(verify_recorded_input(C3 / f, proto["data_sha256"][f]), f"{f} SHA-256 differs (checked before load)")
        meta, tasks, plan = build()
        g3._stop(reference_matches(plan["commitment"], proto["plan"]["commitment"]), "C6 plan commitment differs")
        x = np.load(C3 / EEG, allow_pickle=False)
        g3._stop(x.dtype == np.float32 and x.shape == (21600, 20, 512) and bool(np.all(np.isfinite(x))), "features")
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
        X = torch.as_tensor(x, device=dev)
        del x
        y01 = torch.as_tensor(meta["label"].astype(np.int64) - 1, device=dev)
        classes = {"EEGNet": dg.EEGNetG, "Shallow": dg.GroupShallow}
        gdir = PRIVATE / "groups"
        gdir.mkdir(parents=True, exist_ok=True)
        results, t0, n_new, peak = {}, time.time(), 0, 0.0
        log(f"start {wait_for_resources(torch)}; groups per model {len(plan['groups'])}")
        for model in c6.MODELS:
            for i, g in enumerate(plan["groups"]):
                key = (model, g["subject"], g["fold"])
                f = gdir / f"{model}_{g['subject']}_{g['fold']}.npz"
                if f.exists():
                    z = np.load(f)
                    if str(z["commitment"]) == plan["commitment"]:
                        results[key] = {k: z[k] for k in z.files if k != "commitment"}
                        continue
                res = None
                for attempt in range(OOM_RETRIES + 1):
                    wait_for_resources(torch)
                    torch.cuda.reset_peak_memory_stats()
                    tg = time.time()
                    try:
                        res = c6.run_group(classes[model], g, X, y01, dev)
                        break
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        log(f"{model} group {i + 1}: CUDA out of memory (attempt {attempt + 1}); recomputing")
                        time.sleep(300)
                g3._stop(res is not None, f"{model} group {i + 1}: out of memory", "STOPPED_RESOURCE")
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
                np.savez(f, commitment=np.array(plan["commitment"]), **res)
                results[key] = res
                n_new += 1
                if (i + 1) % 20 == 0 or i + 1 == len(plan["groups"]):
                    log(f"{model} group {i + 1}/{len(plan['groups'])} {time.time() - tg:.0f} s (total {time.time() - t0:.0f} s, peak {peak:.1f} GB)")
        recs, aa = {}, {}
        for model in c6.MODELS:
            recs[model], aa[model] = c6.pair_records(plan, results, model)
            g3._stop(max(aa[model]) == 0.0, f"{model} A/A determinism check failed: {max(aa[model])}")
        z = np.load(C3 / TSLR)
        tslr = {int(k): 0.5 * ((a - b) + (c - d)) for k, a, b, c, d in zip(z["k"], z["pAA"], z["pAB"], z["pBB"], z["pBA"])}
        public = c6_analysis.analyse(recs, aa, tslr)
        for model in c6.MODELS:
            keys = sorted({k for r in recs[model] for k in r})
            np.savez(PRIVATE / f"R7_C6_{model}_pair_records_PRIVATE.npz",
                     **{k: np.array([float(r[k]) for r in recs[model]]) for k in keys})
        out.update({"status": "COMPLETED", "results": public, "plan": proto["plan"],
                    "runtime": {"wall_s_this_invocation": time.time() - t0, "groups_trained_this_invocation": n_new,
                                "gpu_peak_allocated_gb": peak}})
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
    log(f"status {out['status']}")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute"):
        m.add_argument(flag, action="store_true")
    a = ap.parse_args()
    sys.exit(cmd_plan_only() if a.plan_only else cmd_execute())
