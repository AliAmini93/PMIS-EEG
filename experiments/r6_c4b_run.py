#!/usr/bin/env python3
"""Lee2019 shifted-copy EEGNet refits."""

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
from experiments.pmis_r6 import c2, c4b, lee2019
from evaluation import r6_c4b_analysis as c4b_analysis  # noqa: E402

PROTOCOL_ID = "R6-C4b-OVERLAP-EPISODE-EEGNET-LEE2019-v1"
PROTOCOL = CONFIG_ROOT / "R6_C4b_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r6_c4b"
C3 = ROOT / "data" / "r6_c3"
C4 = ROOT / "data" / "r6_c4"
PUBLIC = ROOT / "results" / "R6_C4b_results.json"
EEGNET_FILE = "lee2019_win_eegnet_4_40_128hz.npy"
TSLR_RECORDS = "R6_C4_pair_records_PRIVATE.npz"
FRACTION_DEN = 4  # outcome-blind quarter: 10 pairs per task -> two GPU groups (5 pairs x 8 arms) per task
GPU_CAP_GB = 7.5
OOM_RETRIES = 6
GPU_MIN_FREE_GB = 10.0
HOST_MIN_AVAILABLE_MB = 4096
MAX_WAIT_S = 6 * 3600


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R6_C4b_run.log", "a") as fh:
        fh.write(line + "\n")


def load_meta():
    z = np.load(C3 / "lee2019_meta.npz")
    return {k: z[k] for k in ("subject", "session", "run", "pos", "label")}


def build():
    meta = load_meta()
    tasks = lee2019.build_tasks(meta)
    plan = c4b.build_plan(tasks, fraction_den=FRACTION_DEN)
    plan["commitment"] = c4b.commitment(plan, tasks)
    return meta, tasks, plan


def summary_of(plan):
    arms = {}
    for g in plan["groups"]:
        for a in g["arms"]:
            arms[a["arm"]] = arms.get(a["arm"], 0) + 1
    return {"commitment": plan["commitment"], "n_pairs": plan["n_pairs"], "n_arms": plan["n_arms"],
            "n_groups": len(plan["groups"]), "arms_by_type": arms,
            "max_group_size": max(len(g["arms"]) for g in plan["groups"])}


def cmd_plan_only():
    _, _, plan = build()
    print(json.dumps(summary_of(plan), indent=1))


def verify():
    import torch
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    g3._stop(torch.__version__ == proto["environment"]["torch"] and torch.cuda.is_available(), "torch/CUDA")
    return proto


def wait_for_resources(torch, log_every=600):
    t0 = time.time()
    last = 0.0
    while True:
        free, _ = torch.cuda.mem_get_info()
        st = g3.resource_state()
        ok = (free / 2**30 >= GPU_MIN_FREE_GB and st["mem_available_mb"] >= HOST_MIN_AVAILABLE_MB
              and not (st["psi_full_avg10"] > 5.0))
        if ok:
            return {"gpu_free_gb": free / 2**30, **st}
        if time.time() - t0 > MAX_WAIT_S:
            raise g3.G3Stop("STOPPED_RESOURCE", "GPU/host resources unavailable for 6 h (not preempting)")
        if time.time() - last > log_every:
            log(f"waiting for resources: gpu free {free / 2**30:.1f} GB, host avail {st['mem_available_mb']} MB")
            last = time.time()
        time.sleep(60)


def cmd_execute():
    out = {"stage": "R6-C4b", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        import torch
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
        for f, p in ((EEGNET_FILE, C4 / EEGNET_FILE), ("lee2019_meta.npz", C3 / "lee2019_meta.npz"), (TSLR_RECORDS, C4 / TSLR_RECORDS)):
            g3._stop(verify_recorded_input(p, proto["data_sha256"][f]), f"{f} SHA-256 differs (checked before load)")
        meta, tasks, plan = build()
        g3._stop(reference_matches(plan["commitment"], proto["plan"]["commitment"]), "C4b plan commitment differs")
        win = np.load(C4 / EEGNET_FILE, mmap_mode="r")
        g3._stop(win.dtype == np.float32 and win.shape == (21600, 6, 20, 256), "feature shape")
        x = c4b.stack(win)
        del win
        g3._stop(bool(np.all(np.isfinite(x))), "non-finite features")
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)  # per-window channel z-score
        X = torch.as_tensor(x, device=dev)
        del x
        y = meta["label"].astype(np.int64)
        y01 = torch.as_tensor(np.tile(y - 1, len(c4b.VARIANTS)), device=dev)
        tmap = {(int(t["subject"]), int(t["fold"])): t for t in tasks}
        gdir = PRIVATE / "groups"
        gdir.mkdir(parents=True, exist_ok=True)
        results, t0, n_new = {}, time.time(), 0
        res_start = wait_for_resources(torch)
        log(f"start resources {res_start}; groups {len(plan['groups'])}, arms {plan['n_arms']}")
        peak = 0.0
        for i, g in enumerate(plan["groups"]):
            key = (g["subject"], g["fold"], g["group"])
            f = gdir / f"g_{g['subject']}_{g['fold']}_{g['group']}.npz"
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
                    res = c2.run_group(g, tmap[(g["subject"], g["fold"])], X, y01, dev)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"group {i + 1}: CUDA out of memory (attempt {attempt + 1}); discarded, waiting, recomputing")
                    time.sleep(300)
            g3._stop(res is not None, f"group {i + 1}: out of memory after {OOM_RETRIES + 1} attempts", "STOPPED_RESOURCE")
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            np.savez(f, commitment=np.array(plan["commitment"]), **res)
            results[key] = res
            n_new += 1
            log(f"group {i + 1}/{len(plan['groups'])} arms {len(g['arms'])} {time.time() - tg:.0f} s "
                f"(total {time.time() - t0:.0f} s, peak {peak:.1f} GB)")
        g3._stop(len(results) == len(plan["groups"]), "missing groups")
        recs, aa = c4b.pair_records(plan, tmap, results)
        g3._stop(len(aa) == len(tasks) and max(aa) == 0.0, f"A/A determinism check failed: max {max(aa)}")
        z = np.load(C4 / TSLR_RECORDS)
        tslr = {int(z["k"][i]): {v: 0.5 * ((z[f"{v}_pAA"][i] - z[f"{v}_pAB"][i]) + (z[f"{v}_pBB"][i] - z[f"{v}_pBA"][i]))
                                 for v in c4b.VARIANTS} for i in range(z["k"].size)}
        public = c4b_analysis.analyse(recs, tslr, aa)
        keys = sorted({k for r in recs for k in r})
        np.savez(PRIVATE / "R6_C4b_pair_records_PRIVATE.npz",
                 **{k: np.array([float(r.get(k, np.nan)) for r in recs]) for k in keys})
        out.update({"status": "COMPLETED", "results": public, "plan": summary_of(plan),
                    "runtime": {"wall_s_this_invocation": time.time() - t0, "groups_trained_this_invocation": n_new,
                                "gpu_peak_allocated_gb": peak}, "resources_end": g3.resource_state()})
        code = 0
    except g3.G3Stop as e:
        out.update({"status": e.status, "stop_message": e.message})
        log(f"STOP {e.status}: {e.message}")
        code = 2
    except Exception as e:
        out.update({"status": "BLOCKED", "stop_message": f"{type(e).__name__}: {e}"})
        log("STOP BLOCKED: " + traceback.format_exc())
        code = 2
    if out.get("status") == "COMPLETED" or not PUBLIC.exists():
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
