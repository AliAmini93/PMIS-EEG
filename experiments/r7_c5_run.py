#!/usr/bin/env python3
"""Lee2019 fine temporal shifts and retention of the exact-copy effect."""

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
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches, verify_recorded_input, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity
from experiments.gpu_resources import select_device

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r6 import c2, lee2019  # noqa: E402
from experiments.pmis_r7 import c5
from evaluation import r7_c5_analysis as c5_analysis  # noqa: E402

PROTOCOL_ID = "R7-C5-FINE-SHIFT-LEE2019-v1"
PROTOCOL = CONFIG_ROOT / "R7_C5_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r7_c5"
C3, C4 = ROOT / "data" / "r6_c3", ROOT / "data" / "r6_c4"
PUBLIC = ROOT / "results" / "R7_C5_results.json"
FILES = {"c4_cov": (C4, "lee2019_win_covs_8_30.npy"), "c4_eeg": (C4, "lee2019_win_eegnet_4_40_128hz.npy"),
         "c5_cov": (PRIVATE, "lee2019_shift_covs_8_30.npy"), "c5_eeg": (PRIVATE, "lee2019_shift_eegnet_4_40_128hz.npy"),
         "meta": (C3, "lee2019_meta.npz")}
GPU_CAP_GB, GPU_MIN_FREE_GB, HOST_MIN_AVAILABLE_MB, MAX_WAIT_S, OOM_RETRIES = 7.5, 10.0, 4096, 6 * 3600, 6


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R7_C5_run.log", "a") as fh:
        fh.write(line + "\n")


def tasks_and_plan():
    z = np.load(C3 / "lee2019_meta.npz")
    meta = {k: z[k] for k in ("subject", "session", "run", "pos", "label")}
    tasks = lee2019.build_tasks(meta)
    plan = c5.build_eegnet_plan(tasks)
    plan["commitment"] = c5.commitment(tasks, plan)
    return meta, tasks, plan


def cmd_plan_only():
    _, tasks, plan = tasks_and_plan()
    print(json.dumps({"commitment": plan["commitment"], "pairs": plan["n_pairs"], "eegnet_arms": plan["n_arms"],
                      "eegnet_groups": len(plan["groups"]), "tslr_fits": plan["n_pairs"] * 2 * len(c5.VARIANTS)}))


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    for key, (d, f) in FILES.items():
        g3._stop(verify_recorded_input(d / f, proto["data_sha256"][f]), f"{f} SHA-256 differs (checked before load)")
    meta, tasks, plan = tasks_and_plan()
    g3._stop(reference_matches(plan["commitment"], proto["plan"]["commitment"]), "C5 plan commitment differs")
    return proto, meta, tasks, plan


def _write(out):
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    if out.get("status") == "COMPLETED" or not PUBLIC.exists():
        PUBLIC.write_text(json.dumps(out, indent=2, default=float) + "\n")


def cmd_execute_tslr():
    out = {"stage": "R7-C5-tslr", "protocol_id": PROTOCOL_ID}
    try:
        proto, meta, tasks, _ = verify()
        out["port_manifest_sha256"] = port_identity()
        log(f"TSLR PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        start = g3.resource_state()
        workers = g3.start_guard(start)
        cov = c5.stack(np.load(C4 / FILES["c4_cov"][1], mmap_mode="r"), np.load(PRIVATE / FILES["c5_cov"][1], mmap_mode="r"))
        g3._stop(bool(np.all(np.isfinite(cov))), "non-finite covariances")
        c5.G["covs"], c5.G["y"] = cov, np.tile(meta["label"].astype(np.int64), len(c5.VARIANTS))
        from multiprocessing import get_context
        pool = get_context("fork").Pool(workers)
        res, t0 = [], time.time()
        try:
            for i, r in enumerate(pool.imap_unordered(c5.run_task_tslr, tasks, chunksize=2)):
                res.append(r)
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    g3.run_guard(g3.resource_state())
                    log(f"TSLR task {i + 1}/{len(tasks)} {time.time() - t0:.0f} s")
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()
        recs = sorted([dict(x, subject=r["subject"], fold=r["fold"]) for r in res for x in r["records"]], key=lambda q: q["k"])
        g3._stop(len(recs) == proto["plan"]["pairs"] and sum(r["fits"] for r in res) == proto["plan"]["tslr_fits"], "counts")
        g3._stop(all(np.isfinite(v) for r in recs for v in r.values() if isinstance(v, float)), "non-finite output")
        keys = sorted({k for r in recs for k in r})
        np.savez(PRIVATE / "R7_C5_tslr_records_PRIVATE.npz", **{k: np.array([float(r[k]) for r in recs]) for k in keys})
        log(f"TSLR done {time.time() - t0:.0f} s, newton max {max(r['newton_iter_max'] for r in res)}")
        return 0
    except g3.G3Stop as e:
        log(f"TSLR STOP {e.status}: {e.message}")
        _write(dict(out, status=e.status, stop_message=e.message))
        return 2
    except Exception as e:
        log("TSLR STOP BLOCKED: " + traceback.format_exc())
        _write(dict(out, status="BLOCKED", stop_message=f"{type(e).__name__}: {e}"))
        return 2


def wait_for_resources(torch):
    t0, last = time.time(), 0.0
    while True:
        free, _ = torch.cuda.mem_get_info()
        st = g3.resource_state()
        if free / 2**30 >= GPU_MIN_FREE_GB and st["mem_available_mb"] >= HOST_MIN_AVAILABLE_MB and not (st["psi_full_avg10"] > 5.0):
            return
        if time.time() - t0 > MAX_WAIT_S:
            raise g3.G3Stop("STOPPED_RESOURCE", "GPU/host resources unavailable for 6 h (not preempting)")
        if time.time() - last > 600:
            log(f"waiting for resources: gpu free {free / 2**30:.1f} GB, host avail {st['mem_available_mb']} MB")
            last = time.time()
        time.sleep(60)


def cmd_execute_eegnet():
    out = {"stage": "R7-C5-eegnet", "protocol_id": PROTOCOL_ID}
    try:
        import torch
        proto, meta, tasks, plan = verify()
        out["port_manifest_sha256"] = port_identity()
        log(f"EEGNET PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dev = select_device()
        _, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / total)
        x = c5.stack(np.load(C4 / FILES["c4_eeg"][1], mmap_mode="r"), np.load(PRIVATE / FILES["c5_eeg"][1], mmap_mode="r"))
        g3._stop(x.dtype == np.float32 and bool(np.all(np.isfinite(x))), "features")
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
        X = torch.as_tensor(x, device=dev)
        del x
        y01 = torch.as_tensor(np.tile(meta["label"].astype(np.int64) - 1, len(c5.VARIANTS)), device=dev)
        tmap = {(int(t["subject"]), int(t["fold"])): t for t in tasks}
        gdir = PRIVATE / "groups"
        gdir.mkdir(parents=True, exist_ok=True)
        results, t0 = {}, time.time()
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
                try:
                    res = c2.run_group(g, tmap[(g["subject"], g["fold"])], X, y01, dev)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"group {i + 1}: CUDA out of memory (attempt {attempt + 1}); recomputing")
                    time.sleep(300)
            g3._stop(res is not None, f"group {i + 1}: out of memory", "STOPPED_RESOURCE")
            np.savez(f, commitment=np.array(plan["commitment"]), **res)
            results[key] = res
            if (i + 1) % 50 == 0 or i + 1 == len(plan["groups"]):
                log(f"EEGNET group {i + 1}/{len(plan['groups'])} (total {time.time() - t0:.0f} s)")
        recs, aa = c5.eegnet_pair_records(plan, tmap, results)
        g3._stop(len(aa) == len(tasks) and max(aa) == 0.0, f"A/A determinism check failed: {max(aa)}")
        keys = sorted({k for r in recs for k in r})
        np.savez(PRIVATE / "R7_C5_eegnet_records_PRIVATE.npz", **{k: np.array([float(r[k]) for r in recs]) for k in keys})
        (PRIVATE / "R7_C5_eegnet_aa.json").write_text(json.dumps(aa))
        log(f"EEGNET done {time.time() - t0:.0f} s")
        return 0
    except g3.G3Stop as e:
        log(f"EEGNET STOP {e.status}: {e.message}")
        _write(dict(out, status=e.status, stop_message=e.message))
        return 2
    except Exception as e:
        log("EEGNET STOP BLOCKED: " + traceback.format_exc())
        _write(dict(out, status="BLOCKED", stop_message=f"{type(e).__name__}: {e}"))
        return 2


def cmd_analyse():
    out = {"stage": "R7-C5", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        proto, meta, tasks, plan = verify()
        out["port_manifest_sha256"] = port_identity()
        load = lambda f: (lambda z: [{k: z[k][i].item() for k in z.files} for i in range(z["k"].size)])(np.load(PRIVATE / f))  # noqa: E731
        t, e = load("R7_C5_tslr_records_PRIVATE.npz"), load("R7_C5_eegnet_records_PRIVATE.npz")
        g3._stop(len(t) == len(e) == proto["plan"]["pairs"], "missing records")
        aa = json.loads((PRIVATE / "R7_C5_eegnet_aa.json").read_text())
        out.update({"status": "COMPLETED", "results": c5_analysis.analyse(t, e, aa), "plan": proto["plan"]})
        code = 0
    except g3.G3Stop as e_:
        out.update({"status": e_.status, "stop_message": e_.message})
        code = 2
    except Exception as e_:
        out.update({"status": "BLOCKED", "stop_message": f"{type(e_).__name__}: {e_}"})
        log("ANALYSE STOP BLOCKED: " + traceback.format_exc())
        code = 2
    _write(out)
    log(f"analyse status {out['status']}")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute-tslr", "--execute-eegnet", "--analyse"):
        m.add_argument(flag, action="store_true")
    a = ap.parse_args()
    fn = {"plan_only": cmd_plan_only, "execute_tslr": cmd_execute_tslr,
          "execute_eegnet": cmd_execute_eegnet, "analyse": cmd_analyse}
    sys.exit(next(f for k, f in fn.items() if getattr(a, k))())
