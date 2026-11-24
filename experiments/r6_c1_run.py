#!/usr/bin/env python3
"""HS-Stroke one-step influence, copy-dose and exact-refit audit."""

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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches, verify_recorded_input, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r6 import c1
from evaluation import r6_c1_analysis as c1_analysis  # noqa: E402

PROTOCOL_ID = "R6-C1-AUDIT-TSLR-MECHANISM-v1"
PROTOCOL = CONFIG_ROOT / "R6_C1_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r6_c1"
PUBLIC = ROOT / "results" / "R6_C1_results.json"

PREDICTIONS = {
    "P1_reference_pathway_minor": "fixed-reference d / exact d in [0.9, 1.1] and |reference-only d| < 0.1 points",
    "P2_first_order_law": "Newton-one-step d / exact d in [0.8, 1.25] (participant means) and pair-level Pearson r >= 0.9",
    "P3_dose_sublinear": "exact d(k) increases with k = 1, 2, 4, 8 and d(8) / d(1) < 8",
    "P4_solver_tolerance_negligible": "|exact-optimum d - G3 (lbfgs tol 1e-4) d| < 0.05 points (participant means)",
}


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R6_C1_run.log", "a") as fh:
        fh.write(line + "\n")


def build_plan():
    inp = g3.load_real_inputs(ROOT)
    plan = g3.make_plan(inp["inst"], inp["dist"], inp["meta"], inp["man"], inp["pairs"])
    g3._stop(reference_matches(plan["commitment"], load_protocol(CONFIG_ROOT / "G3_PROTOCOL_FROZEN.json")["commitments"]["plan_commitment_sha256"]), "G3 plan commitment not reproduced")
    return inp, plan


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    env = g3.environment()
    g3._stop(env["blas_openmp_threads"] == [1], "BLAS must be single threaded")
    for k in ("python", "numpy", "scipy", "sklearn"):
        g3._stop(env[k] == proto["environment"][k], f"{k} differs")
    g3._stop(verify_recorded_input(ROOT / "data/g3/G3_pair_scores_PRIVATE.npz", proto["data_sha256"]["G3_pair_scores_PRIVATE.npz"]),
             "stored G3 pair scores changed")
    return proto


def run_pool(tasks, workers):
    from multiprocessing import get_context
    pool = get_context("fork").Pool(workers)
    out, t0 = [], time.perf_counter()
    min_avail = None
    try:
        for i, res in enumerate(pool.imap_unordered(c1.run_task, tasks, chunksize=1)):
            out.append(res)
            st = g3.resource_state()
            min_avail = st["mem_available_mb"] if min_avail is None else min(min_avail, st["mem_available_mb"])
            g3.run_guard(st)
            log(f"task {i + 1}/{len(tasks)} fits {sum(r['fits'] for r in out)} {time.perf_counter() - t0:.0f} s "
                f"avail {st['mem_available_mb']} MB load {st['loadavg_1']:.1f}")
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()
    return out, time.perf_counter() - t0, min_avail


def cmd_execute():
    out = {"stage": "R6-C1", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        out["protocol_sha256"] = pv.sha256_file(PROTOCOL)
        log(f"PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        start = g3.resource_state()
        workers = g3.start_guard(start)
        log(f"start {start} workers {workers}")
        inp, plan = build_plan()
        covs, pre = g3.preflight_covariances(inp["paths"]["r4_private"] / g3.COVS_FILE, int(inp["dist"]["label"].size),
                                             proto["data_sha256"]["G2_covs_8_30.npy"])
        y = inp["dist"]["label"].astype(np.int64)
        c1.G["covs"], c1.G["y"] = covs, y
        z = dict(np.load(ROOT / "data/g3/G3_pair_scores_PRIVATE.npz"))
        a1_vals, stored = c1_analysis.recompute_g3(z)
        a1 = c1_analysis.compare_g3(a1_vals, json.loads((ROOT / "results/G3_results.json").read_text()))
        log(f"A1 max abs numeric diff vs G3 public: {a1['max_abs_numeric_diff']:.3e}")
        tasks = plan["seeds"][42]["tasks"]
        res, wall, min_avail = run_pool(tasks, workers)
        g3._stop(len(res) == len(tasks), "task count")
        recs = sorted([dict(r_, subject=r["subject"], fold=r["fold"]) for r in res for r_ in r["records"]],
                      key=lambda q: q["k"])
        g3._stop(len(recs) == plan["seeds"][42]["n_pairs"], "pair count")
        g3._stop(all(np.isfinite(v) for r in recs for v in r.values() if isinstance(v, float)), "non-finite output")
        result = c1_analysis.analyse(recs, stored)
        PRIVATE.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in recs for k in r})
        np.savez(PRIVATE / "R6_C1_pair_records_PRIVATE.npz",
                 **{k: np.array([float(r.get(k, np.nan)) for r in recs]) for k in keys})
        rss = {}
        for r in res:
            rss[r["pid"]] = max(rss.get(r["pid"], 0.0), r["max_rss_mb"])
        out.update({"status": "COMPLETED", "preflight": pre, "A1_recompute_G3_public_numbers": a1,
                    **result, "runtime": {"wall_s": wall, "fits": int(sum(r["fits"] for r in res)), "workers": workers,
                                          "task_seconds_sum_h": sum(r["seconds"] for r in res) / 3600,
                                          "peak_rss_per_worker_mb": max(rss.values()),
                                          "min_mem_available_mb": min_avail,
                                          "base_newton_iter_max": int(max(max(r["newton_iter_base"]) for r in res))},
                    "resources_start": start, "resources_end": g3.resource_state()})
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
    m.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    sys.exit(cmd_execute())
