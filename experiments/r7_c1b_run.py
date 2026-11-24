#!/usr/bin/env python3
"""HS-Stroke independent reference-mean and decoder verification."""

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
from experiments.pmis_r7 import c1b
from evaluation import r7_c1b_analysis as c1b_analysis  # noqa: E402

PROTOCOL_ID = "R7-C1b-INDEPENDENT-KARCHER-AUDIT-v1"
PROTOCOL = CONFIG_ROOT / "R7_C1b_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r7_c1b"
PUBLIC = ROOT / "results" / "R7_C1b_results.json"


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R7_C1b_run.log", "a") as fh:
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
    g3._stop(verify_recorded_input(ROOT / "data/g3/G3_pair_scores_PRIVATE.npz", proto["data_sha256"]["G3_pair_scores_PRIVATE.npz"]),
             "stored G3 pair scores changed")
    return proto


def cmd_execute():
    out = {"stage": "R7-C1b", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        out["protocol_sha256"] = pv.sha256_file(PROTOCOL)
        log(f"PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        start = g3.resource_state()
        workers = g3.start_guard(start)
        inp, plan = build_plan()
        covs, pre = g3.preflight_covariances(inp["paths"]["r4_private"] / g3.COVS_FILE, int(inp["dist"]["label"].size),
                                             proto["data_sha256"]["G2_covs_8_30.npy"])
        c1b.G["covs"], c1b.G["y"] = covs, inp["dist"]["label"].astype(np.int64)
        tasks = plan["seeds"][42]["tasks"]
        log(f"start {start} workers {workers} tasks {len(tasks)}")
        from multiprocessing import get_context
        pool = get_context("fork").Pool(workers)
        res, t0, min_avail = [], time.time(), None
        try:
            for i, r in enumerate(pool.imap_unordered(c1b.run_task, tasks, chunksize=1)):
                res.append(r)
                st = g3.resource_state()
                min_avail = st["mem_available_mb"] if min_avail is None else min(min_avail, st["mem_available_mb"])
                g3.run_guard(st)
                log(f"task {i + 1}/{len(tasks)} fits {sum(x['fits'] for x in res)} {time.time() - t0:.0f} s")
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()
        g3._stop(sum(r["fits"] for r in res) == proto["fits"]["total"], "fit count mismatch")
        stored = dict(np.load(ROOT / "data/g3/G3_pair_scores_PRIVATE.npz"))
        recs = sorted([x for r in res for x in r["records"]], key=lambda q: q["k"])
        g3._stop(len(recs) == plan["seeds"][42]["n_pairs"], "pair count")
        g3._stop(all(np.isfinite(v) for r in recs for v in r.values()), "non-finite output")
        mapped = []
        for r in recs:
            k = int(r["k"])
            m = c1b.to_g3_keys(r)
            m.update({"subject": float(stored["s42_subject"][k]), "label": float(stored["s42_label"][k]),
                      "pos_dist": float(stored["s42_pos_dist"][k]), "sign": float(stored["s42_sign"][k])})
            mapped.append(m)
        public = c1b_analysis.analyse(recs, stored, json.loads((ROOT / "results/G3_results.json").read_text()), mapped)
        keys = sorted({k for r in recs for k in r})
        np.savez(PRIVATE / "R7_C1b_arm_records_PRIVATE.npz", **{k: np.array([float(r.get(k, np.nan)) for r in recs]) for k in keys})
        out.update({"status": "COMPLETED", "preflight": pre, "results": public,
                    "runtime": {"wall_s": time.time() - t0, "fits": int(sum(r["fits"] for r in res)), "workers": workers,
                                "karcher_iter_max": int(max(r["karcher_iter_max"] for r in res)),
                                "newton_iter_max": int(max(r["newton_iter_max"] for r in res)),
                                "peak_rss_worker_mb": max(r["max_rss_mb"] for r in res), "min_mem_available_mb": min_avail}})
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
