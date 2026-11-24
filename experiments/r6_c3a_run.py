#!/usr/bin/env python3
"""Lee2019 fixed-support injected-copy swap using exact TS+LR refits."""

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

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r6 import c3, lee2019
from evaluation import r6_c3_analysis as c3_analysis  # noqa: E402

PROTOCOL_ID = "R6-C3a-PMIS-TSLR-LEE2019-v1"
PROTOCOL = CONFIG_ROOT / "R6_C3a_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r6_c3"
PUBLIC = ROOT / "results" / "R6_C3a_results.json"
FEATURES = ("lee2019_covs_8_30.npy", "lee2019_meta.npz")


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R6_C3a_run.log", "a") as fh:
        fh.write(line + "\n")


def load_meta():
    z = np.load(PRIVATE / "lee2019_meta.npz")
    return {k: z[k] for k in ("subject", "session", "run", "pos", "label")}


def cmd_plan_only():
    tasks = lee2019.build_tasks(load_meta())
    print(json.dumps({"tasks": len(tasks), "pairs": sum(len(t["pairs"]) for t in tasks),
                      "commitment": c3.tslr_commitment(tasks)}))


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    env = g3.environment()
    g3._stop(env["blas_openmp_threads"] == [1], "BLAS must be single threaded")
    for f in FEATURES:
        g3._stop(reference_matches(pv.sha256_file(PRIVATE / f), proto["feature_sha256"][f]), f"{f} SHA-256 differs (checked before load)")
    return proto


def cmd_execute():
    out = {"stage": "R6-C3a", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        out["protocol_sha256"] = pv.sha256_file(PROTOCOL)
        log(f"PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        start = g3.resource_state()
        workers = g3.start_guard(start)
        meta = load_meta()
        tasks = lee2019.build_tasks(meta)
        g3._stop(reference_matches(c3.tslr_commitment(tasks), proto["plan"]["commitment"]), "plan commitment differs")
        covs = np.load(PRIVATE / "lee2019_covs_8_30.npy", allow_pickle=False)
        g3._stop(covs.shape == (21600, 20, 20) and bool(np.all(np.isfinite(covs))), "covariance shape/finite")
        c3.G["covs"], c3.G["y"] = covs, meta["label"].astype(np.int64)
        log(f"start {start} workers {workers} tasks {len(tasks)}")
        from multiprocessing import get_context
        pool = get_context("fork").Pool(workers)
        res, t0, min_avail = [], time.time(), None
        try:
            for i, r in enumerate(pool.imap_unordered(c3.run_task_tslr, tasks, chunksize=1)):
                res.append(r)
                st = g3.resource_state()
                min_avail = st["mem_available_mb"] if min_avail is None else min(min_avail, st["mem_available_mb"])
                g3.run_guard(st)
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    log(f"task {i + 1}/{len(tasks)} {time.time() - t0:.0f} s")
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()
        recs = sorted([dict(x, subject=r["subject"], fold=r["fold"]) for r in res for x in r["records"]], key=lambda q: q["k"])
        g3._stop(len(recs) == proto["plan"]["pairs"], "pair count")
        g3._stop(all(np.isfinite(v) for r in recs for v in r.values() if isinstance(v, float)), "non-finite output")
        public = c3_analysis.analyse(recs)
        keys = sorted({k for r in recs for k in r})
        np.savez(PRIVATE / "R6_C3a_pair_records_PRIVATE.npz", **{k: np.array([float(r[k]) for r in recs]) for k in keys})
        out.update({"status": "COMPLETED", "results": public,
                    "runtime": {"wall_s": time.time() - t0, "fits": int(sum(r["fits"] for r in res)), "workers": workers,
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
    for flag in ("--plan-only", "--execute"):
        m.add_argument(flag, action="store_true")
    a = ap.parse_args()
    sys.exit(cmd_plan_only() if a.plan_only else cmd_execute())
