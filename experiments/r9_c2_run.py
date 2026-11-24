#!/usr/bin/env python3
"""HS-Stroke randomized split halves with exact TS+LR refits."""

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
from data_preparation import hs_metadata as ad  # noqa: E402
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r7 import c7  # noqa: E402
from experiments.pmis_r8 import bench  # noqa: E402
from evaluation import r9_split_analysis as split_analysis
from experiments.pmis_r9 import split_bench, split_tslr  # noqa: E402

PROTOCOL_ID = "R9-C2-SPLITHALF-SUPPORT-EXPOSURE-TSLR-HSSTROKE-v1"
PROTOCOL = CONFIG_ROOT / "R9_C2_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r9_c2"
PUBLIC = ROOT / "results" / "R9_C2_results.json"
SEEDS = (42, 43, 44)
C7REF = ROOT / "data" / "r7_c7" / "R7_C7_trial_arrays_PRIVATE.npz"
R9C_PUBLIC = ROOT / "results" / "R9_C_results.json"


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R9_C2_run.log", "a") as fh:
        fh.write(line + "\n")


def load():
    priv = ad.local_paths(ROOT)["r4_private"]
    g3._stop(ad.frozen_check(priv)["all_match"], "R4 metadata differ from frozen hashes")
    inst, dist, meta, man, _ = ad.load_metadata(priv)
    return priv, dist, man, bench.tasks(man["table"], SEEDS)


def plan_summary():
    _, dist, man, tl = load()
    return {"folds": len(tl), "unit_tasks": len(split_tslr.unit_tasks(tl)),
            "aa_folds": int(sum(1 for i in range(len(tl)) if i % 10 == 0)),
            "commitment_identical_to_R9_C": split_bench.commitment(man["table"], man["rows"], dist, tl)}


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    env = g3.environment()
    g3._stop(env["blas_openmp_threads"] == [1], "BLAS must be single threaded")
    return proto


def reference(dist):
    z = np.load(C7REF)
    y = dist["label"]
    ref = {}
    for s in SEEDS:
        ref[s] = {}
        for a in ("L", "P"):
            p = z[f"S2_{s}_r4_p_{a}"]
            pred = z[f"S2_{s}_r4_pred_{a}"]
            ref[s][f"p_{a}"] = p
            ref[s][f"c_{a}"] = np.where(np.isfinite(p), (pred == y).astype(float), np.nan)
    return ref


def cmd_execute():
    out = {"stage": "R9-C2", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        log(f"PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        start = g3.resource_state()
        workers = g3.start_guard(start)
        priv, dist, man, tl = load()
        g3._stop(reference_matches(split_bench.commitment(man["table"], man["rows"], dist, tl), proto["plan"]["commitment_identical_to_R9_C"]),
                 "plan commitment differs")
        g3._stop(verify_recorded_input(C7REF, proto["data_sha256"]["R7_C7_trial_arrays_PRIVATE.npz"]), "R7-C7 arrays changed")
        covs, pre = g3.preflight_covariances(priv / g3.COVS_FILE, int(dist["label"].size), proto["data_sha256"]["G2_covs_8_30.npy"])
        c7.G.update({"covs": covs, "label": dist["label"].astype(np.int64), "rows": man["rows"], "table": man["table"],
                     "dist": dist, "exposed": {s: man[f"exposed_S2_{s}"] for s in SEEDS}})
        tasks = split_tslr.unit_tasks(tl)
        log(f"start {start} workers {workers} unit tasks {len(tasks)}")
        from multiprocessing import get_context
        pool = get_context("fork").Pool(workers)
        res, t0, min_avail = [], time.time(), None
        try:
            for i, r in enumerate(pool.imap_unordered(split_tslr.run_unit, tasks, chunksize=1)):
                res.append(r)
                st = g3.resource_state()
                min_avail = st["mem_available_mb"] if min_avail is None else min(min_avail, st["mem_available_mb"])
                g3.run_guard(st)
                if (i + 1) % 6 == 0 or i + 1 == len(tasks):
                    log(f"unit task {i + 1}/{len(tasks)} {time.time() - t0:.0f} s")
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()
        n = int(dist["label"].size)
        A, half, aa = split_tslr.assemble(res, n, SEEDS)
        PRIVATE.mkdir(parents=True, exist_ok=True)
        np.savez(PRIVATE / "R9_C2_trial_arrays_PRIVATE.npz", **{f"{s}_{k}": v for s, d in A.items() for k, v in d.items()},
                 **{f"{s}_half": half[s] for s in SEEDS})
        g3._stop(sum(len(r["folds"]) for r in res) == len(tl), "folds missing")
        g3._stop(len(aa) == proto["plan"]["aa_folds"] and max(aa) == 0.0, f"A/A refit not exact ({max(aa) if aa else None})")
        ref = reference(dist)
        for s in SEEDS:
            for a in ("L", "P"):
                m = np.isfinite(A[s][f"p_{a}"])
                g3._stop(np.array_equal(m, np.isfinite(ref[s][f"p_{a}"])), f"scored set differs from R7-C7 ({s}, {a})")
                g3._stop(np.array_equal(A[s][f"p_{a}"][m], ref[s][f"p_{a}"][m]) and
                         np.array_equal(A[s][f"c_{a}"][m], ref[s][f"c_{a}"][m]), f"{a} does not reproduce R7-C7 (seed {s})")
        results = split_analysis.analyse(A, half, dist, SEEDS, ref)
        results["descriptive"]["reproduction_of_R7_C7_arms"] = results["descriptive"].pop("reproduction_of_R8B_arms")
        r9c = json.loads(R9C_PUBLIC.read_text())["results"]
        results["side_by_side_EEGNet_R9_C"] = {
            "C1": r9c["family_holm"][split_analysis.FAMILY[0]]["mean_pp"],
            "C2": r9c["family_holm"][split_analysis.FAMILY[1]]["mean_pp"],
            "support_share_half": r9c["descriptive"]["support_share_of_total_half_ratio_of_means"],
            "full_L_minus_P_exposed": r9c["descriptive"]["full_LEAKED_minus_PLACEBO_BA_exposed"]["mean_pp"]}
        out.update({"status": "COMPLETED", "preflight": pre, "plan": proto["plan"], "results": results,
                    "runtime": {"wall_s": time.time() - t0, "unit_tasks": len(res), "workers": workers,
                                "single_class_fits": int(sum(r["single_class_fits"] for r in res)),
                                "aa_checks": len(aa), "aa_max_abs": max(aa),
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
    print(out["status"])
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute"):
        m.add_argument(flag, action="store_true")
    a = ap.parse_args()
    if a.plan_only:
        print(json.dumps(plan_summary()))
        sys.exit(0)
    sys.exit(cmd_execute())
