#!/usr/bin/env python3
"""HS-Stroke natural benchmark, retrospective comparison and one-step law."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches
from experiments.runtime import (
    CONFIG_ROOT, RUN_ROOT, json_sha256, load_protocol, port_identity, verify_recorded_input,
)

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from data_preparation import hs_metadata as ad  # noqa: E402
from experiments.pmis_r7 import c7
from evaluation import r7_c7_analysis as c7_analysis  # noqa: E402

PROTOCOL_ID = "R7-C7-BENCHMARK-INFLATION-TSLR-v1.1"
PROTOCOL = CONFIG_ROOT / "R7_C7_PROTOCOL_v1_1.json"
PRIVATE = ROOT / "data" / "r7_c7"
PUBLIC = ROOT / "results" / "R7_C7_results.json"
R4_PRED = "G3_tslr_predictions_PRIVATE.npz"
TRIAL_ARRAYS = PRIVATE / "R7_C7_trial_arrays_PRIVATE.npz"


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R7_C7_run.log", "a") as fh:
        fh.write(line + "\n")


def load():
    paths = ad.local_paths(ROOT)
    priv = paths["r4_private"]
    frozen = ad.frozen_check(priv)
    g3._stop(frozen["all_match"], "R4 metadata or vendored code differ from frozen hashes")
    inst, dist, meta, man, _ = ad.load_metadata(priv)
    man = dict(man)
    with np.load(priv / "G2_manifest.npz", allow_pickle=False) as z:  # S1 masks too (the adapter loads S2 only)
        man.update({k: z[k] for k in z.files if k.startswith("exposed_")})
    return paths, priv, dist, man


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    env = g3.environment()
    g3._stop(env["blas_openmp_threads"] == [1], "BLAS must be single threaded")
    return proto


def reproduce_legacy(A, priv, dist, man, proto):
    g3._stop(verify_recorded_input(priv / R4_PRED, proto["data_sha256"][R4_PRED]),
             "Regenerated legacy benchmark predictions differ from the frozen reference")
    with np.load(priv / R4_PRED, allow_pickle=False) as arrays:
        r4_pred = dict(arrays)
    r4_public = json.loads((priv / "G3_tslr_results.json").read_text())
    g3._stop(reference_matches(json_sha256(r4_public["splits"]), proto["legacy_aggregate_splits_sha256"]),
             "Regenerated legacy benchmark contrasts differ from the frozen reference")
    return c7_analysis.analyse(A, dist, man, r4_pred, r4_public)


def cmd_analyse():
    """Recover the analysis of a completed fit using the frozen array-content receipt."""
    proto = verify()
    g3._stop(verify_recorded_input(TRIAL_ARRAYS, proto["trial_array_reference_sha256"]),
             "Completed benchmark arrays differ from the frozen reference")
    _, priv, dist, man = load()
    A = {}
    with np.load(TRIAL_ARRAYS, allow_pickle=False) as arrays:
        for split, name in c7_analysis.SPLITS.items():
            for seed in c7_analysis.SEEDS:
                prefix = f"{name}_{seed}_"
                A[split, seed] = {key.removeprefix(prefix): arrays[key]
                                  for key in arrays.files if key.startswith(prefix)}
    result = reproduce_legacy(A, priv, dist, man, proto)
    out = {"stage": "R7-C7", "protocol_id": PROTOCOL_ID, "port_manifest_sha256": port_identity(),
           "protocol_sha256": pv.sha256_file(PROTOCOL), "status": "COMPLETED", "results": result}
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(out["status"])
    return 0


def cmd_execute():
    out = {"stage": "R7-C7", "protocol_id": PROTOCOL_ID, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        out["protocol_sha256"] = pv.sha256_file(PROTOCOL)
        log(f"PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        start = g3.resource_state()
        workers = g3.start_guard(start)
        paths, priv, dist, man = load()
        covs, pre = g3.preflight_covariances(priv / g3.COVS_FILE, int(dist["label"].size), proto["data_sha256"]["G2_covs_8_30.npy"])
        c7.G.update({"covs": covs, "label": dist["label"].astype(np.int64), "rows": man["rows"], "table": man["table"]})
        tasks = sorted({(int(r[0]), int(r[1]), int(r[2])) for r in man["table"]})
        g3._stop({"S1": sum(1 for x in tasks if x[0] == 1), "S2": sum(1 for x in tasks if x[0] == 2)} == proto["tasks"], "task count")
        log(f"start {start} workers {workers} tasks {len(tasks)}")
        from multiprocessing import get_context
        pool = get_context("fork").Pool(workers)
        res, t0, min_avail = [], time.time(), None
        try:
            for i, r in enumerate(pool.imap_unordered(c7.run_task, tasks, chunksize=4)):
                res.append(r)
                if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                    st = g3.resource_state()
                    min_avail = st["mem_available_mb"] if min_avail is None else min(min_avail, st["mem_available_mb"])
                    g3.run_guard(st)
                    log(f"task {i + 1}/{len(tasks)} {time.time() - t0:.0f} s")
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()
        A = c7_analysis.assemble(res, int(dist["label"].size))
        PRIVATE.mkdir(parents=True, exist_ok=True)
        np.savez(TRIAL_ARRAYS,
                 **{f"S{sp}_{sd}_{k}": v for (sp, sd), d in A.items() for k, v in d.items()})
        public = reproduce_legacy(A, priv, dist, man, proto)
        out.update({"status": "COMPLETED", "preflight": pre, "results": public,
                    "runtime": {"wall_s": time.time() - t0, "tasks": len(res), "workers": workers,
                                "single_class_r4_fits": int(sum(r["single_class_r4_fits"] for r in res)),
                                "law_skipped_folds": int(sum(r["law_skipped_folds"] for r in res)),
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
    m.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    sys.exit(cmd_analyse() if a.analyse else cmd_execute())
