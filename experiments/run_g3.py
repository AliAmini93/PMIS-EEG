#!/usr/bin/env python3
"""HS-Stroke exact TS+LR identity swaps with placebo, neighbour, bystander and base controls."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.dont_write_bytecode = True
from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
import numpy as np  # noqa: E402

from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402

PUBLIC = ROOT / "results" / "G3_results.json"
PRIVATE_DIR = ROOT / "data" / "g3"



def log(msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE_DIR / "G3_run.log", "a") as fh:
        fh.write(line + "\n")


def build_plan():
    inp = g3.load_real_inputs(ROOT)
    plan = g3.make_plan(inp["inst"], inp["dist"], inp["meta"], inp["man"], inp["pairs"])
    return inp, plan


def cmd_plan_only():
    inp, plan = build_plan()
    print(json.dumps({"commitment": plan["commitment"], "fit_counts": plan["fit_counts"],
                      "checks": plan["checks"]}, indent=1))


def cmd_verify_only():
    protocol = load_protocol(CONFIG_ROOT / g3.PROTOCOL_JSON)
    ver = g3.verify_frozen(ROOT, protocol)
    print("PORT_PROTOCOL_VERIFIED", ver["port_manifest_sha256"])


def save_private(results, private):
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for seed in sorted({r["seed"] for r in results}):
        recs = sorted([dict(rec, subject=r["subject"], fold=r["fold"]) for r in results if r["seed"] == seed
                       for rec in r["records"]], key=lambda x: x["k"])
        keys = sorted({k for rec in recs for k in rec})
        for k in keys:
            arrays[f"s{seed}_{k}"] = np.array([float(rec.get(k, np.nan)) for rec in recs])
    np.savez(PRIVATE_DIR / "G3_pair_scores_PRIVATE.npz", **arrays)
    (PRIVATE_DIR / "G3_participant_effects_PRIVATE.json").write_text(json.dumps(private, indent=1) + "\n")


def cmd_execute():
    t_start = dt.datetime.now(dt.timezone.utc)
    protocol_path = CONFIG_ROOT / g3.PROTOCOL_JSON
    out = {"stage": "R5-G3 frozen PMIS TS+LR", "protocol_id": g3.PROTOCOL_ID,
           "protocol_sha256": pv.sha256_file(protocol_path), "timestamp_utc": t_start.isoformat()}
    try:
        protocol = load_protocol(protocol_path)
        ver = g3.verify_frozen(ROOT, protocol)
        out["port_manifest_sha256"] = ver["port_manifest_sha256"]
        log(f"PORT_PROTOCOL_VERIFIED {ver['port_manifest_sha256']}")
        os.nice(g3.NICE)
        start = g3.resource_state()
        workers = g3.start_guard(start)
        log(f"start guard ok: {start} -> workers {workers}")
        inp, plan = build_plan()
        g3._stop(reference_matches(plan["commitment"], protocol["commitments"]["plan_commitment_sha256"]), "plan commitment differs")
        g3._stop(plan["fit_counts"] == protocol["commitments"]["fit_counts"], "fit counts differ")
        g3._stop(reference_matches(protocol["data_sha256"]["G2_covs_8_30.npy"], pv.historical_sha256("G2_covs_8_30.npy")),
                 "protocol covariance digest differs from provenance")
        log(f"plan commitment ok, fits {plan['fit_counts']['total']}")
        covs, pre = g3.preflight_covariances(inp["paths"]["r4_private"] / g3.COVS_FILE, int(inp["dist"]["label"].size),
                                             protocol["data_sha256"]["G2_covs_8_30.npy"])
        log(f"preflight ok: {pre}")
        y = inp["dist"]["label"].astype(np.int64)
        ex = g3.execute(plan, covs, y, workers, log=log)
        g3._stop(sum(r["stats"]["fits"] for r in ex["results"]) == plan["fit_counts"]["total"], "fit count mismatch")
        public, private = g3.analyse(ex["results"])
        save_private(ex["results"], private)
        end = g3.resource_state()
        out.update({"status": "COMPLETED", "preflight": pre,
                    "resources": {"start": start, "min_mem_available_during_run": ex["min_resource_state"], "end": end,
                                  "workers": workers, "nice": g3.NICE},
                    "runtime": {"wall_s": ex["wall_s"], **g3.run_statistics(ex["results"])},
                    "results": public})
        code = 0
    except g3.G3Stop as e:
        out.update({"status": e.status, "stop_message": e.message})
        log(f"STOP {e.status}: {e.message}")
        code = 2
    except Exception as e:  # any other failure is a fail-closed stop
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
    for flag in ("--plan-only", "--verify-only", "--execute"):
        m.add_argument(flag, action="store_true")
    a = ap.parse_args()
    if a.plan_only:
        cmd_plan_only()
    elif a.verify_only:
        cmd_verify_only()
    else:
        sys.exit(cmd_execute())
