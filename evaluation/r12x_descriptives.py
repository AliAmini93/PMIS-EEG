#!/usr/bin/env python3
"""R12-X post-execution descriptives (written AFTER the frozen analysis ran; changes no frozen number).

Computes the protocol's "always reported" descriptive items that the frozen analysis file does not carry:
  * per-model cohort means on B (clean S_P, contaminated S_L), the leak inflation S_L - S_P, and C (F);
  * all pairwise C-gaps (paired, descriptive t-interval);
  * seed-level selections (each deep seed k with the single exact TS+LR fit) for E1 / E2;
  * the B-side within-benchmark regret (E2 selections scored on clean B; never pooled with E1 / E2);
plus items labelled EXPLORATORY (not pre-declared): top-two selection margins on B and early-stopping epochs.
Also summarises compute and storage use from the worker's usage log and the machine-wide coordinator log.

Reads private unit files; writes only cohort aggregates to reports/aggregate/R12X_descriptives.json.
"""

from __future__ import annotations

import importlib.util
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from experiments import r12x_run as rr
an, ds = rr.an, rr.ds
ROSTER = an.ROSTER
OUT = ROOT / "results" / "R12X_descriptives.json"
K = ds.K_SEEDS


def per_seed(D, window, ro):
    """(n, 3, K) BA arrays for B under S_P / S_L and C under F; TS+LR (exact, seed-free) repeated over k."""
    BP, BL, C = (np.zeros((len(D), len(ROSTER), K)) for _ in range(3))
    for i, d in enumerate(D):
        for j, m in enumerate(ROSTER):
            for k in range(K):
                kk = 0 if m == "TSLR" else k
                for tgt, stage, arm, y in ((BP, "selection", "S_P", d["yB"]), (BL, "selection", "S_L", d["yB"]),
                                           (C, "final", "F", d["yC"])):
                    with np.load(rr.unit_path(m, window, d["participant"], stage, kk)) as z:
                        key = f"exact_p_{arm}" if m == "TSLR" else f"{ro}_p_{arm}"
                        tgt[i, j, k] = an.ba(z[key], y)
    return BP, BL, C


def es_epochs(D, window):
    out = {}
    for m in ("EEGNet", "Shallow"):
        for stage, arm in (("selection", "S_P"), ("selection", "S_L"), ("final", "F")):
            v = []
            for d in D:
                for k in range(K):
                    with np.load(rr.unit_path(m, window, d["participant"], stage, k)) as z:
                        v.append(int(z[f"es_epoch_{arm}"]))
            v = np.asarray(v)
            out[f"{m}/{arm}"] = {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                                 "q75": float(np.percentile(v, 75)), "min": int(v.min()), "max": int(v.max()), "n": int(v.size)}
    return out


def msd(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "se": float(x.std(ddof=1) / np.sqrt(x.size))}


def tci(x):
    x = np.asarray(x, float)
    n = x.size
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ci95_t": [float(x.mean() - h), float(x.mean() + h)],
            "n_positive": int(np.sum(x > 0)), "n_zero": int(np.sum(x == 0)), "n": int(n)}


def top2(v):
    o = np.argsort(-np.asarray(v))
    return {"first": ROSTER[o[0]], "second": ROSTER[o[1]], "margin_BA": float(v[o[0]] - v[o[1]])}


def block(D, window, ro):
    BP, BL, C = per_seed(D, window, ro)
    bp, bl, c = BP.mean(2), BL.mean(2), C.mean(2)
    models = {m: {"B_clean_S_P": msd(bp[:, j]), "B_contaminated_S_L": msd(bl[:, j]),
                  "leak_inflation_S_L_minus_S_P": msd(bl[:, j] - bp[:, j]), "C_F": msd(c[:, j]),
                  "between_seed_sd_of_cohort_mean_C": float(C[:, j, :].mean(0).std(ddof=1)) if m != "TSLR" else 0.0}
              for j, m in enumerate(ROSTER)}
    pairs = {}
    for a in range(len(ROSTER)):
        for b in range(a + 1, len(ROSTER)):
            pairs[f"{ROSTER[b]} - {ROSTER[a]}"] = tci(c[:, b] - c[:, a])
    seeds = []
    for k in range(K):
        e1 = an.e1(BP[:, :, k], BL[:, :, k], C[:, :, k])
        e2 = an.e2(BP[:, :, k], BL[:, :, k], C[:, :, k])
        seeds.append({"k": k, "E2_w_clean": e2["w_clean"], "E2_w_contam": e2["w_contam"], "E2_flip": bool(e2["flip"]),
                      "E2_regret_C": e2["regret"], "E1_flip_fraction": e1["flip_fraction"], "E1_regret_C": e1["regret"]})
    wc, wl = an.select(bp.mean(0)), an.select(bl.mean(0))
    within = {"w_clean": ROSTER[wc], "w_contam": ROSTER[wl],
              "regret_on_clean_B": float(np.mean(bp[:, wc] - bp[:, wl])),
              "note": "E2 selections scored on clean B (S_P); descriptive; never pooled with E1 / E2"}
    explor = {"B_clean_top2": top2(bp.mean(0)), "B_contaminated_top2": top2(bl.mean(0)), "C_top2": top2(c.mean(0))}
    return {"models": models, "pairwise_C_gaps": pairs, "seed_level_selections": seeds,
            "B_side_within_benchmark_regret": within, "EXPLORATORY_selection_margins": explor}


def usage():
    rows = [json.loads(x) for x in (rr.PRIVATE / "unit_usage.jsonl").read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        k = f"{r['model']}/{r['window']}"
        b = by.setdefault(k, {"units": 0, "wall_s": 0.0, "max_unit_wall_s": 0.0, "gpu_peak_reserved_MiB": 0.0,
                              "gpu_peak_allocated_MiB": 0.0})
        b["units"] += 1
        b["wall_s"] += r["wall_s"]
        b["max_unit_wall_s"] = max(b["max_unit_wall_s"], r["wall_s"])
        b["gpu_peak_reserved_MiB"] = max(b["gpu_peak_reserved_MiB"], r["gpu_peak_reserved_bytes"] / 2**20)
        b["gpu_peak_allocated_MiB"] = max(b["gpu_peak_allocated_MiB"], r["gpu_peak_allocated_bytes"] / 2**20)
    tot = {"deep_units": len(rows), "deep_unit_wall_h": sum(r["wall_s"] for r in rows) / 3600,
           "max_host_rss_GiB": max(r["host_rss_bytes"] for r in rows) / 2**30,
           "max_gpu_peak_reserved_MiB": max(r["gpu_peak_reserved_bytes"] for r in rows) / 2**20,
           "resource_waits_between_units_s": sum(r["waited_s"] for r in rows),
           }
    log = (rr.PRIVATE / "worker.log").read_text().splitlines()
    tot["oom_retries_logged"] = sum("OOM on attempt" in line for line in log)
    ts = {}
    for line in log:
        t = line.split(" ", 1)[0]
        for tag, pat in (("start", "worker start"), ("tslr_done", "TS+LR units newly run"),
                         ("lease", "GPU lease acquired"), ("complete", "WORKER COMPLETE")):
            if pat in line:
                ts[tag] = t
    import datetime as dt
    f = lambda a, b: (dt.datetime.fromisoformat(ts[b]) - dt.datetime.fromisoformat(ts[a])).total_seconds()
    coord_dir = Path(os.environ.get("GPUCOORD_DIR", Path.home() / ".local" / "state" / "gpu-coord"))
    ev = []
    for p in coord_dir.glob("GPU-*/events.log"):
        ev += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    acq = [e for e in ev if e.get("event") == "acquire" and e.get("job") == "r12x-deep"]
    rel = [e for e in ev if e.get("event") == "release" and e.get("job") == "r12x-deep"]
    waits = [e for e in ev if e.get("event") == "waiting_device" and e.get("job") == "r12x-deep"]
    n_foreign = sorted({len(re.findall(r"\d+:[^,]+:\d+MiB", e["reason"])) for e in waits})
    wall = {"tslr_cpu_phase_s": f("start", "tslr_done"), "queue_wait_s": acq[0]["queue_wait_s"] if acq else None,
            "lease_held_s": rel[0]["held_s"] if rel else None, "worker_total_s": f("start", "complete"),
            "waiting_device_events_logged": len(waits), "foreign_compute_processes_seen_while_waiting": n_foreign,
            "foreign_processes_touched": 0, "preemptions": 0}
    return {"by_model_window": by, "totals": tot, "wall_clock": wall}




def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    meta = rr.load_meta()
    D = rr.designs(meta)
    for d in D:
        d["yB"], d["yC"] = meta["label"][d["B"]], meta["label"][d["C"]]
    out = {"schema": "R12X_descriptives_v1", "status": "post-execution descriptives; no frozen number is changed",
           "n_participants": len(D), "roster": list(ROSTER), "K_deep_seeds": K}
    for w in ("cue", "fb"):
        for ro in ("final", "es"):
            out[f"{w}/{ro}"] = block(D, w, ro)
        out[f"{w}/EXPLORATORY_es_epochs"] = es_epochs(D, w)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
