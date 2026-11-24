#!/usr/bin/env python3
"""R11-B G0: source-only audit of EXISTING Lee2019 outputs for the ranking-prediction feasibility study.

No model is fitted and no new arm is built. The script reads the git-ignored outputs of the frozen R6-C3a (TS+LR,
300-row base), R6-C3b (EEGNet, 300-row base), R7-C6 (EEGNet and ShallowConvNet, 240-row base + 60 validation) and
R8-D/E (TCFormer, CBraMod; C6 base, one C6 pair per task) runs and computes, per participant and then over 54:

  * near-clean held-out-run BA from the placebo arms (base + one extra copy of a base trial): on all 100 trials of
    the held-out run (C3b, C6, R8-D/E) and on the paired trials only (all sources; C3a keeps pair records only);
  * the single-copy exposed-trial BA contrast c = BA(own-copy arm) - BA(partner-copy arm) on the paired trials;
  * paired comparisons A - B on identical tasks / pair trials: near-clean gap, contrast difference dc, and the
    probability-scale d difference;
  * the implied parameter-free crossing dose delta* = -gap / dc (participant-bootstrap interval of the ratio of
    cohort means, 20,000 draws, seed 11_000).

Naming in all pair records is p<trial><arm> (pAB = trial a in arm B). Only cohort aggregates are written
(reports/aggregate/R11_B_G0_existing.json)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT

from experiments.pmis_r6 import c3, lee2019  # noqa: E402
from experiments.pmis_r7 import c6  # noqa: E402
from experiments.pmis_r8 import de_plan  # noqa: E402

D = ROOT / "data"
OUT = ROOT / "results" / "R11_B_G0_existing.json"
BOOT_B, BOOT_SEED = 20_000, 11_000


def ba(correct, lab) -> float:
    correct, lab = np.asarray(correct, dtype=float), np.asarray(lab)
    return 100.0 * float(np.mean([correct[lab == c].mean() for c in (1, 2)]))


def summ(v) -> dict:
    v = np.asarray(v, dtype=float)
    n, m, sd = v.size, float(v.mean()), float(v.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"n": n, "mean": m, "ci95": [m - h, m + h], "sd": sd, "n_pos": int((v > 0).sum()), "n_neg": int((v < 0).sum())}


def load_meta():
    z = np.load(D / "r6_c3" / "lee2019_meta.npz", allow_pickle=False)
    return {k: z[k] for k in ("subject", "session", "run", "pos", "label")}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    meta = load_meta()
    lab = np.asarray(meta["label"])
    parts = sorted(int(s) for s in np.unique(meta["subject"]))
    tasks = lee2019.build_tasks(meta)
    task_T = {(int(t["subject"]), int(t["fold"])): np.asarray(t["T"], dtype=np.int64) for t in tasks}
    plan6 = c6.build_plan(tasks, lab.astype(np.int64))
    plan8 = de_plan.subset_plan(plan6)
    plan3 = c3.build_eegnet_plan(tasks, fraction_den=4)
    for g in plan3["groups"]:
        g["T"] = task_T[(int(g["subject"]), int(g["fold"]))]

    recs = {}

    def new():
        return {k: {s: [] for s in parts} for k in ("clean", "own", "par", "plc", "lab2", "lab4", "dp")}

    def collect(name, plan, path, key):
        rec = new()
        for g in plan["groups"]:
            s, f = int(g["subject"]), int(g["fold"])
            with np.load(path(g), allow_pickle=False) as z:
                P = z[key]
            T = np.asarray(g["T"], dtype=np.int64)
            yl = lab[T]
            pos = {int(x): i for i, x in enumerate(T)}
            by = {(a["pair"], a["arm"]): m for m, a in enumerate(g["arms"])}
            for j, pr in enumerate(g["pairs"]):
                for arm in ("C", "E"):
                    rec["clean"][s].append(ba(P[by[(j, arm)]] > 0.5, yl))
                ia, ib = pos[int(pr["a"])], pos[int(pr["b"])]
                q = {arm: P[by[(j, arm)]] for arm in ("A", "B", "C", "E")}
                rec["own"][s] += [q["A"][ia] > 0.5, q["B"][ib] > 0.5]
                rec["par"][s] += [q["B"][ia] > 0.5, q["A"][ib] > 0.5]
                rec["plc"][s] += [q["C"][ia] > 0.5, q["E"][ia] > 0.5, q["C"][ib] > 0.5, q["E"][ib] > 0.5]
                rec["lab2"][s] += [int(pr["label"])] * 2
                rec["lab4"][s] += [int(pr["label"])] * 4
                rec["dp"][s].append(0.5 * ((q["A"][ia] - q["B"][ia]) + (q["B"][ib] - q["A"][ib])))
        recs[name] = rec

    for m in ("EEGNet", "Shallow"):
        for ro, key in (("final", "p_final"), ("es", "p_es")):
            collect(f"C6/{m}/{ro}", plan6, lambda g, m=m: D / "r7_c6" / "groups" / f"{m}_{g['subject']}_{g['fold']}.npz", key)
    for m in ("TCFormer", "CBraMod"):
        for ro, key in (("final", "p_final"), ("es", "p_es")):
            collect(f"R8DE/{m}/{ro}", plan8, lambda g, m=m: D / "r8_de" / "groups" / f"{m}_{g['subject']}_{g['fold']}.npz", key)
    collect("C3b/EEGNet300/final", plan3, lambda g: D / "r6_c3b" / "groups" / f"g_{g['subject']}_{g['fold']}_{g['group']}.npz", "p_T")

    z = np.load(D / "r6_c3" / "R6_C3a_pair_records_PRIVATE.npz", allow_pickle=False)
    rec = new()
    for i in range(z["subject"].size):
        s, L = int(z["subject"][i]), int(z["label"][i])
        rec["own"][s] += [z["pAA"][i] > 0.5, z["pBB"][i] > 0.5]
        rec["par"][s] += [z["pAB"][i] > 0.5, z["pBA"][i] > 0.5]
        rec["plc"][s] += [z["pAC"][i] > 0.5, z["pAE"][i] > 0.5, z["pBC"][i] > 0.5, z["pBE"][i] > 0.5]
        rec["lab2"][s] += [L] * 2
        rec["lab4"][s] += [L] * 4
        rec["dp"][s].append(0.5 * ((z["pAA"][i] - z["pAB"][i]) + (z["pBB"][i] - z["pBA"][i])))
    recs["C3a/TSLR300/exact"] = rec

    P = {}   # participant-level vectors
    table = {}
    for name, r in recs.items():
        v = {"own": np.array([ba(r["own"][s], r["lab2"][s]) for s in parts]),
             "par": np.array([ba(r["par"][s], r["lab2"][s]) for s in parts]),
             "plc_pairs": np.array([ba(r["plc"][s], r["lab4"][s]) for s in parts])}
        # probability-scale d, class-balanced within participant (as the frozen C3/C6 summaries)
        dp = []
        for s in parts:
            arr = np.array(r["dp"][s]) * 100.0
            L = np.array(r["lab2"][s][::2])
            dp.append(float(np.mean([arr[L == c].mean() for c in (1, 2)])))
        v["d_p"] = np.array(dp)
        v["c"] = v["own"] - v["par"]
        if any(r["clean"][s] for s in parts):
            v["clean_run"] = np.array([np.mean(r["clean"][s]) for s in parts])
        P[name] = v
        table[name] = {k: summ(x) for k, x in v.items()}
    out = {"stage": "R11-B G0", "fits": 0, "note": "existing outputs only; no new arm or fit",
           "per_model": table}

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(parts), size=(BOOT_B, len(parts)))

    def ratio_boot(gap, dc):
        g, d = gap[idx].mean(1), dc[idx].mean(1)
        r = -g / d
        return {"point": float(-gap.mean() / dc.mean()), "boot_ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                "boot_fraction_dc_nonpositive": float(np.mean(d <= 0))}

    comps = {}
    pairs = [("C6/Shallow/{r}", "C6/EEGNet/{r}", "clean_run", "same tasks, same 240-row base"),
             ("R8DE/TCFormer/{r}", "C6/EEGNet/{r}", "clean_run", "R8-D subset of C6 tasks"),
             ("C6/Shallow/{r}", "R8DE/TCFormer/{r}", "clean_run", "R8-D subset of C6 tasks"),
             ("C6/EEGNet/{r}", "C3a/TSLR300/exact", "plc_pairs", "paired trials only; TS+LR base 300 vs 240"),
             ("C6/Shallow/{r}", "C3a/TSLR300/exact", "plc_pairs", "paired trials only; TS+LR base 300 vs 240")]
    for ro in ("final", "es"):
        for a, b, gk, note in pairs:
            ka, kb = a.format(r=ro), b.format(r=ro)
            A, B = P[ka], P[kb]
            gap = A[gk] - B[gk]
            dc = A["c"] - B["c"]
            comps[f"{ka} - {kb}"] = {"note": note, "gap_measure": gk, "near_clean_gap": summ(gap),
                                      "near_clean_gap_paired_trials": summ(A["plc_pairs"] - B["plc_pairs"]),
                                      "contrast_difference_dc": summ(dc), "d_p_difference": summ(A["d_p"] - B["d_p"]),
                                      "ceiling_dc": summ((100 - A[gk]) - (100 - B[gk])),
                                      "crossing_dose_delta_star": ratio_boot(gap, dc)}
    out["comparisons"] = comps
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in table.items():
        print(k, {kk: (round(vv["mean"], 2), [round(x, 2) for x in vv["ci95"]]) for kk, vv in v.items()})
    for k, v in comps.items():
        print(k, "| gap", round(v["near_clean_gap"]["mean"], 3), [round(x, 2) for x in v["near_clean_gap"]["ci95"]],
              v["near_clean_gap"]["n_pos"], "/", v["near_clean_gap"]["n_neg"],
              "| dc", round(v["contrast_difference_dc"]["mean"], 3), [round(x, 2) for x in v["contrast_difference_dc"]["ci95"]],
              "| d_p", round(v["d_p_difference"]["mean"], 2),
              "| delta*", {kk: (round(vv, 3) if isinstance(vv, float) else [round(x, 3) for x in vv]) for kk, vv in v["crossing_dose_delta_star"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
