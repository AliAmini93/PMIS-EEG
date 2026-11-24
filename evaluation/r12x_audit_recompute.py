#!/usr/bin/env python3
"""R12-X independent recomputation audit (separate code path; written before any outcome existed).

Does NOT import pmis_r12x.analysis. Re-derives B = R5 and C = R6 trial sets from the feature metadata; counts every
unit file per (model, window, stage, seed) against the frozen design; re-checks every A/A pair bit-for-bit from the
saved outputs; checks the C firewall against the design arms; recomputes balanced accuracy, selections (own argmax
with the frozen 0.01-BA tie rule), E1 / E2 / E3 regrets, the bootstrap (vectorised implementation from
scripts/r12x_g1r_calibration.py, same seeds) and classes; then compares every number in
reports/aggregate/R12X_results.json. Writes reports/aggregate/R12X_audit_recompute.json.
"""

from __future__ import annotations

import importlib.util
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, reference_matches

ROOT = RUN_ROOT
from experiments.pmis_r12x import design as ds  # noqa: E402  (partitions only; checked against an independent derivation)

from experiments import r12x_g1r_calibration as cal
from experiments import r12x_run
PRIV = Path(os.environ.get("R12X_PRIVATE_DIR", ROOT / "data" / "r12x"))              # overrides: synthetic tests only
RESULTS = Path(os.environ.get("R12X_RESULTS_PATH", ROOT / "results" / "R12X_results.json"))
PROTO_PATH = Path(os.environ.get("R12X_PROTOCOL_PATH", CONFIG_ROOT / "R12X_PROTOCOL.json"))
AUDIT_OUT = Path(os.environ.get("R12X_AUDIT_PATH", ROOT / "results" / "R12X_audit_recompute.json"))
ROSTER = ("TSLR", "EEGNet", "Shallow")
TOL = 0.01


def ba(p, y):
    c = np.asarray(p) > 0.5
    return 50.0 * (c[y == 1].mean() + c[y == 2].mean())


def pick(v):
    v = np.asarray(v, float)
    for j in range(len(v)):
        if v[j] >= v.max() - TOL:
            return j


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    res = json.loads(RESULTS.read_text())
    proto = load_protocol(PROTO_PATH) if PROTO_PATH.parent == CONFIG_ROOT else json.loads(PROTO_PATH.read_text())
    z = np.load(PRIV / "features" / "meta.npz", allow_pickle=False)
    meta = {k: z[k] for k in z.files}
    parts = sorted(np.unique(meta["participant"]).tolist())
    findings, counts = [], {}
    D, designs = [], []
    for p in parts:
        m = meta["participant"] == p
        idx = np.flatnonzero(m)
        Bi = np.sort(idx[meta["run"][m] == "R5"])
        Ci = np.sort(idx[meta["run"][m] == "R6"])
        d = ds.build_participant(int(p), meta["run"][m], meta["pos"][m], meta["label"][m], idx)
        if not (np.array_equal(d["B"], Bi) and np.array_equal(d["C"], Ci)):
            findings.append(f"B/C derivation differs for participant {p}")
        for name, rows in d["arms"].items():
            if np.intersect1d(rows, Ci).size:
                findings.append(f"C in arm {name} participant {p}")
        if np.intersect1d(d["pool_val"], Ci).size or np.intersect1d(d["A_val"], Ci).size:
            findings.append(f"C in validation rows participant {p}")
        designs.append(d)
        D.append((p, Bi, Ci, meta["label"][Bi], meta["label"][Ci]))
    commit = ds.commitment(designs)
    if not reference_matches(commit, proto["design_commitment"]):
        findings.append("design commitment differs")
    mats, aa_checked = {}, 0
    for w in ("cue", "fb", "precue", "eogcue", "eogfb"):
        for ro in ("final", "es"):
            M = {k: np.full((len(D), 3), np.nan) for k in ("BBP", "BBL", "BC")}
            for i, (p, Bi, Ci, yB, yC) in enumerate(D):
                for j, model in enumerate(ROSTER):
                    if model != "TSLR" and w not in ("cue", "fb"):
                        continue
                    ks = [0] if model == "TSLR" else range(ds.K_SEEDS)
                    acc = {"BBP": [], "BBL": [], "BC": []}
                    for k in ks:
                        for stage in ("selection", "final"):
                            fp = PRIV / "units" / f"{model}_{w}" / f"u_{p}_{stage}_{k}.npz"
                            if not fp.exists():
                                findings.append(f"missing {fp.name} ({model} {w})")
                                continue
                            with np.load(fp) as u:
                                if str(u["commitment"]) != commit:
                                    findings.append(f"foreign commitment {fp.name}")
                                if ro == "final":
                                    counts[(model, w, stage)] = counts.get((model, w, stage), 0) + 1
                                pre = "exact_p_" if model == "TSLR" else f"{ro}_p_"
                                if stage == "selection":
                                    pairs = [("S_L", "AA_L")]
                                    acc["BBP"].append(ba(u[pre + "S_P"], yB))
                                    acc["BBL"].append(ba(u[pre + "S_L"], yB))
                                    if u[pre + "S_P"].shape != (40,):
                                        findings.append(f"selection scored set size {fp.name}")
                                else:
                                    pairs = [("F", "AA_F")]
                                    acc["BC"].append(ba(u[pre + "F"], yC))
                                for a, b in pairs:
                                    keys = ["exact_p_"] if model == "TSLR" else ["final_p_", "es_p_", "es_epoch_"]
                                    for kk in keys:
                                        aa_checked += 1
                                        if not np.array_equal(u[kk + a], u[kk + b]):
                                            findings.append(f"A/A mismatch {fp.name} {kk}{a}")
                    for key, v in acc.items():
                        if v:
                            M[key][i, j] = float(np.mean(v))
            mats[(w, ro)] = M
    expected = {}
    for w in ("cue", "fb", "precue", "eogcue", "eogfb"):
        for stage in ("selection", "final"):
            expected[("TSLR", w, stage)] = len(D)
            if w in ("cue", "fb"):
                for model in ("EEGNet", "Shallow"):
                    expected[(model, w, stage)] = len(D) * ds.K_SEEDS
    for key, n in expected.items():
        if counts.get(key, 0) != n:
            findings.append(f"unit count {key}: {counts.get(key, 0)} != {n}")
    # recompute the frozen analysis for the two deep windows
    comp, mism = {}, []
    n = len(D)
    blk = np.zeros(n, int)
    for g in ("A", "B", "C"):
        mem = sorted(p for p in parts if (p <= 60 and g == "A") or (61 <= p <= 81 and g == "B") or (p >= 82 and g == "C"))
        for r, p in enumerate(mem):
            blk[parts.index(p)] = r % 2
    for (w, ro), M in mats.items():
        if w not in ("cue", "fb"):
            continue
        BBP, BBL, BC = M["BBP"], M["BBL"], M["BC"]
        tp, tl = BBP.sum(0), BBL.sum(0)
        wc = [pick((tp - BBP[i]) / (n - 1)) for i in range(n)]
        wl = [pick((tl - BBL[i]) / (n - 1)) for i in range(n)]
        r1 = np.array([BC[i, wc[i]] - BC[i, wl[i]] for i in range(n)])
        c2, l2 = pick(BBP.mean(0)), pick(BBL.mean(0))
        r2 = BC[:, c2] - BC[:, l2]
        r3 = np.zeros(n)
        for b in (0, 1):
            o = blk != b
            r3[blk == b] = BC[blk == b, pick(BBP[o].mean(0))] - BC[blk == b, pick(BBL[o].mean(0))]
        R1, _ = cal.boot(BBP, BBL, BC, "E1", 20000, 12010)
        R2, _ = cal.boot(BBP, BBL, BC, "E2", 20000, 12010)
        q1, q2 = np.quantile(R1, [0.025, 0.05, 0.975]), np.quantile(R2, [0.025, 0.05, 0.975])
        anyflip = bool(np.any(np.array(wc) != np.array(wl)))
        cls1 = ("NO_FLIP" if not anyflip else "FLIP_AND_REGRET" if q1[1] > 0 else "REVERSE_REGRET" if q1[2] < 0
                else "FLIP_NO_REGRET")
        cls2 = ("NO_FLIP" if c2 == l2 else "FLIP_AND_REGRET" if q2[1] > 0 else "REVERSE_REGRET" if q2[2] < 0
                else "FLIP_NO_REGRET")
        mine = {"E1/regret": float(r1.mean()), "E1/flip_fraction": float(np.mean(np.array(wc) != np.array(wl))),
                "E1/ci95_lo": float(q1[0]), "E1/ci95_hi": float(q1[2]), "E1/lower95_one_sided": float(q1[1]),
                "E1/class": cls1, "E2/regret": float(r2.mean()), "E2/w_clean": ROSTER[c2], "E2/w_contam": ROSTER[l2],
                "E2/ci95_lo": float(q2[0]), "E2/ci95_hi": float(q2[2]), "E2/class": cls2, "E3/regret": float(r3.mean()),
                "mean_BA_B_clean": BBP.mean(0).tolist(), "mean_BA_B_contam": BBL.mean(0).tolist(),
                "mean_BA_C": BC.mean(0).tolist()}
        rep = res["all"][f"{w}/{ro}"]
        theirs = {"E1/regret": rep["E1"]["regret"], "E1/flip_fraction": rep["E1"]["flip_fraction"],
                  "E1/ci95_lo": rep["E1"]["ci95"][0], "E1/ci95_hi": rep["E1"]["ci95"][1],
                  "E1/lower95_one_sided": rep["E1"]["lower95_one_sided"], "E1/class": rep["E1"]["class"],
                  "E2/regret": rep["E2"]["regret"], "E2/w_clean": rep["E2"]["w_clean"], "E2/w_contam": rep["E2"]["w_contam"],
                  "E2/ci95_lo": rep["E2"]["ci95"][0], "E2/ci95_hi": rep["E2"]["ci95"][1], "E2/class": rep["E2"]["class"],
                  "E3/regret": rep["E3"]["regret"]}
        for k, v in theirs.items():
            ok = v == mine[k] if isinstance(v, str) else abs(v - mine[k]) <= 1e-9
            if not ok:
                mism.append(f"{w}/{ro} {k}: reported {v} vs recomputed {mine[k]}")
        comp[f"{w}/{ro}"] = mine
    sp = r12x_run.sensitivity_participants(proto)
    if sp:   # sensitivity (G7 as declared): recompute E1 on the subset with re-derived selections
        keep = [i for i, d in enumerate(D) if d[0] in set(sp)]
        M = mats[("cue", "final")]
        BBP, BBL, BC = M["BBP"][keep], M["BBL"][keep], M["BC"][keep]
        m = len(keep)
        tp, tl = BBP.sum(0), BBL.sum(0)
        wc = [pick((tp - BBP[i]) / (m - 1)) for i in range(m)]
        wl = [pick((tl - BBL[i]) / (m - 1)) for i in range(m)]
        r1 = np.array([BC[i, wc[i]] - BC[i, wl[i]] for i in range(m)])
        Rs, _ = cal.boot(BBP, BBL, BC, "E1", 20000, 12010)
        qs = np.quantile(Rs, [0.025, 0.05, 0.975])
        rep = res["sensitivity_G7_as_declared_descriptive"]["cue/final"]["E1"]
        for k, a, b in (("regret", rep["regret"], float(r1.mean())), ("ci95_lo", rep["ci95"][0], float(qs[0])),
                        ("ci95_hi", rep["ci95"][1], float(qs[2])), ("lower95", rep["lower95_one_sided"], float(qs[1]))):
            if abs(a - b) > 1e-9:
                mism.append(f"sensitivity {k}: reported {a} vs recomputed {b}")
        comp["sensitivity_cue/final"] = {"n": m, "E1/regret": float(r1.mean()), "E1/ci95": [float(qs[0]), float(qs[2])]}
    out = {"schema": "R12X_audit_recompute_v1", "participants": n, "unit_counts": {"/".join(k): v for k, v in counts.items()},
           "aa_pairs_checked": aa_checked, "findings": findings, "mismatches": mism, "recomputed": comp,
           "pass": not findings and not mism}
    AUDIT_OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"pass": out["pass"], "aa_pairs_checked": aa_checked, "findings": len(findings), "mismatches": len(mism)}))


if __name__ == "__main__":
    main()
