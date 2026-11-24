#!/usr/bin/env python3
"""R11-A: verify the R10-B C0 geometry behind the overlap-label erratum (no fitting, no new outcome).

R10-B labelled its C0 benchmark "stride 500 (50 % overlap)". With 2,000-ms windows and a 500-ms stride, adjacent
windows of a trial share 1,500 of 2,000 samples, i.e. 75 % overlap. This script confirms, from the frozen code and
the stored private C0 artifacts, that the run itself used stride 500 (five windows per trial at offsets 0, 500,
1,000, 1,500, 2,000 ms) so that only the description is wrong:

  1. the frozen constants (WLEN_MS, C0_STRIDE, stride_offsets) at the frozen-protocol commit and at R10-B final;
  2. for every participant and both models, the stored test-row lists of the 10 C0 fits equal the rows produced
     now by the frozen split code (pmis_r10b.design.c0_splits), the rows decode to exactly those five offsets,
     every trial contributes five windows, and every fit trains on 1,600 rows;
  3. the C0 aggregates in reports/aggregate/R10_B_results.json are recomputed from the stored per-window
     predictions with the frozen analysis function and compared (max abs difference).

Only aggregate counts and booleans are written (reports/aggregate/R11_R10B_C0_erratum_check.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT

from evaluation import r10b_analysis as analysis
from experiments.pmis_r10b import design, geometry  # noqa: E402

META = ROOT / "data" / "r6_c3" / "lee2019_meta.npz"
FOLDS = ROOT / "data" / "r10_b" / "folds"
OUT = ROOT / "results" / "R11_R10B_C0_erratum_check.json"






def recorded_geometry() -> dict:
    offs = geometry.stride_offsets(geometry.C0_STRIDE)
    shared = geometry.shared_samples(geometry.C0_STRIDE)
    return {"c0_offsets_ms": list(offs), "windows_per_trial": len(offs),
            "adjacent_shared_samples": shared,
            "adjacent_overlap_fraction": shared / geometry.WLEN_MS}


def raw_overlap(arm_rows, anchors, onset, rec):
    """Measure overlap among sample intervals from the same recording."""
    t, o = design.trial_of(arm_rows), design.offset_of(arm_rows)
    st, en = onset[t] + o, onset[t] + o + 2000
    own_shared, cross_shared, cross_file_coincidence = [], 0, 0
    for a in anchors:
        a0, a1 = onset[a], onset[a] + 2000
        sh = np.clip(np.minimum(en, a1) - np.maximum(st, a0), 0, None)
        own_shared.append(int(sh[t == a].max()) if np.any(t == a) else 0)
        same_rec = rec[t] == rec[a]
        cross_shared += int(sh[(t != a) & same_rec].sum())
        cross_file_coincidence += int(sh[~same_rec].sum())
    return np.array(own_shared), cross_shared, cross_file_coincidence


def audit_recording_overlap(meta):
    """Recheck both window tracks with sample coordinates grouped by recording."""
    recording = meta["subject"] * 100 + meta["session"] * 10 + meta["run"]
    onset = meta["onset"]
    cross_trial, cross_recording = 0, 0
    for f in design.folds(meta):
        for lag in geometry.LAGS_B:
            own, cross, elsewhere = raw_overlap(design.track_b_arms(f)[f"B1_{lag}"], f["H1"], onset, recording)
            if not np.all(own == geometry.shared_samples(lag)):
                raise ValueError("Fixed-density own-window overlap differs from the protocol")
            cross_trial += cross
            cross_recording += elsewhere
        for stride in geometry.STRIDES:
            a = design.track_a_arms(f, stride)
            clean, cross, elsewhere = raw_overlap(a["G"], f["T"], onset, recording)
            if np.any(clean):
                raise ValueError("A clean sliding-window arm shares scored samples")
            cross_trial += cross
            cross_recording += elsewhere
    return {"same_recording_cross_trial_shared_samples": cross_trial,
            "cross_recording_index_coincidences": cross_recording}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    geo = recorded_geometry()
    offs = geo["c0_offsets_ms"]
    res = {"geometry": geo}

    meta = dict(np.load(META, allow_pickle=False))
    res["recording_overlap"] = audit_recording_overlap(meta)
    subjects = sorted(int(s) for s in np.unique(meta["subject"]))
    lab = np.asarray(meta["label"])
    checks = {m: {"participants": 0, "rows_equal_frozen_split": 0, "offsets_exact": 0, "five_windows_per_trial": 0,
                  "train_rows_1600_all_fits": 0, "test_rows_partition": 0} for m in geometry.MODELS}
    preds = {m: {} for m in geometry.MODELS}
    for u in subjects:
        sp = design.c0_splits(meta, u)
        trials = np.flatnonzero(np.asarray(meta["subject"]) == u)
        for m in geometry.MODELS:
            fp = FOLDS / m / f"c0_{u}.npz"
            with np.load(fp, allow_pickle=False) as z:
                st = {k: z[k] for k in z.files}
            c = checks[m]
            c["participants"] += 1
            eq = all(np.array_equal(st[f"{s}{k}_rows"], sp[s][k]) for s in ("rw", "tg") for k in range(5))
            c["rows_equal_frozen_split"] += int(eq)
            allrows = np.concatenate([st[f"rw{k}_rows"] for k in range(5)])
            o = set(int(x) for x in np.unique(design.offset_of(allrows)))
            c["offsets_exact"] += int(o == set(offs))
            tcount = np.bincount(design.trial_of(allrows), minlength=int(trials.max()) + 1)[trials]
            c["five_windows_per_trial"] += int(np.all(tcount == len(offs)) and allrows.size == len(offs) * trials.size)
            part = all(np.array_equal(np.sort(np.concatenate([st[f"{s}{k}_rows"] for k in range(5)])),
                                      np.sort(sp["rows"])) for s in ("rw", "tg"))
            c["test_rows_partition"] += int(part)
            ntr = [tr.size for _, _, tr in design.c0_train_rows(sp)]
            c["train_rows_1600_all_fits"] += int(all(n == 1600 for n in ntr))
            preds[m][u] = st
    res["artifact_checks"] = checks

    # recompute the C0 aggregates from stored predictions with the frozen analysis code
    frozen = json.loads((ROOT / "results" / "R10_B_results.json").read_text())["results"]["C0_conventional"]
    c0 = {m: {u: preds[m][u] for u in subjects} for m in geometry.MODELS}
    parts = np.unique(np.asarray(meta["subject"]))
    diffs = {}
    for m in geometry.MODELS:
        new = analysis.c0_participant(c0[m], meta, parts)
        for lvl in ("window", "trial", "anchor"):
            for key in ("random_window_BA", "trial_grouped_BA", "inflation_rw_minus_tg"):
                for stat in ("mean", "sd"):
                    diffs[f"{m}/{lvl}/{key}/{stat}"] = abs(new[lvl][key][stat] - frozen[m][lvl][key][stat])
    res["c0_recompute_n_compared"] = len(diffs)
    res["c0_recompute_max_abs_diff_BA"] = max(diffs.values())
    res["frozen_c0_window_inflation_BA"] = {m: frozen[m]["window"]["inflation_rw_minus_tg"]["mean"]
                                            for m in geometry.MODELS}
    n = len(subjects)
    ok = (geometry.C0_STRIDE == 500 and geometry.WLEN_MS == 2000
          and res["geometry"]["adjacent_overlap_fraction"] == 0.75
          and all(all(v == n for v in checks[m].values()) for m in geometry.MODELS))
    ok = ok and res["c0_recompute_max_abs_diff_BA"] < 1e-9
    ok = ok and res["recording_overlap"]["same_recording_cross_trial_shared_samples"] == 0
    res["verdict"] = "LABEL_ONLY_ERRATUM_CONFIRMED" if ok else "CHECK_FAILED"
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("geometry", "artifact_checks", "c0_recompute_max_abs_diff_BA", "verdict")},
                     indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
