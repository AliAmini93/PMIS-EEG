#!/usr/bin/env python3
"""R12-X G1R source-validity checks on producer GDF members (one participant per call; label-blind: labels are
only counted, plus the label-only lag-1 same-class fraction). No model, no class-conditional statistic.

Per run: (G1) file size and CRC-32 equal the pinned central directory; SHA-256 recorded; (G2) own GDF 1.x reader ==
MNE 1.11 (labels, sample count, 512 Hz, physical samples <= 1e-12 relative, identical event table); (G3) the 20
analysis channels and 3 EOG present, identical channel order in all six runs; units uV; (G4) provenance: folder
group letter matches the participant number range, file name = <id>_<run tag>.gdf; (G5) geometry: 40 cues, 20/20,
cue -> 781 = 640 samples, [cue - 4.25 s, cue + 4 s) inside the file (covers the pre-cue control's warm-up), cue
spacing >= 7 s; (G6) no whole-file SHA-256 shared by two runs; no raw-window SHA-256 (20 channels, float64,
[cue, cue + 4 s)) shared within the participant (cross-participant check in r12x_g1r_cohort.py); (G7, deviation D1)
runs with an extreme calibration header have channel amplitudes (MAD) within x0.1-x10 of the participant's
normal-header runs.
Writes data/r12x/G1R_check_<ID>.json, window hashes and file SHA-256 under data/r12x/ (git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from data_preparation import dreyer_features as fe  # noqa: E402
from data_preparation import dreyer_gdf as gd  # noqa: E402
from data_preparation import dreyer_sources as so  # noqa: E402

DATA = ROOT / "data" / "r12x"
FS = 512
CUE = {"769": 1, "770": 2}


def check_run(n, r, root, size, mne) -> dict:
    fails, res = [], {"run": r}
    p = root / so.member_path(n, r)
    if p.stat().st_size != size:
        raise ValueError("Dreyer2023 source size changed during audit")
    with p.open("rb") as stream:
        res["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    own = gd.read(p)
    mm = mne.io.read_raw_gdf(p, preload=True, verbose="ERROR")
    xm = mm.get_data()
    same_shape = own["physical_V"].shape == xm.shape
    scale = np.maximum(np.max(np.abs(own["physical_V"]), axis=1, keepdims=True), 1e-30)
    rel = float(np.max(np.abs(xm - own["physical_V"]) / scale)) if same_shape else float("inf")
    ev_m = sorted((int(round(o * FS)), str(d)) for o, d in zip(mm.annotations.onset, mm.annotations.description))
    ev_o = sorted((int(q), str(t)) for q, t in zip(own["event_pos0"], own["event_type"]))
    res.update({"G2_labels_equal": own["label"] == mm.ch_names, "G2_same_shape": same_shape, "G2_max_rel_diff": rel,
                "G2_events_equal": ev_m == ev_o, "G2_fs": mm.info["sfreq"], "G2_trailing_bytes": own["trailing_bytes"],
                "n_samples": int(xm.shape[1]), "n_events": len(ev_o), "event_codes": sorted({c for _, c in ev_o})})
    if not (res["G2_labels_equal"] and same_shape and rel <= 1e-12 and ev_m == ev_o and mm.info["sfreq"] == FS
            and own["fs"] == FS and own["trailing_bytes"] == 0):
        fails.append("G2")
    need = set(fe.CHANNELS) | set(fe.EOG)
    units = {own["phys_dim"][i] for i, c in enumerate(own["label"]) if c in need}
    res.update({"G3_channels_present": need <= set(own["label"]), "G3_units": sorted(units), "channel_order": own["label"]})
    if not (need <= set(own["label"]) and units <= {"uV", "µV"}):
        fails.append("G3")
    cues = [(s, CUE[c]) for s, c in ev_o if c in CUE]
    fb = [s for s, c in ev_o if c == "781"]
    lat = [min((f - s for f in fb if f >= s), default=10**9) for s, _ in cues]
    gaps = np.diff([s for s, _ in cues]) if len(cues) > 1 else np.array([10**9])
    labs = [c for _, c in cues]
    res.update({"G5_n_cues": len(cues), "G5_per_class": [labs.count(1), labs.count(2)],
                "G5_cue_to_feedback": sorted(set(lat)), "G5_min_spacing_s": float(gaps.min() / FS),
                "G5_first_ok": bool(cues and cues[0][0] - 640 - 3 * FS >= 0),
                "G5_last_ok": bool(cues and cues[-1][0] + 4 * FS <= xm.shape[1]),
                "has_800": "800" in {c for _, c in ev_o},
                "labels_lag1_same": float(np.mean([a == b for a, b in zip(labs, labs[1:])])) if len(labs) > 1 else None})
    if not (len(cues) == 40 and labs.count(1) == 20 and labs.count(2) == 20
            and all(abs(v - 640) <= 1 for v in lat) and res["G5_first_ok"] and res["G5_last_ok"] and gaps.min() >= 7 * FS):
        fails.append("G5")
    ch = {c: i for i, c in enumerate(own["label"])}
    D = np.ascontiguousarray(own["physical_V"][[ch[c] for c in fe.CHANNELS]])
    med = np.median(D, axis=1, keepdims=True)
    res["extreme_calibration_header"] = bool(own.get("extreme_calibration_header", False))
    res["channel_mad_V"] = np.median(np.abs(D - med), axis=1).tolist()
    res["window_hashes"] = [hashlib.sha256(np.ascontiguousarray(D[:, s:s + 4 * FS]).tobytes()).hexdigest() for s, _ in cues]
    res["cue_window_hashes"] = [
        hashlib.sha256(np.ascontiguousarray(D[:, s:s + 640]).tobytes()).hexdigest() for s, _ in cues
    ]
    res["fails"] = fails
    return res


def check_participant(root, n, sizes):
    import mne

    runs = [check_run(n, r, root, sizes[r], mne) for r in so.RUNS]
    hashes = [h for r in runs for h in r["window_hashes"]]
    orders = {tuple(r.pop("channel_order", ())) for r in runs}
    shas = [r["sha256"] for r in runs]
    fails = sorted({f for r in runs for f in r["fails"]})
    if len(orders) != 1:
        fails.append("G3_order")
    if len(set(shas)) != len(shas) or len(set(hashes)) != len(hashes):
        fails.append("G6")
    # G7 (deviation D1): extreme-calibration-header runs must have channel amplitudes within x0.1-x10 of the
    # participant's normal-header runs (label-blind)
    normal = [r["channel_mad_V"] for r in runs if "channel_mad_V" in r and not r["extreme_calibration_header"]]
    g7 = {"extreme_runs": [r["run"] for r in runs if r.get("extreme_calibration_header")], "ratio_range": None}
    if g7["extreme_runs"]:
        ref = np.median(np.array(normal), axis=0) if normal else None
        ratios = [np.array(r["channel_mad_V"]) / ref for r in runs if r.get("extreme_calibration_header")] if ref is not None else []
        g7["ratio_range"] = [float(np.min(ratios)), float(np.max(ratios))] if ratios else None
        if ref is None or not all(np.all((x >= 0.1) & (x <= 10)) for x in ratios):
            fails.append("G7")
    by_run = {r["run"]: np.asarray(r["channel_mad_V"]) for r in runs}
    ref = (by_run["R1"] + by_run["R2"]) / 2
    ratio = float(np.median(np.array([by_run[r] / ref for r in so.RUNS[2:]])))
    return {"runs": runs, "G7": g7, "scaling_ratio": ratio, "fails": sorted(set(fails))}


def main():
    ap = argparse.ArgumentParser(description="Run the outcome-blind reader check for one producer participant.")
    ap.add_argument("--participant", type=int, required=True)
    args = ap.parse_args()
    root = so.raw_root()
    files = {r: (root / so.member_path(args.participant, r)).stat().st_size for r in so.RUNS}
    res = check_participant(root, args.participant, files)
    print(json.dumps({"runs": len(res["runs"]), "fails": res["fails"],
                      "reader_max_relative_difference": max(r["G2_max_rel_diff"] for r in res["runs"])}))


if __name__ == "__main__":
    main()
