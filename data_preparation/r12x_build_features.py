#!/usr/bin/env python3
"""R12-X trial-local feature build from producer GDF runs (pmis_r12x.features; frozen recipe). No model, no label
statistic: labels are copied into the metadata table only.

--geometry-only --participant N : label-blind technical validation on one participant's files (window bounds,
    reader agreement, finiteness, shapes, OAS positive-definiteness); writes data/r12x/G1_feature_geometry_<ID>.json
--all                            : full build for every G1R signal-eligible participant (data/r12x/G1R_eligible_PRIVATE.json);
    every raw file's SHA-256 must equal the verified G1R table before it is read; writes data/r12x/features/*.npy +
    meta.npz + build audit with SHA-256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from data_preparation import dreyer_features as fe  # noqa: E402
from data_preparation import dreyer_sources as so  # noqa: E402

DATA = ROOT / "data" / "r12x"
OUT = DATA / "features"
MIN_AVAIL = 8 * 2**30


def wait_mem():
    while True:
        avail = int([ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0].split()[1]) * 1024
        if avail >= MIN_AVAIL:
            return
        time.sleep(60)


def gdf_path(root: Path, n: int, run: str) -> Path:
    return root / so.member_path(n, run)


def participant(root: Path, n: int, mne, sha_table: dict | None = None) -> tuple[dict, list]:
    feats, meta = {}, []
    for r in so.RUNS:
        wait_mem()
        path = gdf_path(root, n, r)
        if sha_table is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha_table[f"{so.producer_id(n)}_{r}"]:
            raise RuntimeError(f"raw file differs from the verified G1R table: {path.name}")
        rr = fe.read_run(path, mne)
        for pos, (onset, lab) in enumerate(rr["cues"]):
            f = fe.trial_features(rr["x"], rr["channels"], onset)
            for k, v in f.items():
                feats.setdefault(k, []).append(v)
            meta.append((n, r, pos, lab, onset))
    return {k: np.stack(v) for k, v in feats.items()}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--participant", type=int)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    import mne
    warnings.simplefilter("ignore")
    root = so.raw_root()
    if a.geometry_only:
        if a.participant is None:
            ap.error("--geometry-only requires --participant")
        n = a.participant
        DATA.mkdir(parents=True, exist_ok=True)
        f, meta = participant(root, n, mne)
        geo = {"participant": so.producer_id(n), "n_trials": len(meta),
               "trials_per_run": {r: sum(1 for m in meta if m[1] == r) for r in so.RUNS},
               "shapes": {k: list(v.shape) for k, v in f.items()},
               "all_finite": {k: bool(np.all(np.isfinite(v))) for k, v in f.items()},
               "cov_min_eigenvalue_positive": {k: bool(np.linalg.eigvalsh(v).min() > 0) for k, v in f.items()
                                               if k.startswith("cov_")},
               "dtypes": {k: str(v.dtype) for k, v in f.items()}}
        (DATA / f"G1_feature_geometry_{so.producer_id(n)}.json").write_text(json.dumps(geo, indent=1) + "\n")
        print(json.dumps(geo, indent=1))
        return
    if not a.all:
        ap.error("choose --geometry-only or --all")
    elig = json.loads((DATA / "G1R_eligible_PRIVATE.json").read_text())   # signal-level eligible (G1R cohort check)
    OUT.mkdir(parents=True, exist_ok=True)
    proto = load_protocol(CONFIG_ROOT / "R12X_PROTOCOL.json")
    eligible_sha = hashlib.sha256(json.dumps(elig["eligible"]).encode()).hexdigest()
    raw_sha = hashlib.sha256(json.dumps(elig["raw_sha256"], sort_keys=True).encode()).hexdigest()
    if not reference_matches(eligible_sha, proto["participants_sha256"]) or not reference_matches(raw_sha, proto["cohort"]["raw_sha256_table_sha256"]):
        raise ValueError("Dreyer2023 eligibility receipt differs from the frozen raw cohort")
    allf, allm = {}, []
    for n in elig["eligible"]:
        f, meta = participant(root, n, mne, elig["raw_sha256"])
        for k, v in f.items():
            allf.setdefault(k, []).append(v)
        allm += meta
    audit = {"files": {}}
    for k, v in allf.items():
        arr = np.concatenate(v)
        np.save(OUT / f"{k}.npy", arr)
        audit["files"][f"{k}.npy"] = hashlib.sha256((OUT / f"{k}.npy").read_bytes()).hexdigest()
    m = np.array(allm, dtype=object)
    np.savez(OUT / "meta.npz", participant=m[:, 0].astype(np.int64), run=m[:, 1].astype(str),
             pos=m[:, 2].astype(np.int64), label=m[:, 3].astype(np.int64), onset=m[:, 4].astype(np.int64))
    audit["files"]["meta.npz"] = hashlib.sha256((OUT / "meta.npz").read_bytes()).hexdigest()
    audit["eligible_set_sha256"] = hashlib.sha256(json.dumps(elig["eligible"]).encode()).hexdigest()
    audit["raw_sha256_table_sha256"] = hashlib.sha256(json.dumps(elig["raw_sha256"], sort_keys=True).encode()).hexdigest()
    audit["n_trials"] = len(allm)
    if not all(reference_matches(audit["files"][name], digest) for name, digest in proto["feature_sha256"].items()):
        raise ValueError("Dreyer2023 feature files differ from the original")
    (OUT / "build_audit.json").write_text(json.dumps(audit, indent=1) + "\n")


if __name__ == "__main__":
    main()
