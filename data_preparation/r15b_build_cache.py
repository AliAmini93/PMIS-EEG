#!/usr/bin/env python3
"""R15-B: build model-format arrays from Dreyer2023 fitting runs R1-R4 only (CPU; no model).

- First proves that the loader refuses R5 and R6 (OutcomeRunRefused) without touching any file.
- Checks the eligible set and the raw SHA-256 table against the committed R12-X G1R digests.
- Hashes every file before reading it.
- Writes per-participant arrays to the private work directory (outside Git), and a cohort-level audit to
  reports/aggregate/R15B_fitrun_cache_audit.json.

The audit holds counts and digests only. It contains no participant-level values and no participant identifiers
beyond the committed exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
import numpy as np  # noqa: E402

from data_preparation import feasibility as D  # noqa: E402

from data_preparation.dreyer_sources import raw_root

ELIGIBLE = ROOT / "data" / "r12x" / "G1R_eligible_PRIVATE.json"
WORK = ROOT / "data" / "feasibility"
CACHE = WORK / "cache"


def one(n):
    import mne
    elig = D.check_cohort(ELIGIBLE)
    out = CACHE / f"{D.producer_id(n)}.npz"
    if out.exists():
        z = np.load(out)
        return n, {k: list(z[k].shape) for k in z.files}, hashlib.sha256(out.read_bytes()).hexdigest()
    runs = D.load_participant(raw_root(), n, elig["raw_sha256"], mne)
    arr = D.participant_arrays(runs)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **arr)
    tmp.rename(out)
    return n, {k: list(v.shape) for k, v in arr.items()}, hashlib.sha256(out.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build the fitting-run-only feasibility cache.")
    ap.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = ap.parse_args()
    refused = []
    for r in ("R5", "R6"):
        try:
            D.gdf_path(Path("/nonexistent"), 1, r)
        except D.OutcomeRunRefused:
            refused.append(r)
    assert refused == ["R5", "R6"], refused
    elig = D.check_cohort(ELIGIBLE)
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = list(elig["eligible"])
    shapes, digests = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, sh, dg in ex.map(one, ids):
            shapes[n], digests[n] = sh, dg
            print(f"{len(shapes)}/{len(ids)} participant caches complete", flush=True)
    keys = sorted({k for s in shapes.values() for k in s})
    ntr = [shapes[n]["y"][0] for n in ids]
    audit = {"schema": "R15B_fitrun_cache_audit_v1", "runs_read": list(D.FIT_RUNS), "runs_refused_by_loader": refused,
             "eligible_set_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
             "raw_sha256_table_sha256": D.json_sha256(elig["raw_sha256"]),
             "participants": len(ids), "raw_files_hashed_and_read": 4 * len(ids), "trials_total": int(sum(ntr)),
             "trials_per_participant": {"min": int(min(ntr)), "max": int(max(ntr))},
             "arrays": {k: shapes[ids[0]][k][1:] for k in keys},
             "cache_set_sha256": hashlib.sha256(json.dumps([digests[n] for n in ids]).encode()).hexdigest(),
             "formats": D.FORMATS, "windows_samples_512": D.WINDOWS,
             "note": "per-participant arrays live in the private work directory only"}
    (ROOT / "results" / "R15B_fitrun_cache_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({k: audit[k] for k in ("participants", "trials_total", "trials_per_participant", "arrays")}))


if __name__ == "__main__":
    main()
