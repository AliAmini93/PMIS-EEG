#!/usr/bin/env python3
"""R12-X G1R feature validation (label-blind; no model): shapes, dtypes, finiteness, OAS positive-definiteness, meta
consistency (83 x 6 runs x 40 trials, 20/20 per run), eligible-set agreement, and the A/B/C design firewall for every
participant (pmis_r12x.design asserts). Writes reports/aggregate/R12X_G1R_feature_validation.json."""

from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from experiments import r12x_run as run


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    F = run.FEAT
    meta = run.load_meta()
    elig = json.loads((run.PRIVATE / "G1R_eligible_PRIVATE.json").read_text())["eligible"]
    parts = sorted(np.unique(meta["participant"]).tolist())
    out = {"schema": "R12X_G1R_feature_validation_v1", "n_trials": int(meta["label"].size), "participants": len(parts),
           "participants_equal_eligible_set": parts == sorted(elig), "arrays": {}}
    cnt_ok = True
    for p in parts:
        m = meta["participant"] == p
        for r in ("R1", "R2", "R3", "R4", "R5", "R6"):
            mr = m & (meta["run"] == r)
            lab = meta["label"][mr]
            cnt_ok &= bool(mr.sum() == 40 and (lab == 1).sum() == 20 and (lab == 2).sum() == 20
                           and np.array_equal(np.sort(meta["pos"][mr]), np.arange(40)))
    out["per_run_40_trials_20_20"] = cnt_ok
    for f in sorted(F.glob("*.npy")):
        a = np.load(f, mmap_mode="r")
        rec = {"shape": list(a.shape), "dtype": str(a.dtype), "finite": bool(np.all(np.isfinite(a)))}
        if f.name.startswith("cov_"):
            rec["min_eigenvalue_positive"] = bool(np.linalg.eigvalsh(np.asarray(a)).min() > 0)
        rec["rows_match_meta"] = a.shape[0] == meta["label"].size
        out["arrays"][f.name] = rec
    D = run.designs(meta)                      # asserts the C firewall, class balance, sizes for every participant
    out["designs_built_with_firewall_asserts"] = len(D)
    out["design_commitment"] = run.ds.commitment(D)
    out["participants_sha256"] = run.participants_digest(meta)
    out["pass"] = bool(out["participants_equal_eligible_set"] and cnt_ok and len(D) == len(parts)
                       and all(v["finite"] and v["rows_match_meta"] and v.get("min_eigenvalue_positive", True)
                               for v in out["arrays"].values()))
    (ROOT / "results" / "R12X_G1R_feature_validation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
