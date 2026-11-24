#!/usr/bin/env python3
"""Baseline decodability and ceiling dependence across the identity-swap results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.runtime import verify_recorded_input, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from evaluation import r8_ceiling as ceiling  # noqa: E402

PROTOCOL_ID = "R8-C-CEILING-AUDIT-v1"
PROTOCOL = CONFIG_ROOT / "R8_C_PROTOCOL.json"
PUBLIC = ROOT / "results" / "R8_C_results.json"
D = ROOT / "data"
SOURCES = {  # name: (file, key prefix, cohort, model)
    "HS-Stroke_TSLR_G3": ("g3/G3_pair_scores_PRIVATE.npz", "s42_", "HS-Stroke", "TS+LR"),
    "HS-Stroke_EEGNet_C2": ("r6_c2/R6_C2_pair_records_PRIVATE.npz", "", "HS-Stroke", "EEGNet"),
    "Lee2019_TSLR_C3a": ("r6_c3/R6_C3a_pair_records_PRIVATE.npz", "", "Lee2019", "TS+LR"),
    "Lee2019_EEGNet_C3b": ("r6_c3b/R6_C3b_pair_records_PRIVATE.npz", "", "Lee2019", "EEGNet"),
    "Lee2019_EEGNet_final_C6": ("r7_c6/R7_C6_EEGNet_pair_records_PRIVATE.npz", "final_", "Lee2019", "EEGNet"),
    "Lee2019_EEGNet_earlystop_C6": ("r7_c6/R7_C6_EEGNet_pair_records_PRIVATE.npz", "es_", "Lee2019", "EEGNet ES"),
    "Lee2019_Shallow_final_C6": ("r7_c6/R7_C6_Shallow_pair_records_PRIVATE.npz", "final_", "Lee2019", "ShallowConvNet"),
    "Lee2019_Shallow_earlystop_C6": ("r7_c6/R7_C6_Shallow_pair_records_PRIVATE.npz", "es_", "Lee2019", "ShallowConvNet ES"),
    "Shin2022_TSLR_R8A": ("r8_a/R8_A_pair_records_PRIVATE.npz", "t_", "Shin2022", "TS+LR"),
    "Shin2022_EEGNet_R8A": ("r8_a/R8_A_pair_records_PRIVATE.npz", "e_", "Shin2022", "EEGNet"),
    "Lee2019_TCFormer_final_R8D": ("r8_de/R8_DE_TCFormer_pair_records_PRIVATE.npz", "final_", "Lee2019", "TCFormer"),
    "Lee2019_TCFormer_earlystop_R8D": ("r8_de/R8_DE_TCFormer_pair_records_PRIVATE.npz", "es_", "Lee2019", "TCFormer ES"),
    "Lee2019_CBraMod_FT_final_R8E": ("r8_de/R8_DE_CBraMod_pair_records_PRIVATE.npz", "final_", "Lee2019", "CBraMod fine-tuned"),
    "Lee2019_CBraMod_FT_earlystop_R8E": ("r8_de/R8_DE_CBraMod_pair_records_PRIVATE.npz", "es_", "Lee2019", "CBraMod fine-tuned ES"),
    "Lee2019_CBraMod_probe_R8E1": ("r8_de/R8_E1_pair_records_PRIVATE.npz", "", "Lee2019", "CBraMod frozen probe"),
}
EXPECTED_D = {  # published participant-level primary d (points): each source must reproduce it (fail-closed key check)
    "HS-Stroke_TSLR_G3": ("G3_results.json", ("results", "primary", "mean_points")),
    "HS-Stroke_EEGNet_C2": ("R6_C2_results.json", ("results", "primary", "mean_points")),
    "Lee2019_TSLR_C3a": ("R6_C3a_results.json", ("results", "primary", "mean_points")),
    "Lee2019_EEGNet_C3b": ("R6_C3b_results.json", ("results", "primary", "mean_points")),
    "Lee2019_EEGNet_final_C6": ("R7_C6_results.json", ("results", "family_holm", "EEGNet_final_d", "mean_points")),
    "Lee2019_EEGNet_earlystop_C6": ("R7_C6_results.json", ("results", "family_holm", "EEGNet_es_d", "mean_points")),
    "Lee2019_Shallow_final_C6": ("R7_C6_results.json", ("results", "family_holm", "Shallow_final_d", "mean_points")),
    "Lee2019_Shallow_earlystop_C6": ("R7_C6_results.json", ("results", "family_holm", "Shallow_es_d", "mean_points")),
    "Shin2022_TSLR_R8A": ("R8_A_results.json", ("results", "family_holm", "H1_TSLR_d", "mean_points")),
    "Shin2022_EEGNet_R8A": ("R8_A_results.json", ("results", "family_holm", "H2_EEGNet_d", "mean_points")),
    "Lee2019_TCFormer_final_R8D": ("R8_DE_results.json", ("results", "TCFormer", "family_holm", "TCFormer_final_d", "mean_points")),
    "Lee2019_TCFormer_earlystop_R8D": ("R8_DE_results.json", ("results", "TCFormer", "family_holm", "TCFormer_es_d", "mean_points")),
    "Lee2019_CBraMod_FT_final_R8E": ("R8_DE_results.json", ("results", "CBraMod", "family_holm", "CBraMod_final_d", "mean_points")),
    "Lee2019_CBraMod_FT_earlystop_R8E": ("R8_DE_results.json", ("results", "CBraMod", "family_holm", "CBraMod_es_d", "mean_points")),
    "Lee2019_CBraMod_probe_R8E1": ("R8_DE_results.json", ("results", "CBraMod_probe", "family", "CBraMod_probe_d", "mean_points")),
}


def expected_d(name):
    f, path = EXPECTED_D[name]
    v = json.loads((ROOT / "results" / f).read_text())
    for k in path:
        v = v[k]
    return float(v)


def load(name):
    f, pre, _, _ = SOURCES[name]
    z = np.load(D / f)
    subj = z[pre + "subject"] if (pre + "subject") in z.files else z["subject"]
    lab = z[pre + "label"] if (pre + "label") in z.files else z["label"]
    return {"subject": subj, "label": lab.astype(int), **{q: z[pre + q] for q in ("pAA", "pAB", "pBA", "pBB")}}


def cmd_execute():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN", "protocol / worktree")
    for n, (f, *_) in SOURCES.items():
        g3._stop(verify_recorded_input(D / f, proto["source_sha256"][n]), f"{n} source changed")
    res = {n: dict(ceiling.audit(load(n)), cohort=SOURCES[n][2], model=SOURCES[n][3]) for n in SOURCES}
    for n, r in res.items():
        r["published_d_points"] = expected_d(n)
        g3._stop(abs(r["d_prob_mean_points"] - r["published_d_points"]) < 1e-6, f"{n}: records do not reproduce the published d")
    PUBLIC.write_text(json.dumps({"stage": "R8-C", "protocol_id": PROTOCOL_ID, "port_manifest_sha256": port_identity(),
                                  "status": "COMPLETED", "results": res}, indent=2, default=float) + "\n")
    print({n: r.get("verdict") for n, r in res.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    sys.exit(cmd_execute())
