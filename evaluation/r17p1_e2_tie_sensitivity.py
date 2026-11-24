"""Recompute the held-out-run E2 erratum from matching saved fits."""

from __future__ import annotations

import argparse

import numpy as np

from evaluation import r12x_analysis as analysis
from experiments import r12x_run as selection
from experiments.runtime import RUN_ROOT, write_json


def compare(BBP, BBL, BC):
    point = analysis.e2(BBP, BBL, BC)
    res = {}
    for label, archived in (("as_implemented_sums", True), ("as_declared_means", False)):
        stats = analysis.bootstrap(BBP, BBL, BC, "E2", legacy_e2_sums=archived)
        res[label] = {k: v for k, v in stats.items() if k != "dist"}
        res[label]["class"] = analysis.classify(point["flip"], stats["lower95_one_sided"], stats["ci95"])
    return {"point_regret": point["regret"], **res}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    protocol = selection.verify()
    selection.check_feature(protocol, "meta.npz")
    meta = selection.load_meta()
    designs = selection.designs(meta)
    if selection.ds.commitment(designs) != protocol["design_commitment"]:
        raise ValueError("Dreyer2023 design differs from its frozen commitment")
    for d in designs:
        d["yB"], d["yC"] = meta["label"][d["B"]], meta["label"][d["C"]]
    mats = selection.collect(designs, protocol)
    cells = {f"{window}/{readout}": compare(*m) for (window, readout), m in mats.items()}
    members = set(selection.sensitivity_participants(protocol))
    keep = [i for i, d in enumerate(designs) if d["participant"] in members]
    if len(keep) != len(members):
        raise ValueError("Sensitivity cohort differs from the audited design")
    sensitivity = compare(*(m[keep] for m in mats["cue", "final"]))
    out = {"tie_tolerance_ba_points": analysis.TIE_TOL, "bootstrap_draws": analysis.N_BOOT,
           "cells": cells, "sensitivity_cue_final": sensitivity}
    path = RUN_ROOT / "results" / "R17P1_E2_tie_sensitivity.json"
    write_json(path, out)
    print(path.name)


if __name__ == "__main__":
    main()
