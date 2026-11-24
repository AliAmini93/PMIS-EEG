"""R7-C5 analysis (frozen before execution): shift-decay curves and retained fractions per model."""

from __future__ import annotations

import numpy as np

from evaluation.r6_indep_stats import participant_values
from evaluation.r6_mc_stats import sign_flip_mc, summary

from experiments.pmis_r7.c5 import SHIFT_MS, VARIANTS

MIN_DENOM = 0.005  # guard: fractions not computed if any participant's d(D0) <= 0.5 points (per model)
FAMILY = ("EEGNet_minus_TSLR_fraction_D62.5", "EEGNet_minus_TSLR_fraction_D125", "EEGNet_minus_TSLR_fraction_D250",
          "EEGNet_minus_TSLR_fraction_D500", "H3a_EEGNet_fraction_D62.5_above_0.5", "H3b_EEGNet_fraction_D250_below_0.5")


def dv(r, v):
    return 0.5 * ((r[f"{v}_pAA"] - r[f"{v}_pAB"]) + (r[f"{v}_pBB"] - r[f"{v}_pBA"]))


def holm(p):
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def fractions(recs):
    subj = np.array([r["subject"] for r in recs])
    lab = np.array([r["label"] for r in recs])
    pv = {v: participant_values(subj, lab, np.array([dv(r, v) for r in recs])) for v, _, _ in VARIANTS}
    keys = sorted(pv["D0"])
    den = np.array([pv["D0"][k] for k in keys])
    return keys, {v: np.array([pv[v][k] for k in keys]) / den for v, _, _ in VARIANTS}, float(den.min())


def analyse(tslr_recs, eeg_recs, aa):
    out = {"curves": {}, "family_holm": {}}
    for name, recs in (("TSLR", tslr_recs), ("EEGNet", eeg_recs)):
        subj = np.array([r["subject"] for r in recs])
        lab = np.array([r["label"] for r in recs])
        out["curves"][name] = {v: dict(summary(subj, lab, np.array([dv(r, v) for r in recs]), test=False), shift_ms=SHIFT_MS[v])
                               for v, _, _ in VARIANTS}
    kt, ft, mt = fractions(tslr_recs)
    ke, fe, me = fractions(eeg_recs)
    assert kt == ke
    computable = min(mt, me) > MIN_DENOM
    out["fractions"] = {"computable": computable, "min_denominator_points": {"TSLR": mt * 100, "EEGNet": me * 100}}
    if computable:
        for v in ("D62.5", "D125", "D250", "D500", "D1000"):
            out["fractions"][v] = {"median_TSLR": float(np.median(ft[v])), "median_EEGNet": float(np.median(fe[v])),
                                   "mean_TSLR": float(np.mean(ft[v])), "mean_EEGNet": float(np.mean(fe[v]))}
        fam = {}
        for v in ("D62.5", "D125", "D250", "D500"):
            dlt = fe[v] - ft[v]
            fam[f"EEGNet_minus_TSLR_fraction_{v}"] = {"mean": float(dlt.mean()), "n_positive": int((dlt > 0).sum()),
                                                      "n_negative": int((dlt < 0).sum()), "sign_flip_p_monte_carlo": sign_flip_mc(dlt)}
        for key, v, sgn in (("H3a_EEGNet_fraction_D62.5_above_0.5", "D62.5", 1), ("H3b_EEGNet_fraction_D250_below_0.5", "D250", -1)):
            dlt = sgn * (fe[v] - 0.5)
            fam[key] = {"mean_fraction": float(fe[v].mean()), "n_in_declared_direction": int((dlt > 0).sum()),
                        "sign_flip_p_monte_carlo": sign_flip_mc(dlt), "direction_met": bool(dlt.mean() > 0)}
        adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in fam.items()})
        for k in fam:
            fam[k]["holm_p_family"] = adj[k]
        out["family_holm"] = fam
    half = {}
    for name in ("TSLR", "EEGNet"):
        if any("mean_points" not in out["curves"][name][v] for v, _, _ in VARIANTS):
            half[name] = {"first_shift_ms_below_half_of_d0": None, "computable": False}
            continue
        m = [(SHIFT_MS[v], out["curves"][name][v]["mean_points"]) for v, _, _ in VARIANTS]
        d0 = m[0][1]
        below = [s for s, x in m if x < 0.5 * d0]
        half[name] = {"first_shift_ms_below_half_of_d0": below[0] if below else None}
    out["half_decay"] = half
    out["AA_determinism"] = {"n_checks": len(aa), "max_abs_probability_difference": float(max(aa)) if aa else None}
    out["pair_counts"] = {"TSLR": len(tslr_recs), "EEGNet": len(eeg_recs)}
    return out
