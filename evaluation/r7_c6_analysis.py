"""R7-C6 analysis (frozen before execution)."""

from __future__ import annotations

import numpy as np

from evaluation.r6_mc_stats import summary

FAMILY = ("EEGNet_final_d", "EEGNet_es_d", "Shallow_final_d", "Shallow_es_d",
          "EEGNet_es_minus_final", "Shallow_es_minus_final")


def d(r, ro):
    return 0.5 * ((r[f"{ro}_pAA"] - r[f"{ro}_pAB"]) + (r[f"{ro}_pBB"] - r[f"{ro}_pBA"]))


def plc(r, ro):
    return 0.5 * r["sign"] * ((r[f"{ro}_pAC"] - r[f"{ro}_pAE"]) + (r[f"{ro}_pBE"] - r[f"{ro}_pBC"]))


def bac(r, ro):
    c = lambda q: float(r[f"{ro}_p{q}"] > 0.5)  # noqa: E731
    return 0.5 * ((c("AA") - c("AB")) + (c("BB") - c("BA")))


def S(recs, fn, test=True):
    return summary(np.array([r["subject"] for r in recs]), np.array([r["label"] for r in recs]),
                   np.array([fn(r) for r in recs], dtype=np.float64), test=test)


def holm(p):
    keys = sorted(p, key=lambda k: p[k])
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out


def analyse(recs_by_model: dict, aa_by_model: dict, tslr_by_k: dict) -> dict:
    fam, desc = {}, {}
    for m, recs in recs_by_model.items():
        for ro in ("final", "es"):
            fam[f"{m}_{ro}_d"] = S(recs, lambda r, ro=ro: d(r, ro))
            desc[f"{m}_{ro}_placebo"] = S(recs, lambda r, ro=ro: plc(r, ro))
            desc[f"{m}_{ro}_balanced_accuracy_contrast"] = S(recs, lambda r, ro=ro: bac(r, ro), test=False)
            desc[f"{m}_{ro}_context_partner_arm_correct_class_probability"] = S(
                recs, lambda r, ro=ro: 0.5 * (r[f"{ro}_pAB"] + r[f"{ro}_pBA"]), test=False)
            desc[f"{m}_{ro}_off_target_abs_change"] = S(recs, lambda r, ro=ro: r[f"{ro}_off"], test=False)
        fam[f"{m}_es_minus_final"] = S(recs, lambda r: d(r, "es") - d(r, "final"))
        ep = np.array([[r["es_epoch_A"], r["es_epoch_B"]] for r in recs]).ravel()
        desc[f"{m}_early_stopping_epoch"] = {"median": float(np.median(ep)), "q10_q90": [float(np.percentile(ep, 10)), float(np.percentile(ep, 90))],
                                             "fraction_at_final_epoch": float(np.mean(ep == 100))}
        desc[f"{m}_AA_determinism_max_abs"] = float(max(aa_by_model[m]))
        desc[f"{m}_TSLR_C3a_same_pairs_descriptive"] = S(recs, lambda r: tslr_by_k[int(r["k"])], test=False)
    adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in fam.items()})
    for k in fam:
        fam[k]["holm_p_family"] = adj[k]
    if "EEGNet" in recs_by_model and "Shallow" in recs_by_model:
        e = {int(r["k"]): r for r in recs_by_model["EEGNet"]}
        both = [dict(r, d_e=d(e[int(r["k"])], "final"), d_s=d(r, "final")) for r in recs_by_model["Shallow"]]
        desc["Shallow_minus_EEGNet_final_d"] = S(both, lambda r: r["d_s"] - r["d_e"], test=False)
    return {"family_holm": fam, "descriptive": desc,
            "pair_counts": {m: len(v) for m, v in recs_by_model.items()}}
