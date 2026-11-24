"""R8-D / R8-E analysis (frozen before execution). Participant-level (class-balanced within participant, participants
weighted equally; participants lacking a class are dropped and counted by pmis_r6.mc_stats.summary), Monte Carlo
sign-flip (1e6 draws, seed 7499), t95 CI; Holm within each experiment's family.

Families: R8-D {TCFormer_final_d, TCFormer_es_d, TCFormer_es_minus_final}; R8-E2 {CBraMod_final_d, CBraMod_es_d,
CBraMod_es_minus_final}; R8-E1 {CBraMod_probe_d} (single test). Every cross-model comparison is DESCRIPTIVE (no p-value):
paired participant-level differences on identical pairs against EEGNet-8,2 / ShallowConvNet (C6 records) and TS+LR (C3a).
"""

from __future__ import annotations

import numpy as np

from evaluation.r6_mc_stats import summary
from evaluation.r7_c6_analysis import bac, d, holm, plc


def _S(recs, fn, test=True):
    return summary(np.array([r["subject"] for r in recs]), np.array([r["label"] for r in recs]),
                   np.array([fn(r) for r in recs], dtype=np.float64), test=test)


def d_plain(r):
    return 0.5 * ((r["pAA"] - r["pAB"]) + (r["pBB"] - r["pBA"]))


def plc_plain(r):
    return 0.5 * r["sign"] * ((r["pAC"] - r["pAE"]) + (r["pBE"] - r["pBC"]))


def bac_plain(r):
    c = lambda q: float(r[f"p{q}"] > 0.5)  # noqa: E731
    return 0.5 * ((c("AA") - c("AB")) + (c("BB") - c("BA")))


def analyse_deep(model: str, recs: list, aa: list, c6_recs: dict, tslr_by_k: dict) -> dict:
    """c6_recs: {'EEGNet': {k: rec}, 'Shallow': {k: rec}} (C6 pair records keyed by pair id k)."""
    fam, desc = {}, {}
    for ro in ("final", "es"):
        fam[f"{model}_{ro}_d"] = _S(recs, lambda r, ro=ro: d(r, ro))
        desc[f"{ro}_placebo"] = _S(recs, lambda r, ro=ro: plc(r, ro))
        desc[f"{ro}_balanced_accuracy_contrast"] = _S(recs, lambda r, ro=ro: bac(r, ro), test=False)
        desc[f"{ro}_partner_arm_correct_class_probability"] = _S(recs, lambda r, ro=ro: 0.5 * (r[f"{ro}_pAB"] + r[f"{ro}_pBA"]), test=False)
        desc[f"{ro}_off_target_abs_change"] = _S(recs, lambda r, ro=ro: r[f"{ro}_off"], test=False)
        for other in ("EEGNet", "Shallow"):
            o = c6_recs[other]
            desc[f"{other}_{ro}_d_same_pairs"] = _S(recs, lambda r, ro=ro, o=o: d(o[int(r["k"])], ro), test=False)
            desc[f"{model}_minus_{other}_{ro}_d"] = _S(recs, lambda r, ro=ro, o=o: d(r, ro) - d(o[int(r["k"])], ro), test=False)
        desc[f"{model}_{ro}_minus_TSLR_d"] = _S(recs, lambda r, ro=ro: d(r, ro) - tslr_by_k[int(r["k"])], test=False)
    fam[f"{model}_es_minus_final"] = _S(recs, lambda r: d(r, "es") - d(r, "final"))
    desc["TSLR_C3a_d_same_pairs"] = _S(recs, lambda r: tslr_by_k[int(r["k"])], test=False)
    ep = np.array([[r["es_epoch_A"], r["es_epoch_B"]] for r in recs]).ravel()
    desc["early_stopping_epoch"] = {"median": float(np.median(ep)), "q10_q90": [float(np.percentile(ep, 10)), float(np.percentile(ep, 90))],
                                    "fraction_at_final_epoch": float(np.mean(ep == ep.max()))}
    desc["AA_determinism_max_abs"] = float(max(aa))
    adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in fam.items()})
    for k in fam:
        fam[k]["holm_p_family"] = adj[k]
    return {"family_holm": fam, "descriptive": desc, "n_pairs": len(recs)}


def analyse_probe(recs: list, tslr_by_k: dict, eegnet_by_k: dict, subset_ks: set) -> dict:
    fam = {"CBraMod_probe_d": _S(recs, d_plain)}
    sub = [r for r in recs if int(r["k"]) in subset_ks]
    e3 = [r for r in recs if int(r["k"]) in eegnet_by_k]  # C3b trained EEGNet on a quarter of the C3 pairs
    desc = {"placebo": _S(recs, plc_plain), "balanced_accuracy_contrast": _S(recs, bac_plain, test=False),
            "partner_arm_correct_class_probability": _S(recs, lambda r: 0.5 * (r["pAB"] + r["pBA"]), test=False),
            "off_target_abs_change": _S(recs, lambda r: r["off"], test=False),
            "TSLR_C3a_d_same_pairs": _S(recs, lambda r: tslr_by_k[int(r["k"])], test=False),
            "probe_minus_TSLR_d": _S(recs, lambda r: d_plain(r) - tslr_by_k[int(r["k"])], test=False),
            "EEGNet_C3b_d_same_pairs": _S(e3, lambda r: eegnet_by_k[int(r["k"])], test=False),
            "probe_d_on_C3b_pairs": _S(e3, d_plain, test=False),
            "probe_minus_EEGNet_C3b_d": _S(e3, lambda r: d_plain(r) - eegnet_by_k[int(r["k"])], test=False),
            "probe_d_on_R8DE_subset_pairs": _S(sub, d_plain, test=False)}
    return {"family": fam, "descriptive": desc, "n_pairs": len(recs)}
