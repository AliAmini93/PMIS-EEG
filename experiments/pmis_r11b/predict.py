"""R11-B frozen predictions, derived ONLY from the committed G0 aggregate of existing outputs
(reports/aggregate/R11_B_G0_existing.json; its SHA-256 is pinned by the protocol). Nothing here reads an R11-B
outcome.

Single-copy exposed-trial BA contrast c_m,ro (own-copy arm vs partner-copy arm, rows and support fixed):
  EEGNet, ShallowConvNet: R7-C6 (identical 240-row training base, identical tasks, final and min-val-loss epochs);
  TS+LR: R6-C3a (exact optimum; 300-row base: a declared mismatch with the 240-row R11-B base).
Additive (superposition) prediction for the pairwise distortion at dose delta (BA points, A - B):
  D_pred(delta) = delta * dc_AB,ro, with dc_AB,ro the participant-paired contrast difference from G0.
Leaked-arm gap: G_L_pred(delta) = gap_prior + D_pred(delta); crossing dose delta* = -gap_prior / dc (participant-
bootstrap 95 % interval from G0). gap_prior: near-clean placebo-arm BA on the whole held-out run (Shallow - EEGNet:
identical tasks) or on the paired trials (pairs with TS+LR: TS+LR has pair records only; bases differ).
Competitors: ZERO (no differential distortion, D = 0) and CEILING (every exposed scored trial becomes correct for
every model; evaluated per participant from the observed PLACEBO arm; its a-priori form is D = -delta * gap_prior).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.pmis_r11b import design as ds

G0_PATH = "results/R11_B_G0_existing.json"
MODELS = ("TSLR", "EEGNet", "Shallow")
READOUTS = ("final", "es")
PAIRS = (("Shallow", "EEGNet"), ("EEGNet", "TSLR"), ("Shallow", "TSLR"))
PRIMARY_PAIR = ("Shallow", "EEGNet")
SESOI = 2.0          # BA points, vendored R4 / Paper A value (also the TOST margin and the discrimination minimum)
SRC = {("EEGNet", "final"): "C6/EEGNet/final", ("EEGNet", "es"): "C6/EEGNet/es",
       ("Shallow", "final"): "C6/Shallow/final", ("Shallow", "es"): "C6/Shallow/es",
       ("TSLR", "final"): "C3a/TSLR300/exact", ("TSLR", "es"): "C3a/TSLR300/exact"}


def g0_digest(root: Path) -> str:
    return hashlib.sha256((root / G0_PATH).read_bytes()).hexdigest()


def _comp_key(a: str, b: str, ro: str) -> str:
    return f"{SRC[(a, ro)]} - {SRC[(b, ro)]}"


def frozen_predictions(root: Path) -> dict:
    g0 = json.loads((root / G0_PATH).read_text())
    pm, comps = g0["per_model"], g0["comparisons"]
    out = {"models": {}, "pairs": {}, "selection": {}}
    for m in MODELS:
        for ro in READOUTS:
            r = pm[SRC[(m, ro)]]
            out["models"][f"{m}/{ro}"] = {
                "c": r["c"]["mean"], "c_ci95": r["c"]["ci95"], "d_p": r["d_p"]["mean"],
                "placebo_BA_paired_trials": r["plc_pairs"]["mean"],
                "clean_BA_whole_run": r["clean_run"]["mean"] if "clean_run" in r else None,
                "inflation_pred": {str(d): d * r["c"]["mean"] for d in ds.DOSES},
                "source": SRC[(m, ro)]}
    for a, b in PAIRS:
        for ro in READOUTS:
            c = comps[_comp_key(a, b, ro)]
            dc, gap = c["contrast_difference_dc"]["mean"], c["near_clean_gap"]["mean"]
            gap_ci = c["near_clean_gap"]["ci95"]
            D = {str(d): d * dc for d in ds.DOSES}
            ceil_prior = {str(d): -d * gap for d in ds.DOSES}
            G_L = {str(d): gap + d * dc for d in ds.DOSES}
            disc_zero = max(abs(v) for v in D.values()) >= SESOI
            disc_ceil = max(abs(D[k] - ceil_prior[k]) for k in D) >= SESOI
            xs = c["crossing_dose_delta_star"]
            crossing_testable = bool(gap_ci[1] < 0 and dc > 0 and xs["boot_ci95"][0] > 0 and xs["point"] <= 1.0)
            out["pairs"][f"{a}-{b}/{ro}"] = {
                "dc": dc, "dc_ci95": c["contrast_difference_dc"]["ci95"], "gap_prior": gap, "gap_prior_ci95": gap_ci,
                "gap_measure": c["gap_measure"], "note": c["note"],
                "D_pred": D, "G_L_pred": G_L, "ceiling_prior_D": ceil_prior,
                "D_p_pred": {str(d): d * c["d_p_difference"]["mean"] for d in ds.DOSES},
                "discriminating_vs_ZERO": bool(disc_zero), "discriminating_vs_CEILING": bool(disc_ceil),
                "delta_star_pred": xs["point"], "delta_star_pred_ci95": xs["boot_ci95"],
                "crossing_test_eligible": crossing_testable,
                "primary": (a, b) == PRIMARY_PAIR}
    # predicted top-1 / regret for the 3-model roster (paired-trial placebo BA as the common prior level)
    for ro in READOUTS:
        base = {m: out["models"][f"{m}/{ro}"]["placebo_BA_paired_trials"] for m in MODELS}
        rows = {}
        for d in (0.0,) + ds.DOSES:
            L = {m: base[m] + d * out["models"][f"{m}/{ro}"]["c"] for m in MODELS}
            wl, wp = max(L, key=L.get), max(base, key=base.get)
            rows[str(d)] = {"predicted_L_BA": L, "winner_L": wl, "winner_P": wp, "regret_on_P": base[wp] - base[wl],
                            "regret_relevant": bool(base[wp] - base[wl] >= SESOI)}
        out["selection"][ro] = rows
    out["g0_sha256"] = g0_digest(root)
    return out


def protocol_predictions() -> dict:
    """The saved outcome-blind hypotheses used in the dose experiment."""
    from experiments.runtime import CONFIG_ROOT, load_protocol

    return load_protocol(CONFIG_ROOT / "R11_LEE_RANKING_PROTOCOL.json")["frozen_predictions"]
