"""R11-B frozen analysis (participant = unit, n = 54). Fixed before any R11-B fit.

Inputs per model / replicate / read-out: per-arm vectors of correct-class probability over all 21,600 trials in the
frozen fold order (every trial is scored exactly once: in the fold that holds out its run). Correct = p > 0.5.
Per participant, BA = class-balanced mean correctness over the participant's 400 scored trials (or over a declared
subset: exposed S_delta trials, unexposed trials, H1 u H2), in points.

Pairs A - B: G_AB(arm) = BA_A(arm) - BA_B(arm); distortion D_AB(delta) = G_AB(L_delta) - G_AB(P).
Inference: t95; two-sided Monte Carlo sign-flip (1e6 sign vectors, seed 11399); Holm within declared families;
SESOI 2 BA points; TOST accuracy = the 90 % CI of the cohort-mean prediction error inside +/- 2 at every dose.
Stitched OWN / OTHER scores (split-half decomposition at delta = 0.4) are per-trial counterfactuals, NOT the
performance of any deployable model.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from experiments.pmis_r11b import design as ds
from experiments.pmis_r11b.predict import MODELS, PAIRS, PRIMARY_PAIR, READOUTS, SESOI

SIGNFLIP_B, SIGNFLIP_SEED, ALPHA = 1_000_000, 11_399, 0.05
BOOT_B, BOOT_SEED = 20_000, 11_010
GRID = (0.0,) + ds.DOSES   # 0.0 = the PLACEBO arm (no scored-trial copy; 100 base copies)


# ------------------------------------------------------------------------------------------------ inference
def signflip_p(d, B: int = SIGNFLIP_B, seed: int = SIGNFLIP_SEED) -> float:
    d = np.asarray(d, dtype=np.float64)
    n, obs = d.size, abs(float(np.mean(d)))
    rng = np.random.default_rng(seed)
    hits, done, chunk = 0, 0, 50_000
    while done < B:
        m = min(chunk, B - done)
        s = rng.integers(0, 2, size=(m, n), dtype=np.int8) * 2 - 1
        hits += int(np.sum(np.abs(s @ d) / n >= obs - 1e-12))
        done += m
    return float((1 + hits) / (1 + B))


def summ(d, test: bool = True) -> dict:
    d = np.asarray(d, dtype=np.float64)
    n, m, sd = d.size, float(d.mean()), float(d.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    h90 = float(stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n))
    out = {"n": n, "mean": m, "ci95": [m - h, m + h], "ci90": [m - h90, m + h90], "sd": sd,
           "n_positive": int(np.sum(d > 0)), "n_negative": int(np.sum(d < 0))}
    if test:
        out["signflip_p"] = signflip_p(d)
    return out


def holm(fam: dict) -> dict:
    keys = sorted(fam, key=lambda k: fam[k]["signflip_p"])
    m, run = len(keys), 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * fam[k]["signflip_p"]))
        fam[k]["holm_p"] = run
        fam[k]["holm_reject"] = bool(run < ALPHA)
    return fam


# ------------------------------------------------------------------------------------------------ participant BA
class Trials:
    """Per-trial metadata aligned with the score vectors (frozen fold order)."""

    def __init__(self, folds: list[dict], labels: np.ndarray, subjects: np.ndarray):
        T = np.concatenate([f["T"] for f in folds])
        self.T, self.lab, self.subj = T, np.asarray(labels)[T], np.asarray(subjects)[T]
        self.parts = np.unique(self.subj)
        self.exposed = {d: np.concatenate([np.isin(f["T"], f["S"][d]) for f in folds]) for d in ds.DOSES}
        self.h1 = np.concatenate([np.isin(f["T"], f["H1"]) for f in folds])
        self.h2 = np.concatenate([np.isin(f["T"], f["H2"]) for f in folds])
        self._idx = {(int(u), c): np.flatnonzero((self.subj == u) & (self.lab == c)) for u in self.parts for c in ds.LABELS}

    def ba(self, correct: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        out = np.empty(self.parts.size)
        for i, u in enumerate(self.parts):
            v = []
            for c in ds.LABELS:
                ix = self._idx[(int(u), c)]
                if mask is not None:
                    ix = ix[mask[ix]]
                v.append(np.mean(correct[ix]))
            out[i] = 100.0 * float(np.mean(v))
        return out


def correct(p: np.ndarray) -> np.ndarray:
    return (np.asarray(p) > 0.5).astype(np.float64)


def ceiling_correct(p_placebo: np.ndarray, exposed: np.ndarray) -> np.ndarray:
    """CEILING competitor: the PLACEBO arm's decisions with every exposed scored trial set correct."""
    c = correct(p_placebo)
    c[exposed] = 1.0
    return c


# ------------------------------------------------------------------------------------------------ building blocks
def arm_table(S: dict, tr: Trials) -> dict:
    """S: model -> read-out -> arm -> p vector. Returns participant BA per model/read-out/arm (all trials)."""
    return {m: {ro: {a: tr.ba(correct(v)) for a, v in S[m][ro].items()} for ro in S[m]} for m in S}


def ro_of(m: str, ro: str) -> str:
    return "exact" if m == "TSLR" else ro


def distortion(BA: dict, a: str, b: str, ro: str) -> dict:
    A, B = BA[a][ro_of(a, ro)], BA[b][ro_of(b, ro)]
    G = {arm: A[arm] - B[arm] for arm in A if arm in B}
    D = {d: G[ds.DOSE_TAG[d]] - G["P"] for d in ds.DOSES}
    return {"G": G, "D": D}


def prediction_test(D_obs: dict, pred: dict, competitors: dict, disc: dict) -> dict:
    """D_obs: dose -> participant vector; pred: dose -> number; competitors: name -> dose -> vector or number."""
    L0 = np.sum([(D_obs[d] - pred[d]) ** 2 for d in ds.DOSES], axis=0)
    comps = {}
    for name, f in competitors.items():
        Lb = np.sum([(D_obs[d] - f[d]) ** 2 for d in ds.DOSES], axis=0)
        comps[name] = {**summ(Lb - L0), "discriminating_a_priori": bool(disc[name])}
    acc = {}
    for d in ds.DOSES:
        e = summ(D_obs[d] - pred[d], test=False)
        acc[str(d)] = {"observed_mean": float(np.mean(D_obs[d])), "observed_ci95": summ(D_obs[d], test=False)["ci95"],
                       "predicted": float(pred[d]), "error_mean": e["mean"], "error_ci90": e["ci90"],
                       "within_margin": bool(e["ci90"][0] > -SESOI and e["ci90"][1] < SESOI)}
    return {"comparisons": comps, "accuracy": acc, "all_within_margin": all(a["within_margin"] for a in acc.values()),
            "loss_prediction": summ(L0, test=False)}


def classify(test: dict) -> str:
    comps = {b: c for b, c in test["comparisons"].items() if c["discriminating_a_priori"]}
    if any(c["holm_reject"] and c["mean"] < 0 for c in comps.values()):
        return "CONTRADICTED"
    won = [c["holm_reject"] and c["mean"] > 0 for c in comps.values()]
    if comps and all(won):
        return "SUPPORTED_STRONG" if test["all_within_margin"] else "SUPPORTED_RELATIVE"
    if any(won):
        return "PARTIAL"
    return "NOT_SUPPORTED"


def reversal_class(gP: dict, gL: dict, d_holm_reject: bool) -> str:
    """R10-A rule for the order change P -> L (gap summaries with ci95)."""
    if np.sign(gP["mean"]) == np.sign(gL["mean"]) or gP["mean"] == 0 or gL["mean"] == 0:
        return "NO_REVERSAL"
    if not d_holm_reject:
        return "DESCRIPTIVE_REVERSAL"
    excl = lambda s: s["ci95"][0] > 0 or s["ci95"][1] < 0  # noqa: E731
    return "SUPPORTED_REVERSAL" if excl(gP) and excl(gL) else "SUPPORTED_CROSSING"


def crossing(G: dict, pred_ci: list, parts_n: int) -> dict:
    """Observed crossing dose from the cohort-mean leaked-arm gap over GRID (0 = PLACEBO), piecewise linear."""
    xs = list(GRID)
    keys = ["P"] + [ds.DOSE_TAG[d] for d in ds.DOSES]
    means = [float(np.mean(G[k])) for k in keys]

    def first_cross(ms):
        if ms[0] >= 0:
            return None, "NOT_EVALUABLE_PREMISE"
        for i in range(1, len(ms)):
            if ms[i] >= 0:
                x0, x1, y0, y1 = xs[i - 1], xs[i], ms[i - 1], ms[i]
                return x0 + (0 - y0) * (x1 - x0) / (y1 - y0), "CROSSED"
        return None, "NO_CROSSING_BY_1.0"

    dstar, status = first_cross(means)
    if status == "CROSSED":
        cls = "CROSSING_AS_PREDICTED" if pred_ci[0] <= dstar <= pred_ci[1] else (
            "CROSSING_EARLIER_THAN_PREDICTED" if dstar < pred_ci[0] else "CROSSING_LATER_THAN_PREDICTED")
    else:
        cls = status
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, parts_n, size=(BOOT_B, parts_n))
    M = np.stack([np.asarray(G[k])[idx].mean(1) for k in keys], axis=1)
    bs = []
    n_none = 0
    for row in M:
        v, st = first_cross(list(row))
        if v is None:
            n_none += 1
        else:
            bs.append(v)
    boot = {"ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 100 else None,
            "fraction_no_crossing_or_premise_failed": n_none / BOOT_B}
    return {"cohort_mean_G_by_dose": dict(zip([str(x) for x in xs], means)), "delta_star_observed": dstar,
            "status": status, "class": cls, "predicted_ci95": pred_ci, "bootstrap": boot}


def selection(BA: dict, ro: str, tr_parts_n: int) -> dict:
    models = list(MODELS)
    lv = lambda m, arm: BA[m][ro_of(m, ro)][arm]  # noqa: E731
    rng = np.random.default_rng(BOOT_SEED + 1)
    idx = rng.integers(0, tr_parts_n, size=(BOOT_B, tr_parts_n))
    out = {}
    P = {m: float(np.mean(lv(m, "P"))) for m in models}
    C = {m: float(np.mean(lv(m, "C"))) for m in models}
    wp, wc = max(P, key=P.get), max(C, key=C.get)
    for d in ds.DOSES:
        a = ds.DOSE_TAG[d]
        L = {m: float(np.mean(lv(m, a))) for m in models}
        wl = max(L, key=L.get)
        bL = np.stack([lv(m, a)[idx].mean(1) for m in models], 1)
        bP = np.stack([lv(m, "P")[idx].mean(1) for m in models], 1)
        wlb, wpb = bL.argmax(1), bP.argmax(1)
        reg_b = bP[np.arange(BOOT_B), wpb] - bP[np.arange(BOOT_B), wlb]
        out[str(d)] = {"means_L": L, "winner_L": wl, "winner_P": wp, "winner_C": wc,
                       "regret_on_P": P[wp] - P[wl], "regret_on_C": C[wc] - C[wl],
                       "regret_relevant_on_P": bool(P[wp] - P[wl] >= SESOI),
                       "bootstrap_P_winner_L_differs_from_winner_P": float(np.mean(wlb != wpb)),
                       "bootstrap_regret_on_P_ci95": [float(np.percentile(reg_b, 2.5)), float(np.percentile(reg_b, 97.5))]}
    return {"means_P": P, "means_C": C, "by_dose": out}


# ------------------------------------------------------------------------------------------------ full analysis
def analyse_replicate(S: dict, tr: Trials, preds: dict) -> dict:
    """S: model -> read-out -> arm -> p vector (one training-seed replicate; TS+LR under read-out 'exact')."""
    BA = arm_table(S, tr)
    n = tr.parts.size
    out = {"absolute_BA": {m: {ro: {a: summ(v, test=False) for a, v in BA[m][ro].items()} for ro in BA[m]} for m in BA},
           "validity_clean_lower_ci_above_50": {f"{m}/{ro}": bool(summ(BA[m][ro]["C"], test=False)["ci95"][0] > 50)
                                                for m in BA for ro in BA[m]}}
    # per-model inflation and the CEILING form, descriptive (accuracy vs delta * c_m)
    infl = {}
    for m in MODELS:
        for ro in READOUTS:
            r = ro_of(m, ro)
            if f"{m}/{ro}" in infl or r not in BA[m]:
                continue
            c_m = preds["models"][f"{m}/{ro}"]["c"]
            row = {}
            for d in ds.DOSES:
                a = ds.DOSE_TAG[d]
                obs = BA[m][r][a] - BA[m][r]["P"]
                ceil = tr.ba(ceiling_correct(S[m][r]["P"], tr.exposed[d])) - BA[m][r]["P"]
                row[str(d)] = {"L_minus_P": summ(obs, test=False), "pred_delta_c": d * c_m,
                               "error_vs_pred_ci90": summ(obs - d * c_m, test=False)["ci90"],
                               "ceiling_mean": float(np.mean(ceil)),
                               "exposed_L_minus_P": summ(tr.ba(correct(S[m][r][a]), tr.exposed[d]) - tr.ba(correct(S[m][r]["P"]), tr.exposed[d]), test=False),
                               "unexposed_L_minus_P": None if d == 1.0 else summ(tr.ba(correct(S[m][r][a]), ~tr.exposed[d]) - tr.ba(correct(S[m][r]["P"]), ~tr.exposed[d]), test=False)}
            row["placebo_P_minus_C"] = summ(BA[m][r]["P"] - BA[m][r]["C"], test=False)
            infl[f"{m}/{ro}"] = row
    out["per_model_inflation"] = infl

    pairs, fam1, fam2, dose_tests, tests = {}, {}, {}, {}, {}
    for a, b in PAIRS:
        for ro in READOUTS:
            key = f"{a}-{b}/{ro}"
            pr = preds["pairs"][key]
            dd = distortion(BA, a, b, ro)
            ra, rb = ro_of(a, ro), ro_of(b, ro)
            ceil = {}
            for d in ds.DOSES:
                ca = tr.ba(ceiling_correct(S[a][ra]["P"], tr.exposed[d])) - BA[a][ra]["P"]
                cb = tr.ba(ceiling_correct(S[b][rb]["P"], tr.exposed[d])) - BA[b][rb]["P"]
                ceil[d] = ca - cb
            pred = {d: pr["D_pred"][str(d)] for d in ds.DOSES}
            t = prediction_test(dd["D"], pred, {"ZERO": {d: 0.0 for d in ds.DOSES}, "CEILING": ceil},
                                {"ZERO": pr["discriminating_vs_ZERO"], "CEILING": pr["discriminating_vs_CEILING"]})
            tests[key] = t
            for name, c in t["comparisons"].items():
                if c["discriminating_a_priori"]:
                    (fam1 if (a, b) == PRIMARY_PAIR else fam2)[f"{key}:{name}"] = c
            dfam = {str(d): summ(dd["D"][d]) for d in ds.DOSES}
            holm(dfam)
            dose_tests[key] = dfam
            pairs[key] = {"gaps": {arm: summ(v, test=False) for arm, v in dd["G"].items()},
                          "D_placebo_P_minus_C": summ(dd["G"]["P"] - dd["G"]["C"], test=False),
                          "D_by_dose": dfam,
                          "D_exposed": {str(d): summ((tr.ba(correct(S[a][ra][ds.DOSE_TAG[d]]), tr.exposed[d]) - tr.ba(correct(S[b][rb][ds.DOSE_TAG[d]]), tr.exposed[d]))
                                                     - (tr.ba(correct(S[a][ra]["P"]), tr.exposed[d]) - tr.ba(correct(S[b][rb]["P"]), tr.exposed[d])), test=False) for d in ds.DOSES},
                          "D_p_scale": {str(d): summ(_p_scale_D(S, tr, a, b, ra, rb, d), test=False) for d in ds.DOSES},
                          "D_p_pred": pr["D_p_pred"],
                          "reversal_P_to_L": {str(d): reversal_class(summ(dd["G"]["P"], test=False), summ(dd["G"][ds.DOSE_TAG[d]], test=False),
                                                                     dfam[str(d)]["holm_reject"]) for d in ds.DOSES},
                          "reversal_C_to_L_descriptive": {str(d): reversal_class(summ(dd["G"]["C"], test=False), summ(dd["G"][ds.DOSE_TAG[d]], test=False),
                                                                                 dfam[str(d)]["holm_reject"]) for d in ds.DOSES}}
            if pr["crossing_test_eligible"]:
                pairs[key]["crossing"] = crossing(dd["G"], pr["delta_star_pred_ci95"], n)
    holm(fam1)
    holm(fam2)
    cls = {k: classify(t) for k, t in tests.items()}
    out["prediction_tests"] = tests
    out["families"] = {"H1_primary": fam1, "H1_secondary_TSLR_pairs": fam2}
    out["classification"] = cls
    out["pairs"] = pairs
    out["selection"] = {ro: selection(BA, ro, n) for ro in READOUTS}
    for ro in READOUTS:
        for d, row in out["selection"][ro]["by_dose"].items():
            wl, wp = row["winner_L"], row["winner_P"]
            if wl == wp:
                row["top1_change"] = "NONE"
            else:
                k = next((f"{x}-{y}/{ro}" for x, y in PAIRS if {x, y} == {wl, wp}), None)
                rc = pairs[k]["reversal_P_to_L"][d] if k else "NOT_A_DECLARED_PAIR"
                row["top1_change"] = "SUPPORTED" if rc == "SUPPORTED_REVERSAL" else f"DESCRIPTIVE ({rc})"
    out["decomposition_delta_0.4"] = decomposition(S, tr, preds)
    out["verdict"] = verdict(cls, pairs)
    return out


def _p_scale_D(S, tr, a, b, ra, rb, d) -> np.ndarray:
    """Participant class-balanced correct-class-probability distortion (points)."""
    arm = ds.DOSE_TAG[d]
    va = np.asarray(S[a][ra][arm]) - np.asarray(S[a][ra]["P"])
    vb = np.asarray(S[b][rb][arm]) - np.asarray(S[b][rb]["P"])
    return tr.ba(va - vb)


def decomposition(S: dict, tr: Trials, preds: dict) -> dict:
    """Split-half crossover at delta = 0.4 on H1 u H2: E = OWN - OTHER (exposure at fixed support of 40 same-run
    copies), S = OTHER - P (support). Stitched per-trial counterfactuals, not deployable models."""
    m12 = tr.h1 | tr.h2
    out = {}
    for m in MODELS:
        for ro in READOUTS:
            r = ro_of(m, ro)
            if f"{m}/{r}" in out:
                continue
            p1, p2, pp = (np.asarray(S[m][r][k]) for k in ("L40", "L40c", "P"))
            own = np.where(tr.h1, p1, p2)
            oth = np.where(tr.h1, p2, p1)
            E = tr.ba(correct(own), m12) - tr.ba(correct(oth), m12)
            Sup = tr.ba(correct(oth), m12) - tr.ba(correct(pp), m12)
            c_m = preds["models"][f"{m}/{ro}"]["c"]
            out[f"{m}/{r}"] = {"E_exposure_fixed_support": summ(E), "S_support": summ(Sup),
                               "E_minus_single_copy_c": summ(E - c_m, test=False), "single_copy_c": c_m,
                               "_E": E, "_S": Sup}
    pairs = {}
    for a, b in PAIRS:
        for ro in READOUTS:
            ka, kb = f"{a}/{ro_of(a, ro)}", f"{b}/{ro_of(b, ro)}"
            dE = out[ka]["_E"] - out[kb]["_E"]
            dS = out[ka]["_S"] - out[kb]["_S"]
            dc = preds["pairs"][f"{a}-{b}/{ro}"]["dc"]
            pairs[f"{a}-{b}/{ro}"] = {"dE": summ(dE, test=False), "dS": summ(dS, test=False), "dc_prior": dc,
                                      "dE_minus_dc_ci90": summ(dE - dc, test=False)["ci90"]}
    for v in out.values():
        v.pop("_E")
        v.pop("_S")
    return {"per_model": out, "pairs": pairs, "label": "stitched per-trial counterfactuals; not deployable models"}


RANK = {"RANKING_PREDICTION_SUPPORTED": 3, "RANKING_PREDICTION_PARTIAL": 2, "RANKING_PREDICTION_NOT_SUPPORTED": 1,
        "RANKING_PREDICTION_CONTRADICTED": 0}


def verdict(cls: dict, pairs: dict) -> dict:
    a, b = PRIMARY_PAIR
    cf, ce = cls[f"{a}-{b}/final"], cls[f"{a}-{b}/es"]
    x = pairs[f"{a}-{b}/es"].get("crossing", {}).get("class", "NOT_ELIGIBLE")
    good = ("SUPPORTED_STRONG", "SUPPORTED_RELATIVE")
    if "CONTRADICTED" in (cf, ce):
        v = "RANKING_PREDICTION_CONTRADICTED"
    elif cf == ce == "SUPPORTED_STRONG" and x == "CROSSING_AS_PREDICTED":
        v = "RANKING_PREDICTION_SUPPORTED"
    elif cf in good or ce in good or x == "CROSSING_AS_PREDICTED":
        v = "RANKING_PREDICTION_PARTIAL"
    else:
        v = "RANKING_PREDICTION_NOT_SUPPORTED"
    return {"verdict": v, "primary_final": cf, "primary_es": ce, "primary_es_crossing": x}


def combine(r1: dict, r2: dict | None) -> dict:
    if r2 is None:
        return {"final_verdict": r1["verdict"]["verdict"], "R2": "NOT_COMPLETED"}
    v1, v2 = r1["verdict"], r2["verdict"]
    worst = min((v1["verdict"], v2["verdict"]), key=lambda k: RANK[k])
    same = all(v1[k] == v2[k] for k in v1)
    sel_same = all(r1["selection"][ro]["by_dose"][d]["winner_L"] == r2["selection"][ro]["by_dose"][d]["winner_L"]
                   and r1["selection"][ro]["by_dose"][d]["regret_relevant_on_P"] == r2["selection"][ro]["by_dose"][d]["regret_relevant_on_P"]
                   and r1["selection"][ro]["by_dose"][d]["winner_P"] == r2["selection"][ro]["by_dose"][d]["winner_P"]
                   for ro in READOUTS for d in r1["selection"][ro]["by_dose"])
    return {"final_verdict": worst, "R1": v1, "R2": v2, "primary_classes_identical": same,
            "selection_identical": sel_same, "seed_robust": bool(same and sel_same)}
