"""R10-B frozen analysis (participant = unit, n = 54). Everything here is fixed before any real-data outcome.

Per participant, quantities are class-balanced means over that participant's 400 anchors (4 folds x 100), in points
(x100) of correct-class probability (p) or of correctness (BA). Crossover scores: OWN(a) = the anchor's score in the
arm that contains its own trial's extra window(s); OTHER(a) = its score in the paired arm that contains the other
half's windows instead. OWN / OTHER are per-anchor crossover scores, NOT the performance of one deployable model.

Inference: t95 interval; two-sided Monte Carlo sign-flip test (1e6 sign vectors, seed SIGNFLIP_SEED); Holm within
each declared family. Retention ratios rho_u = E_u(lag) / E_u(0) are formed per participant only if every
participant's denominator exceeds GUARD points for that model (else the declared residual-form fallback is used).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from experiments.pmis_r10b import geometry as geo

SIGNFLIP_B, SIGNFLIP_SEED, GUARD, ACC_MARGIN, ALPHA = 1_000_000, 10_399, 0.5, 0.10, 0.05
DISCRIM_MIN = 0.05  # a comparison is informative only if prediction and baseline differ by >= 0.05 at some condition


# ------------------------------------------------------------------------------------------------ inference
def signflip_p(d, B: int = SIGNFLIP_B, seed: int = SIGNFLIP_SEED) -> float:
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    obs = abs(d.mean())
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
    n = d.size
    m, sd = float(d.mean()), float(d.std(ddof=1))
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


# ------------------------------------------------------------------------------------------------ per-participant
def class_balanced(v: np.ndarray, lab: np.ndarray, subj: np.ndarray, parts: np.ndarray) -> np.ndarray:
    out = np.empty(parts.size)
    for i, u in enumerate(parts):
        m = subj == u
        out[i] = 100.0 * np.mean([np.mean(v[m & (lab == c)]) for c in (1, 2)])
    return out


def crossover(own_1, own_2, half):
    """OWN / OTHER per anchor from paired arms (arm 1 holds H1's windows, arm 2 holds H2's)."""
    own = np.where(half == 1, own_1, own_2)
    other = np.where(half == 1, own_2, own_1)
    return own, other


def participant_table(S: dict, half, lab, subj, parts) -> dict:
    """S: arm -> per-anchor score vector (all folds concatenated). Returns participant-level quantities."""
    cb = lambda v: class_balanced(v, lab, subj, parts)  # noqa: E731
    out = {"B": {}, "A": {}}
    for lag in geo.LAGS_B:
        own, oth = crossover(S[f"B1_{lag}"], S[f"B2_{lag}"], half)
        out["B"][lag] = {"E": cb(own - oth), "S": cb(oth - S["Z"]), "I": cb(own - S["Z"]), "OWN": cb(own), "OTHER": cb(oth)}
    out["B"]["Z"] = cb(S["Z"])
    for s in geo.STRIDES:
        oa, ta = crossover(S[f"A{s}_A1"], S[f"A{s}_A2"], half)
        ox, tx = crossover(S[f"A{s}_X1"], S[f"A{s}_X2"], half)
        G, L = S[f"A{s}_G"], S[f"A{s}_L"]
        out["A"][s] = {"H1_L_minus_G": cb(L - G), "E_A": cb(oa - ta), "E_X": cb(ox - tx), "support_other_minus_G": cb(ta - G),
                       "half_own_minus_G": cb(oa - G), "G": cb(G), "L": cb(L)}
    return out


# ------------------------------------------------------------------------------------------------ prediction tests
def retention(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray | None, bool]:
    ok = bool(np.all(den > GUARD))
    return (num / den if ok else None), ok


def loss(obs: dict, pred: dict, conds) -> np.ndarray:
    return np.sum([(obs[c] - pred[c]) ** 2 for c in conds], axis=0)


def loss_resid(num: dict, den: np.ndarray, pred: dict, conds) -> np.ndarray:
    """Declared fallback if the guard fails: residual form sum (E(c) - f(c) E(0))^2 / E(0)-free scale."""
    return np.sum([(num[c] - pred[c] * den) ** 2 for c in conds], axis=0)


def discriminating(pred: dict, base: dict, conds) -> bool:
    return bool(max(abs(pred[c] - base[c]) for c in conds) >= DISCRIM_MIN)


def prediction_test(num: dict, den: np.ndarray, pred: dict, baselines: dict, conds) -> dict:
    rho = {}
    ok = bool(np.all(den > GUARD))
    if ok:
        rho = {c: num[c] / den for c in conds}
        L0 = loss(rho, pred, conds)
        Lb = {b: loss(rho, f, conds) for b, f in baselines.items()}
    else:
        L0 = loss_resid(num, den, pred, conds)
        Lb = {b: loss_resid(num, den, f, conds) for b, f in baselines.items()}
    res = {"guard_passed": ok, "scale": "participant retention ratio" if ok else "residual fallback",
           "comparisons": {b: {**summ(Lb[b] - L0), "discriminating_a_priori": discriminating(pred, baselines[b], conds)}
                           for b in baselines},
           "loss_C5": summ(L0, test=False)}
    if ok:
        acc = {}
        for c in conds:
            e = summ(rho[c] - pred[c], test=False)
            acc[c] = {"observed_mean": float(np.mean(rho[c])), "observed_ci95": summ(rho[c], test=False)["ci95"],
                      "predicted": pred[c], "error_mean": e["mean"], "error_ci90": e["ci90"],
                      "within_margin": bool(e["ci90"][0] > -ACC_MARGIN and e["ci90"][1] < ACC_MARGIN)}
        res["accuracy"] = acc
        res["all_within_margin"] = all(a["within_margin"] for a in acc.values())
        obs_c = [np.mean(rho[c]) for c in conds]
        res["spearman_pred_vs_observed_descriptive"] = float(stats.spearmanr([pred[c] for c in conds], obs_c)[0])
    return res


def classify(test: dict) -> str:
    comps = {b: c for b, c in test["comparisons"].items() if c["discriminating_a_priori"]}
    if any(c["holm_reject"] and c["mean"] < 0 for c in comps.values()):
        return "CONTRADICTED"  # a baseline predicts significantly better than the C5 curve
    won = [c["holm_reject"] and c["mean"] > 0 for c in comps.values()]
    if comps and all(won):
        return "SUPPORTED_STRONG" if test.get("all_within_margin") else "SUPPORTED_RELATIVE"
    if any(won):
        return "PARTIAL"
    return "NOT_SUPPORTED"


def overall_verdict(cls_b: dict, cls_a: dict) -> str:
    """Frozen overall rule. EEGNet carries the discriminating test (its curve departs most from overlap)."""
    good = ("SUPPORTED_STRONG", "SUPPORTED_RELATIVE")
    if cls_b["EEGNet"] == "SUPPORTED_STRONG" and cls_b["TSLR"] in good + ("PARTIAL",) and cls_a["EEGNet"] in good:
        return "PREDICTIVE_TRANSFER_SUPPORTED"
    if cls_b["EEGNet"] in good:
        return "PARTIAL_TRANSFER"
    return "NO_PREDICTIVE_TRANSFER"


# ------------------------------------------------------------------------------------------------ full analysis
def stack_scores(fold_out: list[dict], key: str) -> dict:
    """fold_out: per fold (frozen order) dict of p_<arm> / c_<arm>. key 'p' or 'c'."""
    arms = sorted({k[2:] for k in fold_out[0] if k.startswith(key + "_")})
    return {a: np.concatenate([fo[f"{key}_{a}"] for fo in fold_out]) for a in arms}


def c0_participant(c0: dict, meta: dict, parts: np.ndarray) -> dict:
    """c0: subject -> {scheme k rows / p}. Window-level, trial-aggregated and anchor-only BA per scheme."""
    lab, N = np.asarray(meta["label"]), 21600
    res = {sch: {"window": [], "trial": [], "anchor": []} for sch in ("rw", "tg")}
    for u in parts:
        z = c0[int(u)]
        for sch in ("rw", "tg"):
            rows = np.concatenate([z[f"{sch}{k}_rows"] for k in range(5)])
            p = np.concatenate([z[f"{sch}{k}_p"] for k in range(5)])
            t, off = rows % N, np.asarray(geo.OFFSETS)[rows // N]
            yl = lab[t]
            corr = (p > 0.5).astype(float)
            res[sch]["window"].append(100 * np.mean([corr[yl == c].mean() for c in (1, 2)]))
            ut = np.unique(t)
            mp = np.array([p[t == x].mean() for x in ut])
            tc = (mp > 0.5).astype(float)
            res[sch]["trial"].append(100 * np.mean([tc[lab[ut] == c].mean() for c in (1, 2)]))
            a = off == 0
            res[sch]["anchor"].append(100 * np.mean([corr[a & (yl == c)].mean() for c in (1, 2)]))
    out = {}
    for lvl in ("window", "trial", "anchor"):
        rw, tg = np.array(res["rw"][lvl]), np.array(res["tg"][lvl])
        out[lvl] = {"random_window_BA": summ(rw, test=False), "trial_grouped_BA": summ(tg, test=False),
                    "inflation_rw_minus_tg": summ(rw - tg)}
    return out


def analyse_all(scores: dict, folds: list[dict], meta: dict, curves: dict, c0: dict | None) -> dict:
    """scores: model -> list of per-fold dicts (frozen fold order). curves: frozen knots per model."""
    lab_all, subj_all = np.asarray(meta["label"]), np.asarray(meta["subject"])
    T = np.concatenate([f["T"] for f in folds])
    half = np.concatenate([np.where(np.isin(f["T"], f["H1"]), 1, 2) for f in folds])
    lab, subj = lab_all[T], subj_all[T]
    parts = np.unique(subj)
    P, C = {}, {}
    for m in geo.MODELS:
        P[m] = participant_table(stack_scores(scores[m], "p"), half, lab, subj, parts)
        C[m] = participant_table(stack_scores(scores[m], "c"), half, lab, subj, parts)
    out = {"n_participants": int(parts.size), "descriptive": {}, "families": {}}

    # descriptive curves
    for m in geo.MODELS:
        d = {"TrackB": {}, "TrackA": {}}
        for lag in geo.LAGS_B:
            d["TrackB"][lag] = {q: {"p": summ(P[m]["B"][lag][q], test=False), "BA": summ(C[m]["B"][lag][q], test=False)}
                                for q in ("E", "S", "I", "OWN", "OTHER")}
        d["TrackB"]["Z_level"] = {"p": summ(P[m]["B"]["Z"], test=False), "BA": summ(C[m]["B"]["Z"], test=False)}
        for s in geo.STRIDES:
            d["TrackA"][s] = {q: {"p": summ(P[m]["A"][s][q], test=False), "BA": summ(C[m]["A"][s][q], test=False)}
                              for q in P[m]["A"][s]}
        # retention on both scales as ratios of cohort means (descriptive; the tested retention is per participant, p scale)
        d["retention_ratio_of_cohort_means"] = {
            sc: {"TrackB": {lag: float(np.mean(T_[m]["B"][lag]["E"]) / np.mean(T_[m]["B"][0]["E"])) for lag in geo.LAGS_B_NONZERO},
                 "TrackA": {s: float(np.mean(T_[m]["A"][s]["E_A"]) / np.mean(T_[m]["A"][s]["E_X"])) for s in geo.STRIDES}}
            for sc, T_ in (("p", P), ("BA", C))}
        out["descriptive"][m] = d

    # H1: realistic window-level contamination vs trial-grouped control (BA primary), Holm over 6
    out["families"]["H1"] = holm({f"{m}_stride{s}": summ(C[m]["A"][s]["H1_L_minus_G"]) for m in geo.MODELS for s in geo.STRIDES})

    # H2: matched overlap effect (p primary), Holm over 12
    fam = {}
    for m in geo.MODELS:
        e2000 = P[m]["B"][2000]["E"]
        for lag in geo.LAGS_B[1:-1]:
            fam[f"{m}_E{lag}_minus_E2000"] = summ(P[m]["B"][lag]["E"] - e2000)
        fam[f"{m}_E2000"] = summ(e2000)
    out["families"]["H2"] = holm(fam)

    # H3 (Track B, primary) and H3-A (Track A, secondary)
    tests_b, tests_a = {}, {}
    for m in geo.MODELS:
        pb = geo.track_b_prediction(curves[m])
        num = {lag: P[m]["B"][lag]["E"] for lag in geo.LAGS_B_NONZERO}
        base_b = {b: {lag: geo.baseline_b(b, lag) for lag in geo.LAGS_B_NONZERO} for b in geo.BASELINES_B}
        tests_b[m] = prediction_test(num, P[m]["B"][0]["E"], pb, base_b, geo.LAGS_B_NONZERO)
        pa = {s: geo.track_a_prediction(curves[m], s, "additive") for s in geo.STRIDES}
        numa = {s: P[m]["A"][s]["E_A"] for s in geo.STRIDES}
        dena = np.min([P[m]["A"][s]["E_X"] for s in geo.STRIDES], axis=0)
        base_a = {b: {s: geo.baseline_a(b, s) for s in geo.STRIDES} for b in geo.BASELINES_A}
        if np.all(dena > GUARD):
            rho_a = {s: numa[s] / P[m]["A"][s]["E_X"] for s in geo.STRIDES}
            tests_a[m] = prediction_test(rho_a, np.ones(parts.size), pa, base_a, geo.STRIDES)  # rho_a already a ratio
            tests_a[m]["scale"] = "participant ratio E_A(s) / E_X(s)"
        else:
            tests_a[m] = {"guard_passed": False, "note": "E_X <= guard for some participant: H3-A not computable"}
        # secondary functionals (descriptive; reported in full, never used for a decision)
        if tests_a[m].get("guard_passed", True) and "accuracy" in tests_a[m]:
            obs = {s: float(np.mean(rho_a[s])) for s in geo.STRIDES}
            tests_a[m]["secondary_functionals_descriptive"] = {
                fn: {"predicted": {s: geo.track_a_prediction(curves[m], s, fn) for s in geo.STRIDES},
                     "abs_error_cohort": {s: abs(obs[s] - geo.track_a_prediction(curves[m], s, fn)) for s in geo.STRIDES}}
                for fn in ("additive", "nearest", "saturating")}
    holm({f"{m}_vs_{b}": tests_b[m]["comparisons"][b] for m in geo.MODELS for b in geo.BASELINES_B})
    holm({f"{m}_vs_{b}": tests_a[m]["comparisons"][b] for m in geo.MODELS for b in geo.BASELINES_A if "comparisons" in tests_a[m]})
    cls_b = {m: classify(tests_b[m]) for m in geo.MODELS}
    cls_a = {m: (classify(tests_a[m]) if "comparisons" in tests_a[m] else "NOT_COMPUTABLE") for m in geo.MODELS}
    out["families"]["H3_trackB"] = {"tests": tests_b, "classification": cls_b}
    out["families"]["H3A_trackA"] = {"tests": tests_a, "classification": cls_a}

    # H4: model generality (Holm over 7): curve swap (Track B) + retention differences on matched anchors and arms
    fam = {}
    rho = {}
    for m in geo.MODELS:
        den = P[m]["B"][0]["E"]
        rho[m] = {lag: P[m]["B"][lag]["E"] / den for lag in geo.LAGS_B_NONZERO} if np.all(den > GUARD) else None
    if rho["EEGNet"] is not None and rho["TSLR"] is not None:
        for m, o in (("EEGNet", "TSLR"), ("TSLR", "EEGNet")):
            own = geo.track_b_prediction(curves[m])
            oth = geo.track_b_prediction(curves[o])
            fam[f"{m}_own_curve_beats_{o}_curve"] = summ(loss(rho[m], oth, geo.LAGS_B_NONZERO) - loss(rho[m], own, geo.LAGS_B_NONZERO))
        for lag in geo.LAGS_B_NONZERO[:-1]:
            fam[f"EEGNet_minus_TSLR_retention_{lag}"] = summ(rho["EEGNet"][lag] - rho["TSLR"][lag])
        out["families"]["H4"] = holm(fam)
        out["families"]["H4"]["_predicted_differences_C5"] = {
            lag: geo.track_b_prediction(curves["EEGNet"])[lag] - geo.track_b_prediction(curves["TSLR"])[lag] for lag in geo.LAGS_B_NONZERO[:-1]}
    else:
        out["families"]["H4"] = {"note": "retention guard failed for a model: H4 not computable"}

    out["verdict"] = overall_verdict(cls_b, cls_a)
    if c0 is not None:
        out["C0_conventional"] = {m: c0_participant(c0[m], meta, parts) for m in geo.MODELS}
    return out
