"""R12-X frozen analysis: selection policies, cross-fitted / in-sample / 2-block policy regret, participant
bootstrap with re-selection, sign-flip test, Holm, classification. Pure numpy; no model code.
Decision rule (calibrated pre-outcome on synthetic data, reports/R12X_G1R_STATISTICAL_CALIBRATION.md): bootstrap
percentile -- FLIP_AND_REGRET iff any flip and the 5th percentile > 0; REVERSE_REGRET iff any flip and the 97.5th
percentile < 0. The LOO sign-flip test failed calibration and is descriptive only.

Inputs are seed-averaged balanced accuracies in percent, shape (n_participants, n_models), models in ROSTER order:
  BBP: on B, selection-stage placebo arm S_P (clean policy)     BBL: on B, contaminated arm S_L
  BC : on C, clean final refit F
"""

from __future__ import annotations

import numpy as np

ROSTER = ("TSLR", "EEGNet", "Shallow")
TIE_TOL = 0.01          # BA points; ties resolved by ROSTER order
SESOI = 2.0
N_BOOT, BOOT_SEED = 20000, 12010
N_FLIP, FLIP_SEED = 1_000_000, 12399


def ba(p_correct: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy (%) of correct-class probabilities (correct iff p > 0.5)."""
    c = np.asarray(p_correct) > 0.5
    return float(100.0 * np.mean([c[y == k].mean() for k in np.unique(y)]))


def select(scores: np.ndarray) -> int:
    """Index of the selected model for a vector of cohort scores (first in roster order within TIE_TOL of max)."""
    return int(np.flatnonzero(scores >= scores.max() - TIE_TOL)[0])


def _select_rows(S: np.ndarray) -> np.ndarray:
    """Row-wise select for a (k, M) score matrix."""
    return np.argmax(S >= S.max(axis=1, keepdims=True) - TIE_TOL, axis=1)


def e2(BBP, BBL, BC) -> dict:
    wc, wl = select(BBP.mean(0)), select(BBL.mean(0))
    r = BC[:, wc] - BC[:, wl]
    return {"w_clean": ROSTER[wc], "w_contam": ROSTER[wl], "flip": wc != wl, "regret": float(r.mean()), "r": r}


def e1(BBP, BBL, BC) -> dict:
    n = BBP.shape[0]
    tp, tl = BBP.sum(0), BBL.sum(0)
    wc = _select_rows((tp[None] - BBP) / (n - 1))
    wl = _select_rows((tl[None] - BBL) / (n - 1))
    r = BC[np.arange(n), wc] - BC[np.arange(n), wl]
    return {"w_clean": [ROSTER[i] for i in wc], "w_contam": [ROSTER[i] for i in wl], "flip_fraction": float(np.mean(wc != wl)),
            "any_flip": bool(np.any(wc != wl)), "regret": float(r.mean()), "r": r}


def e3(BBP, BBL, BC, block: np.ndarray) -> dict:
    r = np.zeros(BBP.shape[0])
    flips, sel = [], {}
    for b in (0, 1):
        other, me = block != b, block == b
        wc, wl = select(BBP[other].mean(0)), select(BBL[other].mean(0))
        r[me] = BC[me, wc] - BC[me, wl]
        flips.append(wc != wl)
        sel[f"block{b}_evaluated"] = {"w_clean": ROSTER[wc], "w_contam": ROSTER[wl]}
    return {"selections": sel, "flips": flips, "regret": float(r.mean()), "r": r}


def bootstrap(BBP, BBL, BC, kind: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED, *, legacy_e2_sums: bool = False) -> dict:
    """Participant bootstrap; the selection is re-run inside every resample. kind 'E1' (leave-one-identity-out:
    all copies of the evaluated identity are excluded from its selection) or 'E2' (in-sample)."""
    if kind not in ("E1", "E2"):
        raise ValueError("kind must be E1 or E2")
    n = BBP.shape[0]
    if n < 2 or n_boot < 1:
        raise ValueError("bootstrap requires two participants and at least one draw")
    rng = np.random.default_rng(seed)
    out, anyflip, undefined = np.empty(n_boot), np.zeros(n_boot, bool), 0
    idx = np.arange(n)
    for b in range(n_boot):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        if kind == "E2":
            scale = 1 if legacy_e2_sums else n
            wc, wl = select((c @ BBP) / scale), select((c @ BBL) / scale)
            out[b] = c @ (BC[:, wc] - BC[:, wl]) / n
            anyflip[b] = wc != wl
            continue
        u = idx[c > 0]
        den = n - c[u]
        if np.any(den == 0):
            undefined += 1
            out[b] = 0.0
            continue
        wc = _select_rows((c @ BBP - c[u, None] * BBP[u]) / den[:, None])
        wl = _select_rows((c @ BBL - c[u, None] * BBL[u]) / den[:, None])
        out[b] = np.sum(c[u] * (BC[u, wc] - BC[u, wl])) / n
        anyflip[b] = np.any(wc != wl)
    return {"dist": out, "ci95": [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))],
            "lower95_one_sided": float(np.quantile(out, 0.05)),
            "boot_tail_fraction_le_0": float((1 + np.sum(out <= 0)) / (n_boot + 1)),
            "p_any_flip": float(anyflip.mean()), "mass_at_zero": float(np.mean(out == 0.0)),
            "p_negative": float(np.mean(out < 0)), "undefined_resamples": undefined}


def signflip_p(r: np.ndarray, n_draw: int = N_FLIP, seed: int = FLIP_SEED, chunk: int = 50_000) -> float:
    """One-sided Monte-Carlo sign-flip p for mean(r) > 0."""
    rng = np.random.default_rng(seed)
    obs, ge, done = float(np.mean(r)), 0, 0
    while done < n_draw:
        m = min(chunk, n_draw - done)
        s = rng.choice((-1.0, 1.0), size=(m, r.size))
        ge += int(np.sum((s @ r) / r.size >= obs - 1e-12))
        done += m
    return (1 + ge) / (n_draw + 1)


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for j, k in enumerate(keys):
        run = max(run, min(1.0, (m - j) * p[k]))
        out[k] = run
    return out


def classify(any_flip: bool, lower95_one_sided: float, ci95: list[float]) -> str:
    if not any_flip:
        return "NO_FLIP"
    if lower95_one_sided > 0:
        return "FLIP_AND_REGRET"
    if ci95[1] < 0:
        return "REVERSE_REGRET"
    return "FLIP_NO_REGRET"


def blocks_from_metadata(participants: list[int], group_of) -> np.ndarray:
    """E3 blocks: within each original dataset group, sorted participant numbers; odd ranks -> 0, even -> 1."""
    blk = np.zeros(len(participants), dtype=np.int64)
    for g in sorted({group_of(p) for p in participants}):
        mem = sorted(p for p in participants if group_of(p) == g)
        for rank, p in enumerate(mem):
            blk[participants.index(p)] = rank % 2
    return blk


def analyse(BBP, BBL, BC, block, n_boot: int = N_BOOT, *, legacy_e2_sums: bool = False) -> dict:
    a1, a2, a3 = e1(BBP, BBL, BC), e2(BBP, BBL, BC), e3(BBP, BBL, BC, block)
    b1 = bootstrap(BBP, BBL, BC, "E1", n_boot)
    b2 = bootstrap(BBP, BBL, BC, "E2", n_boot, legacy_e2_sums=legacy_e2_sums)
    res = {"E1": {k: v for k, v in a1.items() if k != "r"} | {k: v for k, v in b1.items() if k != "dist"},
           "E2": {k: v for k, v in a2.items() if k != "r"} | {k: v for k, v in b2.items() if k != "dist"},
           "E3": {k: v for k, v in a3.items() if k != "r"}}
    res["E1"]["class"] = classify(a1["any_flip"], b1["lower95_one_sided"], b1["ci95"])
    res["E2"]["class"] = classify(a2["flip"], b2["lower95_one_sided"], b2["ci95"])
    # Holm input: the bootstrap tail fraction (an approximate p; the decision rule was calibrated at the 5 % level in
    # reports/R12X_G1R_STATISTICAL_CALIBRATION.md). Never reported as an exact p-value.
    for k, b in (("E1", b1), ("E2", b2)):
        res[k]["p_decision"] = b["boot_tail_fraction_le_0"]
    res["E2"]["signflip_p_given_selection_calibrated"] = signflip_p(a2["r"]) if a2["flip"] else None
    # descriptive only: the LOO sign-flip test was NOT calibrated (false-regret rate 0.104 under H0a)
    res["E1"]["signflip_p_UNCALIBRATED_descriptive"] = signflip_p(a1["r"]) if a1["any_flip"] else None
    for k in ("E1", "E2"):
        res[k]["regret_ge_SESOI_established"] = bool(res[k]["ci95"][0] >= SESOI)
    return res
