#!/usr/bin/env python3
"""R12-X G1R synthetic calibration of the selection-regret inference (pre-registered in
research/R12X_G1R_CALIBRATION_PREREGISTRATION.md). Synthetic data only; no Dreyer2023 data are read.

Point estimates use the frozen pmis_r12x.analysis e1 / e2 code paths. The bootstrap is a vectorised re-implementation
whose draws equal analysis.bootstrap exactly (asserted at start on random inputs with the same seed).
Writes reports/aggregate/R12X_G1R_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from evaluation import r12x_analysis as an  # noqa: E402

OUT = ROOT / "results" / "R12X_G1R_calibration.json"
N, N_SIM, N_BOOT, N_FLIP = 86, 2000, 2000, 10_000
TAU, SD_U, SD_S, SD_W, SD_ETA, GRID = 6.2, 10.0, 5.0, 4.0, 3.0, 1.25
REGIMES = {
    "H0a_no_contamination_effect": dict(mu=(70, 70, 70), lam=(0, 0, 0), delta=(0, 0, 0), het=None, h0=True),
    "H0b_near_tie_flip_equal_C": dict(mu=(70, 67, 69.5), lam=(5, 8, 12), delta=(0, 0, 0.5), het=None, h0=True),
    "H0c_same_best_on_C": dict(mu=(72, 68, 70), lam=(5, 8, 12), delta=(0, 0, 2), het=None, h0=True),
    "H0d_skewed_heterogeneous_gap": dict(mu=(72, 68, 70), lam=(5, 8, 12), delta=(0, 0, 2), het="skew", h0=True),
    "H0e_partial_loo_flips": dict(mu=(71, 67, 70), lam=(5, 8, 6.2), delta=(0, 0, 1), het=None, h0=True),
    "H1_adverse_choice": dict(mu=(72, 68, 70), lam=(5, 8, 12), delta=(0, 0, 0), het=None, h0=False),
    "H1_strong": dict(mu=(73, 68, 70), lam=(5, 8, 14), delta=(0, 0, 0), het=None, h0=False),
    "NEG_beneficial_contamination": dict(mu=(72, 68, 70), lam=(5, 8, 12), delta=(0, 0, 4), het=None, h0=False),
    "NOFLIP_equal_inflation": dict(mu=(72, 68, 70), lam=(8, 8, 8), delta=(0, 0, 0), het=None, h0=False),
}


def rnd(x):
    return np.clip(np.round(x / GRID) * GRID, 0, 100)


def generate(cfg, rng):
    mu, lam, delta = (np.array(cfg[k], float) for k in ("mu", "lam", "delta"))
    u = rng.normal(0, SD_U, (N, 1))
    v = rng.normal(0, TAU / np.sqrt(2), (N, 3))
    a = mu + u + v
    bbp = a + rng.normal(0, SD_S, (N, 1)) + rng.normal(0, SD_W, (N, 3))
    bbl = bbp + lam + rng.normal(0, SD_ETA, (N, 3))
    het = np.zeros((N, 3))
    if cfg["het"] == "skew":
        het[:, 2] = np.where(rng.random(N) < 0.2, 8.0, -2.0)
    bc = a + delta + het + rng.normal(0, SD_S, (N, 1)) + rng.normal(0, SD_W, (N, 3))
    return rnd(bbp), rnd(bbl), rnd(bc), mu + delta


def sel_rows(S):
    return np.argmax(S >= S.max(axis=-1, keepdims=True) - an.TIE_TOL, axis=-1)


def boot(BBP, BBL, BC, kind, n_boot, seed):
    n = BBP.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))
    Cm = np.zeros((n_boot, n))
    np.add.at(Cm, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    if kind == "E2":
        wc, wl = sel_rows((Cm @ BBP) / n), sel_rows((Cm @ BBL) / n)
        gap = BC[:, wc] - BC[:, wl]                                     # (n, B)
        R = np.einsum("bi,ib->b", Cm, gap) / n
        return R, wc != wl
    den = (n - Cm)[:, :, None]
    den[den == 0] = np.nan
    sp = (Cm @ BBP)[:, None, :] - Cm[:, :, None] * BBP[None]
    sl = (Cm @ BBL)[:, None, :] - Cm[:, :, None] * BBL[None]
    wc, wl = sel_rows(sp / den), sel_rows(sl / den)                     # (B, n)
    gap = BC[np.arange(n)[None, :], wc] - BC[np.arange(n)[None, :], wl]
    R = (Cm * gap).sum(1) / n
    return R, ((wc != wl) & (Cm > 0)).any(1)


def signflip(r, n_draw, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice((-1.0, 1.0), size=(n_draw, r.size))
    st = s @ r / r.size
    obs = r.mean()
    return (1 + np.sum(st >= obs - 1e-12)) / (n_draw + 1), (1 + np.sum(st <= obs + 1e-12)) / (n_draw + 1)


def one(args):
    name, k = args
    cfg = REGIMES[name]
    rng = np.random.default_rng([12500, list(REGIMES).index(name), k])
    BBP, BBL, BC, muC = generate(cfg, rng)
    e1, e2 = an.e1(BBP, BBL, BC), an.e2(BBP, BBL, BC)
    wc1 = np.array([an.ROSTER.index(w) for w in e1["w_clean"]])
    wl1 = np.array([an.ROSTER.index(w) for w in e1["w_contam"]])
    th1 = float(np.mean(muC[wc1] - muC[wl1]))
    th2 = float(muC[an.ROSTER.index(e2["w_clean"])] - muC[an.ROSTER.index(e2["w_contam"])])
    R1, _ = boot(BBP, BBL, BC, "E1", N_BOOT, 1000 + k)
    R2, _ = boot(BBP, BBL, BC, "E2", N_BOOT, 2000 + k)
    q1 = np.nanquantile(R1, [0.025, 0.05, 0.975])
    q2 = np.nanquantile(R2, [0.025, 0.05, 0.975])
    p1, p1l = signflip(e1["r"], N_FLIP, 3000 + k)
    p2, p2l = signflip(e2["r"], N_FLIP, 4000 + k)
    f1, f2 = e1["any_flip"], e2["flip"]
    return {"theta_E1": th1, "theta_E2": th2, "flip_E1": f1, "flip_frac_E1": e1["flip_fraction"], "flip_E2": f2,
            "A_regret": f1 and q1[1] > 0, "A_reverse": f1 and q1[2] < 0,
            "B_regret": f1 and p1 < 0.05, "B_reverse": f1 and p1l < 0.025,
            "E2boot_regret": f2 and q2[1] > 0, "E2boot_reverse": f2 and q2[2] < 0,
            "E2sf_regret": f2 and p2 < 0.05, "E2sf_reverse": f2 and p2l < 0.025,
            "cov_E1": bool(q1[0] <= th1 <= q1[2]), "cov_E2": bool(q2[0] <= th2 <= q2[2]),
            "est_E1": e1["regret"], "est_E2": e2["regret"]}


def check_equivalence():
    rng = np.random.default_rng(7)
    for kind in ("E1", "E2"):
        BBP, BBL, BC, _ = generate(REGIMES["H1_adverse_choice"], rng)
        ref = an.bootstrap(BBP, BBL, BC, kind, n_boot=300, seed=99)["dist"]
        mine, _ = boot(BBP, BBL, BC, kind, 300, 99)
        assert np.allclose(ref, mine, atol=1e-12, equal_nan=False), f"bootstrap implementations differ ({kind})"


def main():
    ap = argparse.ArgumentParser(description="Reproduce the pre-outcome synthetic inference calibration.")
    ap.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    check_equivalence()
    jobs = [(name, k) for name in REGIMES for k in range(N_SIM)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = pool.map(one, jobs, chunksize=50)
    out = {"schema": "R12X_G1R_calibration_v1", "n_participants": N, "n_sim": N_SIM, "n_boot": N_BOOT,
           "n_signflip": N_FLIP, "bootstrap_equivalence_with_frozen_code": True, "regimes": {}}
    se = lambda p: float(np.sqrt(p * (1 - p) / N_SIM))  # noqa: E731
    for name in REGIMES:
        rr = [r for (nm, _), r in zip(jobs, res) if nm == name]
        agg = {k: float(np.mean([r[k] for r in rr])) for k in rr[0] if isinstance(rr[0][k], (bool, np.bool_))}
        agg.update({f"mean_{k}": float(np.mean([r[k] for r in rr])) for k in ("theta_E1", "theta_E2", "est_E1",
                                                                              "est_E2", "flip_frac_E1")})
        agg["mc_se_at_5pct"] = se(0.05)
        out["regimes"][name] = {"truth_h0": REGIMES[name]["h0"], **agg}
    h0 = [v for v in out["regimes"].values() if v["truth_h0"]]
    for rule in ("A", "B", "E2boot", "E2sf"):
        out[f"rule_{rule}_max_false_regret_H0"] = max(v[f"{rule}_regret"] for v in h0)
        out[f"rule_{rule}_max_false_reverse_H0"] = max(v[f"{rule}_reverse"] for v in h0)
        out[f"rule_{rule}_calibrated"] = bool(out[f"rule_{rule}_max_false_regret_H0"] <= 0.060
                                              and out[f"rule_{rule}_max_false_reverse_H0"] <= 0.035)
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "regimes"}, indent=1))
    for name, v in out["regimes"].items():
        print(name, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})


if __name__ == "__main__":
    main()
