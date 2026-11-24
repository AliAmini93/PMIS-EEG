#!/usr/bin/env python3
"""R15-C0 synthetic studies S1-S4 (pre-specified in research/R15C0_SIMULATION_PLAN.md). Synthetic data only.

Usage: r15c0_sims.py S1|S2|S3|S4|all [--quick]
Writes reports/aggregate/R15C0_<study>.json. CPU only (numpy), run under nice; no GPU.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
from experiments.pmis_r15c0 import sim  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

AGG = ROOT / "results"
QUICK = "--quick" in sys.argv
N_REP = 60 if QUICK else 400
N_BOOT = 200 if QUICK else 1000
N_BOOT_S3 = 100 if QUICK else 500
WORKERS = 6

# design profiles: trials on B / C per participant, session interaction SD (logit)
DESIGNS = {
    "dreyer_within_day_40": dict(TB=40, TC=40, sw=0.10),
    "yang_within_day_1day_40": dict(TB=40, TC=40, sw=0.10),
    "yang_within_day_3days_120": dict(TB=120, TC=120, sw=0.10),
    "yang_cross_day_200_sw025": dict(TB=200, TC=200, sw=0.25),
    "yang_cross_day_200_sw040": dict(TB=200, TC=200, sw=0.40),
}


def pmap(fn, jobs):
    os.nice(10)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        return pool.map(fn, jobs, chunksize=max(1, len(jobs) // (WORKERS * 8)))


# ------------------------------------------------------------------------------------------------- S1
S1_ROSTER = ("TSLR_like", "CNN_a", "CNN_b_leader", "FM_pre", "FM_rand")
S1_DOSES = (0.25, 0.5, 0.75, 1.0)


def s1_cells():
    for r_int, lead, n, TB in itertools.product((0.95, 0.99, 0.999, 1.0), (66.0, 70.0, 74.0), (51, 83), (40, 200)):
        yield dict(r_int=r_int, lead=lead, n=n, TB=TB)


def s1_scenario(c):
    return sim.Scenario("S1", S1_ROSTER, accB=(64.0, 68.0, c["lead"], 68.0, 54.0),
                        r=(0.6, 0.9, c["r_int"], c["r_int"], c["r_int"]), n=c["n"], TB=c["TB"], TC=c["TB"],
                        sw=0.10)


def s1_one(args):
    ci, c, rep = args
    sc = s1_scenario(c)
    rng = np.random.default_rng([15001, ci, rep])
    out = sim.simulate(sc, rng, doses=S1_DOSES)
    res = {}
    wc = sim.sel_rows(out["BBP"][0.5].mean(0)[None])[0]
    for d in S1_DOSES:
        S = out["BBL"][d].mean(0)
        tm = sim.tie_mask(S[None])[0]
        fwd = int(np.flatnonzero(tm)[0])
        rev = int(np.flatnonzero(tm)[-1])
        top2 = np.sort(S)[-2:]
        loo = sim.loo_means(out["BBL"][d])
        ltm = sim.tie_mask(loo)
        res[d] = dict(tie=bool(tm.sum() > 1), decisive=fwd != rev, flip=fwd != wc, sel=fwd,
                      margin=float(top2[1] - top2[0]), max_score=float(S.max()),
                      loo_tie_frac=float((ltm.sum(1) > 1).mean()))
    return ci, res


def run_s1():
    cells = list(s1_cells())
    jobs = [(ci, c, k) for ci, c in enumerate(cells) for k in range(N_REP)]
    res = pmap(s1_one, jobs)
    out = {"schema": "R15C0_S1_v1", "synthetic_only": True, "roster": S1_ROSTER, "n_rep": N_REP,
           "fixed": {"accB": "TSLR_like 64, CNN_a 68, CNN_b_leader <lead>, FM_pre 68, FM_rand 54",
                     "r": "TSLR_like 0.6, CNN_a 0.9, others <r_int>", "sw": 0.10, "K": 5}, "cells": []}
    for ci, c in enumerate(cells):
        rr = [r for i, r in res if i == ci]
        sc = s1_scenario(c)
        row = dict(c)
        for d in S1_DOSES:
            pc, pl, _ = sim.population_scores(sc, d, n_mc=200_000)
            sel = np.bincount([r[d]["sel"] for r in rr], minlength=len(S1_ROSTER)) / len(rr)
            row[f"d{d}"] = dict(
                p_tie=float(np.mean([r[d]["tie"] for r in rr])),
                p_tiebreak_decisive=float(np.mean([r[d]["decisive"] for r in rr])),
                p_flip=float(np.mean([r[d]["flip"] for r in rr])),
                mean_top2_margin=float(np.mean([r[d]["margin"] for r in rr])),
                mean_max_score=float(np.mean([r[d]["max_score"] for r in rr])),
                mean_loo_tie_frac=float(np.mean([r[d]["loo_tie_frac"] for r in rr])),
                selected_share=dict(zip(S1_ROSTER, map(float, sel))),
                population_contaminated=dict(zip(S1_ROSTER, map(float, np.round(pl, 3)))),
                population_clean=dict(zip(S1_ROSTER, map(float, np.round(pc, 3)))))
        out["cells"].append(row)
    return out


# ------------------------------------------------------------------------------------------------- S2
S2_ROSTER = ("leader", "challenger", "other")
DELTAS = (0, 1, 2, 3, 4, 6)
FLIPM = (-2.0, -0.5, 0.5, 2.0, 5.0)
R_LEADER = 0.4


def challenger_params(base: sim.Scenario, F, delta=0.5):
    """Solve for the challenger's memorisation (then spill-over) so that its population contaminated score
    exceeds the leader's by F BA points at dose delta (F < 0: the leader stays ahead by |F|)."""
    def gap(rh, sh):
        sc = sim.with_(base, r=(R_LEADER, rh, 0.6), s=(0.0, sh, 0.0))
        _, pl, _ = sim.population_scores(sc, delta, n_mc=100_000, seed=5)
        return pl[1] - pl[0] - F
    if gap(R_LEADER, 0.0) >= 0:     # the challenger already wins at equal memorisation: lower its rate
        rh = brentq(lambda x: gap(x, 0.0), 0.0, R_LEADER)
        return rh, 0.0
    if gap(1.0, 0.0) >= 0:
        return brentq(lambda x: gap(x, 0.0), R_LEADER, 1.0), 0.0
    return 1.0, brentq(lambda y: gap(1.0, y), 0.0, 6.0)


def s2_base(design, n, Delta, sv=0.3, sr=0.0, accL=72.0, cshift=0.0):
    d = DESIGNS[design]
    accB = (accL, accL - Delta, 64.0)
    accC = (accL, accL - Delta + cshift, 64.0)
    return sim.Scenario("S2", S2_ROSTER, accB=accB, accC=accC, n=n, TB=d["TB"], TC=d["TC"], sw=d["sw"], sv=sv,
                        sr=sr)


def s2_cells():
    cells = []
    for design, n, Delta, F in itertools.product(DESIGNS, (51, 83), DELTAS, FLIPM):
        cells.append(dict(family="main", design=design, n=n, Delta=Delta, F=F, sv=0.3, sr=0.0, cshift=0.0))
    for design, n, sv, Delta in itertools.product(("dreyer_within_day_40", "yang_cross_day_200_sw025"), (51, 83),
                                                 (0.15, 0.5), (2, 3)):
        cells.append(dict(family="sens_sv", design=design, n=n, Delta=Delta, F=2.0, sv=sv, sr=0.0, cshift=0.0))
    for design, n, Delta in itertools.product(("dreyer_within_day_40", "yang_cross_day_200_sw025"), (51, 83), (2, 3)):
        cells.append(dict(family="sens_het", design=design, n=n, Delta=Delta, F=2.0, sv=0.3, sr=0.5, cshift=0.0))
    for design, n in itertools.product(DESIGNS, (51, 83)):
        cells.append(dict(family="null_no_effect", design=design, n=n, Delta=0, F=None, sv=0.3, sr=0.0, cshift=0.0))
        # no contamination effect, but a 2-BA true gap: any flip is seed / sampling noise (reported separately)
        cells.append(dict(family="noise_flip_no_effect_gap2", design=design, n=n, Delta=2, F=None, sv=0.3, sr=0.0,
                          cshift=0.0))
        cells.append(dict(family="null_same_best_on_C", design=design, n=n, Delta=2, F=2.0, sv=0.3, sr=0.0,
                          cshift=4.0))
        cells.append(dict(family="null_heterogeneous_gap", design=design, n=n, Delta=0, F=2.0, sv=0.5, sr=0.5,
                          cshift=0.0))
    return cells


def s2_scenario(c):
    base = s2_base(c["design"], c["n"], c["Delta"], c["sv"], c["sr"], cshift=c["cshift"])
    if c["F"] is None:
        return sim.with_(base, r=(0.0, 0.0, 0.0), s=(0.0, 0.0, 0.0)), None, None
    rh, sh = challenger_params(base, c["F"])
    return sim.with_(base, r=(R_LEADER, rh, 0.6), s=(0.0, sh, 0.0)), rh, sh


def decide(BBP, BBL, BC, muC, Cm, tieaware):
    if tieaware:
        mc, ml, r = sim.e1_tieaware(BBP, BBL, BC)
        anyflip = bool((mc != ml).any())
        theta = float(np.mean((mc * muC).sum(1) / mc.sum(1) - (ml * muC).sum(1) / ml.sum(1)))
    else:
        wc, wl, r = sim.e1(BBP, BBL, BC)
        anyflip = bool((wc != wl).any())
        theta = float(np.mean(muC[wc] - muC[wl]))
    R, _ = sim.boot_e1(BBP, BBL, BC, Cm, tieaware=tieaware)
    q025, q05, q975 = np.quantile(R, [0.025, 0.05, 0.975])
    return dict(flip=anyflip, est=float(r.mean()), theta=theta, A=anyflip and q05 > 0, rev=anyflip and q975 < 0,
                cover=bool(q025 <= theta <= q975), S1=anyflip and q025 >= 1, S2=anyflip and q025 >= 2,
                S3=anyflip and q025 >= 3, tie=bool((sim.tie_mask(BBL.mean(0)[None])[0]).sum() > 1))


def s2_one(args):
    ci, sc, rep, muC = args
    rng = np.random.default_rng([15002, ci, rep])
    out = sim.simulate(sc, rng, doses=(0.5, 1.0))
    Cm = sim.boot_counts(np.random.default_rng([15102, ci, rep]), sc.n, N_BOOT)
    res = {}
    for d in (0.5, 1.0):
        for ta in (False, True):
            res[(d, ta)] = decide(out["BBP"][d], out["BBL"][d], out["BC"], muC, Cm, ta)
    return ci, res


def run_s2(cells=None, schema="R15C0_S2_v1", tag=15002):
    cells = cells or s2_cells()
    scs, extra = [], []
    for c in cells:
        sc, rh, sh = s2_scenario(c)
        _, _, muC = sim.population_scores(sc, 0.5, n_mc=200_000)
        scs.append((sc, muC))
        pcl05, pl05, _ = sim.population_scores(sc, 0.5, n_mc=200_000, seed=1)
        _, pl10, _ = sim.population_scores(sc, 1.0, n_mc=200_000, seed=1)
        extra.append(dict(r_challenger=rh, s_challenger=sh, pop_clean_B=pcl05.round(3).tolist(),
                          pop_contam_B_d05=pl05.round(3).tolist(), pop_contam_B_d10=pl10.round(3).tolist(),
                          pop_C=muC.round(3).tolist()))
    jobs = [(ci, sc, k, muC) for ci, (sc, muC) in enumerate(scs) for k in range(N_REP)]
    res = pmap(s2_one, jobs)
    by = {}
    for ci, r in res:
        by.setdefault(ci, []).append(r)
    out = {"schema": schema, "synthetic_only": True, "roster": S2_ROSTER, "n_rep": N_REP, "n_boot": N_BOOT,
           "r_leader": R_LEADER, "designs": DESIGNS, "cells": []}
    for ci, c in enumerate(cells):
        row = dict(c, **extra[ci])
        for (d, ta) in ((0.5, False), (0.5, True), (1.0, False), (1.0, True)):
            rr = [r[(d, ta)] for r in by[ci]]
            key = f"d{d}_{'tieaware' if ta else 'roster'}"
            row[key] = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
        out["cells"].append(row)
    return out


# ------------------------------------------------------------------------------------------------- S4
def s4_cells():
    cells = []
    for mech, design, n in itertools.product(("memorisation_diff", "spillover_diff", "equal_memorisation"),
                                             ("dreyer_within_day_40", "yang_cross_day_200_sw025"), (51, 83)):
        cells.append(dict(mech=mech, design=design, n=n, Delta=2))
    return cells


def s4_scenario(c):
    base = s2_base(c["design"], c["n"], c["Delta"])
    if c["mech"] == "memorisation_diff":
        return sim.with_(base, r=(0.4, 1.0, 0.6), s=(0.0, 0.0, 0.0))
    if c["mech"] == "spillover_diff":
        return sim.with_(base, r=(0.6, 0.6, 0.6), s=(0.0, 0.6, 0.0))
    return sim.with_(base, r=(0.9, 0.9, 0.6), s=(0.0, 0.0, 0.0))


S4_DOSES = (0.25, 0.5, 0.75, 1.0)


def s4_one(args):
    ci, sc, rep, muC = args
    rng = np.random.default_rng([15004, ci, rep])
    out = sim.simulate(sc, rng, doses=S4_DOSES)
    Cm = sim.boot_counts(np.random.default_rng([15104, ci, rep]), sc.n, N_BOOT)
    return ci, {d: decide(out["BBP"][d], out["BBL"][d], out["BC"], muC, Cm, True) for d in S4_DOSES}


def run_s4():
    cells = s4_cells()
    scs = []
    for c in cells:
        sc = s4_scenario(c)
        _, _, muC = sim.population_scores(sc, 0.5, n_mc=200_000)
        scs.append((sc, muC))
    jobs = [(ci, sc, k, muC) for ci, (sc, muC) in enumerate(scs) for k in range(N_REP)]
    res = pmap(s4_one, jobs)
    out = {"schema": "R15C0_S4_v1", "synthetic_only": True, "n_rep": N_REP, "n_boot": N_BOOT, "rule": "A-tie",
           "cells": []}
    for ci, c in enumerate(cells):
        sc, muC = scs[ci]
        row = dict(c)
        for d in S4_DOSES:
            pc, pl, _ = sim.population_scores(sc, d, n_mc=200_000, seed=2)
            rr = [r[d] for i, r in res if i == ci]
            row[f"d{d}"] = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
            row[f"d{d}"]["pop_flip_margin"] = float(pl[1] - pl[0])
        out["cells"].append(row)
    return out


# ------------------------------------------------------------------------------------------------- S3
S3_TB = (40, 200)


def s3_cells():
    cells = []
    for gap, r0, dr, kap, n, TB in itertools.product((0, 7, 14), (0.5, 0.8, 0.95, 1.0), (-0.10, 0.0, 0.10), (0.0, 0.5),
                                                     (51, 83), S3_TB):
        if r0 >= 0.95 and dr > 0:
            continue
        cells.append(dict(gap=gap, r0=r0, dr=dr, kappa=kap, n=n, TB=TB))
    return cells


def s3_scenario(c):
    rp = min(1.0, c["r0"] + c["dr"])
    return sim.Scenario("S3", ("pre", "rand"), accB=(68.0, 68.0 - c["gap"]), r=(rp, c["r0"]),
                        kappa=(c["kappa"], c["kappa"]), n=c["n"], TB=c["TB"], TC=40, sw=0.0)


def s3_truth(sc, n_mc=400_000, seed=9):
    """Population fix-rate per model over its own errors, and on common support (both controls err)."""
    rng = np.random.default_rng(seed)
    etaB, _ = sim.intercepts(sc)
    r, kap = sc.vec("r", 0.0), sc.vec("kappa", 0.0)
    u = rng.normal(0, sc.su, (n_mc, 1))
    v = rng.normal(0, sc.sv, (n_mc, 2))
    d = rng.normal(0, sc.sd, (n_mc, 1))
    p = sim.expit(etaB + u + v + d + rng.normal(0, sc.se, (n_mc, 2)))
    rr = sim.rate(r, 0.0, kap * d)
    q = 1 - p
    phi = (q * rr).sum(0) / q.sum(0)
    w = q[:, 0] * q[:, 1]
    phi_cs = (w[:, None] * rr).sum(0) / w.sum()
    theta_ba = 100 * (q * rr).mean(0)
    return dict(phi=phi.tolist(), phi_cs=phi_cs.tolist(), theta_ba=theta_ba.tolist())


def s3_one(args):
    ci, sc, rep = args
    rng = np.random.default_rng([15003, ci, rep])
    tr = sim.simulate(sc, rng, doses=(0.5,), want_trials=True)["trials"]
    own, sib = tr["own"], tr["sib"]                                  # (n, 2, K, TB)
    n = sc.n
    # per-participant sums for ratio estimators
    o_m, s_m = own.mean(2), sib.mean(2)                               # (n, 2, TB) seed means
    ba = 100 * (o_m - s_m).mean(2)                                    # (n, 2) BA-scale own-copy effect
    num_naive, den_naive = (o_m - s_m).sum(2), (1 - s_m).sum(2)       # (n, 2)
    sup = (sib[:, :, :2].mean(2) < 0.5).all(1)                        # (n, TB): both models' seeds 1-2 err
    o3, s3 = own[:, :, 2:].mean(2), sib[:, :, 2:].mean(2)
    num_cs = ((o3 - s3) * sup[:, None]).sum(2)
    den_cs = ((1 - s3) * sup[:, None]).sum(2)
    Cm = sim.boot_counts(np.random.default_rng([15103, ci, rep]), n, N_BOOT_S3)

    def ratio_diff(num, den):
        est = num.sum(0) / np.maximum(den.sum(0), 1e-9)
        bn, bd = Cm @ num, Cm @ den
        b = bn / np.maximum(bd, 1e-9)
        return est, b

    e_n, b_n = ratio_diff(num_naive, den_naive)
    e_c, b_c = ratio_diff(num_cs, den_cs)
    d_ba = ba[:, 0] - ba[:, 1]
    b_ba = (Cm @ d_ba) / n
    res = dict(ba_pre=float(ba[:, 0].mean()), ba_rand=float(ba[:, 1].mean()), d_ba=float(d_ba.mean()),
               d_ba_ci90=np.quantile(b_ba, [0.05, 0.95]).tolist(),
               phi_naive=e_n.tolist(), d_naive=float(e_n[0] - e_n[1]),
               d_naive_ci90=np.quantile(b_n[:, 0] - b_n[:, 1], [0.05, 0.95]).tolist(),
               phi_cs=e_c.tolist(), d_cs=float(e_c[0] - e_c[1]),
               d_cs_ci90=np.quantile(b_c[:, 0] - b_c[:, 1], [0.05, 0.95]).tolist(),
               support_trials_per_participant=float(sup.sum(1).mean()))
    return ci, res


def run_s3():
    cells = s3_cells()
    scs = [s3_scenario(c) for c in cells]
    truths = [s3_truth(sc) for sc in scs]
    jobs = [(ci, sc, k) for ci, sc in enumerate(scs) for k in range(N_REP)]
    res = pmap(s3_one, jobs)
    by = {}
    for ci, r in res:
        by.setdefault(ci, []).append(r)
    margins_phi, margins_ba = (0.05, 0.067, 0.10), (2.0, 5.0)
    out = {"schema": "R15C0_S3_v1", "synthetic_only": True, "n_rep": N_REP, "n_boot": N_BOOT_S3,
           "margins_phi": margins_phi, "margins_ba": margins_ba, "cells": []}
    for ci, c in enumerate(cells):
        rr, t = by[ci], truths[ci]
        row = dict(c, truth=t)
        td_phi, td_cs, td_ba = t["phi"][0] - t["phi"][1], t["phi_cs"][0] - t["phi_cs"][1], t["theta_ba"][0] - t["theta_ba"][1]
        row["truth_diff"] = dict(phi=td_phi, phi_cs=td_cs, ba=td_ba)
        for key, truth in (("naive", td_phi), ("cs", td_cs)):
            ests = np.array([x[f"d_{key}"] for x in rr])
            cis = np.array([x[f"d_{key}_ci90"] for x in rr])
            row[key] = dict(mean_est=float(ests.mean()), bias_vs_own_truth=float(ests.mean() - truth),
                            bias_vs_zero=float(ests.mean()), sd=float(ests.std()),
                            cover90=float(np.mean((cis[:, 0] <= truth) & (truth <= cis[:, 1]))),
                            tost={str(m): float(np.mean((cis[:, 0] > -m) & (cis[:, 1] < m))) for m in margins_phi},
                            ceiling_both_ge_095=float(np.mean([min(x[f"phi_{key}"]) >= 0.95 for x in rr])))
        e = np.array([x["d_ba"] for x in rr])
        cis = np.array([x["d_ba_ci90"] for x in rr])
        row["ba"] = dict(mean_est=float(e.mean()), truth=td_ba, sd=float(e.std()),
                         tost={str(m): float(np.mean((cis[:, 0] > -m) & (cis[:, 1] < m))) for m in margins_ba})
        row["support_trials_per_participant"] = float(np.mean([x["support_trials_per_participant"] for x in rr]))
        out["cells"].append(row)
    return out


def group_key(study, c):
    """Cells are stored grouped (each list <= 200 entries, the repository privacy guard on long JSON lists)."""
    if study == "S2":
        return f"{c['family']}|{c['design']}"
    if study in ("S3", "S3b"):
        return f"n{c['n']}|TB{c['TB']}"
    return "all"


def grouped(study, out):
    g = {}
    for c in out.pop("cells"):
        g.setdefault(group_key(study, c), []).append(c)
    out["cells_by_group"] = g
    return out


def main():
    ap = argparse.ArgumentParser(description="Run the post hoc synthetic selection study.")
    ap.add_argument("study", choices=("S1", "S2", "S3", "S3b", "S4", "all"))
    ap.add_argument("--quick", action="store_true", help="reduced synthetic test only")
    which = ap.parse_args().study
    AGG.mkdir(parents=True, exist_ok=True)
    if which == "S3b":
        # DISCLOSED EXTENSION (added after the pre-specified runs started; see the plan's amendment note): the S3 grid
        # at 120 selection-run trials per participant, the trial count of the proposed three-day within-day design.
        global S3_TB
        S3_TB = (120,)
        which = "S3"
        out = run_s3()
        out["extension"] = "S3b: TB = 120 only; disclosed amendment, criteria unchanged"
        out["quick"] = QUICK
        out = grouped("S3b", out)
        (AGG / f"R15C0_S3b{'_quick' if QUICK else ''}.json").write_text(json.dumps(out, indent=1) + "\n")
        print("S3b done")
        return
    studies = {"S1": run_s1, "S2": run_s2, "S3": run_s3, "S4": run_s4}
    todo = list(studies) if which == "all" else [which]
    for s in todo:
        t0 = time.time()
        out = studies[s]()
        out["seconds"] = round(time.time() - t0, 1)
        out["quick"] = QUICK
        out = grouped(s, out)
        name = f"R15C0_{s}{'_quick' if QUICK else ''}.json"
        (AGG / name).write_text(json.dumps(out, indent=1) + "\n")
        print(s, "done", out["seconds"], "s ->", name, flush=True)


if __name__ == "__main__":
    main()
