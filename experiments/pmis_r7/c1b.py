"""R7-C1b: complete independent audit of every G3 seed-42 arm with a convergent Karcher solver (all 14 participants).

Every G3 seed-42 arm (primary A/B for all 8,080 pairs; multiset A/B for all pairs; C1 neighbour G for 2,210 pairs;
C2 placebo C/E for 4,890 pairs) is refitted with pmis_r7.karcher_ls.IndepTSLR2 (line-search Karcher reference +
damped-Newton exact L2-logistic; no R4/sklearn code) and scored on the pair's two trials. G3's frozen plan is rebuilt
read-only (commitment-checked). The earlier defective audit (R6-C1 A3) is preserved unchanged.
"""

from __future__ import annotations

import os
import resource
import time

import numpy as np

from models.tslr import arm_extra_copy, arm_multiset

from models.karcher_ls import IndepTSLR2

G: dict = {}


def _fit(u, cnt, ab):
    m = IndepTSLR2().fit(G["covs"][u], G["y"][u], cnt)
    return m.p_correct(G["covs"][ab], G["y"][ab]), m.karcher["grad_norm"], m.karcher["iterations"], m.newton_iter


def run_task(task: dict) -> dict:
    t0 = time.perf_counter()
    base = task["base_canon"]
    ones = np.ones(base.size, dtype=np.int64)
    recs, jmax, kit, nit, n = [], 0.0, 0, 0, 0
    for pr in task["pairs"]:
        a, b = pr["a"], pr["b"]
        ab = np.array([a, b], dtype=np.int64)
        r = {"k": pr["k"]}
        arms = [("A", arm_multiset(base, ones, a)), ("B", arm_multiset(base, ones, b)),
                ("mA", arm_multiset(base, task["base_multi_cnt"], a)), ("mB", arm_multiset(base, task["base_multi_cnt"], b))]
        if pr["g"] >= 0:
            arms.append(("G", arm_multiset(base, ones, pr["g"])))
        if pr["c"] >= 0:
            arms += [("C", arm_extra_copy(base, ones, pr["c"])), ("E", arm_extra_copy(base, ones, pr["e"]))]
        for name, (u, cnt) in arms:
            p, jn, ki, ni = _fit(u, cnt, ab)
            r[f"{name}_pa"], r[f"{name}_pb"], r[f"{name}_J"] = float(p[0]), float(p[1]), jn
            jmax, kit, nit, n = max(jmax, jn), max(kit, ki), max(nit, ni), n + 1
        recs.append(r)
    return {"subject": task["subject"], "fold": task["fold"], "records": recs, "fits": n, "grad_norm_max": jmax,
            "karcher_iter_max": kit, "newton_iter_max": nit, "seconds": time.perf_counter() - t0, "pid": os.getpid(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def to_g3_keys(r: dict) -> dict:
    """Map independent arm scores to the G3 pair-score key names (pAA = p(a | C + a'), ...)."""
    out = {"pAA": r["A_pa"], "pBA": r["A_pb"], "pAB": r["B_pa"], "pBB": r["B_pb"],
           "mAA": r["mA_pa"], "mBA": r["mA_pb"], "mAB": r["mB_pa"], "mBB": r["mB_pb"]}
    for q in ("AA", "AB", "BA", "BB"):
        out[f"c{q}"] = float(out[f"p{q}"] > 0.5)
    if "G_pa" in r:
        out.update({"pAG": r["G_pa"], "pBG": r["G_pb"]})
    if "C_pa" in r:
        out.update({"pAC": r["C_pa"], "pBC": r["C_pb"], "pAE": r["E_pa"], "pBE": r["E_pb"]})
    return out
