"""R9-C2: the R9-C cross-fitted split-half decomposition for the frozen R4 TS+LR decoder (CPU, deterministic).

Same R4 HS-Stroke S2 arms, the same fold seeds (42-44) and the IDENTICAL halves H1 / H2 as R9-C (split_bench.split_halves
with the same outcome-blind RNG), so the two decoders are compared on the same exposed trials and the same split arms:
  L1 = LEAKED with H2's copies replaced by PLACEBO's rows at the same positions (exposes H1 only); L2 the mirror image.
Every arm (L, L1, L2, P) is fitted with the frozen, vendored R4 TSLRw exactly as R4 / R7-C7 fit it (multiplicity-weighted
Karcher reference recomputed per arm, tangent map, sklearn lbfgs L2-LR; R4's single-class rule), through R7-C7's own fit
function c7._r4, so L and P must reproduce R7-C7's stored S2 arrays for seeds 42-44 bit for bit (a fail-closed check).
Outputs per scored trial: correct-class probability p_* and correctness c_* for each arm.
"""

from __future__ import annotations

import os
import resource
import time

import numpy as np

from experiments.pmis_r7 import c7
from experiments.pmis_r8 import bench
from experiments.pmis_r9 import split_bench

ARMS = split_bench.ARMS  # ("L", "L1", "L2", "P")
G = c7.G  # shared globals: covs, label, rows, table, dist, exposed (seed -> mask)


def fit_arm(tr, T):
    """Frozen R4 TSLRw fit of one training multiset (distinct-trial indices with repeats), scored on T."""
    pred, pc, single = c7._r4(np.asarray(tr, dtype=np.int64), T)
    return pc, (pred == G["label"][T]).astype(float), bool(single)


def run_unit(task):
    """task = (seed, unit, ((fold, aa_flag), ...)). Returns per-fold arrays for every arm."""
    seed, unit, folds = task
    t0 = time.perf_counter()
    out, n_single = [], 0
    for f, aa in folds:
        arms = bench.fold_arms(G["table"], G["rows"], 2, seed, unit, f)
        h1, h2 = split_bench.split_halves(arms, G["dist"], (seed, unit, f))
        L1, L2 = split_bench.make_split_arms(arms, h1, h2)
        split_bench.check_split(arms, L1, L2, h1, h2, G["exposed"][seed])
        T = arms["T"]
        res = {"fold": int(f), "T": T, "H1": h1, "H2": h2}
        for name, tr in zip(ARMS, (arms["L"], L1, L2, arms["P"])):
            pc, cc, single = fit_arm(tr, T)
            res[f"p_{name}"], res[f"c_{name}"] = pc, cc
            n_single += int(single)
        if aa:  # A/A: an independent second fit of LEAKED must be exactly equal
            pc2, _, _ = fit_arm(arms["L"].copy(), T)
            res["aa_max_abs"] = float(np.max(np.abs(pc2 - res["p_L"])))
        for k, v in res.items():
            if k.startswith(("p_", "c_")) and not np.all(np.isfinite(v)):
                raise FloatingPointError(f"non-finite {k} (seed {seed}, unit {unit}, fold {f})")
        out.append(res)
    return {"seed": int(seed), "unit": int(unit), "folds": out, "single_class_fits": n_single,
            "seconds": time.perf_counter() - t0, "pid": os.getpid(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def unit_tasks(task_list):
    """Group the R9-C fold list (seed, unit, fold) by (seed, unit); A/A on every 10th fold in R9-C's global order."""
    groups: dict = {}
    for i, (s, u, f) in enumerate(task_list):
        groups.setdefault((s, u), []).append((int(f), i % 10 == 0))
    return [(s, u, tuple(v)) for (s, u), v in sorted(groups.items())]


def assemble(results, n_distinct, seeds):
    """Trial arrays per seed over distinct trials (NaN where unscored) and the half label (1 = H1, 2 = H2, 0 otherwise)."""
    A = {s: {f"{q}_{a}": np.full(n_distinct, np.nan) for q in ("p", "c") for a in ARMS} for s in seeds}
    half = {s: np.zeros(n_distinct, dtype=np.int64) for s in seeds}
    aa = []
    for r in results:
        s = r["seed"]
        for fr in r["folds"]:
            T = fr["T"]
            for key in A[s]:
                if not np.all(np.isnan(A[s][key][T])):
                    raise AssertionError("trial scored twice")
                A[s][key][T] = fr[key]
            half[s][fr["H1"]] = 1
            half[s][fr["H2"]] = 2
            if "aa_max_abs" in fr:
                aa.append(fr["aa_max_abs"])
    return A, half, aa
