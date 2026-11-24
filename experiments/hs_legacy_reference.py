#!/usr/bin/env python3
"""R4-G3a: frozen TS+LR evaluation of the G2 manifest (protocol: aggregate_results/G3_PROTOCOL.json).

For every split (S1, S2) x seed (42..46) x unit x fold and arm a in {L, C, P, LCB}: fit TSLRw on the
arm's training multiset (each copy is a row; tangent reference = multiplicity-weighted Riemannian mean),
predict the shared scored set T_f. LEAKED additionally predicts every test-fold instance for the
descriptive "as-reported" (instance-weighted) score. --dry-run fits one fold of each split for timing
and prints NO accuracy.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from data_preparation import hs_metadata
from experiments.pmis_g2.g3 import preflight_covariances
from experiments.runtime import RUN_ROOT
from models.vendor_r4.r4_models import TSLRw
from models.vendor_r4 import r4_stats as rs

ARMS = ("T", "L", "C", "P", "LCB", "TEST_INST")
FIT_ARMS = ("L", "C", "P", "LCB")
SPLIT_NAMES = {1: "S1", 2: "S2"}
G = {}


def _init(priv):
    G["covs"] = np.load(Path(priv) / "G2_covs_8_30.npy")
    G["label"] = np.load(Path(priv) / "G2_distinct.npz")["label"]
    man = np.load(Path(priv) / "G2_manifest.npz")
    G["rows"], G["table"] = man["rows"], man["table"]


def _get(sel, arm):
    r = sel[sel[:, 4] == ARMS.index(arm)][0]
    return G["rows"][r[5]:r[6]]


def run_task(task):
    split, seed, unit = task
    t = G["table"]
    sel_all = t[(t[:, 0] == split) & (t[:, 1] == seed) & (t[:, 2] == unit)]
    out = []
    for f in np.unique(sel_all[:, 3]):
        sel = sel_all[sel_all[:, 3] == f]
        T = _get(sel, "T")
        res = {"fold": int(f), "T": T}
        for arm in FIT_ARMS:
            tr = _get(sel, arm)
            u, cnt = np.unique(tr, return_counts=True)
            classes = np.unique(G["label"][u])
            if classes.size < 2:  # pre-specified degenerate rule: predict the only training class; counted
                res["single_class_fits"] = res.get("single_class_fits", 0) + 1
                predict = lambda c, k=int(classes[0]): np.full(len(c), k, dtype=np.int8)  # noqa: E731
            else:
                predict = TSLRw().fit(G["covs"][u], G["label"][u], cnt).predict
            res[arm] = np.asarray(predict(G["covs"][T])).astype(np.int8)
            if arm == "L":
                ti = _get(sel, "TEST_INST")
                ut, inv = np.unique(ti, return_inverse=True)
                res["L_inst_true"] = G["label"][ti].astype(np.int8)
                res["L_inst_pred"] = np.asarray(predict(G["covs"][ut])).astype(np.int8)[inv]
        out.append(res)
    return split, seed, unit, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    priv = hs_metadata.local_paths(RUN_ROOT)["r4_private"]
    if not hs_metadata.frozen_check(priv)["all_match"]:
        raise ValueError("HS-Stroke frozen metadata differ")
    _, dist, _, _, _ = hs_metadata.load_metadata(priv)
    from data_preparation.hs_integrity import historical_sha256

    preflight_covariances(priv / "G2_covs_8_30.npy", len(dist["label"]),
                          historical_sha256("G2_covs_8_30.npy"))
    _init(priv)
    dist = np.load(Path(priv) / "G2_distinct.npz")
    subj = dist["subject"]
    man = np.load(Path(priv) / "G2_manifest.npz")
    tasks = sorted({(int(r[0]), int(r[1]), int(r[2])) for r in G["table"]})
    if a.dry_run:
        picks = [next(x for x in tasks if x[0] == 1), next(x for x in tasks if x[0] == 2)]
        for tk in picks:
            t0 = time.time()
            run_task(tk)
            n = sum(1 for x in tasks if x[0] == tk[0])
            print(f"dry-run split {SPLIT_NAMES[tk[0]]}: one unit (5 folds x 4 arms) {time.time() - t0:.1f} s; "
                  f"{n} units -> projected {n * (time.time() - t0) / 60 / a.workers:.1f} min on {a.workers} workers")
        return
    pred = {}
    inst = {}
    single_class = 0
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(priv,)) as pool:
        for k, (split, seed, unit, out) in enumerate(pool.imap_unordered(run_task, tasks, chunksize=4)):
            for res in out:
                single_class += res.get("single_class_fits", 0)
                for arm in FIT_ARMS:
                    key = (split, seed, arm)
                    if key not in pred:
                        pred[key] = np.full(dist["label"].size, -1, dtype=np.int8)
                    if (pred[key][res["T"]] != -1).any():
                        raise AssertionError("trial scored twice")
                    pred[key][res["T"]] = res[arm]
                inst.setdefault((split, seed), []).append((res["L_inst_true"], res["L_inst_pred"],
                                                           np.full(res["L_inst_true"].size, unit)))
            if k % 100 == 0:
                print(f"  {k + 1}/{len(tasks)} units, {time.time() - t0:.0f} s", flush=True)
    for key, v in pred.items():
        if (v == -1).any():
            raise AssertionError(f"unscored trials in {key}")
    np.savez(Path(priv) / "G3_tslr_predictions_PRIVATE.npz",
             **{f"{SPLIT_NAMES[s]}_{seed}_{arm}": v for (s, seed, arm), v in sorted(pred.items())})
    y = dist["label"]
    participants = np.unique(subj)
    seeds = sorted({k[1] for k in pred})
    agg = {"stage": "R4-G3-TSLR", "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "model": "TS+LR (Paper A M3; OAS cov 8-30 Hz MI window; weighted Riemannian reference; LR C=1)",
           "n_participants": int(participants.size), "seeds": seeds, "runtime_s": round(time.time() - t0, 1),
           "degenerate_single_class_fits": int(single_class),
           "splits": {}}

    def ba_per_participant(split, seed, arm, mask=None):
        out = []
        for p in participants:
            m = subj == p if mask is None else (subj == p) & mask
            out.append(100.0 * rs.balanced_accuracy(y[m], pred[(split, seed, arm)][m]) if m.sum() and len(np.unique(y[m])) == 2 else np.nan)
        return np.array(out)

    primary_p = {}
    for split in (1, 2):
        name = SPLIT_NAMES[split]
        ba = {arm: np.mean([ba_per_participant(split, s, arm) for s in seeds], axis=0) for arm in FIT_ARMS}
        ba42 = {arm: ba_per_participant(split, 42, arm) for arm in FIT_ARMS}
        exp_ba = {}
        for subset in ("exposed", "unexposed"):
            vals = {arm: [] for arm in ("L", "P", "C")}
            for s in seeds:
                e = man[f"exposed_{name}_{s}"]
                m = e if subset == "exposed" else ~e
                for arm in vals:
                    vals[arm].append(ba_per_participant(split, s, arm, m))
            exp_ba[subset] = {arm: np.nanmean(v, axis=0) for arm, v in vals.items()}
        # as-reported: LEAKED scored on every test-fold instance (duplicate-weighted) vs CLEAN deduplicated
        asrep = []
        for s in seeds:
            tr, pr, un = (np.concatenate(z) for z in zip(*inst[(split, s)]))
            # map unit -> participant
            if split == 1:
                ses_subj = {int(se): int(sb) for se, sb in zip(dist["session"], subj)}
                part = np.array([ses_subj[int(u)] for u in un])
            else:
                part = un
            asrep.append([100.0 * rs.balanced_accuracy(tr[part == p], pr[part == p]) for p in participants])
        asrep = np.mean(asrep, axis=0)
        d_primary = ba["L"] - ba["P"]
        primary_p[name] = rs.sign_flip_p(d_primary)
        agg["splits"][name] = {
            "absolute_BA_mean_over_participants": {arm: round(float(np.mean(v)), 3) for arm, v in ba.items()},
            "primary_LEAKED_minus_PLACEBO": rs.summarise(d_primary),
            "PLACEBO_minus_CLEAN": rs.summarise(ba["P"] - ba["C"]),
            "LEAKED_minus_CLEAN": rs.summarise(ba["L"] - ba["C"]),
            "learning_curve_CLEAN_minus_LCB": rs.summarise(ba["C"] - ba["LCB"]),
            "exposed_only_LEAKED_minus_PLACEBO": rs.summarise(exp_ba["exposed"]["L"] - exp_ba["exposed"]["P"]),
            "exposed_only_PLACEBO_minus_CLEAN": rs.summarise(exp_ba["exposed"]["P"] - exp_ba["exposed"]["C"]),
            "unexposed_only_LEAKED_minus_PLACEBO": rs.summarise(exp_ba["unexposed"]["L"] - exp_ba["unexposed"]["P"]),
            "producer_seed42_only_LEAKED_minus_PLACEBO": rs.summarise(ba42["L"] - ba42["P"]),
            "as_reported_LEAKED_dup_weighted_minus_CLEAN_dedup_DIFFERENT_TEST_SETS": rs.summarise(asrep - ba["C"]),
            "seed_range_primary_mean_pp": [round(float(x), 3) for x in sorted(
                np.mean(ba_per_participant(split, s, "L") - ba_per_participant(split, s, "P")) for s in seeds)],
        }
    adj = rs.holm(primary_p)
    for name in adj:
        agg["splits"][name]["primary_LEAKED_minus_PLACEBO"]["holm_p_over_S1_S2"] = adj[name]
    (priv / "G3_tslr_results.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps({n: {k: (v["mean_pp"], v["ci95_pp"], v["n_positive"], v["label"]) for k, v in d.items()
                          if isinstance(v, dict) and "mean_pp" in v} for n, d in agg["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
