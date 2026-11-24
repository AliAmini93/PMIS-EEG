"""R5-G3 frozen PMIS execution: exact per-arm TS+LR on the frozen R4 OAS covariances (CPU only).

Phases (all in this module; the CLI is scripts/run_g3.py):
  plan       metadata only: SHA-verified R4 metadata, S2 replay, pairs (seed 42: the R5 G1 table;
             seeds 43-46: the R5 G1 rule re-run read-only), every pair re-validated by the G2 contract,
             outcome-blind control selection, fold tasks, and a commitment digest. No feature is read.
  preflight  the covariance file's SHA-256 is verified BEFORE it is loaded; shape/finite/symmetric/SPD.
  execute    every arm is fitted from scratch with the vendored R4 TSLRw (fresh multiplicity-weighted
             Karcher reference + fresh LogisticRegression(C=1.0, max_iter=2000) per arm); P(correct
             class) and hard labels are predicted. Any fit failure, non-finite output or identical
             reference between paired arms raises and STOPS the run (fail closed).
  analyse    frozen estimands, participant-level inference, Holm family, descriptive controls.
Participant- and trial-level scores stay in git-ignored data/; only cohort aggregates are public.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import resource
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_preparation import hs_pairs
from experiments.runtime import port_identity, verify_recorded_input

from data_preparation import hs_metadata as ad
from data_preparation import hs_integrity as pv
from evaluation.g2_estimands import correct_class_probability, exact_two_sided_sign_flip, holm, participant_class_balanced, sign_flip_floor, t_interval
from models.tslr import arm_extra_copy, arm_multiset, fit_arm
from models.vendor_r4 import r4_models as rm

PROTOCOL_ID = "R5-G3-PMIS-TSLR-v1"
PROTOCOL_JSON = "G3_PROTOCOL_FROZEN.json"
PRIMARY_SEED = 42
SENSITIVITY_SEEDS = (43, 44, 45, 46)
POSITION_CUT = 6
REQUIRED_LABELS = (1, 2)
COVS_FILE = "G2_covs_8_30.npy"
N_CHANNELS = 11
MAX_WORKERS = 4
NICE = 10
START_MIN_AVAILABLE_MB = 4096
RUN_MIN_AVAILABLE_MB = 2048
START_MAX_PSI_FULL_AVG10 = 1.0
RUN_MAX_PSI_FULL_AVG10 = 5.0
SECONDARY_FAMILY = (
    "S1_balanced_accuracy_contrast",
    "S2_C1_same_run_minus_different_run_neighbour",
    "S3_C1_identity_minus_neighbour_contrast",
    "S4_C2_placebo_nontest_swap",
    "S5_C4_multiset_base_identity_swap",
    "S6_position_near_le6",
    "S7_position_far_gt6",
)
# files whose SHA-256 the frozen protocol pins (the executable and every in-repo dependency)


class G3Stop(RuntimeError):
    """Fail-closed stop. `status` is BLOCKED (assertion/provenance) or STOPPED_RESOURCE."""

    def __init__(self, status: str, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message

    def __reduce__(self):  # survives pickling from pool workers
        return (G3Stop, (self.status, self.message))


def _stop(ok: bool, message: str, status: str = "BLOCKED") -> None:
    if not ok:
        raise G3Stop(status, message)


# ------------------------------------------------------------------------------------------ provenance


def environment() -> dict:
    import scipy
    import sklearn
    import threadpoolctl
    from threadpoolctl import threadpool_info
    return {"interpreter": sys.executable, "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__,
            "thread_env": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "MKL_NUM_THREADS")},
            "blas_openmp_threads": sorted({t["num_threads"] for t in threadpool_info()})}




def verify_frozen(root: Path, protocol: dict) -> dict:
    _stop(protocol.get("status") == "FROZEN" and protocol.get("protocol_id") == PROTOCOL_ID, "protocol not frozen")
    env = environment()
    for k in ("python", "numpy", "scipy", "sklearn"):
        _stop(env[k] == protocol["environment"][k], f"{k} version differs from the frozen protocol")
    _stop(env["blas_openmp_threads"] == [1], "BLAS/OpenMP not single-threaded")
    return {"port_manifest_sha256": port_identity(), "environment": env}


# ------------------------------------------------------------------------------------------- resources
def resource_state() -> dict:
    mem = {ln.split(":")[0]: int(ln.split()[1]) // 1024 for ln in open("/proc/meminfo")}
    psi = {}
    try:
        for ln in open("/proc/pressure/memory"):
            kind, *kv = ln.split()
            psi[kind] = float(dict(x.split("=") for x in kv)["avg10"])
    except OSError:
        psi = {"some": float("nan"), "full": float("nan")}
    return {"mem_available_mb": mem["MemAvailable"], "swap_free_mb": mem.get("SwapFree", 0),
            "psi_some_avg10": psi.get("some", float("nan")), "psi_full_avg10": psi.get("full", float("nan")),
            "loadavg_1": os.getloadavg()[0], "cpus": os.cpu_count()}


def start_guard(state: dict) -> int:
    """Pre-declared start rule; returns the number of workers (<= 4, never more than idle cores)."""
    _stop(state["mem_available_mb"] >= START_MIN_AVAILABLE_MB, "MemAvailable below 4096 MB", "STOPPED_RESOURCE")
    _stop(not (state["psi_full_avg10"] > START_MAX_PSI_FULL_AVG10), "memory pressure (PSI full avg10 > 1.0)",
          "STOPPED_RESOURCE")
    idle = int(state["cpus"] - np.ceil(state["loadavg_1"]))
    return int(max(1, min(MAX_WORKERS, idle)))


def run_guard(state: dict) -> None:
    _stop(state["mem_available_mb"] >= RUN_MIN_AVAILABLE_MB, "MemAvailable fell below 2048 MB", "STOPPED_RESOURCE")
    _stop(not (state["psi_full_avg10"] > RUN_MAX_PSI_FULL_AVG10), "memory pressure (PSI full avg10 > 5.0)",
          "STOPPED_RESOURCE")


# ------------------------------------------------------------------------------------------------ plan


def load_real_inputs(root: Path) -> dict:
    """Generate the pair tables from the verified local metadata, without opening feature arrays."""
    paths = ad.local_paths(root)
    priv = paths["r4_private"]
    _stop(ad.frozen_check(priv)["all_match"], "HS-Stroke metadata differ from frozen inputs")
    inst, dist, meta, man, _ = ad.load_metadata(priv)
    _stop(ad.integrity(inst, dist)["all_pass"], "metadata integrity failed")
    stems = ad.stems_of(inst, meta)
    pairs = {seed: hs_pairs.run_seed(inst, dist, stems, seed, man=man)[0]
             for seed in (PRIMARY_SEED,) + SENSITIVITY_SEEDS}
    return {"paths": paths, "inst": inst, "dist": dist, "meta": meta, "man": man, "pairs": pairs}


def make_plan(inst, dist, meta, man, pairs_by_seed: dict) -> dict:
    """Fold tasks for every seed; every pair re-validated; seed-42 controls selected outcome-blind."""
    hashes = dist["hash"].astype(str)
    stems = ad.stems_of(inst, meta)
    plan = {"seeds": {}, "checks": {}}
    for seed in sorted(pairs_by_seed):
        pairs = pairs_by_seed[seed]
        rep = ad.replay_seed(inst, dist, stems, man, seed)
        _stop(rep["T_equal"] == rep["C_equal"] == rep["n_folds"] and rep["whole_file"] and rep["exposed_equal_manifest"]
              and rep["scored"] == int(dist["label"].size), f"seed {seed}: fold replay differs from manifest")
        val = ad.validate_pairs(pairs, inst, dist, rep, hashes)
        _stop(val["contract_pass"]["canonical"] == val["contract_pass"]["multiset"] == len(pairs)
              and not val["contract_fail_counts"] and sum(val["field_mismatch_counts"].values()) == 0
              and val["trial_in_two_pairs_of_a_fold"] == 0, f"seed {seed}: pair contract failure")
        if seed == PRIMARY_SEED:
            _stop(not val["control_arm_fail_counts"], "control-arm contract failure")
        ctrl = {c["pair_index"]: c for c in val["controls"]}
        by_fold = collections.defaultdict(list)
        for k, p in enumerate(pairs):
            by_fold[(p["subject"], p["fold"])].append(k)
        tasks = []
        for (subj, f) in sorted(by_fold):
            v = rep["views"][(subj, f)]
            T = np.asarray(v.T, dtype=np.int64)
            base_c = np.sort(inst["distinct"][v.base_canon]).astype(np.int64)
            bm_ids, bm_cnt = np.unique(inst["distinct"][v.base_multi], return_counts=True)
            _stop(np.array_equal(bm_ids, base_c), "multiset base support differs from canonical base")
            entries = []
            for k in by_fold[(subj, f)]:
                p = pairs[k]
                e = {"k": k, "a": int(p["a"]), "b": int(p["b"]), "label": int(p["label"]),
                     "session": int(p["session"]), "pos_dist": int(p["pos_dist"]), "g": -1, "c": -1, "e": -1, "sign": 0}
                if seed == PRIMARY_SEED:
                    c = ctrl[k]
                    e.update({"g": int(c.get("neighbour_g", -1)), "c": int(c.get("placebo_c", -1)),
                              "e": int(c.get("placebo_e", -1)), "sign": int(c.get("placebo_sign", 0))})
                    if e["c"] >= 0:
                        _stop(e["c"] in set(base_c.tolist()) and e["e"] in set(base_c.tolist()),
                              "placebo trial not a base identity")
                    if e["g"] >= 0:
                        _stop(e["g"] in set(T.tolist()), "neighbour not scored")
                entries.append(e)
            tasks.append({"seed": seed, "subject": int(subj), "fold": int(f), "T": T,
                          "T_session": dist["session"][T].astype(np.int64), "T_label": dist["label"][T].astype(np.int64),
                          "unexposed": np.array([not v.alias.get(int(t)) for t in T], dtype=bool),
                          "base_canon": base_c, "base_multi_cnt": bm_cnt.astype(np.int64),
                          "controls": seed == PRIMARY_SEED, "pairs": entries})
        plan["seeds"][seed] = {"tasks": tasks, "n_pairs": len(pairs)}
        plan["checks"][str(seed)] = {"pairs": len(pairs), "contract_pass_both_bases": True,
                                     "manifest_cells_equal": rep["n_folds"],
                                     "control_arm_pass_counts": val["control_arm_pass_counts"] if seed == PRIMARY_SEED else None}
    plan["commitment"] = plan_commitment(plan)
    plan["fit_counts"] = fit_counts(plan)
    return plan


def plan_commitment(plan: dict) -> str:
    h = hashlib.sha256()
    for seed in sorted(plan["seeds"]):
        for t in plan["seeds"][seed]["tasks"]:
            h.update(json.dumps([seed, t["subject"], t["fold"], t["controls"]]).encode())
            for arr in (t["T"], t["base_canon"], t["base_multi_cnt"], t["unexposed"].astype(np.int8)):
                h.update(np.ascontiguousarray(arr).tobytes())
            h.update(json.dumps(t["pairs"], sort_keys=True).encode())
    return h.hexdigest()


def fit_counts(plan: dict) -> dict:
    out = {}
    for seed, s in plan["seeds"].items():
        pairs = [p for t in s["tasks"] for p in t["pairs"]]
        c = {"primary_canonical": 2 * len(pairs)}
        if seed == PRIMARY_SEED:
            c["C4_multiset"] = 2 * len(pairs)
            c["C1_neighbour"] = sum(p["g"] >= 0 for p in pairs)
            c["C2_placebo"] = 2 * sum(p["c"] >= 0 for p in pairs)
        out[str(seed)] = c
    out["total"] = int(sum(v for s in out.values() for v in s.values()))
    return out


# ------------------------------------------------------------------------------------------- preflight
def preflight_covariances(path: Path, n_distinct: int, expected_sha256: str | None) -> tuple[np.ndarray, dict]:
    """Check the optional historical SHA-256 before loading, then validate array structure."""
    _stop(path.name == COVS_FILE, "unexpected feature file")
    _stop(verify_recorded_input(path, expected_sha256), "covariance file SHA-256 differs from the R4-frozen value")
    covs = np.load(path, allow_pickle=False)
    return covs, check_covariances(covs, n_distinct) | {"sha256_verified_before_load": expected_sha256 is not None}


def check_covariances(covs: np.ndarray, n_distinct: int) -> dict:
    _stop(covs.dtype == np.float64 and covs.shape == (n_distinct, N_CHANNELS, N_CHANNELS), "covariance shape/dtype")
    _stop(bool(np.all(np.isfinite(covs))), "non-finite covariance")
    asym = float(np.max(np.abs(covs - np.swapaxes(covs, 1, 2))) / max(1e-300, float(np.max(np.abs(covs)))))
    _stop(asym < 1e-12, "covariances not symmetric")
    min_eig = float(np.linalg.eigvalsh(covs).min())
    _stop(min_eig > 0, "covariance not positive definite")
    return {"shape": list(covs.shape), "dtype": str(covs.dtype), "all_finite": True,
            "max_relative_asymmetry": asym, "all_positive_definite": True}


# --------------------------------------------------------------------------------------------- execute
G: dict = {}


@dataclass
class ArmResult:
    ref: np.ndarray
    p: np.ndarray
    correct: np.ndarray
    n_iter: int
    conv_warnings: int
    seconds: float


def fit_score(base_ids, base_mult, score_ids, add=None, extra_copy_of=None) -> ArmResult:
    """One exact arm: fresh reference + fresh classifier (vendored TSLRw), then P(correct) and labels."""
    from sklearn.exceptions import ConvergenceWarning
    covs, y = G["covs"], G["y"]
    t0 = time.perf_counter()
    if add is not None:
        u, cnt = arm_multiset(base_ids, base_mult, add)
    else:
        u, cnt = arm_extra_copy(base_ids, base_mult, extra_copy_of)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = fit_arm(covs[u], y[u], cnt)
    feats = rm.tangent_space_b(covs[score_ids], model.ref)
    proba = model.clf.predict_proba(feats)
    pred = model.clf.predict(feats)
    ys = y[score_ids]
    p = correct_class_probability(proba, model.clf.classes_, ys)
    _stop(bool(np.all(np.isfinite(p))) and bool(np.all(np.isfinite(model.ref))), "non-finite model output")
    n_conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return ArmResult(model.ref, p, pred == ys, int(model.clf.n_iter_[0]), n_conv, time.perf_counter() - t0)


def run_task(task: dict) -> dict:
    T = task["T"]
    pos = {int(t): i for i, t in enumerate(T.tolist())}
    ones = np.ones(task["base_canon"].size, dtype=np.int64)
    um = task["unexposed"]
    stats = {"fits": 0, "seconds": [], "n_iter_max": 0, "n_iter_at_cap": 0, "conv_warnings": 0, "ref_verified": 0}
    recs = []

    def note(r: ArmResult):
        stats["fits"] += 1
        stats["seconds"].append(r.seconds)
        stats["n_iter_max"] = max(stats["n_iter_max"], r.n_iter)
        stats["n_iter_at_cap"] += int(r.n_iter >= 2000)
        stats["conv_warnings"] += r.conv_warnings
        return r

    for j, pr in enumerate(task["pairs"]):
        a, b = pr["a"], pr["b"]
        A = note(fit_score(task["base_canon"], ones, T, add=a))
        B = note(fit_score(task["base_canon"], ones, T, add=b))
        _stop(not np.array_equal(A.ref, B.ref), "paired arms share an identical reference")
        if j == 0:  # independent recomputation of the arm reference (exactness audit)
            u, cnt = arm_multiset(task["base_canon"], ones, a)
            _stop(np.array_equal(rm.riemann_mean_w(G["covs"][u], cnt), A.ref), "arm reference not reproducible")
            stats["ref_verified"] += 1
        ia, ib = pos[a], pos[b]
        sc = um & (task["T_session"] == pr["session"]) & (task["T_label"] == pr["label"])
        rec = {"k": pr["k"], "label": pr["label"], "pos_dist": pr["pos_dist"],
               "pAA": A.p[ia], "pAB": B.p[ia], "pBA": A.p[ib], "pBB": B.p[ib],
               "cAA": bool(A.correct[ia]), "cAB": bool(B.correct[ia]),
               "cBA": bool(A.correct[ib]), "cBB": bool(B.correct[ib]),
               "off_fold": float(np.mean(np.abs(A.p[um] - B.p[um]))) if um.any() else np.nan,
               "off_session_class": float(np.mean(np.abs(A.p[sc] - B.p[sc]))) if sc.any() else np.nan,
               "n_off_fold": int(um.sum()), "n_off_session_class": int(sc.sum())}
        if task["controls"]:
            ab = np.array([a, b], dtype=np.int64)
            mA = note(fit_score(task["base_canon"], task["base_multi_cnt"], ab, add=a))
            mB = note(fit_score(task["base_canon"], task["base_multi_cnt"], ab, add=b))
            _stop(not np.array_equal(mA.ref, mB.ref), "multiset paired arms share an identical reference")
            rec.update({"mAA": mA.p[0], "mAB": mB.p[0], "mBA": mA.p[1], "mBB": mB.p[1]})
            if pr["g"] >= 0:
                Gm = note(fit_score(task["base_canon"], ones, ab, add=pr["g"]))
                rec.update({"pAG": Gm.p[0], "pBG": Gm.p[1]})
            if pr["c"] >= 0:
                Cm = note(fit_score(task["base_canon"], ones, ab, extra_copy_of=pr["c"]))
                Em = note(fit_score(task["base_canon"], ones, ab, extra_copy_of=pr["e"]))
                _stop(not np.array_equal(Cm.ref, Em.ref), "placebo arms share an identical reference")
                rec.update({"pAC": Cm.p[0], "pBC": Cm.p[1], "pAE": Em.p[0], "pBE": Em.p[1], "sign": pr["sign"]})
        recs.append(rec)
    return {"seed": task["seed"], "subject": task["subject"], "fold": task["fold"], "records": recs,
            "stats": stats, "pid": os.getpid(),
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}


def execute(plan: dict, covs: np.ndarray, y: np.ndarray, workers: int, log=print, monitor=True) -> dict:
    """Run every task; resource guard after each task; any exception stops the pool (fail closed)."""
    G["covs"], G["y"] = covs, y
    tasks = [t for seed in sorted(plan["seeds"]) for t in plan["seeds"][seed]["tasks"]]
    t0 = time.perf_counter()
    results, min_state = [], None
    if workers <= 1:
        it = map(run_task, tasks)
        pool = None
    else:
        from multiprocessing import get_context
        pool = get_context("fork").Pool(workers)
        it = pool.imap_unordered(run_task, tasks, chunksize=1)
    try:
        for i, res in enumerate(it):
            results.append(res)
            if monitor:
                st = resource_state()
                if min_state is None or st["mem_available_mb"] < min_state["mem_available_mb"]:
                    min_state = st
                run_guard(st)
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                log(f"tasks {i + 1}/{len(tasks)}  fits {sum(r['stats']['fits'] for r in results)}  "
                    f"{time.perf_counter() - t0:.0f} s")
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {"results": results, "wall_s": time.perf_counter() - t0, "min_resource_state": min_state,
            "workers": workers}


# --------------------------------------------------------------------------------------------- analyse
@dataclass(frozen=True)
class V:
    participant: str
    label: int
    value: float


def summarise(items: list[V], scale: float = 100.0, with_test: bool = True) -> dict:
    """Class-balanced participant effects; participants lacking a class are dropped and counted.
    with_test=False (descriptive quantities) omits the sign-flip p by the frozen reporting rule."""
    labs = collections.defaultdict(set)
    for it in items:
        labs[it.participant].add(it.label)
    keep = {p for p, s in labs.items() if s == set(REQUIRED_LABELS)}
    kept = [it for it in items if it.participant in keep]
    out = {"n_pairs": len(kept), "n_participants_dropped_missing_class": len(set(labs) - keep)}
    if len(keep) < 2:
        return out | {"n_participants": len(keep), "inference": "not enabled"}
    eff = participant_class_balanced(kept, REQUIRED_LABELS, value=lambda s: s.value)
    vals = np.array(list(eff.values()), dtype=np.float64)
    lo, hi = t_interval(vals, 0.95)
    out.update({"n_participants": int(vals.size), "mean_points": float(vals.mean()) * scale,
                "ci95_points": [lo * scale, hi * scale], "ci95_excludes_zero": bool(lo > 0 or hi < 0),
                "sd_between_participants_points": float(vals.std(ddof=1)) * scale,
                "n_participants_positive": int((vals > 0).sum()), "n_participants_negative": int((vals < 0).sum()),
                "inference_enabled": bool(vals.size >= 10)})
    if with_test:
        out.update({"exact_sign_flip_p": exact_two_sided_sign_flip(vals),
                    "sign_flip_floor": sign_flip_floor(int(vals.size))})
    return out


def _items(recs, fn, cond=lambda r: True):
    return [V(r["participant"], r["label"], float(fn(r))) for r in recs if cond(r)]


def d_of(r):
    return 0.5 * ((r["pAA"] - r["pAB"]) + (r["pBB"] - r["pBA"]))


def analyse(results: list[dict]) -> tuple[dict, dict]:
    by_seed = collections.defaultdict(list)
    for res in results:
        for r in res["records"]:
            by_seed[res["seed"]].append(dict(r, participant=str(res["subject"]), fold=res["fold"]))
    for s in by_seed:
        by_seed[s].sort(key=lambda r: r["k"])
    r42 = by_seed[PRIMARY_SEED]
    has = lambda key: (lambda r: key in r)  # noqa: E731
    primary = summarise(_items(r42, d_of))
    sec = {
        "S1_balanced_accuracy_contrast": summarise(_items(
            r42, lambda r: 0.5 * ((r["cAA"] - r["cAB"]) + (r["cBB"] - r["cBA"])))),
        "S2_C1_same_run_minus_different_run_neighbour": summarise(_items(
            r42, lambda r: 0.5 * ((r["pAB"] - r["pAG"]) + (r["pBA"] - r["pBG"])), has("pAG"))),
        "S3_C1_identity_minus_neighbour_contrast": summarise(_items(
            r42, lambda r: d_of(r) - 0.5 * ((r["pAB"] - r["pAG"]) + (r["pBA"] - r["pBG"])), has("pAG"))),
        "S4_C2_placebo_nontest_swap": summarise(_items(
            r42, lambda r: 0.5 * r["sign"] * ((r["pAC"] - r["pAE"]) + (r["pBE"] - r["pBC"])), has("pAC"))),
        "S5_C4_multiset_base_identity_swap": summarise(_items(
            r42, lambda r: 0.5 * ((r["mAA"] - r["mAB"]) + (r["mBB"] - r["mBA"])), has("mAA"))),
        "S6_position_near_le6": summarise(_items(r42, d_of, lambda r: r["pos_dist"] <= POSITION_CUT)),
        "S7_position_far_gt6": summarise(_items(r42, d_of, lambda r: r["pos_dist"] > POSITION_CUT)),
    }
    pvals = {k: v["exact_sign_flip_p"] for k, v in sec.items() if "exact_sign_flip_p" in v}
    adj = holm(pvals)
    for k in sec:
        sec[k]["holm_p_secondary_family"] = adj.get(k)
    desc = {
        "D1_C1_subset_primary_identity_swap": summarise(_items(r42, d_of, has("pAG")), with_test=False),
        "D2_C1_identity_vs_different_run_neighbour": summarise(_items(
            r42, lambda r: 0.5 * ((r["pAA"] - r["pAG"]) + (r["pBB"] - r["pBG"])), has("pAG")), with_test=False),
        "D3_C3_on_target_abs_change": summarise(_items(
            r42, lambda r: 0.5 * (abs(r["pAA"] - r["pAB"]) + abs(r["pBB"] - r["pBA"]))), with_test=False),
        "D3_C3_off_target_abs_change_unexposed_same_fold": summarise(_items(
            r42, lambda r: r["off_fold"], lambda r: np.isfinite(r["off_fold"])), with_test=False),
        "D3_C3_off_target_abs_change_unexposed_same_session_class": summarise(_items(
            r42, lambda r: r["off_session_class"], lambda r: np.isfinite(r["off_session_class"])), with_test=False),
        "D4_context_partner_arm_correct_class_probability": summarise(_items(
            r42, lambda r: 0.5 * (r["pAB"] + r["pBA"])), with_test=False),
        "D4_context_own_arm_correct_class_probability": summarise(_items(
            r42, lambda r: 0.5 * (r["pAA"] + r["pBB"])), with_test=False),
        "D4_context_partner_arm_balanced_accuracy": summarise(_items(
            r42, lambda r: 0.5 * (float(r["cAB"]) + float(r["cBA"]))), with_test=False),
    }
    # D5 within-participant slope of d on |pos_a - pos_b| (descriptive)
    slopes = []
    for p in sorted({r["participant"] for r in r42}):
        x = np.array([r["pos_dist"] for r in r42 if r["participant"] == p], dtype=float)
        yv = np.array([d_of(r) for r in r42 if r["participant"] == p], dtype=float)
        if x.size >= 3 and np.ptp(x) > 0:
            slopes.append(float(np.polyfit(x, yv, 1)[0]))
    desc["D5_position_slope_points_per_trial"] = {"n_participants": len(slopes)}
    if len(slopes) >= 2:
        lo, hi = t_interval(slopes, 0.95)
        desc["D5_position_slope_points_per_trial"].update({"mean": float(np.mean(slopes)) * 100,
                                                           "ci95": [lo * 100, hi * 100]})
    # D6 placebo dispersion vs primary
    pl = [r for r in r42 if "pAC" in r]
    d_pl = [0.5 * r["sign"] * ((r["pAC"] - r["pAE"]) + (r["pBE"] - r["pBC"])) for r in pl]
    d_sub = [d_of(r) for r in pl]
    sd = lambda v: float(np.std(v, ddof=1)) * 100 if len(v) >= 2 else None  # noqa: E731
    desc["D6_C2_dispersion"] = {
        "n_pairs": len(pl), "pair_sd_placebo_points": sd(d_pl), "pair_sd_primary_same_pairs_points": sd(d_sub),
        "between_participant_sd_placebo_points": sec["S4_C2_placebo_nontest_swap"].get("sd_between_participants_points"),
        "between_participant_sd_primary_points": primary.get("sd_between_participants_points")}
    # D7 fold-seed sensitivity (primary estimand only)
    seeds = {str(s): summarise(_items(by_seed[s], d_of), with_test=False) for s in sorted(by_seed)}
    means = [v["mean_points"] for v in seeds.values() if "mean_points" in v]
    desc["D7_fold_seed_sensitivity_primary"] = {"per_seed": seeds,
                                                "range_of_means_points": [min(means), max(means)] if means else None}
    public = {"primary": primary, "secondary": sec, "descriptive": desc,
              "pair_counts": {"seed42": len(r42), "C1": sum("pAG" in r for r in r42),
                              "C2": len(pl), "per_seed": {str(s): len(v) for s, v in by_seed.items()}}}
    private = {"participant_effects_primary": {
        p: float(v) for p, v in participant_class_balanced(_items(r42, d_of), REQUIRED_LABELS,
                                                           value=lambda s: s.value).items()}}
    return public, private


def run_statistics(results: list[dict]) -> dict:
    secs = np.array([s for r in results for s in r["stats"]["seconds"]])
    rss = {}
    for r in results:
        rss[r["pid"]] = max(rss.get(r["pid"], 0.0), r["max_rss_mb"])
    return {"fits": int(sum(r["stats"]["fits"] for r in results)),
            "per_fit_seconds": {"median": float(np.median(secs)), "p95": float(np.percentile(secs, 95)),
                                "max": float(secs.max()), "sum_hours": float(secs.sum() / 3600)},
            "lr_n_iter_max": int(max(r["stats"]["n_iter_max"] for r in results)),
            "lr_fits_at_max_iter_2000": int(sum(r["stats"]["n_iter_at_cap"] for r in results)),
            "convergence_warnings": int(sum(r["stats"]["conv_warnings"] for r in results)),
            "reference_independent_recomputations_passed": int(sum(r["stats"]["ref_verified"] for r in results)),
            "worker_processes": len(rss), "peak_rss_per_worker_mb_max": max(rss.values()),
            "peak_rss_workers_sum_mb": float(sum(rss.values()))}
