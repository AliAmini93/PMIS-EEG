#!/usr/bin/env python3
"""Dreyer2023 held-out-run policy fitting, confound controls and regret analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, read_json, port_identity

ROOT = RUN_ROOT
from experiments import gpu_resources as gpucoord  # noqa: E402
from evaluation import r12x_analysis as an  # noqa: E402
from experiments.pmis_r12x import design as ds  # noqa: E402
from experiments.pmis_r12x import execute as ex  # noqa: E402
from experiments.pmis_r12x import guards as gu  # noqa: E402
from data_preparation import dreyer_sources as so  # noqa: E402

PROTOCOL = CONFIG_ROOT / "R12X_PROTOCOL.json"
PRIVATE = Path(os.environ.get("R12X_PRIVATE_DIR", ROOT / "data" / "r12x"))       # override: synthetic tests only
FEAT = PRIVATE / "features"
PUBLIC = Path(os.environ.get("R12X_RESULTS_PATH", ROOT / "results" / "R12X_results.json"))
POLICY = {"time_limits": "none", "gpu_memory": "no per-process cap; wait until free GPU memory >= measured need",
          "initial_need_free_mib": 6144, "need_rule": "max(initial, 1.25 x largest observed peak reserved + 512 MiB)",
          "host": "wait until MemAvailable >= 8 GiB and PSI full avg10 <= 5", "oom": "3 identical attempts, then stop",
          "nice": 10, "deterministic": True}
WINDOW_DEEP = {"cue": (20, 160), "fb": (20, 512)}
TSLR_WINDOWS = ("cue", "fb", "precue", "eogcue", "eogfb")
DEEP = ("EEGNet", "Shallow")


class Stop(RuntimeError):
    pass


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_meta() -> dict:
    with np.load(FEAT / "meta.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def designs(meta: dict) -> list[dict]:
    out = []
    for p in sorted(np.unique(meta["participant"]).tolist()):
        m = meta["participant"] == p
        out.append(ds.build_participant(int(p), meta["run"][m], meta["pos"][m], meta["label"][m], np.flatnonzero(m)))
    return out


def participants_digest(meta) -> str:
    return hashlib.sha256(json.dumps(sorted(np.unique(meta["participant"]).tolist())).encode()).hexdigest()


def verify() -> dict:
    return load_protocol(PROTOCOL)


def check_feature(proto: dict, name: str) -> Path:
    p = FEAT / name
    if not reference_matches(sha(p), proto["feature_sha256"][name]):
        raise Stop(f"feature file differs from protocol: {name}")
    return p


def unit_path(model, window, participant, stage, k):
    return PRIVATE / "units" / f"{model}_{window}" / f"u_{participant}_{stage}_{k}.npz"


def save_atomic(fp: Path, **arrays):
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_name(fp.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, fp)


def unit_done(fp: Path, commit: str) -> bool:
    if not fp.exists():
        return False
    with np.load(fp) as z:
        return str(z["commitment"]) == commit


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str):
        with open(self.path, "a") as fh:
            fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}\n")


def write_progress(**kw):
    p = PRIVATE / "worker_progress.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated": dt.datetime.now(dt.timezone.utc).isoformat(), **kw}, indent=1))
    tmp.replace(p)


def tslr_stage(proto, D, y12, commit, log):
    n_done = 0
    for window in TSLR_WINDOWS:
        covs = np.load(check_feature(proto, f"cov_{window}.npy"), allow_pickle=False)
        if not np.all(np.isfinite(covs)):
            raise Stop(f"non-finite covariances ({window})")
        for d in D:
            for stage in ("selection", "final"):
                fp = unit_path("TSLR", window, d["participant"], stage, 0)
                if unit_done(fp, commit):
                    continue
                save_atomic(fp, commitment=np.array(commit), **ex.run_tslr_unit(d, stage, covs, y12))
                n_done += 1
        log(f"TS+LR {window} complete")
    return n_done


def deep_stage(proto, D, y12, commit, log):
    import torch
    from models import deep_group as dg
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    is_oom = lambda e: isinstance(e, torch.OutOfMemoryError)  # noqa: E731
    units = [(m, w, d, s, k) for m in DEEP for w in WINDOW_DEEP for d in D for s in ("selection", "final")
             for k in range(ds.K_SEEDS)]
    todo = [u for u in units if not unit_done(unit_path(u[0], u[1], u[2]["participant"], u[3], u[4]), commit)]
    log(f"deep units: {len(units)} total, {len(todo)} to run")
    need = POLICY["initial_need_free_mib"] * 2**20
    usage_path = PRIVATE / "unit_usage.jsonl"
    dev = gpucoord.select_device()
    gu.wait_safe(torch, need, log)
    cls = {"EEGNet": dg.EEGNetG, "Shallow": dg.GroupShallow}
    # determinism smoke test (synthetic): the same unit twice must be bit-identical
    rng = np.random.default_rng(0)
    lab = np.concatenate([rng.permutation([1] * 20 + [2] * 20) for _ in range(6)])
    sd = ds.build_participant(1, np.repeat(np.array(ds.RUNS_A + (ds.RUN_B, ds.RUN_C)), 40),
                              np.tile(np.arange(40), 6), lab, np.arange(240))
    ys = torch.as_tensor(lab - 1, device=dev)
    for m in DEEP:
        for w, (C, T) in WINDOW_DEEP.items():
            Xs = torch.as_tensor(rng.standard_normal((240, C, T)).astype(np.float32), device=dev)
            a = ex.run_deep_unit(cls[m], sd, "selection", 0, Xs, ys, dev, epochs=3)
            b = ex.run_deep_unit(cls[m], sd, "selection", 0, Xs, ys, dev, epochs=3)
            if any(not np.array_equal(a[k], b[k]) for k in a):
                raise Stop(f"non-deterministic smoke test ({m} {w})")
            need = max(need, int(1.25 * torch.cuda.max_memory_reserved()) + 512 * 2**20)
            del Xs
    log(f"smoke test deterministic; need {need // 2**20} MiB")
    X_cache, t_gpu, done = {}, 0.0, 0
    for (m, w, d, stage, k) in todo:
        if w not in X_cache:
            X_cache.clear()
            torch.cuda.empty_cache()
            x = np.load(check_feature(proto, f"eeg_{w}.npy"), allow_pickle=False)
            if x.dtype != np.float32 or x.shape[1:] != WINDOW_DEEP[w] or not np.all(np.isfinite(x)):
                raise Stop(f"deep features ({w})")
            x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
            X_cache[w] = torch.as_tensor(x, device=dev)
            del x
            y01 = torch.as_tensor(y12 - 1, device=dev)
        fp = unit_path(m, w, d["participant"], stage, k)
        if unit_done(fp, commit):
            continue
        waited = gu.wait_safe(torch, need, log)

        def recover():
            torch.cuda.empty_cache()
            gu.wait_safe(torch, need, log)

        with gu.UnitMeter(torch) as meter:
            res = gu.with_oom_retry(lambda: ex.run_deep_unit(cls[m], d, stage, k, X_cache[w], y01, dev),
                                    is_oom, recover, log)
        save_atomic(fp, commitment=np.array(commit), **res)
        need = max(need, int(1.25 * meter.rec["gpu_peak_reserved_bytes"]) + 512 * 2**20)
        t_gpu += meter.rec["wall_s"]
        done += 1
        with open(usage_path, "a") as fh:
            fh.write(json.dumps({"model": m, "window": w, "stage": stage, "k": k, "waited_s": round(waited, 1),
                                 **meter.rec}) + "\n")
        if done % 20 == 0 or done == len(todo):
            write_progress(stage="deep", done=done, todo=len(todo), gpu_unit_hours=round(t_gpu / 3600, 3))
            log(f"deep {done}/{len(todo)}; unit GPU time {t_gpu / 3600:.3f} h")
    return done


def cmd_worker(*, tslr_only=False) -> int:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    log = Log(PRIVATE / "worker.log")
    try:
        with gpucoord.JobLock("r12x-experiment"):
            proto = verify()
            log(f"worker start; protocol {proto['protocol_id']} verified with port manifest {port_identity()}")
            os.nice(10)
            check_feature(proto, "meta.npz")
            meta = load_meta()
            if not reference_matches(participants_digest(meta), proto["participants_sha256"]):
                raise Stop("participants differ from protocol")
            D = designs(meta)
            commit = ds.commitment(D)
            if not reference_matches(commit, proto["design_commitment"]):
                raise Stop("design commitment differs")
            y12 = meta["label"].astype(np.int64)
            write_progress(stage="tslr")
            log(f"TS+LR units newly run: {tslr_stage(proto, D, y12, commit, log)}")
            if tslr_only:
                write_progress(stage="tslr_complete")
                return 0
            write_progress(stage="deep")
            log(f"deep units newly run: {deep_stage(proto, D, y12, commit, log)}")
            write_progress(stage="complete")
            (PRIVATE / "WORKER_COMPLETE").write_text(dt.datetime.now(dt.timezone.utc).isoformat())
            log("WORKER COMPLETE")
            return 0
    except gpucoord.AlreadyRunning as e:
        log(f"refused: {e}")
        return 3
    except (Stop, gu.ResourceStop, ex.AAStop, FloatingPointError) as e:
        log(f"STOP {type(e).__name__}: {e}")
        write_progress(stage="stopped", reason=f"{type(e).__name__}: {e}")
        return 2
    except Exception:
        log("STOP unexpected: " + traceback.format_exc())
        write_progress(stage="stopped", reason="unexpected exception (see worker.log)")
        return 2


def collect(D, proto) -> dict:
    """Seed-averaged BA matrices per (window, read-out); a missing unit is a stop (no partial analysis)."""
    commit = ds.commitment(D)
    out = {}
    for window in ("cue", "fb"):
        for ro in ("final", "es"):
            BBP, BBL, BC = (np.zeros((len(D), 3)) for _ in range(3))
            for i, d in enumerate(D):
                for j, m in enumerate(an.ROSTER):
                    ks = [0] if m == "TSLR" else range(ds.K_SEEDS)
                    for tgt, stage, arm, y in ((BBP, "selection", "S_P", d["yB"]), (BBL, "selection", "S_L", d["yB"]),
                                               (BC, "final", "F", d["yC"])):
                        vals = []
                        for k in ks:
                            fp = unit_path(m, window, d["participant"], stage, k)
                            if not unit_done(fp, commit):
                                raise Stop(f"missing or foreign unit {fp.name}")
                            with np.load(fp) as z:
                                key = f"exact_p_{arm}" if m == "TSLR" else f"{ro}_p_{arm}"
                                vals.append(an.ba(z[key], y))
                        tgt[i, j] = float(np.mean(vals))
            out[(window, ro)] = (BBP, BBL, BC)
    return out


def sensitivity_participants(proto):
    receipt = read_json(PRIVATE / "G1R_eligible_PRIVATE.json")
    participants = receipt["sensitivity_G7_as_declared"]
    expected = proto["sensitivity_membership"]
    digest = hashlib.sha256(json.dumps(participants).encode()).hexdigest()
    if len(participants) != expected["count"] or not reference_matches(digest, expected["sha256"]):
        raise Stop("Dreyer2023 sensitivity membership differs from the frozen audit")
    return participants


def cmd_analyse() -> int:
    proto = verify()
    if PUBLIC.exists():
        raise Stop("analysis already run (runs once)")
    check_feature(proto, "meta.npz")
    meta = load_meta()
    D = designs(meta)
    commit = ds.commitment(D)
    if not reference_matches(commit, proto["design_commitment"]):
        raise Stop("design commitment differs")
    for d in D:
        d["yB"], d["yC"] = meta["label"][d["B"]], meta["label"][d["C"]]
    block = an.blocks_from_metadata([d["participant"] for d in D], so.group_of)
    mats = collect(D, proto)
    res = {f"{w}/{ro}": an.analyse(*mats[(w, ro)], block) for (w, ro) in mats}
    sec_p = {"E1 cue/es": res["cue/es"]["E1"]["p_decision"], "E1 fb/final": res["fb/final"]["E1"]["p_decision"],
             "E2 cue/final": res["cue/final"]["E2"]["p_decision"]}
    controls = {}
    for w in ("precue", "eogcue", "eogfb"):
        vals = {"B_S_P": [], "C_F": []}
        for d in D:
            for stage, arm, y, key in (("selection", "S_P", d["yB"], "B_S_P"), ("final", "F", d["yC"], "C_F")):
                unit = unit_path("TSLR", w, d["participant"], stage, 0)
                if not unit_done(unit, commit):
                    raise Stop(f"missing or foreign {w} control fit")
                with np.load(unit, allow_pickle=False) as z:
                    vals[key].append(an.ba(z[f"exact_p_{arm}"], y))
        controls[w] = {k: {"mean_BA": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n": len(v)} for k, v in vals.items()}
    sens = {}
    sp = sensitivity_participants(proto)
    if sp:
        keep = [i for i, d in enumerate(D) if d["participant"] in set(sp)]
        if len(keep) != len(sp):
            raise Stop("sensitivity participants not all in the cohort")
        bs = an.blocks_from_metadata([D[i]["participant"] for i in keep], so.group_of)
        for key in (("cue", "final"),):
            BBP, BBL, BC = mats[key]
            r = an.analyse(BBP[keep], BBL[keep], BC[keep], bs)
            sens["/".join(key)] = {"n": len(keep), "E1": r["E1"], "E2": r["E2"]}
    lag1 = [float(np.mean(meta["label"][d["C"]][1:] == meta["label"][d["C"]][:-1])) for d in D]
    out = {"protocol_id": proto["protocol_id"], "n_participants": len(D),
           "primary": {"estimand": "E1 LOO cross-fitted policy regret, cue window, final epoch", **res["cue/final"]["E1"]},
           "secondary_holm": an.holm(sec_p), "all": res, "controls_descriptive": controls,
           "sensitivity_G7_as_declared_descriptive": sens,
           "label_lag1_same_class_fraction_C_mean": float(np.mean(lag1))}
    PUBLIC.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan-only", action="store_true")
    g.add_argument("--worker", action="store_true", help="fit TS+LR and neural arms")
    g.add_argument("--execute-tslr", action="store_true", help="fit only the recorded CPU TS+LR stage")
    g.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.plan_only:
        meta = load_meta()
        D = designs(meta)
        print(json.dumps({"participants": len(D), "participants_sha256": participants_digest(meta),
                          "design_commitment": ds.commitment(D), "port_manifest_sha256": port_identity()}, indent=1))
        return 0
    if a.worker or a.execute_tslr:
        return cmd_worker(tslr_only=a.execute_tslr)
    try:
        return cmd_analyse()
    except Stop as e:
        print(f"STOP: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
