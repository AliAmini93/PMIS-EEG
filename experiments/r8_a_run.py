#!/usr/bin/env python3
"""Shin2022 natural exact-copy swap with TS+LR and EEGNet."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, json_sha256, port_identity
from experiments.gpu_resources import select_device

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r8 import shin
from evaluation import r8_shin_analysis as shin_analysis
from models import shin_models  # noqa: E402

PROTOCOL_ID = "R8-A-PMIS-SHIN2022-NATURAL-v1.3"
PROTOCOL = CONFIG_ROOT / "R8_A_PROTOCOL_v1_3.json"
PRIVATE = ROOT / "data" / "r8_a"
PUBLIC = ROOT / "results" / "R8_A_results.json"
SEEDS = tuple(8201 + i for i in range(20))


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "R8_A_run.log", "a") as fh:
        fh.write(line + "\n")


def inputs():
    return ROOT / "data/shin2022/r2_shin_windows_v1.npz", ROOT / "data/shin2022/recording_registry.json"


def load(check=None):
    w, r = inputs()
    if check is not None:
        g3._stop(reference_matches(pv.sha256_file(w), check["windows"]), "input SHA-256 differs")
    z = dict(np.load(w, allow_pickle=False))
    reg = json.loads(r.read_text())["rows"]
    if check is not None:
        g3._stop(reference_matches(json_sha256(reg), check["registry"]), "recording registry commitment differs")
    tasks = shin.build_tasks(z, reg)
    return z, tasks


def cmd_plan_only():
    proto = load_protocol(PROTOCOL)
    _, tasks = load(proto["portable_input_sha256"])
    g3._stop(reference_matches(shin.commitment(tasks), proto["plan"]["commitment"]), "plan commitment differs")
    print(json.dumps({"participants": len(tasks), "pairs": sum(len(t["pairs"]) for t in tasks),
                      "arms": sum(2 * t["T"].size for t in tasks), "commitment": shin.commitment(tasks)}))


def verify():
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    z, tasks = load(proto["portable_input_sha256"])
    g3._stop(reference_matches(shin.commitment(tasks), proto["plan"]["commitment"]), "plan commitment differs")
    return proto, z, tasks


def run(mode):
    out = {"stage": f"R8-A-{mode}", "protocol_id": PROTOCOL_ID}
    try:
        proto, z, tasks = verify()
        out["port_manifest_sha256"] = port_identity()
        log(f"{mode} PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}")
        os.nice(10)
        y = z["label"].astype(np.int64)
        t0 = time.time()
        if mode == "tslr":
            covs = z["cov830_FB_EARLY"]
            used = np.unique(np.concatenate([np.r_[t["T"], t["base"], t["placebo_arms"]] for t in tasks]))
            g3._stop(bool(np.all(np.isfinite(covs[used]))) and np.linalg.eigvalsh(covs[used]).min() > 0, "covariances of used rows")
            res = {}
            for t in tasks:
                P, NT, it = shin_models.tslr_task(t, covs, y)
                res[t["subject"]] = {"P": P, "NT": NT}
                log(f"tslr {t['subject']} arms {P.shape[0]} newton max {it} {time.time() - t0:.0f} s")
            np.savez(PRIVATE / "R8_A_tslr_PRIVATE.npz", **{f"{s}_{k}": v for s, d in res.items() for k, v in d.items()})
        elif mode == "eegnet":
            import torch
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            dev = select_device()
            _, total = torch.cuda.mem_get_info()
            torch.cuda.set_per_process_memory_fraction(min(0.9, 8.0 * 2**30 / total))
            x = z["eeg440_FB_EARLY"].astype(np.float32)
            used = np.unique(np.concatenate([np.r_[t["T"], t["base"], t["placebo_arms"]] for t in tasks]))
            g3._stop(bool(np.all(np.isfinite(x[used]))), "features of used rows")
            x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)
            X = torch.as_tensor(x, device=dev)
            y01 = torch.as_tensor(y - 1, device=dev)
            commit = shin.commitment(tasks)
            res, aa = {}, 0.0
            ck = PRIVATE / "eegnet_ck"
            ck.mkdir(parents=True, exist_ok=True)
            for t in tasks:
                f = ck / f"{t['subject']}.npz"
                if f.exists() and str(np.load(f)["commitment"]) == commit:
                    zz = np.load(f)
                    P, a = zz["P"], float(zz["aa"])
                else:
                    P = None
                    for attempt in range(7):
                        w0, wl = time.time(), 0.0
                        while True:  # wait, never preempt; count memory this process already holds as available to it
                            torch.cuda.empty_cache()
                            avail = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30
                            if avail >= 6.0:
                                break
                            g3._stop(time.time() - w0 < 6 * 3600, "GPU unavailable for 6 h", "STOPPED_RESOURCE")
                            if time.time() - wl > 600:
                                log(f"eegnet waiting for GPU: {avail:.1f} GB available")
                                wl = time.time()
                            time.sleep(60)
                        try:
                            P, a = shin_models.eegnet_task(t, X, y01, dev, SEEDS)
                            break
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            log(f"eegnet {t['subject']}: out of memory (attempt {attempt + 1}); recomputing")
                            time.sleep(120)
                    g3._stop(P is not None, f"{t['subject']}: out of memory", "STOPPED_RESOURCE")
                    np.savez(f, commitment=np.array(commit), P=P, aa=np.array(a))
                res[t["subject"]] = P
                aa = max(aa, a)
                log(f"eegnet {t['subject']} arms {P.shape[0]} aa {a:.1e} {time.time() - t0:.0f} s")
            g3._stop(aa == 0.0, f"A/A determinism failed: {aa}")
            np.savez(PRIVATE / "R8_A_eegnet_PRIVATE.npz", aa=np.array(aa), **{f"{s}_P": v for s, v in res.items()})
        else:  # analyse
            tz, ez = np.load(PRIVATE / "R8_A_tslr_PRIVATE.npz"), np.load(PRIVATE / "R8_A_eegnet_PRIVATE.npz")
            recs = []
            for t in tasks:
                _, extras = shin.arm_rows(t)
                tr = shin.pair_records(t, tz[f"{t['subject']}_P"], extras, y)
                er = shin.pair_records(t, ez[f"{t['subject']}_P"], extras, y)
                NT = tz[f"{t['subject']}_NT"]
                pos = {int(v): i for i, v in enumerate(t["T"].tolist())}
                for pr, a, b in zip(t["pairs"], tr, er):
                    ia, ib = pos[pr["a"]], pos[pr["b"]]
                    r = {"subject": a["subject"], "label": a["label"], "pos_dist": a["pos_dist"], "sign": a["sign"]}
                    r.update({"t_" + k: v for k, v in a.items() if k.startswith("p")})
                    r.update({"e_" + k: v for k, v in b.items() if k.startswith("p")})
                    r.update({"nt_pAA": NT[ia, ia], "nt_pBA": NT[ia, ib], "nt_pAB": NT[ib, ia], "nt_pBB": NT[ib, ib]})
                    recs.append(r)
            out.update({"status": "COMPLETED", "results": shin_analysis.analyse(recs, float(ez["aa"])), "plan": proto["plan"]})
            keys = sorted(k for k in recs[0] if k != "subject")
            np.savez(PRIVATE / "R8_A_pair_records_PRIVATE.npz", subject=np.array([r["subject"] for r in recs]),
                     **{k: np.array([float(r[k]) for r in recs]) for k in keys})
        if mode != "analyse":
            log(f"{mode} done {time.time() - t0:.0f} s")
            return 0
        code = 0
    except g3.G3Stop as e:
        out.update({"status": e.status, "stop_message": e.message})
        log(f"{mode} STOP {e.status}: {e.message}")
        code = 2
    except Exception as e:
        out.update({"status": "BLOCKED", "stop_message": f"{type(e).__name__}: {e}"})
        log(f"{mode} STOP BLOCKED: " + traceback.format_exc())
        code = 2
    if out.get("status") == "COMPLETED" or not PUBLIC.exists():
        PUBLIC.parent.mkdir(parents=True, exist_ok=True)
        PUBLIC.write_text(json.dumps(out, indent=2, default=float) + "\n")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute-tslr", "--execute-eegnet", "--analyse"):
        m.add_argument(flag, action="store_true")
    a = ap.parse_args()
    if a.plan_only:
        sys.exit(cmd_plan_only())
    sys.exit(run("tslr" if a.execute_tslr else "eegnet" if a.execute_eegnet else "analyse"))
