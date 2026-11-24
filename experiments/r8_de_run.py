#!/usr/bin/env python3
"""Lee2019 matched TCFormer and CBraMod training, frozen probes and analysis."""

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

from experiments.runtime import reference_matches, verify_recorded_input, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity
from experiments.gpu_resources import select_device

ROOT = RUN_ROOT
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r6 import lee2019  # noqa: E402
from experiments.pmis_r7 import c6  # noqa: E402
from evaluation import r8_de_analysis as de_analysis
from experiments.pmis_r8 import de_plan
from models import models_de, probe, seq_train  # noqa: E402

PROTOCOL_ID = "R8-DE-TRANSFORMER-FOUNDATION-LEE2019-v1.1"
PROTOCOL = CONFIG_ROOT / "R8_DE_PROTOCOL_v1_1.json"
PRIVATE = ROOT / "data" / "r8_de"
PUBLIC = ROOT / "results" / "R8_DE_results.json"
C3, E = ROOT / "data" / "r6_c3", ROOT / "data" / "r8_e"
EEG128, EEG200 = "lee2019_eegnet_4_40_128hz.npy", "lee2019_cbramod_200hz.npy"
CKPT = ROOT / "weights" / "pretrained_weights.pth"
CKPT_SHA = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
PRIOR = {"C3a": C3 / "R6_C3a_pair_records_PRIVATE.npz", "C3b": ROOT / "data" / "r6_c3b" / "R6_C3b_pair_records_PRIVATE.npz",
         "C6_EEGNet": ROOT / "data" / "r7_c6" / "R7_C6_EEGNet_pair_records_PRIVATE.npz",
         "C6_Shallow": ROOT / "data" / "r7_c6" / "R7_C6_Shallow_pair_records_PRIVATE.npz"}
C6_COMMITMENT_PROTOCOL = CONFIG_ROOT / "R7_C6_PROTOCOL.json"
GPU_MIN_AVAIL_GB, HOST_MIN_MB, MAX_WAIT_S = 3.0, 4096, 6 * 3600
MODELS = ("TCFormer", "CBraMod")


def log(msg, name="R8_DE_run.log"):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / name, "a") as fh:
        fh.write(line + "\n")


def build():
    z = np.load(C3 / "lee2019_meta.npz")
    meta = {k: z[k] for k in ("subject", "session", "run", "pos", "label")}
    tasks = lee2019.build_tasks(meta)
    c6p = c6.build_plan(tasks, meta["label"].astype(np.int64))
    sub = de_plan.subset_plan(c6p)
    return meta, tasks, c6.commitment(c6p), sub, de_plan.commitment(sub)


def plan_summary():
    meta, tasks, c6c, sub, subc = build()
    subj = np.array([g["subject"] for g in sub["groups"]])
    lab = np.array([g["pairs"][0]["label"] for g in sub["groups"]])
    both = int(sum(np.unique(lab[subj == s]).size == 2 for s in np.unique(subj)))
    return {"c6_commitment": c6c, "subset_commitment": subc, "subset_pairs": sub["n_pairs"], "subset_arms_per_model": sub["n_arms"],
            "participants": int(np.unique(subj).size), "participants_with_both_classes": both,
            "fallback_pairs": int(sum(g["pairs"][0]["label"] != 1 + g["fold"] % 2 for g in sub["groups"])),
            "probe_pairs": int(sum(len(t["pairs"]) for t in tasks)), "probe_tasks": len(tasks)}


def cmd_plan_only():
    print(json.dumps(plan_summary()))


def verify():
    import torch
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    g3._stop(torch.__version__ == proto["environment"]["torch"], "torch version")
    return proto


def check_data(proto, names):
    paths = {EEG128: C3 / EEG128, "lee2019_meta.npz": C3 / "lee2019_meta.npz", EEG200: E / EEG200, "cbramod_checkpoint": CKPT, **PRIOR}
    for n in names:
        g3._stop(verify_recorded_input(paths[n], proto["data_sha256"][n]), f"{n} SHA-256 differs (checked before load)")


def setup_torch():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch, select_device()


def wait_gpu(torch, tag):
    t0, last = time.time(), 0.0
    while True:
        torch.cuda.empty_cache()
        avail = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30
        st = g3.resource_state()
        if avail >= GPU_MIN_AVAIL_GB and st["mem_available_mb"] >= HOST_MIN_MB and not (st["psi_full_avg10"] > 5.0):
            return
        g3._stop(time.time() - t0 < MAX_WAIT_S, "GPU/host unavailable for 6 h (not preempting)", "STOPPED_RESOURCE")
        if time.time() - last > 600:
            log(f"{tag} waiting for resources: gpu available {avail:.1f} GB", f"R8_DE_{tag}.log")
            last = time.time()
        time.sleep(60)


def load_x(torch, dev, model):
    if model == "TCFormer":
        x = np.load(C3 / EEG128, allow_pickle=False)
        g3._stop(x.shape == (21600, 20, 512) and bool(np.all(np.isfinite(x))), "128-Hz features")
        x = (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True) + 1e-8)  # C6 per-trial channel z-score
    else:
        x = np.load(E / EEG200, allow_pickle=False)
        g3._stop(x.shape == (21600, 20, 800) and bool(np.all(np.isfinite(x))), "200-Hz CBraMod input")
        x = x / 100.0  # official loader: data / 100
    return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=dev)


def factories(model):
    if model == "TCFormer":
        return models_de.make_tcformer, models_de.tcformer_opt, models_de.TCF_EPOCHS, models_de.TCF_BS, {}
    return (lambda: models_de.make_cbramod_ft(CKPT)), models_de.cbramod_opt, models_de.CBM_EPOCHS, models_de.CBM_BS, \
        {"label_smoothing": 0.1, "clip": 1.0}


def cmd_execute(model, k, n):
    tag = f"{model}_shard{k}of{n}"
    try:
        torch, dev = setup_torch()
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R8_DE_{tag}.log")
        os.nice(10)
        check_data(proto, ["lee2019_meta.npz", EEG128 if model == "TCFormer" else EEG200] + (["cbramod_checkpoint"] if model == "CBraMod" else []))
        meta, tasks, c6c, sub, subc = build()
        g3._stop(reference_matches(c6c, proto["plan"]["c6_commitment"]) and reference_matches(subc, proto["plan"]["subset_commitment"]), "plan commitment differs")
        X = load_x(torch, dev, model)
        y01 = torch.as_tensor(meta["label"].astype(np.int64) - 1, device=dev)
        mk, opt, epochs, bs, kw = factories(model)
        gdir = PRIVATE / "groups"
        gdir.mkdir(parents=True, exist_ok=True)
        mine = de_plan.shard(sub["groups"], k, n)
        t0 = time.time()
        for i, g in enumerate(mine):
            f = gdir / f"{model}_{g['subject']}_{g['fold']}.npz"
            if f.exists() and str(np.load(f)["commitment"]) == subc:
                continue
            res = None
            for attempt in range(7):
                wait_gpu(torch, tag)
                try:
                    res = seq_train.run_group(mk, opt, g, X, y01, dev, epochs, bs, **kw)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"{tag} group {i}: out of memory (attempt {attempt + 1}); identical recompute", f"R8_DE_{tag}.log")
                    time.sleep(120)
            g3._stop(res is not None, "out of memory 7 times", "STOPPED_RESOURCE")
            np.savez(f, commitment=np.array(subc), **res)
            if (i + 1) % 6 == 0 or i + 1 == len(mine):
                log(f"{tag} group {i + 1}/{len(mine)} {time.time() - t0:.0f} s", f"R8_DE_{tag}.log")
        log(f"{tag} DONE", f"R8_DE_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R8_DE_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R8_DE_{tag}.log")
        return 2


def cmd_execute_probe():
    try:
        torch, dev = setup_torch()
        proto = verify()
        log(f"probe PORT_PROTOCOL_VERIFIED {port_identity()}", "R8_DE_probe.log")
        os.nice(10)
        check_data(proto, ["lee2019_meta.npz", EEG200, "cbramod_checkpoint"])
        meta, tasks, c6c, sub, subc = build()
        g3._stop(reference_matches(subc, proto["plan"]["subset_commitment"]), "plan commitment differs")
        from models.vendor_cbramod.cbramod import CBraMod
        torch.manual_seed(0)
        bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        bb.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
        bb.proj_out = torch.nn.Identity()
        bb = bb.to(dev).eval()
        X = load_x(torch, dev, "CBraMod")
        with torch.no_grad():
            emb = torch.cat([bb(X[i:i + 64].view(-1, 20, 4, 200)).mean(dim=(1, 2)) for i in range(0, X.shape[0], 64)])
            emb2 = torch.cat([bb(X[i:i + 64].view(-1, 20, 4, 200)).mean(dim=(1, 2)) for i in range(0, 640, 64)])
        F = emb.double().cpu().numpy()
        g3._stop(bool(np.all(np.isfinite(F))), "non-finite embedding")
        g3._stop(bool(torch.equal(emb[:640], emb2)), "embedding not reproducible")
        np.save(PRIVATE / "R8_E1_embeddings_PRIVATE.npy", F)
        y01 = meta["label"].astype(np.int64) - 1
        recs, st, t0 = [], [], time.time()
        for t in tasks:
            r, s_ = probe.run_task(t, F, y01)
            recs += r
            st.append(s_)
        keys = sorted(recs[0])
        conv = {"n_fits": int(sum(x["n_fits"] for x in st)), "n_stagnated_at_float_floor": int(sum(x["n_stagnated"] for x in st)),
                "max_abs_grad": float(max(x["max_abs_grad"] for x in st)), "newton_max_iter": int(max(x["newton_max_iter"] for x in st))}
        np.savez(PRIVATE / "R8_E1_pair_records_PRIVATE.npz", **{kk: np.array([float(r[kk]) for r in recs]) for kk in keys},
                 newton_max_iter=np.array(conv["newton_max_iter"]), n_fits=np.array(conv["n_fits"]),
                 n_stagnated=np.array(conv["n_stagnated_at_float_floor"]), max_abs_grad=np.array(conv["max_abs_grad"]))
        log(f"probe DONE {len(recs)} pairs, convergence {conv}, {time.time() - t0:.0f} s", "R8_DE_probe.log")
        return 0
    except g3.G3Stop as e:
        log(f"probe STOP {e.status}: {e.message}", "R8_DE_probe.log")
        return 2
    except Exception:
        log("probe STOP BLOCKED: " + traceback.format_exc(), "R8_DE_probe.log")
        return 2


def _records(path, prefix=""):
    z = np.load(path)
    keys = [k for k in z.files if z[k].ndim == 1]
    n = len(z["k"])
    return [{k: (float(z[k][i])) for k in keys} for i in range(n)]


def cmd_analyse():
    out = {"stage": "R8-DE", "protocol_id": PROTOCOL_ID}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        check_data(proto, list(PRIOR))
        meta, tasks, c6c, sub, subc = build()
        g3._stop(reference_matches(subc, proto["plan"]["subset_commitment"]), "plan commitment differs")
        z = np.load(PRIOR["C3a"])
        tslr = {int(k): 0.5 * ((a - b) + (c - d)) for k, a, b, c, d in zip(z["k"], z["pAA"], z["pAB"], z["pBB"], z["pBA"])}
        z = np.load(PRIOR["C3b"])
        eeg3 = {int(k): 0.5 * ((a - b) + (c - d)) for k, a, b, c, d in zip(z["k"], z["pAA"], z["pAB"], z["pBB"], z["pBA"])}
        c6r = {m: {int(r["k"]): r for r in _records(PRIOR[f"C6_{m}"])} for m in ("EEGNet", "Shallow")}
        res = {}
        gdir = PRIVATE / "groups"
        for model in MODELS:
            results, missing = {}, 0
            for g in sub["groups"]:
                f = gdir / f"{model}_{g['subject']}_{g['fold']}.npz"
                if not (f.exists() and str(np.load(f)["commitment"]) == subc):
                    missing += 1
                    continue
                zz = np.load(f)
                results[(model, g["subject"], g["fold"])] = {kk: zz[kk] for kk in zz.files if kk != "commitment"}
            g3._stop(missing == 0, f"{model}: {missing} groups missing")
            recs, aa = c6.pair_records(sub, results, model)
            g3._stop(max(aa) == 0.0, f"{model} A/A determinism check failed: {max(aa)}")
            keys = sorted({kk for r in recs for kk in r})
            np.savez(PRIVATE / f"R8_DE_{model}_pair_records_PRIVATE.npz", **{kk: np.array([float(r[kk]) for r in recs]) for kk in keys})
            res[model] = de_analysis.analyse_deep(model, recs, aa, c6r, tslr)
        pr = _records(PRIVATE / "R8_E1_pair_records_PRIVATE.npz")
        res["CBraMod_probe"] = de_analysis.analyse_probe(pr, tslr, eeg3, {int(g["pairs"][0]["k"]) for g in sub["groups"]})
        zc = np.load(PRIVATE / "R8_E1_pair_records_PRIVATE.npz")
        res["CBraMod_probe"]["convergence"] = {"n_fits": int(zc["n_fits"]), "n_stagnated_at_float_floor": int(zc["n_stagnated"]),
                                               "max_abs_grad": float(zc["max_abs_grad"]), "newton_max_iter": int(zc["newton_max_iter"])}
        out.update({"status": "COMPLETED", "plan": proto["plan"], "results": res})
        code = 0
    except g3.G3Stop as e:
        out.update({"status": e.status, "stop_message": e.message})
        code = 2
    except Exception as e:
        out.update({"status": "BLOCKED", "stop_message": f"{type(e).__name__}: {e}"})
        log("analyse STOP BLOCKED: " + traceback.format_exc())
        code = 2
    PUBLIC.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(out["status"])
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for flag in ("--plan-only", "--execute-probe", "--execute", "--analyse"):
        m.add_argument(flag, action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.plan_only:
        sys.exit(cmd_plan_only())
    if a.execute_probe:
        sys.exit(cmd_execute_probe())
    if a.execute:
        kk, nn = (int(v) for v in a.shard.split("/"))
        sys.exit(cmd_execute(a.model, kk, nn))
    sys.exit(cmd_analyse())
