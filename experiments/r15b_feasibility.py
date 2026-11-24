#!/usr/bin/env python3
"""Dreyer2023 technical feasibility from fitting runs with engagement checks and refits."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
from experiments.runtime import CONFIG_ROOT, RUN_ROOT, file_sha256, read_json
from experiments.gpu_resources import select_device

ROOT = RUN_ROOT
import numpy as np  # noqa: E402

WORK = ROOT / "data" / "feasibility"
CACHE = WORK / "cache"
ELIGIBLE = ROOT / "data" / "r12x" / "G1R_eligible_PRIVATE.json"
RES = Path(os.environ.get("R15B_RES", WORK / "results"))
CHECKPOINTS = read_json(CONFIG_ROOT / "checkpoints.json")
CK = {model: ROOT / "weights" / v["filename"] for model, v in CHECKPOINTS.items()}
PREFIX = {"ST-EEGFormer-small": "st", "LaBraM-Base": "lb", "CBraMod": "cb"}
DEC_CONFIGS = [(m, w, i, a) for m, ws in (("ST-EEGFormer-small", ("w100", "w125")), ("LaBraM-Base", ("w100",)), ("CBraMod", ("w100",)))
               for w in ws for i in ("pre", "rand") for a in ("FT", "LP")]
EPOCHS, BS, WARMUP, WD, LS = 100, 16, 10, 0.05, 0.1
OUT = Path(os.environ.get("R15B_OUT", ROOT / "results" / "R15B_feasibility.json"))
RES_OVERRIDE = os.environ.get("R15B_RES")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def participants():
    from data_preparation import feasibility as D
    return list(D.check_cohort(ELIGIBLE)["eligible"])


def load(n):
    from data_preparation import feasibility as D
    z = np.load(CACHE / f"{D.producer_id(n)}.npz")
    assert set(np.unique(z["run"]).tolist()) <= {1, 2, 3, 4}, "a non-fitting run reached the feasibility stage"
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ training unit
def ba(logits, y):
    p = logits.argmax(1)
    return float(np.mean([(p[y == c] == c).float().mean().item() for c in (0, 1)]))


def train_unit(torch, M, name, init, mode, Xtr, ytr, Xte, yte, seed, dev, epochs=EPOCHS, want_state_hash=False):
    from torch import nn
    import torch.nn.functional as F
    T = Xtr.shape[-1]
    torch.manual_seed(seed)
    model = M.build_classifier(name, CK[name] if init == "pre" else None, M_CH, T, seed=seed).to(dev)
    g = torch.Generator().manual_seed(seed + 7)
    n = Xtr.shape[0]
    steps = (n + BS - 1) // BS
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    if mode == "LP":
        model.eval()
        with torch.no_grad():
            ftr = torch.cat([model.features(Xtr[i:i + 64]) for i in range(0, n, 64)]).float()
            fte = torch.cat([model.features(Xte[i:i + 64]) for i in range(0, Xte.shape[0], 64)]).float()
        torch.manual_seed(seed + 3)
        head = nn.Linear(ftr.shape[1], 2).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=WD)
        base = {"lr": 1e-3}
        groups = [{"lr_scale": 1.0}]
        for gr in opt.param_groups:
            gr["lr_scale"] = 1.0
        fwd_tr, fwd_te, params = (lambda idx: head(ftr[idx])), (lambda: head(fte)), head
        scaler = None
    else:
        spec = M.SPEC[name]
        groups = model.param_groups(weight_decay=WD, **({"layer_decay": spec["layer_decay"]} if spec["layer_decay"] else {}))
        opt = torch.optim.AdamW(groups, lr=spec["base_lr"])
        base = {"lr": spec["base_lr"]}
        fwd_tr, fwd_te, params = (lambda idx: model(Xtr[idx])), (lambda: torch.cat([model(Xte[i:i + 64]) for i in range(0, Xte.shape[0], 64)])), model
        scaler = torch.amp.GradScaler("cuda") if name == "ST-EEGFormer-small" else None   # ST forward is fp16-autocast upstream
    losses = []
    for ep in range(epochs):
        if mode == "FT":
            model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for it in range(steps):
            lr = M.cosine_lr(base["lr"], ep + it / steps, epochs, WARMUP)
            for gr in opt.param_groups:
                gr["lr"] = lr * gr.get("lr_scale", 1.0)
            idx = perm[it * BS:(it + 1) * BS]
            loss = F.cross_entropy(fwd_tr(idx).float(), ytr[idx], label_smoothing=LS)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot += float(loss.detach()) * idx.numel()
        losses.append(tot / n)
        if not np.isfinite(losses[-1]):
            raise FloatingPointError(f"non-finite loss at epoch {ep}")
    if mode == "FT":
        model.eval()
    with torch.no_grad():
        lte = fwd_te().float()
        ltr = torch.cat([fwd_tr(torch.arange(i, min(i + 64, n), device=dev)) for i in range(0, n, 64)]).float()
    torch.cuda.synchronize(dev)
    out = {"ba_test": ba(lte.cpu(), yte.cpu()), "acc_train": float((ltr.argmax(1) == ytr).float().mean()),
           "loss_first": losses[0], "loss_last": losses[-1], "wall_s": time.perf_counter() - t0,
           "peak_mib": torch.cuda.max_memory_allocated(dev) / 2 ** 20, "trainable_params": int(sum(p.numel() for p in params.parameters() if p.requires_grad)),
           "load_report": model.load_report if init == "pre" else None}
    if want_state_hash:
        h = hashlib.sha256()
        for k, v in sorted(params.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        out["state_sha256"] = h.hexdigest()
        out["test_logits_sha256"] = hashlib.sha256(lte.cpu().numpy().tobytes()).hexdigest()
    return out


M_CH = None


# ------------------------------------------------------------------ stages
def stage_S(torch, M, dev, log):
    rows = []
    for name, wins in (("ST-EEGFormer-small", {"w100": 128, "w125": 160, "nat4s": 512}), ("LaBraM-Base", {"w100": 200, "nat4s": 800}),
                       ("CBraMod", {"w100": 200, "nat4s": 800})):
        for w, T in wins.items():
            for init in ("pre", "rand"):
              try:
                torch.manual_seed(0)
                m = M.build_classifier(name, CK[name] if init == "pre" else None, M_CH, T, seed=0).to(dev)
                r = {"model": name, "window": w, "T": T, "init": init, "params": M.n_params(m), "load_report": m.load_report, "status": "ok"}
                for bs in (16, 64):
                    x = torch.randn(bs, 20, T, device=dev)
                    y = torch.randint(0, 2, (bs,), device=dev)
                    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
                    m.train()
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize(dev)
                    t0 = time.perf_counter()
                    for _ in range(10):
                        loss = torch.nn.functional.cross_entropy(m(x).float(), y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    torch.cuda.synchronize(dev)
                    m.eval()
                    with torch.no_grad():
                        o = m(x)
                    r[f"bs{bs}"] = {"train_step_ms": (time.perf_counter() - t0) / 10 * 1e3, "peak_mib": torch.cuda.max_memory_allocated(dev) / 2 ** 20,
                                    "out_shape": list(o.shape), "finite": bool(torch.isfinite(o).all())}
                rows.append(r)
                log(f"S {name} {w} {init} ok bs16 {r['bs16']['train_step_ms']:.1f} ms")
                del m
              except Exception as exc:          # recorded, never dropped
                rows.append({"model": name, "window": w, "T": T, "init": init, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                             "trace": traceback.format_exc()[-1500:]})
                log(f"S {name} {w} {init} FAILED {type(exc).__name__}: {exc}")
    return rows


def stage_A(torch, M, dev, log):
    rows = []
    pid = participants()[0]
    z = load(pid)
    for name in ("ST-EEGFormer-small", "LaBraM-Base", "CBraMod"):
        key = f"{PREFIX[name]}_w100"
        T = z[key].shape[-1]
        g = torch.Generator().manual_seed(99)
        Xs = torch.randn(48, 20, T, generator=g)
        Xs = torch.cat([Xs, Xs[:1]])                       # one exact duplicate row in the training set (pipeline check only)
        ys = torch.cat([torch.arange(48) % 2, torch.tensor([0])])
        sets = {"synthetic_with_duplicate_row": (Xs, ys, torch.randn(16, 20, T, generator=g), torch.arange(16) % 2),
                "fitting_runs_R1R3_to_R4_one_participant": tuple(torch.from_numpy(a) for a in (
                    z[key][z["run"] <= 3], z["y"][z["run"] <= 3], z[key][z["run"] == 4], z["y"][z["run"] == 4]))}
        for sname, (a, b, c, d) in sets.items():
            if name == "ST-EEGFormer-small" and sname.startswith("synthetic"):
                a = (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True)
                c = (c - c.mean(-1, keepdim=True)) / c.std(-1, keepdim=True)
            a, c = a.float().to(dev), c.float().to(dev)
            b, d = b.long().to(dev), d.long().to(dev)
            for mode in ("FT", "LP"):
                try:
                    runs = [train_unit(torch, M, name, "pre", mode, a, b, c, d, seed=11, dev=dev, epochs=10, want_state_hash=True) for _ in range(2)]
                    same = runs[0]["state_sha256"] == runs[1]["state_sha256"] and runs[0]["test_logits_sha256"] == runs[1]["test_logits_sha256"]
                    rows.append({"model": name, "data": sname, "mode": mode, "epochs": 10, "bit_identical": same, "status": "ok",
                                 "state_sha256": [r["state_sha256"] for r in runs], "test_logits_sha256": [r["test_logits_sha256"] for r in runs]})
                    log(f"A {name} {sname} {mode} identical={same}")
                except Exception as exc:
                    rows.append({"model": name, "data": sname, "mode": mode, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc()[-1500:]})
                    log(f"A {name} {sname} {mode} FAILED {type(exc).__name__}: {exc}")
    return rows


def stage_E(torch, M, dev, log, ids):
    st_vit_mod = M._st_vit()  # noqa: F841  (ensures upstream modules are importable before timing)
    from models.vendor_labram.random_masking import random_masking
    tok, tok_rep = M.lb_tokenizer(CK["LaBraM-VQ-NSP"])
    tok = tok.to(dev)
    ich = torch.tensor(M.lb_input_chans(M_CH), device=dev)
    cidx = torch.tensor(M.st_chan_idx(M_CH), device=dev)
    variants = [("pre", None)] + [("rand", s) for s in (0, 1, 2)]
    res = {"tokenizer_load_report": tok_rep, "ST-EEGFormer-small": {}, "LaBraM-Base": {}, "CBraMod": {}}
    for init, seed in variants:
        tag = init if seed is None else f"rand{seed}"
        if seed is not None:
            torch.manual_seed(seed)
        mae, rep_m = M.st_mae(CK["ST-EEGFormer-small"] if init == "pre" else None)
        if seed is not None:
            torch.manual_seed(seed)
        mem, rep_l = M.lb_mem(CK["LaBraM-Base"] if init == "pre" else None)
        if seed is not None:
            torch.manual_seed(seed)
        cbb, rep_c = M.cb_backbone(CK["CBraMod"] if init == "pre" else None)
        if seed is not None:
            torch.manual_seed(seed)
        tokv, _ = M.lb_tokenizer(CK["LaBraM-VQ-NSP"] if init == "pre" else None)
        mae, mem, cbb, tokv = mae.to(dev).eval(), mem.to(dev).eval(), cbb.to(dev).eval(), tokv.to(dev).eval()
        acc = {k: {} for k in ("st_w100", "st_w125", "st_nat", "lb_w100", "lb_nat", "cb_w100", "cb_nat")}
        tokrec = {"lb_w100": {}, "lb_nat": {}}
        codes_all = {"lb_w100": [], "lb_nat": []}
        for n in ids:
            z = load(n)
            with torch.no_grad():
                for key in ("st_w100", "st_w125", "st_nat"):
                    x = torch.from_numpy(z[key]).to(dev)
                    torch.manual_seed(10_000 + n)
                    _, pred, mask = mae(x, cidx.unsqueeze(0).expand(x.shape[0], -1), mask_ratio=0.75)   # the decoder needs (B, C)
                    tgt = mae.patch_embed.patchify_eeg(x).reshape(pred.shape)
                    m = mask.bool()
                    acc[key][n] = (float(((pred - tgt) ** 2).sum(-1)[m].sum()), float((tgt ** 2).sum(-1)[m].sum()))
                for key in ("lb_w100", "lb_nat"):
                    x = torch.from_numpy(z[key]).to(dev)
                    xa = x.view(x.shape[0], 20, -1, 200)
                    codes = tok.get_codebook_indices(xa, ich)
                    torch.manual_seed(20_000 + n)
                    bm = random_masking(xa.flatten(1, 2), mask_ratio=0.5).to(dev)
                    h = mem.student(xa, ich, bm, return_all_patch_tokens=True)[:, 1:]
                    lab = codes[bm]
                    vals = []
                    for head in (mem.lm_head, mem.student.lm_head):     # the public MEM forward uses the outer head
                        logits = head(h[bm]).float()
                        top = logits.topk(5, dim=1).indices
                        vals += [float(torch.nn.functional.cross_entropy(logits, lab, reduction="sum")), float((top[:, 0] == lab).sum()),
                                 float((top == lab[:, None]).any(1).sum())]
                    acc[key][n] = tuple(vals) + (int(lab.numel()),)
                    if key == "lb_nat":       # tokenizer's own objective (spectrum reconstruction); the upstream decoder
                        _, lg = tokv(x, ich)  # mis-reads the shape when there is one patch per channel, so 1-s inputs fail
                        tokrec[key][n] = (float(lg["val/rec_loss"]), float(lg["val/rec_angle_loss"]))
                    if init == "pre":
                        codes_all[key].append(codes.flatten().cpu())
                for key in ("cb_w100", "cb_nat"):
                    x = torch.from_numpy(z[key]).to(dev)
                    xs = x.view(x.shape[0], 20, -1, 200)
                    gen = torch.Generator().manual_seed(30_000 + n)
                    mk = (torch.rand(xs.shape[:3], generator=gen) < 0.5).long().to(dev)
                    y = cbb(xs, mask=mk)
                    mm = mk.to(y.dtype)
                    acc[key][n] = (float((((y - xs) ** 2).sum(-1) * mm).sum()), float(((xs ** 2).sum(-1) * mm).sum()))
        for key, d in acc.items():
            model = {"st": "ST-EEGFormer-small", "lb": "LaBraM-Base", "cb": "CBraMod"}[key[:2]]
            if key.startswith("lb"):
                per = np.array([[v[i] / v[6] for i in range(6)] for v in d.values()])
                tr = np.array(list(tokrec[key].values())) if tokrec[key] else None
                res[model].setdefault(key, {})[tag] = {
                    "outer_lm_head": {"ce_nats_mean": float(per[:, 0].mean()), "top1_mean": float(per[:, 1].mean()), "top5_mean": float(per[:, 2].mean())},
                    "student_lm_head": {"ce_nats_mean": float(per[:, 3].mean()), "top1_mean": float(per[:, 4].mean()), "top5_mean": float(per[:, 5].mean())},
                    "uniform_ce_nats": float(np.log(8192)), "n_participants": len(per), "masked_tokens": int(sum(v[6] for v in d.values())),
                    "tokenizer_self_reconstruction": ({"amplitude_mse_mean": float(tr[:, 0].mean()), "phase_mse_mean": float(tr[:, 1].mean()),
                                                       "note": "std-normalised Fourier amplitude / phase targets (VQ-NSP objective); 'pre' = released tokenizer, 'randK' = random tokenizer"}
                                                      if tr is not None else {"status": "not computable",
                                                                              "reason": "upstream VQ-NSP decoder (NeuralTransformer.forward_features) treats a one-patch-per-channel input as 20 time patches and fails (21 vs 401 tokens)"})}
            else:
                per = np.array([v[0] / v[1] for v in d.values()])
                res[model].setdefault(key, {})[tag] = {"nmse_mean": float(per.mean()), "nmse_median": float(np.median(per)),
                                                      "nmse_min": float(per.min()), "nmse_max": float(per.max()), "n_participants": len(per)}
        if init == "pre":
            for key, cl in codes_all.items():
                c = torch.cat(cl).numpy()
                cnt = np.bincount(c, minlength=8192).astype(float)
                p = cnt / cnt.sum()
                H = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                res["LaBraM-Base"].setdefault(key, {})["codebook"] = {
                    "distinct_codes_used": int((cnt > 0).sum()), "entropy_nats": H, "perplexity": float(np.exp(H)),
                    "majority_code_fraction": float(p.max()), "unigram_ce_nats": H,
                    "note": "unigram baseline = cohort code distribution (optimistic: fitted on the scored tokens)"}
            res["load_reports"] = {"st_mae": rep_m, "lb_mem": rep_l, "cb": rep_c}
        log(f"E {tag} done")
        del mae, mem, cbb, tokv
    return res


def stage_D(torch, M, dev, log, ids, done):
    from data_preparation import feasibility as D
    rows = []
    fh = open(RES / "decodability_units.jsonl", "a")
    for k, n in enumerate(ids):
        z = load(n)
        tr, te = z["run"] <= 3, z["run"] == 4
        ytr, yte = torch.from_numpy(z["y"][tr]).long().to(dev), torch.from_numpy(z["y"][te]).long().to(dev)
        for (name, w, init, mode) in DEC_CONFIGS:
            uid = f"{D.producer_id(n)}|{name}|{w}|{init}|{mode}"
            if uid in done:
                continue
            key = f"{PREFIX[name]}_{w}"
            Xtr, Xte = torch.from_numpy(z[key][tr]).to(dev), torch.from_numpy(z[key][te]).to(dev)
            seed = 1_000 + 7 * n
            try:
                r = train_unit(torch, M, name, init, mode, Xtr, ytr, Xte, yte, seed=seed, dev=dev)
                r.pop("load_report", None)
                r.update({"uid": uid, "status": "ok"})
            except Exception as exc:              # every failure is recorded, never dropped
                r = {"uid": uid, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
        log(f"D participant {k + 1}/{len(ids)}", progress=f"D {k + 1}/{len(ids)} participants")
    fh.close()
    return rows


def tslr_anchor(ids):
    from experiments.pmis_r10b.train import tslr_fit_predict
    out = []
    for n in ids:
        z = load(n)
        y = z["y"] + 1
        tr, te = np.where(z["run"] <= 3)[0], np.where(z["run"] == 4)[0]
        pc = tslr_fit_predict(z["cov_w125"].astype(np.float64), y, tr, te)
        pred_ok = pc > 0.5
        out.append(float(np.mean([pred_ok[y[te] == c].mean() for c in (1, 2)])))
    return out


def summarise(vals):
    v = np.asarray(vals, float)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None, "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()), "frac_above_0.5": float((v > 0.5).mean()),
            "se_mean": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None}


def main():
    global M_CH
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="SAED")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not set(a.stages) <= set("SAED") or not a.stages or a.limit < 0:
        ap.error("use one or more of SAED and a nonnegative participant limit")
    required = set(CK) if "E" in a.stages else set(CK) - {"LaBraM-VQ-NSP"}
    for model in sorted(required):
        if not CK[model].is_file() or file_sha256(CK[model]) != CHECKPOINTS[model]["sha256"]:
            raise ValueError(f"Provide the original verified checkpoint for {model}")
    RES.mkdir(parents=True, exist_ok=True)
    from data_preparation import feasibility as D
    M_CH = list(D.CHANNELS)
    ids = participants()
    if a.limit:
        ids = ids[:a.limit]
    result = json.loads(OUT.read_text()) if OUT.exists() else {"schema": "R15B_feasibility_v1", "human_status": "HUMAN_NOT_VERIFIED"}
    result.update({"scope": "technical feasibility only; synthetic data and Dreyer2023 fitting runs R1-R4; no contamination arm; not frozen",
                   "recipe": {"epochs": EPOCHS, "batch": BS, "warmup_epochs": WARMUP, "weight_decay": WD, "label_smoothing": LS, "mixup": "none",
                              "split": "train R1-R3 (120 trials), score R4 (40 trials), per participant", "read_out": "final epoch"}})
    logf = open(RES / "feasibility.log", "a")

    def log(msg, progress=None, lease=None):
        logf.write(f"{now()} {msg}\n")
        logf.flush()
        print(msg, flush=True)

    if "D" in a.stages:
        t = tslr_anchor(ids)
        result["tslr_anchor_cue_w125"] = summarise(t)
        log(f"TSLR anchor mean BA {np.mean(t):.4f}")
    from experiments import gpu_resources as gpucoord
    with gpucoord.JobLock("r15b-feasibility"):
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        dev = select_device()
        if torch.cuda.mem_get_info()[0] < 8000 * 2**20:
            raise RuntimeError("The assigned GPU needs 8000 MiB free for the recorded screen")
        from models import feasibility as M
        result["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                                 "deterministic_algorithms": True, "cudnn_benchmark": False}

        def log2(msg, progress=None):
            log(msg)
        if "S" in a.stages:
            result["S_synthetic"] = stage_S(torch, M, dev, log2)
            OUT.write_text(json.dumps(result, indent=1) + "\n")
        if "A" in a.stages:
            result["A_determinism"] = stage_A(torch, M, dev, log2)
            OUT.write_text(json.dumps(result, indent=1) + "\n")
        if "E" in a.stages:
            result["E_engagement"] = stage_E(torch, M, dev, log2, ids)
            OUT.write_text(json.dumps(result, indent=1) + "\n")
        if "D" in a.stages:
            done = set()
            p = RES / "decodability_units.jsonl"
            if p.exists():
                done = {json.loads(line)["uid"] for line in p.read_text().splitlines() if json.loads(line)["status"] == "ok"}
            stage_D(torch, M, dev, log2, ids, done)
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            last = {}
            for r in rows:
                last[r["uid"]] = r
            agg = {}
            for (name, w, init, mode) in DEC_CONFIGS:
                us = [r for u, r in last.items() if u.split("|")[1:] == [name, w, init, mode] and u.split("|")[0] in {D.producer_id(n) for n in ids}]
                ok = [r for r in us if r["status"] == "ok"]
                agg[f"{name}|{w}|{init}|{mode}"] = {
                    "units": len(us), "failed": len(us) - len(ok), "failures": sorted({r.get("error", "")[:200] for r in us if r["status"] != "ok"}),
                    "ba_test": summarise([r["ba_test"] for r in ok]) if ok else None,
                    "acc_train": summarise([r["acc_train"] for r in ok]) if ok else None,
                    "frac_train_acc_1": float(np.mean([r["acc_train"] == 1.0 for r in ok])) if ok else None,
                    "wall_s_mean": float(np.mean([r["wall_s"] for r in ok])) if ok else None,
                    "wall_s_total": float(np.sum([r["wall_s"] for r in ok])) if ok else None,
                    "peak_mib_max": float(np.max([r["peak_mib"] for r in ok])) if ok else None,
                    "trainable_params": ok[0]["trainable_params"] if ok else None}
            pairs = {}
            for (name, w, _, mode) in DEC_CONFIGS:
                k = f"{name}|{w}|{mode}"
                if k in pairs:
                    continue
                pre = {u.split("|")[0]: r["ba_test"] for u, r in last.items() if u.split("|")[1:] == [name, w, "pre", mode] and r["status"] == "ok"}
                rnd = {u.split("|")[0]: r["ba_test"] for u, r in last.items() if u.split("|")[1:] == [name, w, "rand", mode] and r["status"] == "ok"}
                common = sorted(set(pre) & set(rnd))
                if common:
                    diff = [pre[c] - rnd[c] for c in common]
                    pairs[k] = {"pretrained_minus_random_ba": summarise(diff)}
            result["D_decodability"] = {"configs": agg, "pretrained_vs_random_paired": pairs, "participants": len(ids)}
        result["finished_utc"] = now()
        OUT.write_text(json.dumps(result, indent=1) + "\n")
    log("all requested stages finished")


if __name__ == "__main__":
    main()
