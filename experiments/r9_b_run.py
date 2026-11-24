#!/usr/bin/env python3
"""Lee2019 CBraMod masked reconstruction, initialization and probe controls."""

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

ROOT = RUN_ROOT
import experiments.r8_de_run as r8  # noqa: E402
from experiments.pmis_g2 import g3  # noqa: E402
from data_preparation import hs_integrity as pv  # noqa: E402
from experiments.pmis_r7 import c6  # noqa: E402
from evaluation.r7_c6_analysis import d as d_ro, holm  # noqa: E402
from evaluation import r8_de_analysis as de_analysis
from models import models_de, probe, seq_train  # noqa: E402
from evaluation.r8_de_analysis import _S, d_plain  # noqa: E402
from models import models_r9, recon  # noqa: E402

PROTOCOL_ID = "R9-B-CBRAMOD-VALIDITY-LEE2019-v1"
PROTOCOL = CONFIG_ROOT / "R9_B_PROTOCOL.json"
PRIVATE = ROOT / "data" / "r9_b"
PUBLIC = ROOT / "results" / "R9_B_results.json"
PUBLIC_B1 = ROOT / "results" / "R9_B1_results.json"
X19 = PRIVATE / "lee2019_cbramod_tueglike19_200hz.npy"
X19_AUDIT = PRIVATE / "lee2019_tueglike19_build_audit.json"
R8_FT = ROOT / "data" / "r8_de" / "R8_DE_CBraMod_pair_records_PRIVATE.npz"
R8_PROBE = ROOT / "data" / "r8_de" / "R8_E1_pair_records_PRIVATE.npz"
VARIANTS = ("random20", "pre19")
GPU_MIN_AVAIL_GB, HOST_MIN_MB, MAX_WAIT_S = 3.0, 4096, 6 * 3600


def log(msg, name="R9_B_run.log"):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / name, "a") as fh:
        fh.write(line + "\n")


DATA = {"lee2019_meta.npz": r8.C3 / "lee2019_meta.npz", r8.EEG200: r8.E / r8.EEG200, "cbramod_checkpoint": r8.CKPT,
        X19.name: X19, "R8_DE_CBraMod_pair_records_PRIVATE.npz": R8_FT, "R8_E1_pair_records_PRIVATE.npz": R8_PROBE, **r8.PRIOR}


def verify():
    import torch
    proto = load_protocol(PROTOCOL)
    g3._stop(proto["status"] == "FROZEN" and proto["protocol_id"] == PROTOCOL_ID, "protocol not frozen")
    g3._stop(torch.__version__ == proto["environment"]["torch"], "torch version")
    return proto


def check_data(proto, names):
    for n in names:
        g3._stop(verify_recorded_input(DATA[n], proto["data_sha256"][n]), f"{n} SHA-256 differs (checked before load)")


def wait_gpu(torch, tag, need=GPU_MIN_AVAIL_GB):
    t0, last = time.time(), 0.0
    while True:
        torch.cuda.empty_cache()
        avail = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30
        st = g3.resource_state()
        if avail >= need and st["mem_available_mb"] >= HOST_MIN_MB and not (st["psi_full_avg10"] > 5.0):
            return
        g3._stop(time.time() - t0 < MAX_WAIT_S, "GPU/host unavailable for 6 h (not preempting)", "STOPPED_RESOURCE")
        if time.time() - last > 600:
            log(f"{tag} waiting for resources: gpu available {avail:.1f} GB", f"R9_B_{tag}.log")
            last = time.time()
        time.sleep(60)


def load_input(torch, dev, which):
    if which == "x20":
        x = np.load(DATA[r8.EEG200], allow_pickle=False)
        g3._stop(x.shape == (21600, 20, 800) and bool(np.all(np.isfinite(x))), "R8 20-channel input")
    else:
        x = np.load(X19, allow_pickle=False)
        g3._stop(x.shape == (21600, 19, 800) and bool(np.all(np.isfinite(x))), "19-channel input")
    return torch.as_tensor(np.ascontiguousarray(x / 100.0, dtype=np.float32), device=dev)


def b3_authorised():
    if not PUBLIC_B1.exists():
        return False
    return json.loads(PUBLIC_B1.read_text()).get("classification") == "MONTAGE_DEGRADED"


def cmd_recon():
    out = {"stage": "R9-B1", "protocol_id": PROTOCOL_ID}
    try:
        torch, dev = r8.setup_torch()
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        log(f"recon PORT_PROTOCOL_VERIFIED {out['port_manifest_sha256']}", "R9_B_recon.log")
        os.nice(10)
        check_data(proto, ["lee2019_meta.npz", r8.EEG200, X19.name, "cbramod_checkpoint"])
        subj = np.load(DATA["lee2019_meta.npz"])["subject"]
        wait_gpu(torch, "recon", 6.0)
        inputs = {"x20": load_input(torch, dev, "x20"), "x19": load_input(torch, dev, "x19")}
        perm = np.random.default_rng(9301).permutation(20)
        inputs["x20_permuted"] = inputs["x20"][:, torch.as_tensor(perm, device=dev)]
        masks = {"x20": recon.make_mask(21600, 20, 9302), "x19": recon.make_mask(21600, 19, 9302)}
        masks["x20_permuted"] = masks["x20"]
        models = {"pretrained": (r8.CKPT, 0), "random_seed0": (None, 0), "random_seed1": (None, 1), "random_seed2": (None, 2)}
        clean = {k: (v.abs().amax(dim=(1, 2)) < 1.0).cpu().numpy() for k, v in inputs.items()}
        errs, res = {}, {}
        for mname, (ck, seed) in models.items():
            bb = models_r9.make_backbone(ck, seed, keep_head=True).to(dev)
            for iname, X in inputs.items():
                e = recon.per_trial_errors(bb, X, masks[iname], dev)
                e2 = recon.per_trial_errors(bb, X[:640], masks[iname][:640], dev)
                g3._stop(all(np.array_equal(e[k][:640], e2[k]) for k in e), f"{mname}/{iname}: reconstruction not reproducible")
                errs[f"{mname}__{iname}"] = e
                res[f"{mname}__{iname}"] = {"masked_all": recon.nmse(e, subj), "masked_clean": recon.nmse(e, subj, clean[iname]),
                                            "unmasked_all": recon.nmse(e, subj, which="u")}
            del bb
            torch.cuda.empty_cache()
        np.savez(PRIVATE / "R9_B1_errors_PRIVATE.npz", **{f"{k}__{q}": v for k, e in errs.items() for q, v in e.items()})
        g = lambda m, i, w: res[f"{m}__{i}"][w]["mean_over_participants"]  # noqa: E731
        rand_min = min(g(f"random_seed{s}", "x19", "masked_clean") for s in (0, 1, 2))
        cls, v1, v2 = recon.classify(g("pretrained", "x19", "masked_clean"), rand_min, g("pretrained", "x20", "masked_all"),
                                     g("pretrained", "x19", "masked_all"))
        out.update({"status": "COMPLETED", "classification": cls, "V1": v1, "V2": v2, "nmse": res,
                    "clean_fraction": {k: float(v.mean()) for k, v in clean.items()}, "channel_permutation_seed": 9301})
        code = 0
    except g3.G3Stop as e:
        out.update({"status": e.status, "stop_message": e.message})
        code = 2
    except Exception as e:
        out.update({"status": "BLOCKED", "stop_message": f"{type(e).__name__}: {e}"})
        log("recon STOP BLOCKED: " + traceback.format_exc(), "R9_B_recon.log")
        code = 2
    PUBLIC_B1.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_B1.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(out["status"], out.get("classification"))
    return code


def cmd_probe(variant):
    tag = f"probe_{variant}"
    try:
        torch, dev = r8.setup_torch()
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R9_B_{tag}.log")
        g3._stop(variant == "random20" or b3_authorised(), "B3 not authorised by the frozen B1 rule")
        os.nice(10)
        check_data(proto, ["lee2019_meta.npz", r8.EEG200 if variant == "random20" else X19.name, "cbramod_checkpoint"])
        meta, tasks, c6c, sub, subc = r8.build()
        g3._stop(reference_matches(subc, proto["plan"]["subset_commitment"]), "plan commitment differs")
        wait_gpu(torch, tag, 4.0)
        X = load_input(torch, dev, "x20" if variant == "random20" else "x19")
        C = X.shape[1]
        bb = models_r9.make_backbone(None if variant == "random20" else r8.CKPT, 0, keep_head=False).to(dev)
        with torch.no_grad():
            emb = torch.cat([bb(X[i:i + 64].view(-1, C, 4, 200)).mean(dim=(1, 2)) for i in range(0, X.shape[0], 64)])
            emb2 = torch.cat([bb(X[i:i + 64].view(-1, C, 4, 200)).mean(dim=(1, 2)) for i in range(0, 640, 64)])
        F = emb.double().cpu().numpy()
        g3._stop(bool(np.all(np.isfinite(F))), "non-finite embedding")
        g3._stop(bool(torch.equal(emb[:640], emb2)), "embedding not reproducible")
        y01 = meta["label"].astype(np.int64) - 1
        recs, st = [], []
        for t in tasks:
            r, s_ = probe.run_task(t, F, y01)
            recs += r
            st.append(s_)
        keys = sorted(recs[0])
        np.savez(PRIVATE / f"R9_B_probe_{variant}_pair_records_PRIVATE.npz", **{k: np.array([float(r[k]) for r in recs]) for k in keys},
                 n_fits=np.array(sum(x["n_fits"] for x in st)), n_stagnated=np.array(sum(x["n_stagnated"] for x in st)),
                 max_abs_grad=np.array(max(x["max_abs_grad"] for x in st)), newton_max_iter=np.array(max(x["newton_max_iter"] for x in st)))
        log(f"{tag} DONE", f"R9_B_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R9_B_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R9_B_{tag}.log")
        return 2


def cmd_ft(variant, k, n):
    tag = f"ft_{variant}_shard{k}of{n}"
    try:
        torch, dev = r8.setup_torch()
        proto = verify()
        log(f"{tag} PORT_PROTOCOL_VERIFIED {port_identity()}", f"R9_B_{tag}.log")
        g3._stop(variant == "random20" or b3_authorised(), "B3 not authorised by the frozen B1 rule")
        os.nice(10)
        check_data(proto, ["lee2019_meta.npz", r8.EEG200 if variant == "random20" else X19.name, "cbramod_checkpoint"])
        meta, tasks, c6c, sub, subc = r8.build()
        g3._stop(reference_matches(c6c, proto["plan"]["c6_commitment"]) and reference_matches(subc, proto["plan"]["subset_commitment"]), "plan commitment differs")
        X = load_input(torch, dev, "x20" if variant == "random20" else "x19")
        y01 = torch.as_tensor(meta["label"].astype(np.int64) - 1, device=dev)
        mk = (lambda: models_r9.make_cbramod(None, 20)) if variant == "random20" else (lambda: models_r9.make_cbramod(r8.CKPT, 19))
        gdir = PRIVATE / "groups"
        gdir.mkdir(parents=True, exist_ok=True)
        mine = r8.de_plan.shard(sub["groups"], k, n)
        t0 = time.time()
        for i, g in enumerate(mine):
            f = gdir / f"{variant}_{g['subject']}_{g['fold']}.npz"
            if f.exists() and str(np.load(f)["commitment"]) == subc:
                continue
            res = None
            for attempt in range(7):
                wait_gpu(torch, tag)
                try:
                    res = seq_train.run_group(mk, models_de.cbramod_opt, g, X, y01, dev, models_de.CBM_EPOCHS, models_de.CBM_BS,
                                              label_smoothing=0.1, clip=1.0)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    log(f"{tag} group {i}: out of memory (attempt {attempt + 1}); identical recompute", f"R9_B_{tag}.log")
                    time.sleep(120)
            g3._stop(res is not None, "out of memory 7 times", "STOPPED_RESOURCE")
            np.savez(f, commitment=np.array(subc), **res)
            if (i + 1) % 6 == 0 or i + 1 == len(mine):
                log(f"{tag} group {i + 1}/{len(mine)} {time.time() - t0:.0f} s", f"R9_B_{tag}.log")
        log(f"{tag} DONE", f"R9_B_{tag}.log")
        return 0
    except g3.G3Stop as e:
        log(f"{tag} STOP {e.status}: {e.message}", f"R9_B_{tag}.log")
        return 2
    except Exception:
        log(f"{tag} STOP BLOCKED: " + traceback.format_exc(), f"R9_B_{tag}.log")
        return 2


def _by_k(path):
    return {int(r["k"]): r for r in r8._records(path)}


def _ft_records(variant, sub, subc):
    results, missing = {}, 0
    for g in sub["groups"]:
        f = PRIVATE / "groups" / f"{variant}_{g['subject']}_{g['fold']}.npz"
        if not (f.exists() and str(np.load(f)["commitment"]) == subc):
            missing += 1
            continue
        zz = np.load(f)
        results[(variant, g["subject"], g["fold"])] = {kk: zz[kk] for kk in zz.files if kk != "commitment"}
    g3._stop(missing == 0, f"{variant}: {missing} groups missing")
    recs, aa = c6.pair_records(sub, results, variant)
    g3._stop(max(aa) == 0.0, f"{variant} A/A determinism check failed: {max(aa)}")
    return recs, aa


def _family(fam):
    adj = holm({k: v["sign_flip_p_monte_carlo"] for k, v in fam.items()})
    for k in fam:
        fam[k]["holm_p_family"] = adj[k]
    return fam


def cmd_analyse():
    out = {"stage": "R9-B", "protocol_id": PROTOCOL_ID}
    try:
        proto = verify()
        out["port_manifest_sha256"] = port_identity()
        check_data(proto, ["R8_DE_CBraMod_pair_records_PRIVATE.npz", "R8_E1_pair_records_PRIVATE.npz", *r8.PRIOR])
        b1 = json.loads(PUBLIC_B1.read_text())
        g3._stop(b1.get("status") == "COMPLETED", "B1 not completed")
        meta, tasks, c6c, sub, subc = r8.build()
        g3._stop(reference_matches(subc, proto["plan"]["subset_commitment"]), "plan commitment differs")
        tslr = {k: d_plain(r) for k, r in _by_k(r8.PRIOR["C3a"]).items()}
        eeg3 = {k: d_plain(r) for k, r in _by_k(r8.PRIOR["C3b"]).items()}
        c6r = {m: _by_k(r8.PRIOR[f"C6_{m}"]) for m in ("EEGNet", "Shallow")}
        pre = _by_k(R8_FT)
        recs, aa = _ft_records("random20", sub, subc)
        np.savez(PRIVATE / "R9_B_random20_pair_records_PRIVATE.npz", **{kk: np.array([float(r[kk]) for r in recs]) for kk in sorted(recs[0])})
        pp = lambda r, ro: 0.5 * (r[f"{ro}_pAB"] + r[f"{ro}_pBA"])  # noqa: E731
        res = {"B1": {"classification": b1["classification"], "V1": b1["V1"], "V2": b1["V2"]}}
        res["B2"] = {"family_holm": _family({
            "B2_H1_random_final_d": _S(recs, lambda r: d_ro(r, "final")),
            "B2_H2_pre_minus_random_final_d": _S(recs, lambda r: d_ro(pre[int(r["k"])], "final") - d_ro(r, "final")),
            "B2_H3_pre_minus_random_es_d": _S(recs, lambda r: d_ro(pre[int(r["k"])], "es") - d_ro(r, "es")),
            "B2_H4_pre_minus_random_final_partner_prob": _S(recs, lambda r: pp(pre[int(r["k"])], "final") - pp(r, "final"))}),
            "random_init_standard_analysis": de_analysis.analyse_deep("CBraModRandom", recs, aa, c6r, tslr),
            "descriptive": {"pretrained_final_d_same_pairs": _S(recs, lambda r: d_ro(pre[int(r["k"])], "final"), test=False),
                            "pretrained_es_d_same_pairs": _S(recs, lambda r: d_ro(pre[int(r["k"])], "es"), test=False),
                            "pre_minus_random_es_partner_prob": _S(recs, lambda r: pp(pre[int(r["k"])], "es") - pp(r, "es"), test=False)}}
        prr = r8._records(PRIVATE / "R9_B_probe_random20_pair_records_PRIVATE.npz")
        ppre = _by_k(R8_PROBE)
        pq = lambda r: 0.5 * (r["pAB"] + r["pBA"])  # noqa: E731
        res["B2_probe"] = {"family_holm": _family({
            "P_H1_pre_minus_random_probe_d": _S(prr, lambda r: d_plain(ppre[int(r["k"])]) - d_plain(r)),
            "P_H2_pre_minus_random_probe_partner_prob": _S(prr, lambda r: pq(ppre[int(r["k"])]) - pq(r))}),
            "random_encoder_probe_standard_analysis": de_analysis.analyse_probe(prr, tslr, eeg3, {int(g["pairs"][0]["k"]) for g in sub["groups"]})}
        zc = np.load(PRIVATE / "R9_B_probe_random20_pair_records_PRIVATE.npz")
        res["B2_probe"]["convergence"] = {"n_fits": int(zc["n_fits"]), "n_stagnated_at_float_floor": int(zc["n_stagnated"]),
                                          "max_abs_grad": float(zc["max_abs_grad"])}
        if b1["classification"] == "MONTAGE_DEGRADED":
            r19, aa19 = _ft_records("pre19", sub, subc)
            p19 = r8._records(PRIVATE / "R9_B_probe_pre19_pair_records_PRIVATE.npz")
            res["B3"] = {"family_holm": _family({
                "B3_H1_pre19_final_d": _S(r19, lambda r: d_ro(r, "final")),
                "B3_H2_pre19_minus_pre20_final_d": _S(r19, lambda r: d_ro(r, "final") - d_ro(pre[int(r["k"])], "final")),
                "B3_H3_pre19_minus_pre20_probe_d": _S(p19, lambda r: d_plain(r) - d_plain(ppre[int(r["k"])]))}),
                "pre19_standard_analysis": de_analysis.analyse_deep("CBraMod19", r19, aa19, c6r, tslr),
                "pre19_probe_standard_analysis": de_analysis.analyse_probe(p19, tslr, eeg3, {int(g["pairs"][0]["k"]) for g in sub["groups"]})}
        else:
            res["B3"] = {"status": f"not run: frozen B1 rule classified the R8 input as {b1['classification']}"}
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
    for flag in ("--plan-only", "--execute-recon", "--execute-probe", "--execute-ft", "--analyse"):
        m.add_argument(flag, action="store_true")
    ap.add_argument("--variant", choices=VARIANTS, default="random20")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.plan_only:
        print(json.dumps(r8.plan_summary()))
        sys.exit(0)
    if a.execute_recon:
        sys.exit(cmd_recon())
    if a.execute_probe:
        sys.exit(cmd_probe(a.variant))
    if a.execute_ft:
        kk, nn = (int(v) for v in a.shard.split("/"))
        sys.exit(cmd_ft(a.variant, kk, nn))
    sys.exit(cmd_analyse())
