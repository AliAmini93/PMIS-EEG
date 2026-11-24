#!/usr/bin/env python3
"""Build the additional Lee2019 offsets for the sliding-window experiment."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, json_sha256

ROOT = RUN_ROOT
from models.vendor_r4.r4_models import causal_bandpass, oas_cov  # noqa: E402
from data_preparation.r6_c3_build_lee2019 import CHANNELS, FS, WLEN, sha_file, wait_mem  # noqa: E402

OFFSETS_MS = (750, 1250, 1750)
OUT = ROOT / "data" / "r10_b"
PARTS = OUT / "parts"
COV = OUT / "lee2019_r10b_covs_8_30.npy"
EEG = OUT / "lee2019_r10b_eegnet_4_40_128hz.npy"


def subject(src, subj):
    wc, we, onsets_all, shas = [], [], [], {}
    for sess in (1, 2):
        name = f"sess{sess:02d}_subj{subj:02d}_EEG_MI.mat"
        wait_mem()
        shas[name] = sha_file(src / name)
        for part in ("EEG_MI_train", "EEG_MI_test"):
            s = sio.loadmat(src / name, squeeze_me=True, struct_as_record=False, variable_names=[part])[part]
            assert int(s.fs) == FS
            chan = [str(c) for c in s.chan]
            x = np.asarray(s.x, dtype=np.float64)[:, [chan.index(c) for c in CHANNELS]]
            onsets = np.asarray(s.t, dtype=np.int64).ravel()
            del s
            for t in onsets:
                c_, e_ = [], []
                for off in OFFSETS_MS:
                    seg = x[t + off:t + off + WLEN].T
                    assert seg.shape == (20, WLEN)
                    c_.append(oas_cov(causal_bandpass(seg, 8.0, 30.0, FS).T))
                    e_.append(resample_poly(causal_bandpass(seg, 4.0, 40.0, FS), 16, 125, axis=1).astype(np.float32))
                wc.append(np.stack(c_))
                we.append(np.stack(e_))
                onsets_all.append(int(t))
            del x
    return {"wcov": np.stack(wc), "weeg": np.stack(we), "onset": np.array(onsets_all), "shas": json.dumps(shas)}


def main():
    argparse.ArgumentParser(description="Build the recorded Lee2019 sliding-window features.").parse_args()
    src = input_path("lee2019_mi")
    meta = np.load(ROOT / "data" / "r6_c3" / "lee2019_meta.npz")
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for subj in range(1, 55):
        f = PARTS / f"subj{subj:02d}.npz"
        if f.exists():
            continue
        d = subject(src, subj)
        assert np.array_equal(d["onset"], meta["onset"][meta["subject"] == subj])
        np.savez(PARTS / f"subj{subj:02d}.tmp.npz", **d)
        (PARTS / f"subj{subj:02d}.tmp.npz").rename(f)
        print(f"subject {subj} done {time.time() - t0:.0f} s", flush=True)
    parts = [np.load(PARTS / f"subj{s:02d}.npz") for s in range(1, 55)]
    for path, key, dt_, shape in ((COV, "wcov", np.float64, (21600, 3, 20, 20)), (EEG, "weeg", np.float32, (21600, 3, 20, 256))):
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=dt_, shape=shape)
        i = 0
        for p in parts:
            a = p[key]
            assert np.all(np.isfinite(a))
            if key == "wcov":
                assert np.linalg.eigvalsh(a.reshape(-1, 20, 20)).min() > 0
            mm[i:i + a.shape[0]] = a
            i += a.shape[0]
        assert i == 21600
        mm.flush()
        del mm
    shas = {}
    for p in parts:
        shas.update(json.loads(str(p["shas"])))
    expected = load_protocol(CONFIG_ROOT / "R6_C3a_PROTOCOL.json")["raw_manifest"]
    if len(shas) != expected["count"] or not reference_matches(json_sha256(shas), expected["sha256"]):
        raise ValueError("Lee2019 window parts used different source files")
    audit = {"offsets_ms": list(OFFSETS_MS), "raw_file_sha256": shas,
             "feature_sha256": {COV.name: sha_file(COV), EEG.name: sha_file(EEG)}, "build_seconds_this_invocation": time.time() - t0}
    (OUT / "lee2019_r10b_build_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit["feature_sha256"]))


if __name__ == "__main__":
    main()
