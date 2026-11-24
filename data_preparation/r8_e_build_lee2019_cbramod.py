#!/usr/bin/env python3
"""Build Lee2019 trial-local CBraMod inputs on the 20-channel montage."""

import hashlib
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from concurrent.futures import ProcessPoolExecutor

from scipy.signal import butter, resample, sosfilt

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
CHANNELS = ("FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6")
FS, LEAD, MI = 1000, 3000, 4000
OUT = ROOT / "data" / "r8_e"
PARTS = OUT / "parts_cbramod_v2"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def mem_wait():
    while True:
        avail = int([ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0].split()[1]) // 1024
        full = float(open("/proc/pressure/memory").read().split("\n")[1].split()[1].split("=")[1])
        if avail >= 8192 and full <= 1.0:
            return
        time.sleep(60)


def one_subject(subj):
    src = input_path("lee2019_mi")
    audit3 = json.loads((ROOT / "data" / "r6_c3" / "lee2019_build_audit.json").read_text())
    sos = butter(5, [0.3 / (FS / 2), 40.0 / (FS / 2)], btype="band", output="sos")
    f = PARTS / f"subj{subj:02d}.npz"
    if f.exists():
        return subj
    eeg, meta, scale = [], [], []
    for sess in (1, 2):
        name = f"sess{sess:02d}_subj{subj:02d}_EEG_MI.mat"
        mem_wait()
        if sha_file(src / name) != audit3["raw_file_sha256"][name]:
            raise SystemExit(f"STOP: raw file SHA-256 differs from the C3 build audit: {name}")
        for part, run_off in (("EEG_MI_train", 0), ("EEG_MI_test", 1)):
            s = sio.loadmat(src / name, squeeze_me=True, struct_as_record=False, variable_names=[part])[part]
            chan = [str(c) for c in s.chan]
            assert int(s.fs) == FS
            x = np.asarray(s.x, dtype=np.float64)[:, [chan.index(c) for c in CHANNELS]]
            onsets = np.asarray(s.t, dtype=np.int64).ravel()
            labels = np.asarray(s.y_dec, dtype=np.int64).ravel()
            del s
            for pos, (t, lab) in enumerate(zip(onsets, labels)):
                w = x[t - LEAD:t + MI].T
                w = w - w.mean(axis=0, keepdims=True)  # common average reference
                w = sosfilt(sos, w, axis=-1)[:, LEAD:]
                eeg.append(resample(w, 800, axis=-1).astype(np.float32))
                scale.append(float(np.median(np.abs(x[t:t + MI]))))
                meta.append((subj, sess, 2 * (sess - 1) + run_off, pos, int(lab), int(t)))
            del x
    e = np.stack(eeg)
    q = float(np.percentile(np.abs(e) / 100.0, 99))
    if not (0.01 < q < 10.0):
        raise SystemExit(f"STOP: implausible amplitude for subject {subj}: p99 |x|/100 = {q}")
    np.savez(PARTS / f"subj{subj:02d}.tmp.npz", eeg=e, meta=np.array(meta, dtype=np.int64), scale=np.array(scale))
    (PARTS / f"subj{subj:02d}.tmp.npz").rename(f)
    return subj


def main():
    argparse.ArgumentParser(description="Build Lee2019 CBraMod motor-montage features.").parse_args()
    audit3 = json.loads((ROOT / "data" / "r6_c3" / "lee2019_build_audit.json").read_text())
    assert tuple(audit3["channels"]) == CHANNELS
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for subj in ex.map(one_subject, range(1, 55)):
            print(f"subject {subj} done {time.time() - t0:.0f} s", flush=True)
    parts = [np.load(PARTS / f"subj{s:02d}.npz") for s in range(1, 55)]
    meta = np.concatenate([p["meta"] for p in parts])
    m3 = np.load(ROOT / "data" / "r6_c3" / "lee2019_meta.npz")
    for j, k in enumerate(("subject", "session", "run", "pos", "label", "onset")):
        if not np.array_equal(meta[:, j], m3[k]):
            raise SystemExit(f"STOP: trial order differs from the C3 cache ({k})")
    eeg = np.concatenate([p["eeg"] for p in parts])
    assert eeg.shape == (21600, 20, 800) and np.all(np.isfinite(eeg))
    q99 = float(np.percentile(np.abs(eeg) / 100.0, 99))
    if not (0.01 < q99 < 10.0):
        raise SystemExit(f"STOP: implausible processed amplitude p99 |x|/100 = {q99}")
    np.save(OUT / "lee2019_cbramod_200hz.npy", eeg)
    sc = np.concatenate([p["scale"] for p in parts])
    audit = {"build_version": "v2 (second-order sections)", "n_trials": int(eeg.shape[0]), "shape": list(eeg.shape), "channels": list(CHANNELS),
             "trial_order_equals_C3_meta": True, "raw_files_verified_against_C3_audit": 108,
             "raw_median_abs_amplitude_uV": {"median": float(np.median(sc)), "p5": float(np.percentile(sc, 5)), "p95": float(np.percentile(sc, 95))},
             "processed_abs_p99_over_100": q99,
             "sha256": {"lee2019_cbramod_200hz.npy": sha_file(OUT / "lee2019_cbramod_200hz.npy")}, "build_seconds": time.time() - t0}
    (OUT / "lee2019_cbramod_build_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
