#!/usr/bin/env python3
"""Build Lee2019 pretraining-format CBraMod inputs on the 19-channel montage."""

import hashlib
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio

from experiments.runtime import CONFIG_ROOT, RUN_ROOT, load_protocol, input_path, port_identity

ROOT = RUN_ROOT
TUEG_ORDER = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz")
FS, FS_OUT, MI = 1000, 200, 800
OUT = ROOT / "data" / "r9_b"
PARTS = OUT / "parts_tueglike19"


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


def process_run(x_uv):
    import mne
    mne.set_log_level("ERROR")
    x = x_uv - x_uv.mean(axis=1, keepdims=True)  # (19, n) common average reference over the 19 channels
    raw = mne.io.RawArray(x * 1e-6, mne.create_info(list(TUEG_ORDER), FS, ch_types="eeg"))
    raw.resample(FS_OUT)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter(60)
    return raw.get_data() * 1e6


def one_subject(subj):
    src = input_path("lee2019_mi")
    audit3 = json.loads((ROOT / "data" / "r6_c3" / "lee2019_build_audit.json").read_text())
    f = PARTS / f"subj{subj:02d}.npz"
    if f.exists():
        return subj
    eeg, meta = [], []
    for sess in (1, 2):
        name = f"sess{sess:02d}_subj{subj:02d}_EEG_MI.mat"
        mem_wait()
        if sha_file(src / name) != audit3["raw_file_sha256"][name]:
            raise SystemExit(f"STOP: raw file SHA-256 differs from the C3 build audit: {name}")
        for part, run_off in (("EEG_MI_train", 0), ("EEG_MI_test", 1)):
            s = sio.loadmat(src / name, squeeze_me=True, struct_as_record=False, variable_names=[part])[part]
            chan = [str(c) for c in s.chan]
            assert int(s.fs) == FS
            x = np.asarray(s.x, dtype=np.float64)[:, [chan.index(c) for c in TUEG_ORDER]].T
            onsets = np.asarray(s.t, dtype=np.int64).ravel()
            labels = np.asarray(s.y_dec, dtype=np.int64).ravel()
            del s
            y = process_run(x)
            del x
            for pos, (t, lab) in enumerate(zip(onsets, labels)):
                a = int(round(t * FS_OUT / FS))
                w = y[:, a:a + MI]
                if w.shape[1] != MI:
                    raise SystemExit(f"STOP: window outside the run (subject {subj}, session {sess}, trial {pos})")
                eeg.append(w.astype(np.float32))
                meta.append((subj, sess, 2 * (sess - 1) + run_off, pos, int(lab), int(t)))
    e = np.stack(eeg)
    q = float(np.percentile(np.abs(e) / 100.0, 99))
    if not (0.01 < q < 10.0):
        raise SystemExit(f"STOP: implausible amplitude for subject {subj}: p99 |x|/100 = {q}")
    PARTS.mkdir(parents=True, exist_ok=True)
    np.savez(PARTS / f"subj{subj:02d}.tmp.npz", eeg=e, meta=np.array(meta, dtype=np.int64))
    (PARTS / f"subj{subj:02d}.tmp.npz").rename(f)
    return subj


def main(workers):
    PARTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for subj in ex.map(one_subject, range(1, 55)):
            print(f"subject {subj} done {time.time() - t0:.0f} s", flush=True)
    parts = [np.load(PARTS / f"subj{s:02d}.npz") for s in range(1, 55)]
    meta = np.concatenate([p["meta"] for p in parts])
    m3 = np.load(ROOT / "data" / "r6_c3" / "lee2019_meta.npz")
    for j, k in enumerate(("subject", "session", "run", "pos", "label", "onset")):
        if not np.array_equal(meta[:, j], m3[k]):
            raise SystemExit(f"STOP: trial order differs from the C3 cache ({k})")
    eeg = np.concatenate([p["eeg"] for p in parts])
    assert eeg.shape == (21600, 19, 800) and np.all(np.isfinite(eeg))
    q99 = float(np.percentile(np.abs(eeg) / 100.0, 99))
    if not (0.01 < q99 < 10.0):
        raise SystemExit(f"STOP: implausible processed amplitude p99 |x|/100 = {q99}")
    np.save(OUT / "lee2019_cbramod_tueglike19_200hz.npy", eeg)
    mx = np.abs(eeg).max(axis=(1, 2))
    audit = {"build": "R9-B TUEG-like 19-channel pretraining-format input", "n_trials": int(eeg.shape[0]), "shape": list(eeg.shape),
             "channels_lee2019_names": list(TUEG_ORDER), "trial_order_equals_C3_meta": True, "raw_files_verified_against_C3_audit": 108,
             "processed_abs_p99_over_100": q99, "fraction_trials_max_abs_below_100uV_pretraining_rule": float(np.mean(mx < 100.0)),
             "sha256": {"lee2019_cbramod_tueglike19_200hz.npy": sha_file(OUT / "lee2019_cbramod_tueglike19_200hz.npy")},
             "build_seconds": time.time() - t0}
    (OUT / "lee2019_tueglike19_build_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Lee2019 CBraMod pretraining-format features.")
    ap.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    main(ap.parse_args().workers)
