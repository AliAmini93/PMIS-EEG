#!/usr/bin/env python3
"""Build Lee2019 4-second features and coarse-window features from the original MATLAB recordings."""

import hashlib
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

CHANNELS = ("FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6")
FS, LEAD, MI = 1000, 3000, 4000
WINDOWS = (("V0.0", 0), ("V0.5", 500), ("V1.0", 1000), ("V1.5", 1500), ("V2.0", 2000), ("P", -2500))
WLEN = 2000
OUT3, OUT4 = ROOT / "data" / "r6_c3", ROOT / "data" / "r6_c4"
PARTS = OUT3 / "parts"
MIN_AVAIL_MB = 8192


def mem_ok():
    avail = int([ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0].split()[1]) // 1024
    full = float(open("/proc/pressure/memory").read().split("\n")[1].split()[1].split("=")[1])
    return avail >= MIN_AVAIL_MB and full <= 1.0, avail, full


def wait_mem():
    while True:
        ok, avail, full = mem_ok()
        if ok:
            return
        print(f"waiting for host memory: avail {avail} MB, psi full {full}", flush=True)
        time.sleep(60)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def subject(src: Path, subj: int) -> dict:
    covs, eeg, wcov, weeg, meta, shas, rawh = [], [], [], [], [], {}, []
    for sess in (1, 2):
        name = f"sess{sess:02d}_subj{subj:02d}_EEG_MI.mat"
        wait_mem()
        shas[name] = sha_file(src / name)
        for part, run_off in (("EEG_MI_train", 0), ("EEG_MI_test", 1)):
            s = sio.loadmat(src / name, squeeze_me=True, struct_as_record=False, variable_names=[part])[part]
            chan = [str(c) for c in s.chan]
            assert int(s.fs) == FS
            x = np.asarray(s.x, dtype=np.float64)[:, [chan.index(c) for c in CHANNELS]]
            onsets = np.asarray(s.t, dtype=np.int64).ravel()
            labels = np.asarray(s.y_dec, dtype=np.int64).ravel()
            del s
            assert onsets.size == labels.size == 100 and set(labels.tolist()) == {1, 2}
            for pos, (t, lab) in enumerate(zip(onsets, labels)):
                w = x[t - LEAD:t + MI].T
                assert w.shape == (20, LEAD + MI), (name, part, pos)
                rawh.append(hashlib.sha256(np.ascontiguousarray(x[t:t + MI]).tobytes()).hexdigest())
                covs.append(oas_cov(causal_bandpass(w, 8.0, 30.0, FS)[:, LEAD:].T))
                eeg.append(resample_poly(causal_bandpass(w, 4.0, 40.0, FS), 16, 125, axis=1)[:, -512:].astype(np.float32))
                wc, we = [], []
                for _, off in WINDOWS:
                    seg = x[t + off:t + off + WLEN].T
                    assert seg.shape == (20, WLEN)
                    wc.append(oas_cov(causal_bandpass(seg, 8.0, 30.0, FS).T))
                    we.append(resample_poly(causal_bandpass(seg, 4.0, 40.0, FS), 16, 125, axis=1).astype(np.float32))
                wcov.append(np.stack(wc))
                weeg.append(np.stack(we))
                meta.append((subj, sess, 2 * (sess - 1) + run_off, pos, int(lab), int(t)))
            del x
    return {"covs": np.stack(covs), "eeg": np.stack(eeg), "wcov": np.stack(wcov), "weeg": np.stack(weeg),
            "meta": np.array(meta, dtype=np.int64), "shas": json.dumps(shas), "rawh": np.array(rawh)}


def raw_manifest(src):
    """Verify the full data release using a commitment to the original file manifest."""
    protocol = load_protocol(CONFIG_ROOT / "R6_C3a_PROTOCOL.json")
    expected = protocol["raw_manifest"]
    paths = sorted(src.glob("sess??_subj??_EEG_MI.mat"))
    if len(paths) != expected["count"]:
        raise ValueError("Lee2019 input must contain the 108 original MI files")
    checksums = {p.name: sha_file(p) for p in paths}
    if not reference_matches(json_sha256(checksums), expected["sha256"]):
        raise ValueError("Lee2019 archive contents differ from the frozen raw-input manifest")
    return checksums


def main():
    argparse.ArgumentParser(description="Build the recorded Lee2019 covariance and EEG features.").parse_args()
    src = input_path("lee2019_mi")
    expected_raw = raw_manifest(src)
    PARTS.mkdir(parents=True, exist_ok=True)
    OUT4.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for subj in range(1, 55):
        f = PARTS / f"subj{subj:02d}.npz"
        if f.exists():
            continue
        d = subject(src, subj)
        np.savez(PARTS / f"subj{subj:02d}.tmp.npz", **d)
        (PARTS / f"subj{subj:02d}.tmp.npz").rename(f)
        print(f"subject {subj} done {time.time() - t0:.0f} s", flush=True)
    parts = [np.load(PARTS / f"subj{s:02d}.npz") for s in range(1, 55)]  # lazy: arrays read on access
    meta = np.concatenate([p["meta"] for p in parts])
    assert meta.shape[0] == 21600
    shas = {}
    for p in parts:
        shas.update(json.loads(str(p["shas"])))
    if shas != expected_raw:
        raise ValueError("Lee2019 cached parts used different source files")
    rawh = np.concatenate([p["rawh"] for p in parts])
    specs = ((OUT3 / "lee2019_covs_8_30.npy", "covs", np.float64, (21600, 20, 20)),
             (OUT3 / "lee2019_eegnet_4_40_128hz.npy", "eeg", np.float32, (21600, 20, 512)),
             (OUT4 / "lee2019_win_covs_8_30.npy", "wcov", np.float64, (21600, len(WINDOWS), 20, 20)),
             (OUT4 / "lee2019_win_eegnet_4_40_128hz.npy", "weeg", np.float32, (21600, len(WINDOWS), 20, 256)))
    for path, key, dt_, shape in specs:
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=dt_, shape=shape)
        i = 0
        for p in parts:
            a = p[key]
            assert np.all(np.isfinite(a))
            if key in ("covs", "wcov"):
                assert np.linalg.eigvalsh(a.reshape(-1, 20, 20)).min() > 0
            mm[i:i + a.shape[0]] = a
            i += a.shape[0]
        assert i == 21600
        mm.flush()
        del mm
    np.savez(OUT3 / "lee2019_meta.npz", subject=meta[:, 0], session=meta[:, 1], run=meta[:, 2], pos=meta[:, 3],
             label=meta[:, 4], onset=meta[:, 5], columns=np.array(["subject", "session", "run", "pos", "label", "onset"]))
    sha = sha_file
    audit = {"raw_file_sha256": shas, "n_trials": int(meta.shape[0]), "n_participants": int(np.unique(meta[:, 0]).size),
             "exact_duplicate_raw_MI_windows": int(rawh.size - np.unique(rawh).size), "channels": list(CHANNELS),
             "feature_sha256": {f: sha(OUT3 / f) for f in ("lee2019_covs_8_30.npy", "lee2019_eegnet_4_40_128hz.npy",
                                                          "lee2019_meta.npz")},
             "build_seconds_this_invocation": time.time() - t0}
    if audit["exact_duplicate_raw_MI_windows"] != 0:
        raise ValueError("Exact scored-window duplicates found in Lee2019")
    expected_features = load_protocol(CONFIG_ROOT / "R6_C3a_PROTOCOL.json")["feature_sha256"]
    if not all(reference_matches(audit["feature_sha256"][key], val) for key, val in expected_features.items()):
        raise ValueError("Lee2019 feature files differ from the frozen references")
    (OUT3 / "lee2019_build_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    audit4 = {"windows": {k: [off / 1000, (off + WLEN) / 1000] for k, off in WINDOWS}, "raw_file_sha256": shas,
              "feature_sha256": {f: sha(OUT4 / f) for f in ("lee2019_win_covs_8_30.npy", "lee2019_win_eegnet_4_40_128hz.npy")}}
    (OUT4 / "lee2019_win_build_audit.json").write_text(json.dumps(audit4, indent=1) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "raw_file_sha256"}, indent=1))


if __name__ == "__main__":
    main()
