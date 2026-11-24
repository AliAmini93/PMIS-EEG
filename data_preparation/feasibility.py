"""R15-B Dreyer2023 loader: fitting runs R1-R4 only.

R5 (the selection run) and R6 (the evaluation run) of the frozen R12-X held-out-run design are outcome-bearing.
This module cannot read them:
- gdf_path() refuses any run outside FIT_RUNS;
- load_participant() asserts on the runs it returns.

Every raw file is hashed before it is read, and the hash must equal the verified R12-X G1R SHA-256 table. The eligible set
and the table can also be checked against user-supplied historical receipts.

Preprocessing never lets data after the window end enter a task window:
- Each trial segment is [cue - 3 s, cue + W). It is filtered causally (steady-state initial conditions, as R12-X), then
  resampled polyphase on that segment alone, and the last W seconds are kept. Nothing after cue + W is read.
- W = 1.0 s or 1.25 s. Both end at or before the feedback onset (cue -> event 781 = 640 samples = 1.25 s). Feedback data
  is never used and nothing is padded.
- Model formats:
  - ST-EEGFormer: notch 50 Hz, band 0.1-75 Hz (the official BCI-IV-2a YAML), 512 -> 128 Hz, z-score per channel and trial.
  - LaBraM: notch 50 Hz, band 0.1-75 Hz, 512 -> 200 Hz, uV / 100 (official README and fine-tuning engine).
  - CBraMod: notch 50 Hz, band 0.3-75 Hz, 512 -> 200 Hz, uV / 100 (pretraining format).
  The official pipelines filter whole recordings with zero-phase filters. That would let post-window (feedback) samples
  into the window, so it is replaced by the causal trial-local filter. This is a documented deviation.
- Engagement segments ("native" format): [cue - 3 s, cue + 1 s), 4 s, filtered causally from the segment start. They are
  used only for label-free reconstruction / masked-code tests, never for decoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.runtime import CONFIG_ROOT, json_sha256, load_protocol, reference_matches

import numpy as np
from scipy.signal import iirnotch, resample_poly, sosfilt, sosfilt_zi, tf2sos, butter

FIT_RUNS = ("R1", "R2", "R3", "R4")
FS = 512
LEAD = 3 * FS
CHANNELS = ("FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6")
WINDOWS = {"w100": 512, "w125": 640}           # samples at 512 Hz after the cue
FORMATS = {"st": {"fs": 128, "up": 1, "down": 4, "band": (0.1, 75.0), "scale": "z"},
           "lb": {"fs": 200, "up": 25, "down": 64, "band": (0.1, 75.0), "scale": 0.01},
           "cb": {"fs": 200, "up": 25, "down": 64, "band": (0.3, 75.0), "scale": 0.01}}


class OutcomeRunRefused(PermissionError):
    pass


def check_cohort(eligible_file: Path) -> dict:
    d = json.loads(Path(eligible_file).read_text())
    protocol = load_protocol(CONFIG_ROOT / "R12X_PROTOCOL.json")
    if len(d["eligible"]) != protocol["n_participants"]:
        raise RuntimeError("unexpected fitting cohort size")
    if not reference_matches(hashlib.sha256(json.dumps(d["eligible"]).encode()).hexdigest(), protocol["participants_sha256"]):
        raise RuntimeError("eligible set differs from the committed R12-X G1R cohort")
    if not reference_matches(hashlib.sha256(json.dumps(d["raw_sha256"], sort_keys=True).encode()).hexdigest(),
                             protocol["cohort"]["raw_sha256_table_sha256"]):
        raise RuntimeError("raw SHA-256 table differs from the committed R12-X G1R table")
    return d


def producer_id(n: int) -> str:
    g = "A" if n <= 60 else ("B" if n <= 81 else "C")
    return f"{g}{n}"


GDF_TAG = {"R1": "R1_acquisition", "R2": "R2_acquisition", "R3": "R3_onlineT", "R4": "R4_onlineT"}


def gdf_path(root: Path, n: int, run: str) -> Path:
    if run not in FIT_RUNS:
        raise OutcomeRunRefused(f"run {run} is outcome-bearing (R12-X selection / evaluation run); R15-B reads R1-R4 only")
    pid = producer_id(n)
    return Path(root) / "BCI Database" / "Signals" / f"DATA {pid[0]}" / pid / f"{pid}_{GDF_TAG[run]}.gdf"


def load_participant(root: Path, n: int, sha_table: dict, mne) -> dict:
    """{run: {"x": (20, N) uV, "cues": [(sample, class 1|2)]}} for R1-R4; every file's SHA-256 checked before reading"""
    from data_preparation.dreyer_features import read_run
    out = {}
    for run in FIT_RUNS:
        p = gdf_path(root, n, run)
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha_table[f"{producer_id(n)}_{run}"]:
            raise RuntimeError(f"raw SHA-256 mismatch {producer_id(n)} {run}")
        r = read_run(p, mne)
        idx = [r["channels"].index(c) for c in CHANNELS]
        out[run] = {"x": r["x"][idx], "cues": r["cues"]}
    assert set(out) == set(FIT_RUNS)
    return out


def _causal(sos, x):
    zi = sosfilt_zi(sos)[:, :, None] * x[:, 0][None, None, :]
    y, _ = sosfilt(sos, x.T, axis=0, zi=zi)
    return y.T


def causal_chain(seg: np.ndarray, band, notch=50.0) -> np.ndarray:
    b, a = iirnotch(notch, 30.0, fs=FS)
    seg = _causal(tf2sos(b, a), seg)
    return _causal(butter(4, list(band), btype="band", fs=FS, output="sos"), seg)


def fmt(seg512: np.ndarray, f: str, keep_samples_512: int) -> np.ndarray:
    """one trial segment (C, T512) ending exactly at the window end -> model format, last window kept"""
    spec = FORMATS[f]
    y = resample_poly(causal_chain(seg512, spec["band"]), spec["up"], spec["down"], axis=1)
    n_keep = keep_samples_512 * spec["up"] // spec["down"]
    y = y[:, -n_keep:]
    if spec["scale"] == "z":
        y = (y - y.mean(axis=1, keepdims=True)) / (y.std(axis=1, keepdims=True) + 1e-8)
    else:
        y = y * spec["scale"]
    return y.astype(np.float32)


def participant_arrays(runs: dict) -> dict:
    """all model formats for one participant's R1-R4 trials; labels 0 = left (769), 1 = right (770)"""
    out = {k: [] for k in ("y", "run", "trial")}
    for run in FIT_RUNS:
        x, cues = runs[run]["x"], runs[run]["cues"]
        for t, (s, c) in enumerate(cues):
            if s - LEAD < 0 or s + WINDOWS["w125"] > x.shape[1]:
                raise ValueError("window outside the recording")
            out["y"].append(c - 1)
            out["run"].append(int(run[1]))
            out["trial"].append(t)
            for w, ns in WINDOWS.items():
                seg = x[:, s - LEAD:s + ns]
                for f in ("st", "lb", "cb"):
                    if f != "st" and w == "w125":
                        continue          # 1.25 s is not a whole number of 1-s patches; no padding
                    out.setdefault(f"{f}_{w}", []).append(fmt(seg, f, ns))
            nat = x[:, s - LEAD:s + FS]                # 4-s native segment, label-free engagement tests only
            for f in ("st", "lb", "cb"):
                out.setdefault(f"{f}_nat", []).append(fmt(nat, f, nat.shape[1]))
            cov_seg = x[:, s - LEAD:s + WINDOWS["w125"]]
            out.setdefault("cov_w125", []).append(_cov_830(cov_seg))
    return {k: np.asarray(v) for k, v in out.items()}


def _cov_830(seg):
    """TS+LR anchor feature, as R12-X cue window: causal 4th-order Butterworth 8-30 Hz, OAS covariance of the last 1.25 s (uV^2)"""
    import sys
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from models.vendor_r4.r4_models import causal_bandpass, oas_cov
    f = causal_bandpass(seg, 8.0, 30.0, FS)[:, LEAD:]
    return oas_cov(f.T)
