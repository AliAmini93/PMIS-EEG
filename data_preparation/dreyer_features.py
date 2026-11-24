"""R12-X trial-local feature extraction from producer GDF runs (frozen recipe; no statistic spans trials).

Reader: MNE 1.11 read_raw_gdf (production), cross-checked per file against the own GDF 1.x reader (identical
samples within 1e-12 relative, identical event table) -- a disagreement stops the build.
Windows (relative to the cue event 769 / 770; causal 4th-order Butterworth with steady-state initial conditions,
warm-up from the window start minus 3 s, i.e. from the cross onset of the same trial):
  cue    [0, 1.25 s)   PRIMARY (pre-feedback)      20 sensorimotor channels
  fb     [0, 4 s)      secondary (cue + feedback)   20 channels
  precue [-1.25 s, 0)  temporal control             20 channels (TS+LR only)
  eogcue / eogfb       the same windows, 3 EOG channels, 1-30 Hz (TS+LR only; ocular confound control)
Outputs per window: OAS covariance of the band-passed window (8-30 Hz; EOG 1-30 Hz), in microvolts^2; deep input
4-40 Hz, polyphase 512 -> 128 Hz (1 : 4), last N samples of the filtered segment (cue 160, fb 512), float32, uV.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from models.vendor_r4.r4_models import causal_bandpass, oas_cov

from data_preparation import dreyer_gdf as gd

FS = 512
CHANNELS = ("FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6")
EOG = ("EOG1", "EOG2", "EOG3")
LEAD = 3 * FS
WINDOWS = {"cue": (0, 640, CHANNELS, (8.0, 30.0), 160), "fb": (0, 2048, CHANNELS, (8.0, 30.0), 512),
           "precue": (-640, 0, CHANNELS, (8.0, 30.0), None), "eogcue": (0, 640, EOG, (1.0, 30.0), None),
           "eogfb": (0, 2048, EOG, (1.0, 30.0), None)}
CUE = {"769": 1, "770": 2}


def read_run(path, mne) -> dict:
    """Return physical data (uV) of the needed channels and cue events, after the own-reader cross-check."""
    m = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    own = gd.read(path)
    xm = m.get_data()
    if own["label"] != m.ch_names or own["physical_V"].shape != xm.shape:
        raise RuntimeError(f"reader disagreement (labels / shape): {path}")
    scale = np.maximum(np.max(np.abs(own["physical_V"]), axis=1, keepdims=True), 1e-30)
    if float(np.max(np.abs(xm - own["physical_V"]) / scale)) > 1e-12:
        raise RuntimeError(f"reader disagreement (samples): {path}")
    ev_m = sorted((int(round(o * FS)), str(d)) for o, d in zip(m.annotations.onset, m.annotations.description))
    ev_o = sorted((int(p), str(t)) for p, t in zip(own["event_pos0"], own["event_type"]))  # 0-based (stored 1-based)
    if ev_m != ev_o:
        raise RuntimeError(f"reader disagreement (events): {path}")
    if m.info["sfreq"] != FS:
        raise RuntimeError("sampling rate")
    cues = [(s, CUE[c]) for s, c in ev_m if c in CUE]
    ch = {c: i for i, c in enumerate(m.ch_names)}
    need = CHANNELS + EOG
    return {"x": xm[[ch[c] for c in need]] * 1e6, "channels": need, "cues": cues, "n_samples": xm.shape[1]}


def trial_features(x: np.ndarray, channels: tuple, onset: int) -> dict:
    """All window features for one trial (x: channels x samples in uV, onset = cue sample)."""
    idx = {c: i for i, c in enumerate(channels)}
    out = {}
    for name, (a, b, chs, (lo, hi), n_deep) in WINDOWS.items():
        rows = [idx[c] for c in chs]
        seg = x[rows, onset + a - LEAD:onset + b]
        if seg.shape[1] != b - a + LEAD:
            raise ValueError("window outside the recording")
        f = causal_bandpass(seg, lo, hi, FS)[:, LEAD:]
        out[f"cov_{name}"] = oas_cov(f.T)
        if n_deep is not None:
            d = resample_poly(causal_bandpass(seg, 4.0, 40.0, FS), 1, 4, axis=1)[:, -n_deep:]
            out[f"eeg_{name}"] = d.astype(np.float32)
    return out
