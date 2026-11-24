"""Rebuild the Shin2022 identity-swap inputs from the original Figshare archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, input_path, load_protocol, json_sha256, file_sha256, write_json
from . import shin_reader

ARCHIVE_MD5 = "c59c1f1a07256d06a2d9feb649409072"
FS, WINDOW, CUE_TO_FEEDBACK = 250, 240, 480
MEMBER = re.compile(r"^(S\d{2})_(\d{8})/[^/]+/[^/]+/[^/]+\.dat$")
CACHE_COLUMNS = ("sha", "block", "subject", "label", "b_main", "eeg440_FB_EARLY", "cov830_FB_EARLY")


def bandpass(signal, low, high):
    sos = butter(4, [low, high], btype="band", fs=FS, output="sos")
    zi = sosfilt_zi(sos)[:, :, None] * signal[0][None, None, :]
    return sosfilt(sos, signal, axis=0, zi=zi)[0]


def covariance(segment):
    centered = segment - segment.mean(0, keepdims=True)
    n, p = centered.shape
    cov = centered.T @ centered / n
    mean = np.trace(cov) / p
    alpha = np.mean(cov ** 2)
    denominator = (n + 1.0) * (alpha - mean ** 2 / p)
    shrinkage = 1.0 if denominator == 0 else min((alpha + mean ** 2) / denominator, 1.0)
    result = (1.0 - shrinkage) * cov + shrinkage * mean * np.eye(p)
    return result + max(1e-10 * mean, 1e-12) * np.eye(p)


def segments(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges[::2], edges[1::2]


def recording_trials(signal, states):
    cues, _ = segments(states["TargetCode"] > 0)
    starts, ends = segments(states["Feedback"] == 1)
    eeg = bandpass(signal, 4, 40)
    ts = bandpass(signal, 8, 30)
    for i, (start, end) in enumerate(zip(starts, ends)):
        previous = cues[cues <= start]
        cue = int(previous[-1]) if previous.size else -1
        if cue >= 0 and start - cue != CUE_TO_FEEDBACK:
            raise ValueError("Unexpected cue-to-feedback latency in the Shin2022 archive")
        windows = (cue - WINDOW, cue, start - WINDOW, start)
        good = end < len(signal) and (end - start) / FS >= 0.96 and all(
            x >= 0 and x + WINDOW <= len(signal) and
            np.isfinite(signal[x:x + WINDOW]).all() and np.all(np.ptp(signal[x:x + WINDOW], axis=0) > 0)
            for x in windows
        )
        label = int(states["TargetCode"][start])
        if good and label not in (1, 2):
            raise ValueError("Unexpected eligible Shin2022 target")
        yield (i, label, good,
               eeg[start:start + WINDOW].T.astype(np.float32) if good else np.zeros((16, WINDOW), np.float32),
               covariance(ts[start:start + WINDOW]) if good else np.full((16, 16), np.nan))


def read_archive(archive: Path):
    with archive.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("Shin2022 archive differs from the Figshare version 1 checksum")
    seen = {}
    rows = []
    cols = {k: [] for k in CACHE_COLUMNS}
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix="shin2022-") as tmp:
        path = Path(tmp) / "recording.dat"
        members = [(i, MEMBER.fullmatch(i.filename)) for i in sorted(z.infolist(), key=lambda i: i.filename)]
        for info, match in members:
            if not match:
                continue
            subject, date = match.groups()
            with z.open(info) as orig, path.open("wb") as local:
                shutil.copyfileobj(orig, local, length=2**20)
            digest = file_sha256(path)
            rows.append({"subject": subject, "annex_sha256": digest})
            if digest in seen:
                if seen[digest] != subject:
                    raise ValueError("Duplicate recording crosses participants")
                continue
            seen[digest] = subject
            h = shin_reader.read_header(path)
            shim = shin_reader._PRIVATE_STORE.pop(id(h))
            del shim
            # Only the acquisition date is used to apply the recording-identity mask.
            date_iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            time = h.params.get("StorageTime")
            if time is None:
                raise ValueError("Missing acquisition date for Shin2022 session validation")
            if date_iso not in time.value and date not in time.value:
                continue
            if h.sampling_rate != FS or h.source_ch != 16:
                raise ValueError("Unexpected Shin2022 signal geometry")
            signal, states = shin_reader.read_data(path, h)
            for block, label, good, eeg, cov in recording_trials(signal, states):
                for k, v in zip(CACHE_COLUMNS, (digest, block, subject, label, good, eeg, cov)):
                    cols[k].append(v)
    if len(rows) != 110 or len(seen) != 101 or len(set(cols["sha"])) != 99:
        raise ValueError("Shin2022 recording counts differ from the frozen input")
    return {k: np.asarray(v) for k, v in cols.items()}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    archive = input_path("shin2022_zip")
    protocol = load_protocol(CONFIG_ROOT / "R8_A_PROTOCOL_v1_3.json")
    z, rows = read_archive(archive)
    work = RUN_ROOT / "data" / "shin2022"
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "r2_shin_windows_v1.npz"
    np.savez(cache, **z)
    checks = protocol["portable_input_sha256"]
    if not reference_matches(file_sha256(cache), checks["windows"]) or not reference_matches(json_sha256(rows), checks["registry"]):
        raise ValueError("Shin2022 rebuilt cache or recording masks differ from the frozen input projection")
    write_json(work / "recording_registry.json", {"rows": rows})
    print(json.dumps({"recordings": len(rows), "eligible_distinct_recordings": 99,
                      "trial_rows": len(z["sha"]), "eligible_trials": int(z["b_main"].sum())}))


if __name__ == "__main__":
    main()
