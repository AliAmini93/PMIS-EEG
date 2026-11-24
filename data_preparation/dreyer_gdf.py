"""Independent, minimal GDF 1.x reader (own code, numpy only) used to audit MNE's reading of the producer GDF files.

GDF 1.25 layout (Schloegl, GDF v1): 256-byte fixed header (version, patient, recording, start time, header bytes
int64, ids, reserved, n_records int64, record duration 2 x uint32, NS uint32), a 256 x NS variable header (label 16,
transducer 80, physical dimension 8, physical min / max float64, digital min / max int64, prefiltering 80, samples
per record uint32, data type uint32, reserved 32), data records, then the event table (mode uint8, event sampling
rate uint24, N uint32, positions uint32 x N (1-based), types uint16 x N [, channels uint16 x N, durations uint32 x N
for mode 3]).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

DTYPE = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<i8", 16: "<f4", 17: "<f8"}
UNIT = {"uV": 1e-6, "µV": 1e-6, "mV": 1e-3, "V": 1.0}


def read(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if not raw[:4] == b"GDF " or not raw[4:5] == b"1":
        raise ValueError("not a GDF 1.x file")
    hb = struct.unpack("<q", raw[184:192])[0]
    n_rec = struct.unpack("<q", raw[236:244])[0]
    num, den = struct.unpack("<II", raw[244:252])
    ns = struct.unpack("<I", raw[252:256])[0]
    if hb != 256 * (ns + 1):
        raise ValueError("GDF header size mismatch")
    v, o = raw[256:hb], 0

    def take(width, fmt=None):
        nonlocal o
        seg = v[o:o + width * ns]
        o += width * ns
        if fmt is None:
            return [seg[i * width:(i + 1) * width].split(b"\x00")[0].decode("latin-1").strip() for i in range(ns)]
        return list(struct.unpack(f"<{ns}{fmt}", seg))

    label, _tr, pdim = take(16), take(80), take(8)
    pmin, pmax, dmin, dmax = take(8, "d"), take(8, "d"), take(8, "q"), take(8, "q")
    _pre, spr, typ = take(80), take(4, "I"), take(4, "I")
    if len(set(typ)) != 1 or len(set(spr)) != 1:
        raise ValueError("mixed data types / samples per record not supported")
    dt = np.dtype(DTYPE[typ[0]])
    rec_samples = spr[0] * ns
    n_data = n_rec * rec_samples * dt.itemsize
    data = np.frombuffer(raw, dtype=dt, count=n_rec * rec_samples, offset=hb).reshape(n_rec, spr[0], ns)
    dig = data.reshape(n_rec * spr[0], ns).T.astype(np.float64)       # (ns, n_samples)
    # Deviation G1R-D1: physical = d * gain + (pmin - dmin * gain), evaluated in float64 as the writer did. The
    # algebraically identical (d - dmin) * gain + pmin cancels catastrophically for the producer's extreme
    # calibration headers (physical +-2^127, digital +-2^62) found in some online runs.
    pmin_a, pmax_a = np.array(pmin, dtype=np.float64), np.array(pmax, dtype=np.float64)
    dmin_a, dmax_a = np.array(dmin, dtype=np.float64), np.array(dmax, dtype=np.float64)
    gain = (pmax_a - pmin_a) / (dmax_a - dmin_a)
    offset = pmin_a - dmin_a * gain
    phys = dig * gain[:, None] + offset[:, None]
    unit = np.array([UNIT.get(u, 1.0) for u in pdim])[:, None]
    et = raw[hb + n_data:]
    mode = et[0]
    fs_ev = int.from_bytes(et[1:4], "little")
    n_ev = struct.unpack("<I", et[4:8])[0]
    pos = np.frombuffer(et, dtype="<u4", count=n_ev, offset=8).astype(np.int64)
    typ_ev = np.frombuffer(et, dtype="<u2", count=n_ev, offset=8 + 4 * n_ev).astype(np.int64)
    fs = spr[0] * den / num if num else float("nan")
    extreme = bool(np.any(np.abs(pmax_a) > 1e37))
    return {"version": raw[:8].decode("latin-1"), "ns": ns, "n_records": n_rec, "fs": fs, "label": label,
            "extreme_calibration_header": extreme,
            "phys_dim": pdim, "physical_V": phys * unit, "event_mode": mode, "event_fs": fs_ev,
            "event_pos0": pos - 1, "event_type": typ_ev, "trailing_bytes": len(et) - (8 + 6 * n_ev)}
