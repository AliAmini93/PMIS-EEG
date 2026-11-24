"""Reconstruct the HS-Stroke release audit, feature arrays and fold manifests locally."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

from data_preparation import hs_manifest, hs_integrity
from experiments.runtime import reference_matches, RUN_ROOT, input_path
from models.vendor_r4.r4_models import causal_bandpass, oas_cov

FS = 256
MI0, MI1 = 9 * FS, 13 * FS


RUN_RE = re.compile(r"run-(\d+)_eeg\.mat$")


CHANNELS = ["FC3", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP4"]


def load_run(path: Path):
    m = sio.loadmat(path, verify_compressed_data_integrity=False)
    X, y = m["EEGdata"], m["EEGdatalabel"].ravel().astype(int)
    ch = [c[0].tolist()[0] for c in m["configuration_channel"][0] if np.asarray(c[1]).sum()]
    if ch != CHANNELS or X.shape[:2] != (11, 4352) or X.shape[2] != y.size or X.dtype != np.float64:
        raise ValueError(f"unexpected layout in {path.name}")
    if not set(np.unique(y)) <= {1, 2}:
        raise ValueError(f"unexpected labels in {path.name}")
    return X, y


def h(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def features(raw_v: np.ndarray):
    """raw_v: 11 x 4352 float64 in volts -> (OAS cov 11x11 of 8-30 Hz MI window, EEGNet 11x512 float32)."""
    uv = raw_v * 1e6
    ts = causal_bandpass(uv, 8.0, 30.0, FS)[:, MI0:MI1]
    cov = oas_cov(ts.T)
    en = causal_bandpass(uv, 4.0, 40.0, FS)[:, MI0:MI1][:, ::2].astype(np.float32)
    return cov, en


def sessions_of(root: Path):
    out = []
    for subj in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub-")):
        for ses in sorted(p for p in subj.iterdir() if p.is_dir() and p.name.startswith("ses-")):
            out.append((subj.name, ses.name))
    return out


def build(v1_root: Path, v3_root: Path, priv: Path):
    sess = sessions_of(v1_root)
    if sess != sessions_of(v3_root):
        raise RuntimeError("v1/v3 session layouts differ")
    subjects = sorted({s for s, _ in sess})
    inst = collections.defaultdict(list)   # instance-level columns
    dist = collections.defaultdict(list)   # distinct-trial columns
    covs, eeg = [], []
    checks = collections.Counter()
    scored_identities = set()
    for si, (subj, ses) in enumerate(sess):
        files1 = sorted((v1_root / subj / ses / "eeg").glob("*.mat"))
        files3 = sorted((v3_root / subj / ses / "eeg").glob("*.mat"))
        if [f.name for f in files1] != [f.name for f in files3]:
            raise RuntimeError("v1/v3 file names differ within a session")
        v3map = {}
        for f in files3:
            X, y = load_run(f)
            k = int(RUN_RE.search(f.name)[1])
            for i in range(X.shape[2]):
                key = h(X[:, :, i])
                if key in v3map:
                    raise RuntimeError("duplicate trial inside corrected v3 session")
                v3map[key] = (k, i, int(y[i]))
        seen = {}          # hash -> distinct id
        first_copy = None  # (raw, distinct id) of one non-canonical instance for the identity check
        offset = 0
        for f in files1:
            X, y = load_run(f)
            k = int(RUN_RE.search(f.name)[1])
            for i in range(X.shape[2]):
                key = h(X[:, :, i])
                lab = int(y[i])
                if key not in v3map:
                    raise RuntimeError("v1 instance absent from corrected v3 session")
                if v3map[key][2] != lab:
                    raise RuntimeError("label conflict between v1 copy and v3 trial")
                if key in seen:
                    d = seen[key]
                    if dist["label"][d] != lab:
                        raise RuntimeError("label conflict among v1 copies")
                    canonical = False
                    dist["multiplicity"][d] += 1
                    if first_copy is None:
                        first_copy = (X[:, :, i].copy(), d)
                else:
                    d = len(dist["label"])
                    seen[key] = d
                    canonical = True
                    scored = h(X[:, MI0:MI1, i])
                    if scored in scored_identities:
                        raise ValueError("The scored block is shared by different historical trial identities")
                    scored_identities.add(scored)
                    v3run, v3idx, _ = v3map[key]
                    if v3run != k:
                        raise RuntimeError("canonical (first-appearance) run differs from corrected-release run")
                    for c, v in (("subject", subjects.index(subj)), ("session", si), ("origin_run", k),
                                 ("v3_idx", v3idx), ("label", lab), ("multiplicity", 1)):
                        dist[c].append(v)
                    dist["hash"].append(key)
                    cv, en = features(X[:, :, i])
                    covs.append(cv)
                    eeg.append(en)
                for c, v in (("subject", subjects.index(subj)), ("session", si), ("file_run", k),
                             ("idx_in_file", i), ("distinct", d), ("label", lab),
                             ("canonical", canonical), ("b_trial_id", offset + i)):
                    inst[c].append(v)
            offset += X.shape[2]
        if set(seen) != set(v3map):
            raise RuntimeError("corrected v3 session is not the distinct set of v1")
        checks["sessions_v3_equals_distinct_v1"] += 1
        if first_copy is not None:
            cv, en = features(first_copy[0])
            d = first_copy[1]
            if not (np.array_equal(cv, covs[d]) and np.array_equal(en, eeg[d])):
                raise RuntimeError("copy features differ from canonical features")
            checks["sessions_copy_feature_identity_checked"] += 1
        if si % 20 == 0:
            print(f"  session {si + 1}/{len(sess)}: {len(inst['label'])} instances, "
                  f"{len(dist['label'])} distinct", flush=True)
    priv.mkdir(parents=True, exist_ok=True)
    np.savez(priv / "G2_instances.npz", **{c: np.asarray(v) for c, v in inst.items()})
    np.savez(priv / "G2_distinct.npz", **{c: np.asarray(v) for c, v in dist.items() if c != "hash"},
             hash=np.asarray(dist["hash"]))
    np.save(priv / "G2_covs_8_30.npy", np.stack(covs))
    np.save(priv / "G2_eegnet_4_40_128hz.npy", np.stack(eeg))
    (priv / "G2_sessions_PRIVATE.json").write_text(json.dumps({"subjects": subjects, "sessions": sess}) + "\n")
    return inst, dist, checks, len(subjects), len(sess)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    v1 = input_path("hs_stroke_v1")
    v3 = input_path("hs_stroke_v3")
    priv = RUN_ROOT / "data" / "hs_stroke"
    if v1 == v3 or not sessions_of(v1):
        raise ValueError("Provide the two original extracted sourcedata trees")
    _, _, counts, n_participants, n_sessions = build(v1, v3, priv)
    summary = hs_manifest.build_manifest(priv)
    expected = {name: hs_integrity.historical_sha256(name) for name in
                (*hs_integrity.METADATA_FILES, "G2_covs_8_30.npy", "G2_eegnet_4_40_128hz.npy")}
    for filename, digest in expected.items():
        if not reference_matches(hs_integrity.sha256_file(priv / filename), digest):
            raise ValueError(f"HS-Stroke rebuilt file differs from the frozen reference: {filename}")
    print(json.dumps({"participants": n_participants, "sessions": n_sessions,
                      "identity_checks": counts, "distinct_trials": summary["n_distinct_scored_per_split_seed"],
                      "frozen_files_verified": sum(digest is not None for digest in expected.values())}))


if __name__ == "__main__":
    main()
