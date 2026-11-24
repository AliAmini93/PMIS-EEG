"""R10-B design: folds, halves, training arms, invariants and per-arm temporal geometry (metadata only; no outcome).

Indexing: a window row id is vi * N + t, with vi = geometry.vi(offset) and t the Lee2019 trial index (N = 21,600).
Folds: per participant, leave-one-run-out (4 folds; the C3/C5 split). T = the held-out run's 100 trials, whose
anchors (offset 0) are scored in every arm; base = the other 300 trials.
Halves: per fold, T is split into H1 / H2, 25 + 25 per class, by numpy default_rng([10301, participant, run, class]).

Track B (fixed density; one window per exposed anchor; base = 300 anchor-offset rows, as in C5):
  B1_lag / B2_lag = base + the lag-offset window of every H1 / H2 trial (350 rows), lags 0 .. 2000 ms
  Z = base + 50 exact duplicates of base rows (25 per class) from the held-out run's session-mate run (placebo)
  AA = B1_0 again (bit-identical A/A).  One CRN group of 16 arms.
Track A (sliding-window geometry, stride s in {1000, 500, 250}; base = every window of the 300 base trials):
  G  (trial-grouped control)          = base windows
  L  (window-level contaminated)      = G + every non-anchor window of all 100 T trials
  A1 / A2 (half, own-trial windows)   = G + every non-anchor window of H1 / H2
  X1 / X2 (half, + exact anchor copy) = A1 / A2 + the anchors of H1 / H2
  CRN groups (equal rows): {A1, A2 (+ AA at s = 1000)}, {X1, X2}, {G}, {L}.
C0 (conventional contextual benchmark, stride 500): per participant every window of all 400 trials; random
window-level 5-fold CV vs trial-grouped 5-fold CV (class-stratified); every window scored once per scheme.
"""

from __future__ import annotations

import hashlib

import numpy as np

from experiments.pmis_r10b import geometry as geo

N_TRIALS = 21600
HALF_STREAM, PLACEBO_STREAM, SEED_STREAM, DROP_STREAM = 10301, 10302, 10310, 10311
C0_WIN_STREAM, C0_TRIAL_STREAM, C0_SEED_STREAM, C0_DROP_STREAM = 10320, 10321, 10322, 10323
LABELS = (1, 2)


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(tuple(int(p) for p in parts)).encode()).digest()[:4], "little") & 0x7FFFFFFF


def row(offset: int, t) -> np.ndarray:
    return geo.vi(offset) * N_TRIALS + np.asarray(t, dtype=np.int64)


def trial_of(r) -> np.ndarray:
    return np.asarray(r, dtype=np.int64) % N_TRIALS


def offset_of(r) -> np.ndarray:
    return np.asarray(geo.OFFSETS, dtype=np.int64)[np.asarray(r, dtype=np.int64) // N_TRIALS]


def _class_pos_order(idx, lab, pos):
    idx = np.asarray(idx, dtype=np.int64)
    return idx[np.lexsort((pos[idx], lab[idx]))]


def folds(meta: dict) -> list[dict]:
    subj, sess, run, pos, lab = (np.asarray(meta[k], dtype=np.int64) for k in ("subject", "session", "run", "pos", "label"))
    out = []
    for s in np.unique(subj):
        for r in range(4):
            T = np.flatnonzero((subj == s) & (run == r))
            base = np.flatnonzero((subj == s) & (run != r))
            assert T.size == 100 and base.size == 300
            T = T[np.argsort(pos[T], kind="stable")]
            h1, h2 = [], []
            for c in LABELS:
                cand = T[lab[T] == c]
                perm = np.random.default_rng([HALF_STREAM, int(s), r, c]).permutation(cand)
                h1 += perm[:25].tolist()
                h2 += perm[25:50].tolist()
            sess_r = int(sess[T[0]])
            plc = []
            for c in LABELS:
                pool = base[(sess[base] == sess_r) & (lab[base] == c)]
                plc += np.random.default_rng([PLACEBO_STREAM, int(s), r, c]).choice(pool, 25, replace=False).tolist()
            out.append({"subject": int(s), "run": r, "session": sess_r, "T": T, "base": np.sort(base),
                        "H1": _class_pos_order(h1, lab, pos), "H2": _class_pos_order(h2, lab, pos),
                        "Z": _class_pos_order(plc, lab, pos),
                        "pair_seed": _seed(SEED_STREAM, s, r), "group_seed": _seed(DROP_STREAM, s, r)})
    return out


# ------------------------------------------------------------------------------------------------ arms
def track_b_arms(f: dict) -> dict:
    base = row(0, f["base"])
    arms = {}
    for lag in geo.LAGS_B:
        arms[f"B1_{lag}"] = np.concatenate([base, row(lag, f["H1"])])
        arms[f"B2_{lag}"] = np.concatenate([base, row(lag, f["H2"])])
    arms["Z"] = np.concatenate([base, row(0, f["Z"])])
    arms["AA"] = arms["B1_0"].copy()
    return arms


TRACK_B_ORDER = tuple([f"B{h}_{lag}" for lag in geo.LAGS_B for h in (1, 2)] + ["Z", "AA"])


def _base_windows(f, stride):
    return np.concatenate([row(o, f["base"]) for o in geo.stride_offsets(stride)])


def _nonanchor(trials, stride):
    return np.concatenate([row(o, trials) for o in geo.stride_offsets(stride)[1:]])


def track_a_arms(f: dict, stride: int) -> dict:
    G = _base_windows(f, stride)
    A1 = np.concatenate([G, _nonanchor(f["H1"], stride)])
    A2 = np.concatenate([G, _nonanchor(f["H2"], stride)])
    arms = {"G": G, "L": np.concatenate([G, _nonanchor(np.concatenate([f["H1"], f["H2"]]), stride)]),
            "A1": A1, "A2": A2, "X1": np.concatenate([A1, row(0, f["H1"])]), "X2": np.concatenate([A2, row(0, f["H2"])])}
    if stride == geo.STRIDES[0]:
        arms["AA"] = A1.copy()
    return arms


def track_a_groups(stride: int) -> tuple[tuple[str, ...], ...]:
    first = ("A1", "A2", "AA") if stride == geo.STRIDES[0] else ("A1", "A2")
    return (first, ("X1", "X2"), ("G",), ("L",))


def c0_splits(meta: dict, subject: int) -> dict:
    """Per participant: every stride-500 window of all 400 trials. 'rw' = random window-level 5-fold, 'tg' =
    trial-grouped 5-fold; both class-stratified. Returns test-row lists per fold and scheme."""
    subj, lab = np.asarray(meta["subject"]), np.asarray(meta["label"])
    trials = np.flatnonzero(subj == subject)
    offs = geo.stride_offsets(geo.C0_STRIDE)
    rows = np.concatenate([row(o, trials) for o in offs])
    out = {"rows": rows, "rw": [], "tg": []}
    rw_fold = np.empty(rows.size, dtype=np.int64)
    for c in LABELS:
        idx = np.flatnonzero(lab[trial_of(rows)] == c)
        perm = np.random.default_rng([C0_WIN_STREAM, int(subject), c]).permutation(idx)
        rw_fold[perm] = np.arange(perm.size) % 5
    tg_trial_fold = {}
    for c in LABELS:
        tc = trials[lab[trials] == c]
        perm = np.random.default_rng([C0_TRIAL_STREAM, int(subject), c]).permutation(tc)
        for i, t in enumerate(perm):
            tg_trial_fold[int(t)] = i % 5
    tg_fold = np.array([tg_trial_fold[int(t)] for t in trial_of(rows)], dtype=np.int64)
    for k in range(5):
        out["rw"].append(np.sort(rows[rw_fold == k]))
        out["tg"].append(np.sort(rows[tg_fold == k]))
    out["pair_seed"] = _seed(C0_SEED_STREAM, subject)
    out["group_seed"] = _seed(C0_DROP_STREAM, subject)
    return out


def c0_train_rows(sp: dict) -> list[tuple[str, int, np.ndarray]]:
    allr = sp["rows"]
    out = []
    for scheme in ("rw", "tg"):
        for k, te in enumerate(sp[scheme]):
            out.append((scheme, k, np.setdiff1d(allr, te, assume_unique=True)))
    return out


# ------------------------------------------------------------------------------------------------ invariants
class DesignError(AssertionError):
    pass


def _req(cond, msg):
    if not cond:
        raise DesignError(msg)


def check_fold(f: dict, meta: dict) -> None:
    lab, run, sess, subj = (np.asarray(meta[k]) for k in ("label", "run", "session", "subject"))
    T, base, H1, H2 = f["T"], f["base"], f["H1"], f["H2"]
    _req(np.unique(subj[np.concatenate([T, base])]).size == 1, "fold mixes participants")
    _req(np.all(run[T] == f["run"]) and not np.any(run[base] == f["run"]), "held-out run leaks into the base")
    _req(np.intersect1d(H1, H2).size == 0 and np.array_equal(np.sort(np.concatenate([H1, H2])), np.sort(T)), "halves")
    for h in (H1, H2):
        _req(np.sum(lab[h] == 1) == 25 and np.sum(lab[h] == 2) == 25, "halves not class-stratified")
    _req(np.all(np.isin(f["Z"], base)) and np.all(sess[f["Z"]] == f["session"]), "placebo rows not session-matched base")
    # class at every extra position is identical across paired arms (CRN position alignment)
    _req(np.array_equal(lab[H1], lab[H2]) and np.array_equal(lab[H1], lab[f["Z"]]), "extra-row class alignment")
    tb = track_b_arms(f)
    n = {k: v.size for k, v in tb.items()}
    _req(len(set(n.values())) == 1 and n["Z"] == 350, "Track B rows differ")
    for k, v in tb.items():
        if k == "Z":  # the placebo adds exact duplicates of 50 base rows: exactly 50 repeated ids
            _req(v.size - np.unique(v).size == 50, "placebo duplicates")
        else:
            _req(np.unique(v).size == v.size, f"{k}: repeated row id")
    for lag in geo.LAGS_B:
        for h, other in (("1", H2), ("2", H1)):
            a = tb[f"B{h}_{lag}"]
            _req(not np.any(np.isin(trial_of(a[300:]), other)), "Track B arm contains the other half's windows")
            _req(np.all(offset_of(a[300:]) == lag), "Track B extra rows at the wrong offset")
            _req(np.all(offset_of(a[:300]) == 0) and np.array_equal(trial_of(a[:300]), base), "Track B base rows")
            _req(np.sum(lab[trial_of(a)] == 1) == 175, "Track B class composition")
    _req(not np.any(np.isin(trial_of(tb["Z"]), T)), "placebo contains held-out-run material")
    _req(np.array_equal(tb["AA"], tb["B1_0"]), "A/A rows")
    for s in geo.STRIDES:
        ta = track_a_arms(f, s)
        nw = len(geo.stride_offsets(s))
        _req(ta["G"].size == 300 * nw and not np.any(np.isin(trial_of(ta["G"]), T)), f"G_{s}: held-out trial in training")
        _req(ta["L"].size == 300 * nw + 100 * (nw - 1) and not np.any((offset_of(ta["L"]) == 0) & np.isin(trial_of(ta["L"]), T)),
             f"L_{s}: anchor in training")
        for h, mine, other in (("1", H1, H2), ("2", H2, H1)):
            A, X = ta[f"A{h}"], ta[f"X{h}"]
            _req(not np.any(np.isin(trial_of(A), other)) and not np.any((offset_of(A) == 0) & np.isin(trial_of(A), T)), f"A{h}_{s}")
            _req(np.array_equal(np.sort(np.setdiff1d(X, A)), np.sort(row(0, mine))), f"X{h}_{s} != A{h} + anchors")
        for g in track_a_groups(s):
            _req(len({ta[a].size for a in g}) == 1, f"Track A group {g} rows differ")
            if len(g) >= 2:
                _req(np.array_equal(lab[trial_of(ta[g[0]])], lab[trial_of(ta[g[1]])]), f"class alignment {g}")
                _req(np.array_equal(offset_of(ta[g[0]]), offset_of(ta[g[1]])), f"offset alignment {g}")
        if "AA" in ta:
            _req(np.array_equal(ta["AA"], ta["A1"]), "Track A A/A rows")


# ------------------------------------------------------------------------------------------------ geometry records
def arm_geometry(arm_rows: np.ndarray, anchors: np.ndarray, T: np.ndarray, lab: np.ndarray) -> dict:
    """Per-anchor temporal exposure in one training arm, summarised (no outcome)."""
    tr, of = trial_of(arm_rows), offset_of(arm_rows)
    held = np.isin(tr, T)
    per = []
    for a in anchors:
        own = of[tr == a]
        per.append((own.size, int(np.sum([geo.shared_samples(o) > 0 for o in own])),
                    int(own.min()) if own.size else -1, geo.shared_samples(int(own.min())) if own.size else 0))
    per = np.array(per)
    exposed = per[:, 0] > 0
    return {"rows": int(arm_rows.size), "distinct_source_trials": int(np.unique(tr).size),
            "distinct_heldout_run_trials": int(np.unique(tr[held]).size), "heldout_run_windows": int(held.sum()),
            "class_counts": [int(np.sum(lab[tr] == c)) for c in LABELS],
            "anchors_exposed": int(exposed.sum()),
            "own_windows_per_exposed_anchor": float(per[exposed, 0].mean()) if exposed.any() else 0.0,
            "overlapping_own_windows_per_exposed_anchor": float(per[exposed, 1].mean()) if exposed.any() else 0.0,
            "min_lag_ms_exposed": sorted(set(per[exposed, 2].tolist())) if exposed.any() else [],
            "shared_samples_nearest_exposed": sorted(set(per[exposed, 3].tolist())) if exposed.any() else [],
            "own_window_density": float(per[:, 0].sum() / arm_rows.size)}


def commitment(fold_list: list[dict], meta: dict) -> str:
    h = hashlib.sha256()
    for f in fold_list:
        h.update(repr((f["subject"], f["run"], f["pair_seed"], f["group_seed"])).encode())
        for k in ("T", "base", "H1", "H2", "Z"):
            h.update(np.ascontiguousarray(f[k]).tobytes())
        for name in TRACK_B_ORDER:
            h.update(np.ascontiguousarray(track_b_arms(f)[name]).tobytes())
        for s in geo.STRIDES:
            ta = track_a_arms(f, s)
            for g in track_a_groups(s):
                for a in g:
                    h.update(repr((s, a)).encode())
                    h.update(np.ascontiguousarray(ta[a]).tobytes())
    for s in np.unique(np.asarray(meta["subject"])):
        sp = c0_splits(meta, int(s))
        h.update(repr((int(s), sp["pair_seed"], sp["group_seed"])).encode())
        for scheme in ("rw", "tg"):
            for te in sp[scheme]:
                h.update(np.ascontiguousarray(te).tobytes())
    return h.hexdigest()
