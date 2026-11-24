"""G2 read-only real-metadata adapter for PMIS (NO EEG, NO features, NO predictions, NO outcomes).

Reads ONLY the R4 metadata tables (G2_instances.npz, G2_distinct.npz, G2_sessions_PRIVATE.json,
G2_manifest.npz) and the R5 G1 private pair table, from paths given in the git-ignored
the runtime workspace. Never opens the covariance/EEGNet feature arrays or any prediction file.

What it verifies:
  1. frozen SHA-256 of the metadata files and of the vendored R4 code; exact npz key whitelists;
  2. metadata integrity (hash bijection, one canonical instance per trial, cumulative copy layout);
  3. S2 sorted-stem fold replay with the vendored frozen `G2c_manifest.fold_of_instances`: whole-file
     folds, every trial scored once, T and C equal to the frozen R4 manifest for all 5 seeds;
  4. the G1 pair table schema, and every seed-42 pair re-validated with the G2 contract
     (`check_base`/`check_pair`) against BOTH base variants, plus field-by-field agreement with an
     independent recomputation (alias = nearest later common training-side file, positions, sizes);
  5. mutation tests on real metadata: corrupted pairs/bases must be rejected;
  6. optional bit-for-bit reproduction of the pair table by the historical matching script;
  7. metadata-only capacities of the pre-declared controls, position strata and fit budgets.
Outputs: cohort-level aggregate JSON (safe for git) and a private JSON under data/ (git-ignored).
"""

from __future__ import annotations

from experiments.runtime import verify_recorded_input

import collections
import datetime as dt
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from data_preparation import hs_integrity as pv
from experiments.pmis_g2.contracts import ContractError, TrialRow, check_base, check_neighbour_arm, check_pair, check_placebo_arms
from data_preparation.hs_manifest import ARMS as R4_ARMS, fold_of_instances

READ_FILES = ("G2_instances.npz", "G2_distinct.npz", "G2_sessions_PRIVATE.json", "G2_manifest.npz")
EXPECTED_KEYS = {
    "G2_instances.npz": {"subject", "session", "file_run", "idx_in_file", "distinct", "label", "canonical",
                         "b_trial_id"},
    "G2_distinct.npz": {"label", "subject", "session", "origin_run", "v3_idx", "multiplicity", "hash"},
    "G2_manifest.npz": {"rows", "table"} | {f"exposed_S{s}_{seed}" for s in (1, 2) for seed in range(42, 47)},
}
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_SEED = 42
N_FOLDS = 5
SPLIT_CODE_S2 = 2
POSITION_CUT = 6                         # G1 metadata median of |pos_a - pos_b|, fixed before any outcome
BUDGETS = (5, 10, 20, 40, 80, 160, None)  # pairs per participant x class; None = all
BUDGET_RNG_KEY = 2027
MUTATION_SAMPLE = 200
MUTATION_RNG_KEY = 4242
NEIGHBOUR_RNG_KEY = 7101                 # outcome-blind control selection (draft, frozen at G3)
PLACEBO_RNG_KEY = 7102
PAIR_RECORD_KEYS = {
    "a", "a_hash", "a_prime_inst", "alias_file_run", "b", "b_hash", "b_prime_inst", "fails_canonical",
    "fails_multiset", "fold", "gap", "label", "n_common_alias_runs", "n_distinct_base", "n_rows_base_canonical",
    "n_rows_base_multiset", "origin_run", "placebo_candidates_same_origin_run",
    "placebo_candidates_session_class_same_alias_file", "pos_a", "pos_b", "pos_dist", "seed", "session",
    "subject"}


# --------------------------------------------------------------------------------------------- inputs
def local_paths(root: Path) -> dict[str, Path]:
    return {"r4_private": root / "data" / "hs_stroke"}


def npz_keys(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as z:
        return {n[:-4] if n.endswith(".npy") else n for n in z.namelist()}


def load_metadata(priv: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Key-whitelisted, pickle-free load of the four metadata files. Fails closed on any extra array."""
    for f in pv.FORBIDDEN_PRIVATE_FILES:
        if f in READ_FILES:
            raise RuntimeError("forbidden file in read list")
    key_ok = {}
    for name, keys in EXPECTED_KEYS.items():
        found = npz_keys(priv / name)
        key_ok[name] = found == keys
        if not key_ok[name]:
            raise RuntimeError(f"BLOCKED: unexpected array set in {name}")
    with np.load(priv / "G2_instances.npz", allow_pickle=False) as z:
        inst = {k: z[k] for k in EXPECTED_KEYS["G2_instances.npz"]}
    with np.load(priv / "G2_distinct.npz", allow_pickle=False) as z:
        dist = {k: z[k] for k in EXPECTED_KEYS["G2_distinct.npz"]}
    meta = json.loads((priv / "G2_sessions_PRIVATE.json").read_text())
    with np.load(priv / "G2_manifest.npz", allow_pickle=False) as z:
        man = {k: z[k] for k in ("rows", "table", "exposed_S2_42", "exposed_S2_43", "exposed_S2_44",
                                 "exposed_S2_45", "exposed_S2_46")}
    return inst, dist, meta, man, key_ok


def frozen_check(priv: Path) -> dict:
    out = {name: verify_recorded_input(priv / name, pv.historical_sha256(name))
           for name in pv.METADATA_FILES}
    out["all_match"] = all(out.values())
    return out


def stems_of(inst: dict, meta: dict) -> np.ndarray:
    """Exactly the S2 split unit of R4 G2c_manifest.main (run-file stem)."""
    sess = meta["sessions"]
    return np.array([f"{sess[s][0]}_{sess[s][1]}_task-motorimagery_run-{int(r):02d}_eeg"
                     for s, r in zip(inst["session"], inst["file_run"])])


# ------------------------------------------------------------------------------------------ integrity
def integrity(inst: dict, dist: dict) -> dict:
    d, canon = inst["distinct"], inst["canonical"].astype(bool)
    n_d = int(dist["label"].size)
    h = dist["hash"].astype(str)
    out = {"n_instances": int(inst["label"].size), "n_distinct": n_d,
           "n_participants": int(np.unique(dist["subject"]).size),
           "n_sessions": int(np.unique(dist["session"]).size)}
    out["hash_format_sha256_hex"] = bool(all(len(x) == 64 and all(c in "0123456789abcdef" for c in x) for x in h))
    out["hash_unique_over_trials"] = bool(np.unique(h).size == n_d)
    out["one_canonical_per_trial"] = bool(np.array_equal(np.sort(d[canon]), np.arange(n_d)))
    out["instance_label_session_subject_match_trial"] = bool(
        np.array_equal(inst["label"], dist["label"][d]) and np.array_equal(inst["session"], dist["session"][d])
        and np.array_equal(inst["subject"], dist["subject"][d]))
    out["canonical_in_origin_file"] = bool(np.array_equal(inst["file_run"][canon], dist["origin_run"][d[canon]]))
    gap = inst["file_run"] - dist["origin_run"][d]
    out["copies_later_same_earlier"] = [int(np.sum(~canon & (gap > 0))), int(np.sum(~canon & (gap == 0))),
                                        int(np.sum(~canon & (gap < 0)))]
    out["multiplicity_equals_instance_count"] = bool(np.array_equal(np.bincount(d, minlength=n_d),
                                                                    dist["multiplicity"]))
    key = inst["session"].astype(np.int64) * 1000 + inst["file_run"].astype(np.int64)
    pos_key = key * 100000 + inst["idx_in_file"].astype(np.int64)
    out["position_unique_within_file"] = bool(np.unique(pos_key).size == pos_key.size)
    trial_file = key * 1000000 + d.astype(np.int64)
    out["at_most_one_copy_per_trial_per_file"] = bool(np.unique(trial_file).size == trial_file.size)
    out["labels_binary"] = bool(set(np.unique(dist["label"]).tolist()) <= {1, 2})
    bools = [k for k, v in out.items() if isinstance(v, bool)]
    out["all_pass"] = bool(all(out[k] for k in bools) and out["copies_later_same_earlier"][1:] == [0, 0])
    return out


# --------------------------------------------------------------------------------------- fold algebra
def canonical_fold(inst: dict, fold: np.ndarray, n_d: int) -> np.ndarray:
    cf = np.full(n_d, -1, dtype=np.int16)
    canon = inst["canonical"].astype(bool)
    cf[inst["distinct"][canon]] = fold[canon]
    if (cf < 0).any():
        raise RuntimeError("trial without canonical instance")
    return cf


class FoldView:
    """Independent recomputation of one participant x fold (does not reuse the G1 code)."""

    def __init__(self, inst, dist, fold, f, subj, members, cf):
        d_all = inst["distinct"]
        self.f, self.subj = f, subj
        self.members = members
        tr = members[fold[members] != f]
        te = members[fold[members] == f]
        self.T = np.flatnonzero((cf == f) & (dist["subject"] == subj)).astype(np.int64)
        self.in_T = np.zeros(dist["label"].size, dtype=bool)
        self.in_T[self.T] = True
        is_x = self.in_T[d_all[tr]]
        self.base_multi = tr[~is_x]
        self.base_canon = self.base_multi[inst["canonical"][self.base_multi].astype(bool)]
        self.test_canon = te[inst["canonical"][te].astype(bool)]
        self.test_side_copies = collections.defaultdict(list)
        for i in te[~inst["canonical"][te].astype(bool)].tolist():
            self.test_side_copies[int(d_all[i])].append(i)
        self.alias: dict[int, dict[int, int]] = collections.defaultdict(dict)
        x_rows = tr[is_x]
        for i in x_rows[np.lexsort((inst["idx_in_file"][x_rows], inst["file_run"][x_rows]))].tolist():
            self.alias[int(d_all[i])].setdefault(int(inst["file_run"][i]), i)
        self.groups = collections.defaultdict(list)
        for dd in self.T.tolist():
            self.groups[(int(dist["session"][dd]), int(dist["origin_run"][dd]), int(dist["label"][dd]))].append(dd)
        self.pool = collections.defaultdict(set)   # non-test trials with a training-side COPY in file r
        nc = self.base_multi[~inst["canonical"][self.base_multi].astype(bool)]
        for i in nc.tolist():
            dd = int(d_all[i])
            self.pool[(int(dist["session"][dd]), int(dist["label"][dd]), int(inst["file_run"][i]))].add(dd)


def make_row(i: int, inst: dict, dist: dict, fold: np.ndarray, f: int, hashes: np.ndarray) -> TrialRow:
    dd = int(inst["distinct"][i])
    return TrialRow(row_id=f"i{i}", trial_id=f"d{dd}", waveform_sha256=str(hashes[dd]),
                    participant=f"P{int(inst['subject'][i])}", session=f"S{int(inst['session'][i])}",
                    origin_run=int(dist["origin_run"][dd]), file_run=int(inst["file_run"][i]),
                    label=int(inst["label"][i]), partition="test" if int(fold[i]) == f else "train",
                    canonical=bool(inst["canonical"][i]), fold=int(fold[i]))


def manifest_arms(man: dict, seed: int, unit: int, f: int) -> dict:
    t = man["table"]
    sel = t[(t[:, 0] == SPLIT_CODE_S2) & (t[:, 1] == seed) & (t[:, 2] == unit) & (t[:, 3] == f)]
    out = {}
    for arm in ("T", "C"):
        r = sel[sel[:, 4] == R4_ARMS.index(arm)]
        if r.shape[0] != 1:
            raise RuntimeError("manifest row missing or duplicated")
        out[arm] = man["rows"][int(r[0, 5]):int(r[0, 6])]
    return out


def replay_seed(inst, dist, stems, man, seed):
    fold = fold_of_instances("S2", seed, inst, stems)
    n_d = int(dist["label"].size)
    cf = canonical_fold(inst, fold, n_d)
    _, inv = np.unique(stems, return_inverse=True)
    lo = np.full(inv.max() + 1, 127, dtype=np.int16)
    hi = np.full(inv.max() + 1, -1, dtype=np.int16)
    np.minimum.at(lo, inv, fold.astype(np.int16))
    np.maximum.at(hi, inv, fold.astype(np.int16))
    whole_file = bool(np.array_equal(lo, hi))
    views, t_eq, c_eq, scored = {}, 0, 0, 0
    exposed = np.zeros(n_d, dtype=bool)
    hT, hC = hashlib.sha256(), hashlib.sha256()
    for subj in np.unique(inst["subject"]).tolist():
        members = np.flatnonzero(inst["subject"] == subj)
        for f in range(N_FOLDS):
            v = FoldView(inst, dist, fold, f, int(subj), members, cf)
            views[(int(subj), f)] = v
            scored += int(v.T.size)
            exposed[[dd for dd in v.alias]] = True
            arms = manifest_arms(man, seed, int(subj), f)
            c_ids = np.sort(inst["distinct"][v.base_canon])
            t_eq += int(np.array_equal(np.sort(arms["T"]), v.T))
            c_eq += int(np.array_equal(np.sort(arms["C"]), c_ids))
            hT.update(np.ascontiguousarray(v.T, dtype=np.int64).tobytes())
            hC.update(np.ascontiguousarray(c_ids, dtype=np.int64).tobytes())
    return {"fold": fold, "cf": cf, "views": views, "whole_file": whole_file, "scored": scored,
            "T_equal": t_eq, "C_equal": c_eq, "n_folds": len(views),
            "exposed_equal_manifest": bool(np.array_equal(exposed, man[f"exposed_S2_{seed}"].astype(bool))),
            "n_exposed": int(exposed.sum()), "T_digest": hT.hexdigest(), "C_digest": hC.hexdigest()}


def capacity(views: dict, dist: dict) -> tuple[collections.Counter, int, int]:
    """Per (participant, label) maximum matching capacity; complete-graph groups only (counted)."""
    per = collections.Counter()
    complete, incomplete = 0, 0
    for (subj, f), v in views.items():
        for (ses, orun, lab), ids in v.groups.items():
            el = [dd for dd in ids if v.alias.get(dd)]
            if len(el) < 2:
                continue
            runs = [set(v.alias[dd]) for dd in el]
            ok = all(runs[i] & runs[j] for i in range(len(el)) for j in range(i + 1, len(el)))
            complete += int(ok)
            incomplete += int(not ok)
            per[(subj, lab)] += len(el) // 2
    return per, complete, incomplete


# ----------------------------------------------------------------------------------- pair validation
def validate_pairs(pairs: list[dict], inst, dist, rep, hashes):
    fold, views = rep["fold"], rep["views"]
    canon_row = np.full(dist["label"].size, -1, dtype=np.int64)
    ci = np.flatnonzero(inst["canonical"].astype(bool))
    canon_row[inst["distinct"][ci]] = ci
    by_fold = collections.defaultdict(list)
    for k, p in enumerate(pairs):
        by_fold[(p["subject"], p["fold"])].append(k)
    fails = collections.Counter()
    field_mismatch = collections.Counter()
    contract_ok = {"canonical": 0, "multiset": 0}
    used = collections.Counter()
    fold_sizes, controls = [], []
    ctrl_ok = collections.Counter()
    ctrl_fail = collections.Counter()
    for key in sorted(by_fold):
        subj, f = key
        v = views[key]
        rows_canon = [make_row(i, inst, dist, fold, f, hashes) for i in v.base_canon.tolist()]
        rows_multi = [make_row(i, inst, dist, fold, f, hashes) for i in v.base_multi.tolist()]
        rows_test = [make_row(i, inst, dist, fold, f, hashes) for i in v.test_canon.tolist()]
        cbs = {"canonical": check_base(base=rows_canon, test=rows_test, canonical_base=True, test_fold=f),
               "multiset": check_base(base=rows_multi, test=rows_test, canonical_base=False, test_fold=f)}
        test_by_trial = {int(r.trial_id[1:]): r for r in rows_test}
        n_distinct_base = int(np.unique(inst["distinct"][v.base_multi]).size)
        fold_sizes.append({"subject": subj, "fold": f, "rows_canonical": int(v.base_canon.size),
                           "rows_multiset": int(v.base_multi.size), "distinct_base": n_distinct_base,
                           "n_scored": int(v.T.size), "n_pairs": len(by_fold[key])})
        unexposed = [dd for dd in v.T.tolist() if not v.alias.get(dd)]
        copy_row = {}
        for i in v.base_multi[~inst["canonical"][v.base_multi].astype(bool)].tolist():
            copy_row.setdefault((int(inst["distinct"][i]), int(inst["file_run"][i])), i)
        for k in by_fold[key]:
            p = pairs[k]
            a, b, ap, bp = p["a"], p["b"], p["a_prime_inst"], p["b_prime_inst"]
            used[(subj, f, a)] += 1
            used[(subj, f, b)] += 1
            ra, rb = test_by_trial.get(a), test_by_trial.get(b)
            if ra is None or rb is None:
                fails["pair_trial_not_scored"] += 1
                continue
            aa = make_row(ap, inst, dist, fold, f, hashes)
            bb = make_row(bp, inst, dist, fold, f, hashes)
            for variant, cb in cbs.items():
                try:
                    check_pair(cb, a=ra, b=rb, alias_a=aa, alias_b=bb, build_arms=False)
                    contract_ok[variant] += 1
                except ContractError as e:
                    fails[f"{variant}:{e.code}"] += 1
            # field-by-field agreement with the independent recomputation
            common = sorted(set(v.alias.get(a, {})) & set(v.alias.get(b, {})))
            r = common[0] if common else -1
            pos_a = int(inst["idx_in_file"][canon_row[a]])
            pos_b = int(inst["idx_in_file"][canon_row[b]])
            ses, orun, lab = int(dist["session"][a]), int(dist["origin_run"][a]), int(dist["label"][a])
            pl = v.pool.get((ses, lab, r), set())
            checks = {
                "subject": p["subject"] == int(dist["subject"][a]) == int(dist["subject"][b]),
                "session": p["session"] == ses == int(dist["session"][b]),
                "origin_run": p["origin_run"] == orun == int(dist["origin_run"][b]),
                "label": p["label"] == lab == int(dist["label"][b]),
                "fold": int(fold[canon_row[a]]) == int(fold[canon_row[b]]) == f,
                "alias_nearest_common_file": p["alias_file_run"] == r,
                "alias_rows": v.alias.get(a, {}).get(r) == ap and v.alias.get(b, {}).get(r) == bp,
                "alias_training_side": int(fold[ap]) != f and int(fold[bp]) != f,
                "gap": p["gap"] == r - orun,
                "n_common_alias_runs": p["n_common_alias_runs"] == len(common),
                "positions": (p["pos_a"], p["pos_b"], p["pos_dist"]) == (pos_a, pos_b, abs(pos_a - pos_b)),
                "base_sizes": (p["n_rows_base_canonical"], p["n_rows_base_multiset"], p["n_distinct_base"]) ==
                              (int(v.base_canon.size), int(v.base_multi.size), n_distinct_base),
                "hashes": p["a_hash"] == str(hashes[a]) and p["b_hash"] == str(hashes[b]),
                "g1_fail_lists_empty": p["fails_canonical"] == [] and p["fails_multiset"] == [],
                "placebo_counts": p["placebo_candidates_session_class_same_alias_file"] == len(pl) and
                                  p["placebo_candidates_same_origin_run"] ==
                                  sum(1 for x in pl if int(dist["origin_run"][x]) == orun),
            }
            for name, ok in checks.items():
                field_mismatch[name] += int(not ok)
            # control capacities (metadata only)
            g_any = [g for g in v.T.tolist() if g not in (a, b) and v.alias.get(g)
                     and int(dist["session"][g]) == ses and int(dist["label"][g]) == lab
                     and int(dist["origin_run"][g]) != orun]
            g_same_file = [g for g in g_any if r in v.alias[g]]
            bys_same = [u for u in unexposed if int(dist["session"][u]) == ses and int(dist["label"][u]) == lab]
            rec = {"pair_index": k, "subject": subj, "label": lab, "fold": f, "session": ses,
                   "pos_dist": abs(pos_a - pos_b), "placebo_candidates": len(pl),
                   "neighbour_any_file": len(g_any), "neighbour_same_alias_file": len(g_same_file),
                   "bystanders_fold": len(unexposed), "bystanders_session_class": len(bys_same)}
            # outcome-blind selection + contract validation of the pre-declared control arms
            if g_same_file:
                cand = sorted(g_same_file)
                g = cand[int(np.random.default_rng([NEIGHBOUR_RNG_KEY, PRIMARY_SEED, subj, f, a, b]).integers(len(cand)))]
                gp = v.alias[g][r]
                rg, rgp = test_by_trial[g], make_row(gp, inst, dist, fold, f, hashes)
                for variant, cb in cbs.items():
                    for src, src_alias in ((ra, aa), (rb, bb)):
                        try:
                            check_neighbour_arm(cb, a=src, alias_a=src_alias, g=rg, alias_g=rgp)
                            ctrl_ok[f"neighbour_{variant}"] += 1
                        except ContractError as e:
                            ctrl_fail[f"neighbour_{variant}:{e.code}"] += 1
                rec.update({"neighbour_g": g, "neighbour_g_prime_inst": gp})
            if len(pl) >= 2:
                prng = np.random.default_rng([PLACEBO_RNG_KEY, PRIMARY_SEED, subj, f, a, b])
                c, e = (int(x) for x in prng.choice(sorted(pl), size=2, replace=False))
                sign = int(prng.choice([-1, 1]))
                ci, ei = copy_row[(c, r)], copy_row[(e, r)]
                try:
                    check_placebo_arms(cbs["canonical"], c_copy=make_row(ci, inst, dist, fold, f, hashes),
                                       e_copy=make_row(ei, inst, dist, fold, f, hashes),
                                       alias_file=(f"P{subj}", f"S{ses}", r))
                    ctrl_ok["placebo_canonical"] += 1
                except ContractError as e_:
                    ctrl_fail[f"placebo_canonical:{e_.code}"] += 1
                try:
                    check_placebo_arms(cbs["multiset"], c_copy=make_row(ci, inst, dist, fold, f, hashes),
                                       e_copy=make_row(ei, inst, dist, fold, f, hashes),
                                       alias_file=(f"P{subj}", f"S{ses}", r))
                    ctrl_fail["placebo_multiset_UNEXPECTED_PASS"] += 1
                except ContractError as e_:
                    ctrl_ok[f"placebo_multiset_refused:{e_.code}"] += 1
                rec.update({"placebo_c": c, "placebo_e": e, "placebo_c_prime_inst": ci, "placebo_e_prime_inst": ei,
                            "placebo_sign": sign})
            controls.append(rec)
    reused = sum(1 for c in used.values() if c > 1)
    return {"contract_pass": contract_ok, "contract_fail_counts": dict(fails),
            "control_arm_pass_counts": dict(ctrl_ok), "control_arm_fail_counts": dict(ctrl_fail),
            "field_mismatch_counts": dict(field_mismatch), "trial_in_two_pairs_of_a_fold": reused,
            "fold_sizes": fold_sizes, "controls": controls}


# ------------------------------------------------------------------------------------ mutation tests
def mutation_tests(pairs, inst, dist, rep, hashes, n=MUTATION_SAMPLE):
    """Corrupt real pairs/bases in memory; every corruption must be rejected. Counts only."""
    fold, views = rep["fold"], rep["views"]
    rng = np.random.default_rng(MUTATION_RNG_KEY)
    pick = sorted(rng.choice(len(pairs), size=min(n, len(pairs)), replace=False).tolist())
    res = collections.defaultdict(lambda: {"attempted": 0, "rejected": 0})
    cache = {}
    for k in pick:
        p = pairs[k]
        subj, f, a, b, ap, bp = p["subject"], p["fold"], p["a"], p["b"], p["a_prime_inst"], p["b_prime_inst"]
        v = views[(subj, f)]
        if (subj, f) not in cache:
            rows_canon = [make_row(i, inst, dist, fold, f, hashes) for i in v.base_canon.tolist()]
            rows_test = [make_row(i, inst, dist, fold, f, hashes) for i in v.test_canon.tolist()]
            cache[(subj, f)] = (rows_canon, rows_test,
                                check_base(base=rows_canon, test=rows_test, canonical_base=True, test_fold=f),
                                {int(r.trial_id[1:]): r for r in rows_test})
        rows_canon, rows_test, cb, tbt = cache[(subj, f)]
        R = lambda i: make_row(i, inst, dist, fold, f, hashes)  # noqa: E731
        r_file = p["alias_file_run"]

        def attempt(name, fn):
            res[name]["attempted"] += 1
            try:
                fn()
            except ContractError:
                res[name]["rejected"] += 1

        ses, orun, lab = p["session"], p["origin_run"], p["label"]
        other_class = [g for g in v.T.tolist() if v.alias.get(g, {}).get(r_file) is not None
                       and int(dist["session"][g]) == ses and int(dist["label"][g]) != lab]
        if other_class:
            g = other_class[0]
            attempt("partner_other_class_same_alias_file",
                    lambda: check_pair(cb, a=tbt[a], b=tbt[g], alias_a=R(ap), alias_b=R(v.alias[g][r_file])))
        other_run = [g for g in v.T.tolist() if v.alias.get(g, {}).get(r_file) is not None
                     and int(dist["session"][g]) == ses and int(dist["label"][g]) == lab
                     and int(dist["origin_run"][g]) != orun]
        if other_run:
            g = other_run[0]
            attempt("partner_same_class_other_origin_run",
                    lambda: check_pair(cb, a=tbt[a], b=tbt[g], alias_a=R(ap), alias_b=R(v.alias[g][r_file])))
        if v.test_side_copies.get(a):
            attempt("test_side_copy_as_alias",
                    lambda: check_pair(cb, a=tbt[a], b=tbt[b], alias_a=R(v.test_side_copies[a][0]), alias_b=R(bp)))
        other_files = [x for x in v.alias[b] if x != r_file]
        if other_files:
            attempt("aliases_in_different_files",
                    lambda: check_pair(cb, a=tbt[a], b=tbt[b], alias_a=R(ap), alias_b=R(v.alias[b][other_files[0]])))
        attempt("canonical_row_as_alias",
                lambda: check_pair(cb, a=tbt[a], b=tbt[b], alias_a=tbt[a], alias_b=R(bp)))
        attempt("test_copy_injected_into_base",
                lambda: check_base(base=rows_canon + [R(ap)], test=rows_test, canonical_base=False, test_fold=f))
        base_multi_rows = [R(i) for i in v.base_multi.tolist()]
        nc = [r for r in base_multi_rows if not r.canonical]
        if nc:
            victim = nc[0].trial_id
            attempt("multiset_copy_without_canonical_source",
                    lambda: check_base(base=[r for r in base_multi_rows if not (r.trial_id == victim and r.canonical)],
                                       test=rows_test, canonical_base=False, test_fold=f))
        attempt("alternate_id_test_waveform_in_base",
                lambda: check_base(base=rows_canon + [TrialRow(
                    row_id="ghost", trial_id="ghost", waveform_sha256=tbt[a].waveform_sha256,
                    participant=tbt[a].participant, session=tbt[a].session, origin_run=rows_canon[0].origin_run,
                    file_run=rows_canon[0].origin_run, label=lab, partition="train", canonical=True,
                    fold=rows_canon[0].fold)], test=rows_test, canonical_base=True, test_fold=f))
        attempt("fold_label_forged_on_alias",
                lambda: check_pair(cb, a=tbt[a], b=tbt[b], alias_a=R(ap), alias_b=_replace_fold(R(bp), f)))
    out = {k: dict(v) for k, v in sorted(res.items())}
    out["all_rejected"] = all(v["attempted"] == v["rejected"] for v in res.values())
    out["n_pairs_sampled"] = len(pick)
    return out


def _replace_fold(r: TrialRow, f: int) -> TrialRow:
    from dataclasses import replace
    return replace(r, fold=f)


# ------------------------------------------------------------------------------ G1 reproducibility


# ------------------------------------------------------------------------------------------ summaries
def _q(v) -> dict:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "min": float(v.min()), "q1": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "q3": float(np.percentile(v, 75)), "max": float(v.max()),
            "mean": round(float(v.mean()), 3)}


def control_capacity(controls: list[dict]) -> dict:
    def per_participant_both(mask_fn, k=1):
        c = collections.Counter()
        for x in controls:
            if mask_fn(x):
                c[(x["subject"], x["label"])] += 1
        subj = {s for s, _ in c}
        return sum(1 for s in subj if c[(s, 1)] >= k and c[(s, 2)] >= k), \
            _q([min(c[(s, 1)], c[(s, 2)]) for s in {x["subject"] for x in controls}])
    out = {}
    specs = {
        "placebo_nontest_swap_session_class_same_alias_file": lambda x: x["placebo_candidates"] >= 2,
        "neighbour_test_trial_other_run_any_training_file": lambda x: x["neighbour_any_file"] >= 1,
        "neighbour_test_trial_other_run_same_alias_file": lambda x: x["neighbour_same_alias_file"] >= 1,
        "bystander_unexposed_same_session_class": lambda x: x["bystanders_session_class"] >= 1,
        "bystander_unexposed_same_fold": lambda x: x["bystanders_fold"] >= 1,
        "position_near_le_%d" % POSITION_CUT: lambda x: x["pos_dist"] <= POSITION_CUT,
        "position_far_gt_%d" % POSITION_CUT: lambda x: x["pos_dist"] > POSITION_CUT,
    }
    for name, fn in specs.items():
        n_pairs = sum(1 for x in controls if fn(x))
        both1, minq = per_participant_both(fn, 1)
        both10, _ = per_participant_both(fn, 10)
        out[name] = {"pairs": n_pairs, "participants_both_classes_ge1": both1,
                     "participants_both_classes_ge10": both10, "min_class_pairs_per_participant": minq}
    out["neighbour_candidates_per_pair_any_file"] = _q([x["neighbour_any_file"] for x in controls])
    out["bystanders_same_session_class_per_pair"] = _q([x["bystanders_session_class"] for x in controls])
    return out


def budget_coverage(pairs: list[dict]) -> list[dict]:
    """Outcome-blind fit budgets: k pairs per participant x class (frozen RNG on metadata order)."""
    groups = collections.defaultdict(list)
    for k, p in enumerate(pairs):
        groups[(p["subject"], p["label"])].append(k)
    all_sessions = {(p["subject"], p["session"]) for p in pairs}
    all_folds = {(p["subject"], p["fold"]) for p in pairs}
    out = []
    for budget in BUDGETS:
        chosen = []
        for (subj, lab), idx in sorted(groups.items()):
            idx = sorted(idx, key=lambda k: (pairs[k]["fold"], pairs[k]["session"], pairs[k]["origin_run"],
                                             pairs[k]["pos_a"], pairs[k]["pos_b"]))
            perm = np.random.default_rng([BUDGET_RNG_KEY, PRIMARY_SEED, subj, lab]).permutation(len(idx))
            chosen += [idx[i] for i in perm[: (len(idx) if budget is None else budget)]]
        sel = [pairs[k] for k in chosen]
        subj_lab = collections.Counter((p["subject"], p["label"]) for p in sel)
        subjects = {p["subject"] for p in pairs}
        folds_per = collections.defaultdict(set)
        for p in sel:
            folds_per[p["subject"]].add(p["fold"])
        out.append({"pairs_per_participant_class": "all" if budget is None else budget,
                    "pairs": len(sel), "fits_primary_canonical": 2 * len(sel),
                    "participants_with_both_classes": sum(1 for s in subjects
                                                          if subj_lab[(s, 1)] and subj_lab[(s, 2)]),
                    "min_folds_covered_per_participant": min(len(folds_per[s]) for s in subjects),
                    "participant_folds_covered": len({(p["subject"], p["fold"]) for p in sel}),
                    "participant_folds_total": len(all_folds),
                    "sessions_covered": len({(p["subject"], p["session"]) for p in sel}),
                    "sessions_with_pairs_total": len(all_sessions)})
    return out


def pairs_schema(private: dict) -> dict:
    ps = private["per_seed"]
    out = {"seeds": sorted(ps), "stage": private.get("stage")}
    recs = ps[str(PRIMARY_SEED)]["pairs"]
    out["seed42_is_list"] = isinstance(recs, list)
    out["seed42_records"] = len(recs)
    out["record_keys_exact"] = all(set(r) == PAIR_RECORD_KEYS for r in recs)
    ints = PAIR_RECORD_KEYS - {"a_hash", "b_hash", "fails_canonical", "fails_multiset"}
    out["record_types_ok"] = all(all(isinstance(r[k], int) and not isinstance(r[k], bool) for k in ints)
                                 and isinstance(r["a_hash"], str) and isinstance(r["b_hash"], str)
                                 and isinstance(r["fails_canonical"], list) and isinstance(r["fails_multiset"], list)
                                 for r in recs)
    out["record_value_domains_ok"] = all(r["seed"] == PRIMARY_SEED and r["label"] in (1, 2)
                                         and 0 <= r["fold"] < N_FOLDS and r["gap"] >= 1 for r in recs)
    out["sensitivity_seeds_pairs_omitted"] = all(isinstance(ps[s]["pairs"], str) for s in ps if s != str(PRIMARY_SEED))
    return out


def pair_table_digest(pairs: list[dict]) -> str:
    """Commitment to the seed-42 pair selection (index tuples only), for the future G3 freeze."""
    h = hashlib.sha256()
    for p in pairs:
        h.update(json.dumps([p["subject"], p["fold"], p["a"], p["b"], p["a_prime_inst"], p["b_prime_inst"],
                             p["alias_file_run"]]).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ main
