"""Outcome-blind exact matching of HS-Stroke v1 test identities and training-side aliases."""
from __future__ import annotations

import collections
import functools

import numpy as np

from .hs_manifest import ARMS as R4_ARMS, fold_of_instances

PAIRING_SEED = 42


SPLIT_CODE_S2 = 2


N_FOLDS = 5


EXACT_MATCHING_MAX_N = 18


FAIL_KEYS = ("same_trial", "identical_hash", "label", "session_or_subject", "origin_run", "alias_identity",
             "alias_is_canonical", "alias_file_run", "alias_not_later", "alias_in_test_fold", "not_scored",
             "base_contains_test_identity", "rows", "distinct_support", "label_multiset",
             "exposed_identity_A", "exposed_identity_B", "duplicate_row_in_arm")


def canonical_fold(inst: dict, fold: np.ndarray, n_d: int) -> np.ndarray:
    cf = np.full(n_d, -1, dtype=np.int16)
    canon = inst["canonical"].astype(bool)
    cf[inst["distinct"][canon]] = fold[canon]
    if (cf < 0).any():
        raise RuntimeError("a distinct trial has no canonical instance")
    return cf


def enumerate_fold(inst: dict, dist: dict, fold: np.ndarray, f: int, subj: int, members: np.ndarray,
                   cf: np.ndarray) -> dict:
    """Scored test set, base (both variants), training-side alias map and test groups for one
    participant x fold. Alias rows come ONLY from the training side of the fold."""
    d_all = inst["distinct"]
    tr = members[fold[members] != f]
    T = np.flatnonzero((cf == f) & (dist["subject"] == subj)).astype(np.int64)
    in_T = np.zeros(dist["label"].size, dtype=bool)
    in_T[T] = True
    is_x = in_T[d_all[tr]]
    base_multi = tr[~is_x]
    base_canon = base_multi[inst["canonical"][base_multi].astype(bool)]
    x_rows = tr[is_x]
    order = np.lexsort((inst["idx_in_file"][x_rows], inst["file_run"][x_rows]))
    alias: dict[int, dict[int, int]] = collections.defaultdict(dict)
    dup_same_file = 0
    for i in x_rows[order].tolist():
        dd, r = int(d_all[i]), int(inst["file_run"][i])
        if r in alias[dd]:
            dup_same_file += 1
            continue
        alias[dd][r] = i
    groups: dict[tuple, list[int]] = collections.defaultdict(list)
    for dd in T.tolist():
        groups[(int(dist["session"][dd]), int(dist["origin_run"][dd]), int(dist["label"][dd]))].append(dd)
    # placebo pools: non-test distinct trials with a training-side COPY in a given file-run (session, label)
    pool: dict[tuple, set] = collections.defaultdict(set)
    nc = base_multi[~inst["canonical"][base_multi].astype(bool)]
    for i in nc.tolist():
        dd = int(d_all[i])
        pool[(int(dist["session"][dd]), int(dist["label"][dd]), int(inst["file_run"][i]))].add(dd)
    return {"T": T, "in_T": in_T, "base_multi": base_multi, "base_canon": base_canon, "alias": alias,
            "groups": groups, "x_rows": x_rows, "dup_same_file": dup_same_file, "placebo_pool": pool}


def manifest_arms(man, seed: int, unit: int, f: int) -> dict:
    t = man["table"]
    sel = t[(t[:, 0] == SPLIT_CODE_S2) & (t[:, 1] == seed) & (t[:, 2] == unit) & (t[:, 3] == f)]
    out = {}
    for arm in ("T", "C"):
        r = sel[sel[:, 4] == R4_ARMS.index(arm)]
        if r.shape[0] != 1:
            raise RuntimeError("manifest row missing or duplicated")
        out[arm] = man["rows"][int(r[0, 5]):int(r[0, 6])]
    return out


def max_matching_size(n: int, adj: list[int]) -> int:
    """Exact maximum matching of a small general graph by bitmask DP (adj[v] = bitmask of neighbours)."""
    @functools.lru_cache(maxsize=None)
    def best(mask: int) -> int:
        if mask == 0:
            return 0
        v = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << v)
        b = best(rest)
        cand = adj[v] & rest
        while cand:
            u = (cand & -cand).bit_length() - 1
            cand &= cand - 1
            b = max(b, 1 + best(rest & ~(1 << u)))
        return b
    return best((1 << n) - 1)


def pair_group(ids: list[int], alias: dict, pos: dict, rng: np.random.Generator):
    """Outcome-blind pairing inside one (session, origin_run, label) group of eligible test trials.
    Order = canonical trial position, then a seeded permutation; edge = common training-side alias
    file-run. Returns realised pairs (a, b, sorted common alias runs) and matching statistics."""
    ids = sorted(ids, key=lambda dd: (pos[dd], dd))
    ids = [ids[i] for i in rng.permutation(len(ids))]
    n = len(ids)
    runs = [set(alias[dd]) for dd in ids]
    adj = [0] * n
    n_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if runs[i] & runs[j]:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                n_edges += 1
    complete = n_edges == n * (n - 1) // 2
    used = [False] * n
    pairs = []
    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if not used[j] and (adj[i] >> j) & 1:
                used[i] = used[j] = True
                pairs.append((ids[i], ids[j], sorted(runs[i] & runs[j])))
                break
    if complete:
        exact, method = n // 2, "complete_graph"
    elif n <= EXACT_MATCHING_MAX_N:
        exact, method = max_matching_size(n, adj), "bitmask_dp"
    else:
        exact, method = None, "greedy_lower_bound"
    return pairs, {"n_eligible": n, "n_edges": n_edges, "complete": complete, "greedy_pairs": len(pairs),
                   "max_matching": exact, "method": method}


def check_arms(arm_a: np.ndarray, arm_b: np.ndarray, a: int, b: int, inst: dict, dist: dict,
               in_T: np.ndarray) -> list[str]:
    """Invariants on two explicit hypothetical training arms (index multisets over instances)."""
    fails = []
    d = inst["distinct"]
    if arm_a.size != arm_b.size:
        fails.append("rows")
    da, db = set(d[arm_a].tolist()), set(d[arm_b].tolist())
    if len(da) != len(db):
        fails.append("distinct_support")
    if collections.Counter(inst["label"][arm_a].tolist()) != collections.Counter(inst["label"][arm_b].tolist()):
        fails.append("label_multiset")
    if {dd for dd in da if in_T[dd]} != {a}:
        fails.append("exposed_identity_A")
    if {dd for dd in db if in_T[dd]} != {b}:
        fails.append("exposed_identity_B")
    if np.unique(arm_a).size != arm_a.size or np.unique(arm_b).size != arm_b.size:
        fails.append("duplicate_row_in_arm")
    return fails


def check_pair_arms(a: int, b: int, ap: int, bp: int, f: int, fold: np.ndarray, base_rows: np.ndarray,
                    inst: dict, dist: dict, in_T: np.ndarray) -> list[str]:
    """All PMIS pair invariants for test trials a, b with alias instance rows ap, bp on top of base_rows.
    Returns the list of failed invariant names (empty = eligible)."""
    fails = []
    if a == b:
        fails.append("same_trial")
    if dist["hash"][a] == dist["hash"][b]:
        fails.append("identical_hash")
    if dist["label"][a] != dist["label"][b]:
        fails.append("label")
    if dist["session"][a] != dist["session"][b] or dist["subject"][a] != dist["subject"][b]:
        fails.append("session_or_subject")
    if dist["origin_run"][a] != dist["origin_run"][b]:
        fails.append("origin_run")
    if inst["distinct"][ap] != a or inst["distinct"][bp] != b:
        fails.append("alias_identity")
    if inst["canonical"][ap] or inst["canonical"][bp]:
        fails.append("alias_is_canonical")
    if inst["file_run"][ap] != inst["file_run"][bp]:
        fails.append("alias_file_run")
    if inst["file_run"][ap] <= dist["origin_run"][a] or inst["file_run"][bp] <= dist["origin_run"][b]:
        fails.append("alias_not_later")
    if fold[ap] == f or fold[bp] == f:
        fails.append("alias_in_test_fold")
    if not (in_T[a] and in_T[b]):
        fails.append("not_scored")
    base_d = inst["distinct"][base_rows]
    if in_T[base_d].any():
        fails.append("base_contains_test_identity")
    arm_a = np.concatenate([base_rows, np.array([ap], dtype=base_rows.dtype)])
    arm_b = np.concatenate([base_rows, np.array([bp], dtype=base_rows.dtype)])
    fails += check_arms(arm_a, arm_b, a, b, inst, dist, in_T)
    return fails


def run_seed(inst: dict, dist: dict, stems: np.ndarray, seed: int, man=None) -> tuple[list[dict], dict]:
    fold = fold_of_instances("S2", seed, inst, stems)
    n_d = int(dist["label"].size)
    cf = canonical_fold(inst, fold, n_d)
    canon_idx = np.flatnonzero(inst["canonical"].astype(bool))
    pos = {int(inst["distinct"][i]): int(inst["idx_in_file"][i]) for i in canon_idx.tolist()}
    st = collections.Counter()
    st["fail_counts"] = collections.Counter()
    st["gap_hist"], st["n_common_hist"], st["group_method"] = collections.Counter(), collections.Counter(), collections.Counter()
    st["fold_rows_multi"], st["fold_rows_canon"], st["fold_distinct_base"] = [], [], []
    st["manifest_T_equal"], st["manifest_C_equal"], st["manifest_folds"] = 0, 0, 0
    st["scored_total"] = 0
    records = []
    for subj in np.unique(inst["subject"]).tolist():
        members = np.flatnonzero(inst["subject"] == subj)
        for f in range(N_FOLDS):
            fo = enumerate_fold(inst, dist, fold, f, int(subj), members, cf)
            st["scored_total"] += int(fo["T"].size)
            st["dup_same_file"] += fo["dup_same_file"]
            st["fold_rows_multi"].append(int(fo["base_multi"].size))
            st["fold_rows_canon"].append(int(fo["base_canon"].size))
            st["fold_distinct_base"].append(int(np.unique(inst["distinct"][fo["base_multi"]]).size))
            if man is not None:
                arms = manifest_arms(man, seed, int(subj), f)
                st["manifest_folds"] += 1
                st["manifest_T_equal"] += int(np.array_equal(np.sort(arms["T"]), np.sort(fo["T"])))
                st["manifest_C_equal"] += int(np.array_equal(np.sort(arms["C"]),
                                                             np.sort(inst["distinct"][fo["base_canon"]])))
            rng = np.random.default_rng([PAIRING_SEED, seed, int(subj), f])
            for key in sorted(fo["groups"]):
                ids = fo["groups"][key]
                elig = [dd for dd in ids if fo["alias"].get(dd)]
                st["groups"] += 1
                st["test_identities"] += len(ids)
                st["exposed_identities"] += len(elig)
                st["unexposed_identities"] += len(ids) - len(elig)
                if len(elig) < 2:
                    st["groups_lt2_eligible"] += 1
                    st["eligible_singletons"] += len(elig)
                    continue
                pairs, info = pair_group(elig, fo["alias"], pos, rng)
                st["groups_ge2_eligible"] += 1
                st["group_method"][info["method"]] += 1
                st["groups_complete"] += int(info["complete"])
                if info["max_matching"] is not None:
                    st["capacity_exact"] += info["max_matching"]
                    st["greedy_short_of_max"] += int(info["greedy_pairs"] < info["max_matching"])
                else:
                    st["capacity_lower_bound_only"] += info["greedy_pairs"]
                st["eligible_unpaired"] += info["n_eligible"] - 2 * info["greedy_pairs"]
                ses, orun, lab = key
                for a, b, common in pairs:
                    r = common[0]
                    ap, bp = fo["alias"][a][r], fo["alias"][b][r]
                    fm = check_pair_arms(a, b, ap, bp, f, fold, fo["base_multi"], inst, dist, fo["in_T"])
                    fc = check_pair_arms(a, b, ap, bp, f, fold, fo["base_canon"], inst, dist, fo["in_T"])
                    for x in fm + fc:
                        st["fail_counts"][x] += 1
                    st["pairs_checked"] += 1
                    st["pairs_failed"] += int(bool(fm or fc))
                    gap = int(r - orun)
                    st["gap_hist"][gap] += 1
                    st["n_common_hist"][len(common)] += 1
                    n_pl = len(fo["placebo_pool"].get((ses, lab, r), set()))
                    n_pl_run = len({dd for dd in fo["placebo_pool"].get((ses, lab, r), set())
                                    if int(dist["origin_run"][dd]) == orun})
                    records.append({
                        "seed": seed, "subject": int(subj), "session": ses, "fold": f, "origin_run": orun,
                        "label": lab, "a": int(a), "b": int(b), "a_hash": str(dist["hash"][a]),
                        "b_hash": str(dist["hash"][b]), "alias_file_run": int(r), "n_common_alias_runs": len(common),
                        "gap": gap, "pos_a": pos[a], "pos_b": pos[b], "pos_dist": abs(pos[a] - pos[b]),
                        "a_prime_inst": int(ap), "b_prime_inst": int(bp),
                        "n_rows_base_multiset": int(fo["base_multi"].size),
                        "n_rows_base_canonical": int(fo["base_canon"].size),
                        "n_distinct_base": int(np.unique(inst["distinct"][fo["base_multi"]]).size),
                        "placebo_candidates_session_class_same_alias_file": n_pl,
                        "placebo_candidates_same_origin_run": n_pl_run,
                        "fails_multiset": fm, "fails_canonical": fc,
                    })
    if st["scored_total"] != n_d:
        raise RuntimeError("not every distinct trial is scored exactly once")
    return records, st
