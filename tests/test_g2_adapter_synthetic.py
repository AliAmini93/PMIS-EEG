"""The real-metadata adapter exercised end-to-end on a SYNTHETIC cumulative release (no real data)."""

import collections
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from data_preparation import hs_metadata as ad  # noqa: E402
from data_preparation import hs_manifest as g2c  # noqa: E402


def synthetic_release(n_subj=2, n_ses=3, runs=4, per_run=8):
    inst, dist, sessions = collections.defaultdict(list), collections.defaultdict(list), []
    for s in range(n_subj):
        for se in range(n_ses):
            si = len(sessions)
            sessions.append([f"P{s:02d}", f"V{se:02d}"])
            base = len(dist["label"])
            for r in range(1, runs + 1):
                for i in range(per_run):
                    d = len(dist["label"])
                    for k, v in (("subject", s), ("session", si), ("origin_run", r), ("label", 1 + i % 2),
                                 ("v3_idx", i), ("hash", f"{d:064x}"), ("multiplicity", 0)):
                        dist[k].append(v)
            off = 0
            for k in range(1, runs + 1):
                rows = [base + (r - 1) * per_run + i for r in range(1, k + 1) for i in range(per_run)]
                for j, d in enumerate(rows):
                    for key, v in (("subject", s), ("session", si), ("file_run", k), ("idx_in_file", j),
                                   ("distinct", d), ("label", dist["label"][d]),
                                   ("canonical", dist["origin_run"][d] == k), ("b_trial_id", off + j)):
                        inst[key].append(v)
                    dist["multiplicity"][d] += 1
                off += len(rows)
    inst = {k: np.asarray(v) for k, v in inst.items()}
    dist = {k: np.asarray(v) for k, v in dist.items()}
    dist["hash"] = dist["hash"].astype("<U64")
    return inst, dist, {"sessions": sessions}


def synthetic_manifest(inst, dist, stems, seeds=(42,)):
    rows, table, pos, out = [], [], 0, {}
    for seed in seeds:
        fold = g2c.fold_of_instances("S2", seed, inst, stems)
        stats = collections.Counter()
        stats["z_fallback"], stats["lcb_fallback"] = collections.Counter(), collections.Counter()
        exp = np.zeros(dist["label"].size, dtype=bool)
        for unit in np.unique(inst["subject"]):
            members = np.flatnonzero(inst["subject"] == unit)
            for f, arms in g2c.build_unit("S2", seed, int(unit), fold, members, inst, dist, dist["hash"], stats):
                exp[np.intersect1d(arms["T"], arms["L"])] = True
                for arm in g2c.ARMS:
                    v = arms[arm]
                    table.append((2, seed, int(unit), f, g2c.ARMS.index(arm), pos, pos + v.size))
                    rows.append(v)
                    pos += v.size
        out[f"exposed_S2_{seed}"] = exp
    out["rows"] = np.concatenate(rows)
    out["table"] = np.asarray(table, dtype=np.int64)
    return out


def synthetic_pairs(rep, dist, inst):
    """Nearest-common-file pairing on the adapter's own fold views (record schema of R5 G1)."""
    pairs = []
    ci = np.flatnonzero(inst["canonical"])
    pos = {int(inst["distinct"][i]): int(inst["idx_in_file"][i]) for i in ci}
    for (subj, f), v in sorted(rep["views"].items()):
        n_distinct = int(np.unique(inst["distinct"][v.base_multi]).size)
        for key in sorted(v.groups):
            el = [d for d in v.groups[key] if v.alias.get(d)]
            for a, b in zip(el[0::2], el[1::2]):
                common = sorted(set(v.alias[a]) & set(v.alias[b]))
                r = common[0]
                pl = v.pool.get((key[0], key[2], r), set())
                pairs.append({"seed": 42, "subject": subj, "session": key[0], "fold": f, "origin_run": key[1],
                              "label": key[2], "a": a, "b": b, "a_hash": str(dist["hash"][a]),
                              "b_hash": str(dist["hash"][b]), "alias_file_run": r, "n_common_alias_runs": len(common),
                              "gap": r - key[1], "pos_a": pos[a], "pos_b": pos[b], "pos_dist": abs(pos[a] - pos[b]),
                              "a_prime_inst": v.alias[a][r], "b_prime_inst": v.alias[b][r],
                              "n_rows_base_multiset": int(v.base_multi.size),
                              "n_rows_base_canonical": int(v.base_canon.size), "n_distinct_base": n_distinct,
                              "placebo_candidates_session_class_same_alias_file": len(pl),
                              "placebo_candidates_same_origin_run": sum(
                                  1 for x in pl if int(dist["origin_run"][x]) == key[1]),
                              "fails_multiset": [], "fails_canonical": []})
    return pairs


class AdapterOnSyntheticRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst, cls.dist, cls.meta = synthetic_release()
        cls.stems = ad.stems_of(cls.inst, cls.meta)
        cls.man = synthetic_manifest(cls.inst, cls.dist, cls.stems)
        cls.rep = ad.replay_seed(cls.inst, cls.dist, cls.stems, cls.man, 42)
        cls.pairs = synthetic_pairs(cls.rep, cls.dist, cls.inst)
        cls.hashes = cls.dist["hash"].astype(str)

    def test_integrity_and_replay_against_vendored_manifest(self):
        self.assertTrue(ad.integrity(self.inst, self.dist)["all_pass"])
        r = self.rep
        self.assertTrue(r["whole_file"])
        self.assertEqual(r["scored"], self.dist["label"].size)
        self.assertEqual((r["T_equal"], r["C_equal"]), (r["n_folds"], r["n_folds"]))
        self.assertTrue(r["exposed_equal_manifest"])

    def test_every_synthetic_pair_passes_both_bases_with_no_field_mismatch(self):
        self.assertGreater(len(self.pairs), 0)
        val = ad.validate_pairs(self.pairs, self.inst, self.dist, self.rep, self.hashes)
        self.assertEqual(val["contract_pass"], {"canonical": len(self.pairs), "multiset": len(self.pairs)})
        self.assertEqual(val["contract_fail_counts"], {})
        self.assertEqual(sum(val["field_mismatch_counts"].values()), 0)
        self.assertEqual(val["trial_in_two_pairs_of_a_fold"], 0)

    def test_tampered_pair_records_are_detected(self):
        bad = [dict(p) for p in self.pairs]
        bad[0]["alias_file_run"] += 1
        bad[1]["pos_a"] += 1
        bad[2]["b_prime_inst"] = bad[2]["a_prime_inst"]
        val = ad.validate_pairs(bad, self.inst, self.dist, self.rep, self.hashes)
        self.assertGreaterEqual(val["field_mismatch_counts"]["alias_nearest_common_file"], 1)
        self.assertGreaterEqual(val["field_mismatch_counts"]["positions"], 1)
        self.assertGreater(sum(val["contract_fail_counts"].values()), 0)

    def test_mutations_on_metadata_are_all_rejected(self):
        mut = ad.mutation_tests(self.pairs, self.inst, self.dist, self.rep, self.hashes, n=10)
        self.assertTrue(mut["all_rejected"])
        self.assertGreaterEqual(len([k for k in mut if isinstance(mut[k], dict)]), 5)

    def test_schema_budget_and_capacity_helpers(self):
        sch = ad.pairs_schema({"stage": "s", "per_seed": {"42": {"pairs": self.pairs}, "43": {"pairs": "omitted"}}})
        self.assertTrue(sch["record_keys_exact"] and sch["record_types_ok"] and sch["record_value_domains_ok"])
        cap, complete, incomplete = ad.capacity(self.rep["views"], self.dist)
        self.assertEqual(sum(cap.values()), len(self.pairs))
        self.assertEqual(incomplete, 0)
        b = ad.budget_coverage(self.pairs)
        self.assertEqual(b[-1]["pairs"], len(self.pairs))
        self.assertTrue(all(x["pairs"] <= len(self.pairs) for x in b))

    def test_adapter_source_never_opens_feature_or_prediction_files(self):
        from data_preparation import hs_integrity as pv
        src = (ROOT / "data_preparation" / "hs_metadata.py").read_text()
        for name in pv.FORBIDDEN_PRIVATE_FILES:              # referenced only through the constant
            self.assertNotIn(name, src)
        loads = re.findall(r"np\.load\(([^,]+),", src)
        self.assertEqual(len(loads), 3)
        for arg in loads:
            self.assertTrue(any(f in arg for f in ad.READ_FILES), arg)
        self.assertNotIn("loadmat", src)
        for forbidden in ("import torch", "sklearn.linear_model", "LogisticRegression", "scipy.io", "import mne",
                          "TSLR", "fit("):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
