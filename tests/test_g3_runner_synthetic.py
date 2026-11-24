"""G3 runner audited end-to-end on SYNTHETIC metadata and SYNTHETIC SPD covariances (no real data)."""

import copy
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import test_g2_adapter_synthetic as tsyn  # noqa: E402
from data_preparation import hs_metadata as ad  # noqa: E402
from experiments.pmis_g2 import g3  # noqa: E402
from experiments.pmis_g2.contracts import ContractError  # noqa: E402
from models.vendor_r4 import r4_models as rm  # noqa: E402


def synthetic_covs(dist, seed=0, p=11):
    rng = np.random.default_rng(seed)
    n = dist["label"].size
    a = rng.standard_normal((n, p, 3 * p))
    c = a @ np.swapaxes(a, 1, 2) / (3 * p) + 0.1 * np.eye(p)
    c[dist["label"] == 2] += 0.3 * np.diag(np.linspace(0.5, 1.5, p))
    return c


def build(n_subj=3, n_ses=4, seeds=(42, 43)):
    inst, dist, meta = tsyn.synthetic_release(n_subj=n_subj, n_ses=n_ses)
    stems = ad.stems_of(inst, meta)
    man = tsyn.synthetic_manifest(inst, dist, stems, seeds=seeds)
    pairs = {}
    for s in seeds:
        rep = ad.replay_seed(inst, dist, stems, man, s)
        pairs[s] = [dict(p, seed=s) for p in tsyn.synthetic_pairs(rep, dist, inst)]
    plan = g3.make_plan(inst, dist, meta, man, pairs)
    return inst, dist, plan


class RunnerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst, cls.dist, cls.plan = build()
        cls.covs = synthetic_covs(cls.dist)
        cls.y = cls.dist["label"].astype(np.int64)
        cls.ex = g3.execute(cls.plan, cls.covs, cls.y, workers=1, log=lambda m: None, monitor=False)

    def test_plan_has_controls_and_commitment(self):
        pairs = [p for t in self.plan["seeds"][42]["tasks"] for p in t["pairs"]]
        self.assertGreater(len(pairs), 0)
        self.assertGreater(sum(p["g"] >= 0 for p in pairs), 0, "synthetic fixture must exercise C1")
        self.assertGreater(sum(p["c"] >= 0 for p in pairs), 0, "synthetic fixture must exercise C2")
        self.assertEqual(len(self.plan["commitment"]), 64)
        self.assertNotIn("C1_neighbour", self.plan["fit_counts"]["43"])

    def test_every_planned_fit_ran_and_analysis_schema(self):
        self.assertEqual(sum(r["stats"]["fits"] for r in self.ex["results"]), self.plan["fit_counts"]["total"])
        pub, priv = g3.analyse(self.ex["results"])
        self.assertEqual(set(pub), {"primary", "secondary", "descriptive", "pair_counts"})
        self.assertEqual(set(pub["secondary"]), set(g3.SECONDARY_FAMILY))
        self.assertIn("exact_sign_flip_p", pub["primary"])
        self.assertTrue(np.isfinite(pub["primary"]["mean_points"]))
        for k, v in pub["descriptive"].items():
            if isinstance(v, dict) and "mean_points" in v:
                self.assertNotIn("exact_sign_flip_p", v, k)       # descriptive: no p-based claim
        for v in pub["secondary"].values():
            if "exact_sign_flip_p" in v:
                self.assertIsNotNone(v["holm_p_secondary_family"])
        self.assertEqual(len(priv["participant_effects_primary"]), 3)
        st = g3.run_statistics(self.ex["results"])
        self.assertEqual(st["reference_independent_recomputations_passed"], len(self.ex["results"]))

    def test_pool_equals_sequential(self):
        ex2 = g3.execute(self.plan, self.covs, self.y, workers=2, log=lambda m: None, monitor=False)
        key = lambda res: (res["seed"], res["subject"], res["fold"])  # noqa: E731
        a = {key(r): r["records"] for r in self.ex["results"]}
        b = {key(r): r["records"] for r in ex2["results"]}
        self.assertEqual(set(a), set(b))
        for k in a:
            for ra, rb in zip(a[k], b[k]):
                for f in ra:
                    if isinstance(ra[f], float) and np.isnan(ra[f]):
                        self.assertTrue(np.isnan(rb[f]))
                    else:
                        self.assertEqual(ra[f], rb[f], f)

    def test_arm_reference_is_exact_and_per_arm(self):
        g3.G["covs"], g3.G["y"] = self.covs, self.y
        t = self.plan["seeds"][42]["tasks"][0]
        pr = t["pairs"][0]
        ones = np.ones(t["base_canon"].size, dtype=np.int64)
        A = g3.fit_score(t["base_canon"], ones, t["T"], add=pr["a"])
        B = g3.fit_score(t["base_canon"], ones, t["T"], add=pr["b"])
        base_ref = rm.riemann_mean_w(self.covs[t["base_canon"]], ones)
        self.assertFalse(np.array_equal(A.ref, B.ref))
        self.assertFalse(np.array_equal(A.ref, base_ref))
        if pr["c"] >= 0:
            C = g3.fit_score(t["base_canon"], ones, np.array([pr["a"]]), extra_copy_of=pr["c"])
            self.assertFalse(np.array_equal(C.ref, base_ref))


class AnalysisFormulas(unittest.TestCase):
    def rec(self, k, participant, label, **kw):
        base = dict(k=k, label=label, pos_dist=3, pAA=.8, pAB=.6, pBA=.5, pBB=.7, cAA=True, cAB=False, cBA=False,
                    cBB=True, off_fold=.01, off_session_class=np.nan, n_off_fold=5, n_off_session_class=0,
                    mAA=.8, mAB=.7, mBA=.6, mBB=.7, pAG=.55, pBG=.45, pAC=.6, pBC=.5, pAE=.62, pBE=.52, sign=1)
        base.update(kw)
        return base

    def test_estimands_match_hand_computation(self):
        res = []
        for subj in range(3):
            recs = [self.rec(10 * subj + i, None, lab) for i, lab in enumerate((1, 2))]
            res.append({"seed": 42, "subject": subj, "fold": 0, "records": recs, "stats": {}, "pid": 1, "max_rss_mb": 1})
        pub, _ = g3.analyse(res)
        self.assertAlmostEqual(pub["primary"]["mean_points"], 100 * 0.5 * ((.8 - .6) + (.7 - .5)))
        self.assertAlmostEqual(pub["secondary"]["S1_balanced_accuracy_contrast"]["mean_points"], 100.0)
        n = 0.5 * ((.6 - .55) + (.5 - .45))
        self.assertAlmostEqual(pub["secondary"]["S2_C1_same_run_minus_different_run_neighbour"]["mean_points"], 100 * n)
        self.assertAlmostEqual(pub["secondary"]["S3_C1_identity_minus_neighbour_contrast"]["mean_points"],
                               100 * (0.2 - n))
        self.assertAlmostEqual(pub["secondary"]["S4_C2_placebo_nontest_swap"]["mean_points"],
                               100 * 0.5 * ((.6 - .62) + (.52 - .5)))
        self.assertAlmostEqual(pub["secondary"]["S5_C4_multiset_base_identity_swap"]["mean_points"],
                               100 * 0.5 * ((.8 - .7) + (.7 - .6)))
        self.assertAlmostEqual(pub["descriptive"]["D2_C1_identity_vs_different_run_neighbour"]["mean_points"],
                               100 * 0.5 * ((.8 - .55) + (.7 - .45)))
        self.assertEqual(pub["secondary"]["S7_position_far_gt6"]["n_participants"], 0)

    def test_placebo_sign_flips_contribution(self):
        res = [{"seed": 42, "subject": s, "fold": 0, "stats": {}, "pid": 1, "max_rss_mb": 1,
                "records": [self.rec(10 * s + i, None, lab, sign=-1) for i, lab in enumerate((1, 2))]} for s in range(2)]
        pub, _ = g3.analyse(res)
        self.assertAlmostEqual(pub["secondary"]["S4_C2_placebo_nontest_swap"]["mean_points"],
                               -100 * 0.5 * ((.6 - .62) + (.52 - .5)))


class FailClosed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst, cls.dist, cls.plan = build(seeds=(42,))
        cls.y = cls.dist["label"].astype(np.int64)

    def test_nonfinite_covariance_stops_the_run(self):
        covs = synthetic_covs(self.dist)
        t = self.plan["seeds"][42]["tasks"][0]
        covs[t["base_canon"][0]] = np.nan
        with self.assertRaises(Exception):
            g3.execute(self.plan, covs, self.y, workers=1, log=lambda m: None, monitor=False)

    def test_identical_reference_between_paired_arms_stops(self):
        covs = synthetic_covs(self.dist)
        t = self.plan["seeds"][42]["tasks"][0]
        pr = t["pairs"][0]
        covs[pr["b"]] = covs[pr["a"]]
        one = copy.deepcopy(self.plan)
        one["seeds"] = {42: {"tasks": [dict(t, pairs=[pr], controls=False)], "n_pairs": 1}}
        with self.assertRaises(g3.G3Stop):
            g3.execute(one, covs, self.y, workers=1, log=lambda m: None, monitor=False)

    def test_stop_in_worker_reaches_parent(self):
        covs = synthetic_covs(self.dist)
        t = self.plan["seeds"][42]["tasks"][0]
        pr = t["pairs"][0]
        covs[pr["b"]] = covs[pr["a"]]
        one = copy.deepcopy(self.plan)
        one["seeds"] = {42: {"tasks": [dict(t, pairs=[pr], controls=False)] * 2, "n_pairs": 2}}
        with self.assertRaises(g3.G3Stop):
            g3.execute(one, covs, self.y, workers=2, log=lambda m: None, monitor=False)

    def test_exceptions_pickle(self):
        e = pickle.loads(pickle.dumps(g3.G3Stop("BLOCKED", "x")))
        self.assertEqual((e.status, e.message), ("BLOCKED", "x"))
        c = pickle.loads(pickle.dumps(ContractError("code", "msg")))
        self.assertEqual(c.code, "code")

    def test_plan_commitment_changes_with_any_selection(self):
        p2 = copy.deepcopy(self.plan)
        p2["seeds"][42]["tasks"][0]["pairs"][0]["g"] = 10 ** 6
        self.assertNotEqual(g3.plan_commitment(p2), self.plan["commitment"])
        p3 = copy.deepcopy(self.plan)
        p3["seeds"][42]["tasks"][0]["base_canon"] = p3["seeds"][42]["tasks"][0]["base_canon"][1:]
        self.assertNotEqual(g3.plan_commitment(p3), self.plan["commitment"])


class Preflight(unittest.TestCase):
    def test_covariance_checks(self):
        good = synthetic_covs({"label": np.array([1, 2] * 5)})
        self.assertTrue(g3.check_covariances(good, 10)["all_positive_definite"])
        bad_shape = good[:9]
        nan = good.copy(); nan[0, 0, 0] = np.nan
        asym = good.copy(); asym[0, 0, 1] += 1.0
        neg = good.copy(); neg[0] = -np.eye(11)
        for arr in (bad_shape, nan, asym, neg, good.astype(np.float32)):
            with self.assertRaises(g3.G3Stop):
                g3.check_covariances(arr, 10)

    def test_sha_verified_before_load(self):
        good = synthetic_covs({"label": np.array([1, 2] * 5)})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / g3.COVS_FILE
            np.save(path, good)
            from data_preparation.hs_integrity import sha256_file
            covs, info = g3.preflight_covariances(path, 10, sha256_file(path))
            self.assertTrue(info["sha256_verified_before_load"])
            with self.assertRaises(g3.G3Stop):
                g3.preflight_covariances(path, 10, "0" * 64)
            other = Path(d) / "other.npy"
            np.save(other, good)
            with self.assertRaises(g3.G3Stop):
                g3.preflight_covariances(other, 10, sha256_file(other))


class Guards(unittest.TestCase):
    def state(self, **kw):
        s = {"mem_available_mb": 20000, "swap_free_mb": 0, "psi_some_avg10": 0.0, "psi_full_avg10": 0.0,
             "loadavg_1": 5.2, "cpus": 10}
        s.update(kw)
        return s

    def test_start_and_run_guards(self):
        self.assertEqual(g3.start_guard(self.state()), 4)
        self.assertEqual(g3.start_guard(self.state(loadavg_1=8.5)), 1)
        self.assertEqual(g3.start_guard(self.state(loadavg_1=0.0)), 4)
        for bad in (self.state(mem_available_mb=3000), self.state(psi_full_avg10=2.0)):
            with self.assertRaises(g3.G3Stop) as ctx:
                g3.start_guard(bad)
            self.assertEqual(ctx.exception.status, "STOPPED_RESOURCE")
        g3.run_guard(self.state(mem_available_mb=2500))
        for bad in (self.state(mem_available_mb=1000), self.state(psi_full_avg10=6.0)):
            with self.assertRaises(g3.G3Stop):
                g3.run_guard(bad)

    def test_unfrozen_protocol_refused(self):
        with self.assertRaises(g3.G3Stop):
            g3.verify_frozen(ROOT, {"status": "DRAFT", "protocol_id": g3.PROTOCOL_ID})

    def test_runner_source_names_no_other_feature_or_prediction_file(self):
        for f in ("experiments/pmis_g2/g3.py", "experiments/run_g3.py"):
            src = (ROOT / f).read_text()
            for name in ("G2_eegnet_4_40_128hz", "G3_tslr_predictions_PRIVATE", "loadmat", "import torch", "cuda"):
                self.assertNotIn(name, src, (f, name))


if __name__ == "__main__":
    unittest.main()
