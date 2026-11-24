"""R6-C1 audit/decomposition code on SYNTHETIC metadata and SYNTHETIC SPD covariances only."""

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
import test_g3_runner_synthetic as t3  # noqa: E402
from experiments.pmis_g2 import g3  # noqa: E402
from models.tslr import arm_multiset, fit_arm  # noqa: E402
from models.vendor_r4 import r4_models as rm  # noqa: E402
from experiments.pmis_r6 import c1
from evaluation import r6_c1_analysis as c1_analysis, r6_indep_stats as indep_stats  # noqa: E402
from models.indep_tslr import IndepTSLR, karcher_mean, logreg_newton, tangent  # noqa: E402
from models.influence import BaseInfluence  # noqa: E402


def spd(n, p=11, seed=0, shift=0.3):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, p, 3 * p))
    c = a @ np.swapaxes(a, 1, 2) / (3 * p) + 0.1 * np.eye(p)
    y = np.where(np.arange(n) % 2 == 0, 1, 2)
    c[y == 2] += shift * np.diag(np.linspace(0.5, 1.5, p))
    return c, y


class IndependentImplementation(unittest.TestCase):
    def test_karcher_matches_r4(self):
        c, _ = spd(200)
        w = np.random.default_rng(1).integers(1, 4, 200)
        m_ind, it = karcher_mean(c, w)
        m_r4 = rm.riemann_mean_w(c, w)
        self.assertLess(np.linalg.norm(m_ind - m_r4) / np.linalg.norm(m_r4), 1e-7)
        self.assertLess(it, 500)

    def test_tangent_is_isometric_permutation_of_r4(self):
        c, _ = spd(30)
        ref = rm.riemann_mean_w(c, np.ones(30))
        a, b = tangent(c, ref), rm.tangent_space_b(c, ref)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_allclose(np.sort(a, axis=1), np.sort(b, axis=1), atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), rtol=1e-10)

    def test_newton_matches_sklearn_objective(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((400, 20))
        y = (x[:, 0] + 0.5 * rng.standard_normal(400) > 0).astype(float)
        th, _ = logreg_newton(x, y, np.ones(400))
        sk = LogisticRegression(C=1.0, max_iter=2000, tol=1e-10).fit(x, y)
        np.testing.assert_allclose(th[:-1], sk.coef_[0], atol=1e-5)
        self.assertAlmostEqual(th[-1], sk.intercept_[0], places=5)

    def test_weights_equal_expanded_rows(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((100, 5))
        y = (x[:, 0] > 0).astype(float)
        w = rng.integers(1, 4, 100)
        th_w, _ = logreg_newton(x, y, w.astype(float))
        rep = np.repeat(np.arange(100), w)
        th_e, _ = logreg_newton(x[rep], y[rep], np.ones(rep.size))
        np.testing.assert_allclose(th_w, th_e, atol=1e-9)

    def test_full_arm_matches_r4_tslrw(self):
        c, y = spd(300, seed=5)
        mult = np.random.default_rng(6).integers(1, 3, 300)
        ind = IndepTSLR().fit(c, y, mult)
        r4 = fit_arm(c, y, mult)
        cs, ys = spd(40, seed=7)
        p_r4 = c1.correct_class_probability(r4.clf.predict_proba(rm.tangent_space_b(cs, r4.ref)), r4.clf.classes_, ys)
        # R4 uses sklearn lbfgs at its default tol=1e-4: agreement only to solver tolerance ...
        np.testing.assert_allclose(ind.p_correct(cs, ys), p_r4, atol=2e-3)
        # ... while the independent implementation equals a tightly converged sklearn fit on R4's features
        f = rm.tangent_space_b(c, r4.ref)
        rep = np.repeat(np.arange(300), mult)
        tight = LogisticRegression(C=1.0, max_iter=100000, tol=1e-12).fit(f[rep], y[rep])
        p_t = c1.correct_class_probability(tight.predict_proba(rm.tangent_space_b(cs, r4.ref)), tight.classes_, ys)
        np.testing.assert_allclose(ind.p_correct(cs, ys), p_t, atol=1e-6)


class Influence(unittest.TestCase):
    def test_newton_step_predicts_fixed_reference_refit(self):
        c, y = spd(600, seed=8, shift=0.15)
        ref = rm.riemann_mean_w(c, np.ones(600))
        f = rm.tangent_space_b(c, ref)
        y01 = (y == 2).astype(float)
        base = np.arange(500)
        inf = BaseInfluence(f[base], y01[base], np.ones(500))
        za, zb = np.arange(500, 550), np.arange(550, 600)
        q = inf.predict_pair(f[za], y01[za], f[zb], y01[zb], k=1)
        errs, moves = [], []
        for i in range(20):
            ids = np.concatenate([base, [za[i]]])
            th, _ = logreg_newton(f[ids], y01[ids], np.ones(ids.size))
            pz = 1 / (1 + np.exp(-(f[za[i]] @ th[:-1] + th[-1])))
            pc = pz if y01[za[i]] == 1 else 1 - pz
            errs.append(abs(pc - q["nt_pAA"][i]))
            moves.append(abs(pc - q["pA0"][i]))
        self.assertLess(np.mean(errs), 0.1 * np.mean(moves))
        self.assertTrue(np.all(q["nt_pAA"] >= q["pA0"] - 1e-12))  # own copy never lowers own correct-class p

    def test_k_copies_monotone(self):
        c, y = spd(300, seed=9)
        ref = rm.riemann_mean_w(c, np.ones(300))
        f = rm.tangent_space_b(c, ref)
        y01 = (y == 2).astype(float)
        inf = BaseInfluence(f[:280], y01[:280], np.ones(280))
        prev = None
        for k in (1, 2, 4, 8):
            q = inf.predict_pair(f[280:290], y01[280:290], f[290:300], y01[290:300], k=k)
            if prev is not None:
                self.assertTrue(np.all(q["nt_pAA"] >= prev - 1e-12))
            prev = q["nt_pAA"]


class ArmK(unittest.TestCase):
    def test_arm_k(self):
        base = np.array([1, 3, 5, 7])
        ones = np.ones(4, dtype=np.int64)
        u1, c1_ = c1.arm_k(base, ones, 4, 1)
        u, cc = arm_multiset(base, ones, 4)
        np.testing.assert_array_equal(u1, u)
        np.testing.assert_array_equal(c1_, cc)
        u4, c4 = c1.arm_k(base, ones, 4, 4)
        self.assertEqual(int(c4[u4 == 4][0]), 4)
        self.assertEqual(int(c4.sum()), 8)
        with self.assertRaises(ValueError):
            c1.arm_k(base, ones, 3, 2)


class RunTaskSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst, cls.dist, cls.plan = t3.build(seeds=(42,))
        cls.covs = t3.synthetic_covs(cls.dist)
        cls.y = cls.dist["label"].astype(np.int64)
        g3.G["covs"], g3.G["y"] = cls.covs, cls.y
        c1.G["covs"], c1.G["y"] = cls.covs, cls.y
        cls.tasks = cls.plan["seeds"][42]["tasks"]
        cls.res = [c1.run_task(t) for t in cls.tasks]
        cls.g3res = [g3.run_task(t) for t in cls.tasks]

    def test_e1_reproduces_g3_arms_bitwise(self):
        for r, g in zip(self.res, self.g3res):
            for a, b in zip(r["records"], g["records"]):
                for q in ("pAA", "pAB", "pBA", "pBB"):
                    self.assertEqual(a[f"e1_{q}"], b[q])

    def test_one_audit_pair_per_task_and_close(self):
        for r, g in zip(self.res, self.g3res):
            aud = [x for x in r["records"] if x.get("audit") == 1.0]
            self.assertEqual(len(aud), 1)
            gb = {x["k"]: x for x in g["records"]}[aud[0]["k"]]
            for q in ("mAA", "mBB"):
                self.assertLess(abs(aud[0][f"i_{q}"] - gb[q]), 2e-3)
        for r, g in zip(self.res, self.g3res):
            for a, b in zip(r["records"], g["records"]):
                for q in ("pAA", "pAB", "pBA", "pBB"):
                    self.assertLess(abs(a[f"x1_{q}"] - b[q]), 2e-3)

    def test_optimum_arms_agree_across_implementations(self):
        t = self.tasks[0]
        pr = t["pairs"][0]
        ones = np.ones(t["base_canon"].size, dtype=np.int64)
        ab = np.array([pr["a"], pr["b"]])
        u, cnt = arm_multiset(t["base_canon"], ones, pr["a"])
        np.testing.assert_allclose(c1.opt_arm(u, cnt, ab), c1.indep(u, cnt, ab)[0], atol=1e-7)

    def test_audit_selection_is_outcome_blind(self):
        t = self.tasks[0]
        self.assertEqual(c1.audit_index(t), c1.audit_index(dict(t)))

    def test_analysis_schema(self):
        recs = [dict(x, subject=r["subject"], fold=r["fold"]) for r in self.res for x in r["records"]]
        stored = [dict(x, subject=g["subject"]) for g in self.g3res for x in g["records"]]
        out = c1_analysis.analyse(recs, stored)
        self.assertEqual(out["A2_reproduction_of_G3_arms"]["bitwise_equal"],
                         out["A2_reproduction_of_G3_arms"]["values_compared"])
        for k in ("exact", "fixed_reference", "reference_only", "influence_function", "newton_one_step"):
            self.assertIn("mean_points", out["M1_pathway_decomposition_k1"][k])
        self.assertIn("ratio_to_k1", out["M2_dose_response"]["k8"])


class IndependentStats(unittest.TestCase):
    def test_matches_frozen_aggregator(self):
        rng = np.random.default_rng(11)
        subj = np.repeat(np.arange(12), 20)
        lab = np.tile([1, 2], 120)
        val = rng.normal(0.01, 0.02, 240)
        mine = indep_stats.summary(subj, lab, val)
        ref = g3.summarise([g3.V(str(s), int(l), float(v)) for s, l, v in zip(subj, lab, val)])
        self.assertAlmostEqual(mine["mean_points"], ref["mean_points"], places=10)
        np.testing.assert_allclose(mine["ci95_points"], ref["ci95_points"], atol=1e-10)
        self.assertAlmostEqual(mine["exact_sign_flip_p"], ref["exact_sign_flip_p"], places=12)

    def test_sign_flip_floor(self):
        self.assertAlmostEqual(indep_stats.sign_flip_p(np.ones(14)), 2 / 2 ** 14, places=15)
        self.assertEqual(indep_stats.sign_flip_p(np.array([1.0, -1.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
