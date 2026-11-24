"""Vendored frozen R4 code integrity and PMIS TS+LR fit mechanics. Synthetic SPD matrices only."""

import sys
import unittest
from pathlib import Path

import numpy as np

from data_preparation import hs_integrity as provenance  # noqa: E402
from models.tslr import arm_multiset, fit_arm, predict_correct_proba  # noqa: E402
from models.vendor_r4 import r4_models as rm  # noqa: E402


def covs(n, p=11, seed=0, shift=0.0, y=None):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, p, 3 * p))
    c = a @ np.swapaxes(a, 1, 2) / (3 * p) + 0.1 * np.eye(p)
    if y is not None:
        c[np.asarray(y) == 2] += shift * np.diag(np.linspace(0.5, 1.5, p))
    return c


class Vendor(unittest.TestCase):
    def test_vendored_files_match_release_digests(self):
        vendor = Path(rm.__file__).parent
        for name in ("r4_models.py", "r4_stats.py"):
            self.assertEqual(provenance.sha256_file(vendor / name), provenance.FROZEN_R4_CODE_SHA256[name])

    def test_weighted_mean_equals_expanded_mean(self):          # R4 test_r4_g3, re-run on the vendored copy
        c = covs(12, p=5)
        w = np.array([1, 3, 2, 1, 1, 4, 1, 2, 1, 1, 2, 1])
        self.assertTrue(np.allclose(rm.riemann_mean(np.repeat(c, w, axis=0)), rm.riemann_mean_w(c, w), atol=1e-9))

    def test_batched_tangent_space_equals_loop(self):
        c = covs(7, p=5, seed=1)
        ref = rm.riemann_mean(c)
        self.assertTrue(np.allclose(rm.tangent_space(c, ref), rm.tangent_space_b(c, ref), atol=1e-12))

    def test_tslrw_equals_tslr_on_expanded_rows(self):
        y = np.array([1, 2] * 20)
        c = covs(40, p=5, seed=2, shift=0.05, y=y)
        w = np.random.default_rng(3).integers(1, 4, size=40)
        a = rm.TSLRw().fit(c, y, w)
        b = rm.TSLR().fit(np.repeat(c, w, axis=0), np.repeat(y, w))
        self.assertTrue(np.allclose(a.ref, b.ref, atol=1e-9))
        t = covs(10, p=5, seed=4)
        self.assertTrue(np.array_equal(a.predict(t), b.predict(t)))


class ArmMechanics(unittest.TestCase):
    def setUp(self):
        n = 120
        self.y = np.array([1, 2] * (n // 2))
        self.c = covs(n, seed=5, shift=0.2, y=self.y)
        self.base = np.arange(100)
        self.mult = np.random.default_rng(6).integers(1, 4, size=100)
        self.a, self.b = 100, 102           # same class (label 1)
        self.test = np.arange(100, 120)

    def fit(self, alias, mult=None):
        u, cnt = arm_multiset(self.base, self.mult if mult is None else mult, alias)
        return fit_arm(self.c[u], self.y[u], cnt), u, cnt

    def test_arm_multiset_adds_one_row_and_one_identity(self):
        u, cnt = arm_multiset(self.base, self.mult, self.a)
        self.assertEqual((u.size, int(cnt.sum())), (101, int(self.mult.sum()) + 1))
        with self.assertRaises(ValueError):
            arm_multiset(self.base, self.mult, 5)      # alias already in base: support would not grow

    def test_reference_is_rebuilt_per_arm(self):
        ma, ua, ca = self.fit(self.a)
        mb, ub, cb = self.fit(self.b)
        self.assertIsNot(ma.ref, mb.ref)
        self.assertFalse(np.allclose(ma.ref, mb.ref, atol=1e-13))
        self.assertTrue(np.allclose(ma.ref, rm.riemann_mean_w(self.c[ua], ca), atol=0))
        self.assertTrue(np.allclose(mb.ref, rm.riemann_mean_w(self.c[ub], cb), atol=0))
        base_ref = rm.riemann_mean_w(self.c[self.base], self.mult)
        self.assertGreater(np.abs(ma.ref - base_ref).max(), 0.0)    # base-reference shortcut != exact arm
        self.assertIsNot(ma.clf, mb.clf)

    def test_fit_is_deterministic_so_model_randomness_is_fixed_within_pairs(self):
        p1 = predict_correct_proba(self.fit(self.a)[0], self.c[self.test], self.y[self.test])
        p2 = predict_correct_proba(self.fit(self.a)[0], self.c[self.test], self.y[self.test])
        self.assertTrue(np.array_equal(p1, p2))

    def test_correct_probability_consistent_with_frozen_predict(self):
        m = self.fit(self.a)[0]
        p = predict_correct_proba(m, self.c[self.test], self.y[self.test])
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        correct = m.predict(self.c[self.test]) == self.y[self.test]
        self.assertTrue(np.array_equal(correct[p != 0.5], (p > 0.5)[p != 0.5]))

    def test_pair_score_is_symmetric_in_pair_order(self):
        from evaluation.g2_estimands import PairedScore
        ma, mb = self.fit(self.a)[0], self.fit(self.b)[0]
        t = np.array([self.a, self.b])
        pa, pb = (predict_correct_proba(m, self.c[t], self.y[t]) for m in (ma, mb))
        ab = PairedScore("p", 1, pa[0], pb[0], pa[1], pb[1]).identity_swap_gain
        ba = PairedScore("p", 1, pb[1], pa[1], pb[0], pa[0]).identity_swap_gain
        self.assertAlmostEqual(ab, ba, places=15)

    def test_single_class_arm_refused(self):
        with self.assertRaises(ValueError):
            fit_arm(self.c[:10:2], self.y[:10:2], np.ones(5, dtype=int))


if __name__ == "__main__":
    unittest.main()
