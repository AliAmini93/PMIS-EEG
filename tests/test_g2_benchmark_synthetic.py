"""Tiny synthetic run of the benchmark helpers (timing values are not asserted)."""

import sys
import unittest
from pathlib import Path

import numpy as np

from experiments.pmis_g2 import benchmark as bm  # noqa: E402


class BenchmarkHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fold = bm.make_fold(n_distinct=60, n_rows=130, n_test=20, separation=0.15, seed=1)

    def test_fold_shapes_and_spd(self):
        f = self.fold
        self.assertEqual((f.n_distinct, f.n_rows, f.test.size), (60, 130, 20))
        self.assertEqual(int(f.mult.sum()), 130)
        self.assertTrue(np.all(np.linalg.eigvalsh(f.covs) > 0))
        self.assertTrue(all(f.y[a] == f.y[b] and a != b for a, b in f.pairs))

    def test_pair_run_rebuilds_reference(self):
        r = bm.run_pair(self.fold, *self.fold.pairs[0])
        self.assertTrue(r["reference_rebuilt"] and r["finite"])
        self.assertEqual(len(r["wall"]), 2)

    def test_verification_flags(self):
        v = bm.verification_and_components(self.fold, *self.fold.pairs[0])
        self.assertTrue(v["replica_bit_equal_to_vendored_TSLRw_fit"])
        self.assertTrue(v["reference_equals_fresh_weighted_karcher_mean_arm_A"])
        self.assertTrue(v["reference_equals_fresh_weighted_karcher_mean_arm_B"])
        self.assertGreater(v["reference_A_differs_from_B_max_abs"], 0.0)
        self.assertGreater(v["reference_A_differs_from_base_only_max_abs"], 0.0)
        self.assertFalse(v["shortcut_fixed_base_reference_DIFFERENT_ESTIMATOR"]["used"])
        self.assertGreaterEqual(v["karcher_iterations"], 1)

    def test_impossible_multiplicity_refused(self):
        with self.assertRaises(ValueError):
            bm.make_fold(n_distinct=10, n_rows=9, n_test=4, separation=0.0, seed=0)


if __name__ == "__main__":
    unittest.main()
