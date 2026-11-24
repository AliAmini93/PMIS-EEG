"""R10-A (synthetic only): final-epoch grouped trainer equals train_generic; ranking statistics; planted reversals."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from models.vendor_r4 import r4_stats as rs  # noqa: E402
from models import deep_group as dg  # noqa: E402
from evaluation import r10_analysis as analysis
from experiments.pmis_r10 import bench10
from models import hsstroke_deep as deep  # noqa: E402
from evaluation import r10_ranking as rk  # noqa: E402


class Trainer(unittest.TestCase):
    def test_final_equals_train_generic(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(0)
        X = torch.randn(80, 11, 256, generator=g)
        y = torch.arange(80) % 2
        X[y == 1, :3] *= 1.5
        rows = np.stack([np.arange(40), np.r_[np.arange(39), 41], np.arange(40)])
        score, val = np.arange(50, 70), np.arange(70, 80)
        ref = dg.train_generic(dg.GroupShallow, X, y, rows, np.zeros(3, dtype=np.int64), [5], 6, "cpu", score, val, epochs=2)
        m = deep.train_group_final(dg.GroupShallow, X, y, rows, np.zeros(3, dtype=np.int64), [5], 6, "cpu", epochs=2)
        p, c, _ = deep.correct_class(m, X, y, score)
        self.assertTrue(np.array_equal(p, ref["p_final"]))
        self.assertTrue(np.array_equal(p[2], p[0]))  # A/A within a group
        self.assertFalse(np.array_equal(p[1], p[0]))  # one changed row changes the model
        self.assertTrue(set(np.unique(c)) <= {0.0, 1.0})

    def test_part_seeds(self):
        a = bench10.seeds_of("SH1", (42, 0, 1))
        self.assertEqual(a, bench10.seeds_of("SH1", (42, 0, 1)))
        self.assertNotEqual(a, bench10.seeds_of("SH2", (42, 0, 1)))
        self.assertEqual(bench10.seeds_of("SH2", (42, 0, 1)), bench10.seeds_of("EG2", (42, 0, 1)))


def synth(n_part=14, n_trial=200, seeds=(42, 43, 44), acc=None, rng=0):
    """acc[model][arm] -> per-participant accuracy; returns c, p, dist, exposed, half with trials of both classes."""
    r = np.random.default_rng(rng)
    subj = np.repeat(np.arange(n_part), n_trial)
    y = np.tile(np.r_[np.ones(n_trial // 2), 2 * np.ones(n_trial // 2)], n_part).astype(np.int64)
    n = subj.size
    exposed = {s: (np.arange(n) % 3 != 0) for s in seeds}
    half = {s: np.where(exposed[s], 1 + (np.arange(n) % 2), 0) for s in seeds}
    c, p = {}, {}
    for m, arms in acc.items():
        c[m], p[m] = {}, {}
        for a, v in arms.items():
            c[m][a], p[m][a] = {}, {}
            for s in seeds:
                cc = (r.random(n) < np.asarray(v)[subj]).astype(np.float64)
                c[m][a][s], p[m][a][s] = cc, 0.25 + 0.5 * cc
    return c, p, {"subject": subj, "label": y}, exposed, half


def arms_of(base, infl, n=14, noise=0.0, rng=1):
    r = np.random.default_rng(rng)
    b = base + noise * r.standard_normal(n)
    return {"C": b, "P": b, "L1": b + infl / 2, "L2": b + infl / 2, "L": b + infl}


class Ranking(unittest.TestCase):
    def test_ba_matches_r4(self):
        r = np.random.default_rng(3)
        y = r.integers(1, 3, 300)
        pred = r.integers(1, 3, 300)
        subj = np.zeros(300, dtype=int)
        self.assertAlmostEqual(rk.ba_participants((pred == y).astype(float), y, subj, np.array([0]))[0],
                               100 * rs.balanced_accuracy(y, pred), places=12)

    def test_reversal_classes(self):
        g = lambda m, lo, hi: {"mean_pp": m, "ci95_pp": [lo, hi]}  # noqa: E731
        self.assertEqual(rk.reversal_class(g(2, 1, 3), g(3, 2, 4), {}, 0.001), "NO_REVERSAL")
        self.assertEqual(rk.reversal_class(g(-2, -3, -1), g(3, 2, 4), {}, 0.2), "DESCRIPTIVE_REVERSAL")
        self.assertEqual(rk.reversal_class(g(-2, -3, 0.5), g(3, 2, 4), {}, 0.001), "SUPPORTED_CROSSING")
        self.assertEqual(rk.reversal_class(g(-2, -3, -1), g(3, 2, 4), {}, 0.001), "SUPPORTED_REVERSAL")
        self.assertEqual(rk.reversal_class(g(-2, -3, -1), g(3, 2, 4), {}, None), "DESCRIPTIVE_REVERSAL")

    def test_selection_and_kendall(self):
        BA = {"A": {"L": np.full(5, 80.0), "P": np.full(5, 60.0)}, "B": {"L": np.full(5, 70.0), "P": np.full(5, 65.0)}}
        s = rk.selection(BA, "L", "P", ("A", "B"))
        self.assertEqual((s["winner_select_arm"], s["winner_eval_arm"]), ("A", "B"))
        self.assertAlmostEqual(s["regret_cohort_mean_pp"], 5.0)
        self.assertEqual(rk.kendall_distance(["A", "B", "C"], ["C", "B", "A"]), 3)
        self.assertEqual(rk.kendall_distance(["A", "B", "C"], ["A", "B", "C"]), 0)

    def test_f1(self):
        y = np.array([1, 1, 2, 2])
        self.assertAlmostEqual(analysis._f1(np.array([1.0, 1, 1, 1]), y, np.ones(4, bool)), 100.0)
        self.assertAlmostEqual(analysis._f1(np.array([1.0, 0, 1, 0]), y, np.ones(4, bool)), 50.0)


class Planted(unittest.TestCase):
    def run_case(self, acc):
        c, p, dist, exposed, half = synth(acc=acc, n_trial=2000)
        # stitched arms need L1/L2 to differ for own/other: L1 exposes half 1, L2 half 2 (own > other)
        out, _ = analysis.analyse_replicate(c, p, dist, exposed, half, (42, 43, 44), 500, 1)
        return out

    def test_planted_supported_top1_reversal(self):
        acc = {"TSLR": arms_of(0.60, 0.02, noise=0.02, rng=1), "EEGNet": arms_of(0.72, 0.08, noise=0.02, rng=2),
               "Shallow": arms_of(0.68, 0.20, noise=0.02, rng=3)}
        out = self.run_case(acc)
        self.assertEqual(out["top1"]["P"], "EEGNet")
        self.assertEqual(out["top1"]["L"], "Shallow")
        self.assertEqual(out["H2_reversal"]["Shallow_vs_EEGNet"]["P_to_L"], "SUPPORTED_REVERSAL")
        self.assertTrue(out["H2_top1"]["P_to_L"]["supported"])
        self.assertFalse(out["H3_selection"]["select_L_evaluate_P"]["same_winner"])
        self.assertTrue(out["H3_selection"]["select_L_evaluate_P"]["regret_ge_sesoi_2pp"])
        self.assertLess(out["H1_family_D_L_minus_P"]["Shallow_vs_EEGNet"]["holm_p_family"], 0.05)

    def test_planted_no_reversal(self):
        acc = {"TSLR": arms_of(0.60, 0.02, noise=0.02, rng=1), "EEGNet": arms_of(0.75, 0.08, noise=0.02, rng=2),
               "Shallow": arms_of(0.66, 0.02, noise=0.02, rng=3)}
        out = self.run_case(acc)
        self.assertEqual(out["order"]["P"], out["order"]["L"])
        for k in out["H2_reversal"]:
            self.assertEqual(out["H2_reversal"][k]["P_to_L"], "NO_REVERSAL")
        self.assertTrue(out["H3_selection"]["select_L_evaluate_P"]["same_winner"])
        self.assertEqual(out["H3_selection"]["select_L_evaluate_P"]["regret_cohort_mean_pp"], 0.0)

    def test_null_inflation_is_not_supported(self):
        acc = {"TSLR": arms_of(0.60, 0.0, noise=0.02, rng=1), "EEGNet": arms_of(0.70, 0.0, noise=0.02, rng=2),
               "Shallow": arms_of(0.65, 0.0, noise=0.02, rng=3)}
        out = self.run_case(acc)
        for k, v in out["H1_family_D_L_minus_P"].items():
            self.assertLess(abs(v["mean_pp"]), 2.0)
        self.assertIn(out["H4_attribution"]["Shallow_vs_EEGNet"]["distortion_class"],
                      ("NO_SUPPORTED_DISTORTION", "NOT_ATTRIBUTABLE_TO_EXPOSURE", "MIXED", "EXPOSURE_DOMINANT_AT_HALF_DENSITY"))


if __name__ == "__main__":
    unittest.main()
