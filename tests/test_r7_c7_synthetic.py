"""R7-C7 code on a SYNTHETIC LEAKED / PLACEBO / CLEAN manifest and SYNTHETIC SPD covariances only."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from models.vendor_r4 import r4_models as rm  # noqa: E402
from experiments.pmis_r7 import c7
from evaluation import r7_c7_analysis as c7_analysis  # noqa: E402


def synth(n_units=2, seed=0):
    rng = np.random.default_rng(seed)
    n = 220 * n_units
    a = rng.standard_normal((n, 11, 33))
    covs = a @ np.swapaxes(a, 1, 2) / 33 + 0.1 * np.eye(11)
    label = np.where(np.arange(n) % 2 == 0, 1, 2)
    covs[label == 2] += 0.25 * np.diag(np.linspace(0.5, 1.5, 11))
    rows, table, subject = [], [], np.repeat(np.arange(n_units), 220)
    exposed = np.zeros(n, dtype=bool)
    for u in range(n_units):
        o = 220 * u
        T = np.arange(o, o + 40)
        C = np.arange(o + 40, o + 220)
        E = np.repeat(np.arange(o + 40, o + 80), 2)
        X = np.repeat(T[:20], 3)               # 20 exposed test trials, 3 copies each
        Z = np.repeat(np.arange(o + 80, o + 100), 3)
        exposed[T[:20]] = True
        arms = {"T": T, "L": np.r_[C, E, X], "C": C, "P": np.r_[C, E, Z], "LCB": C[:-20], "TEST_INST": np.r_[T, T[:20]]}
        for ai, nm in enumerate(c7.ARMS):
            start = sum(len(r) for r in rows)
            rows.append(arms[nm])
            table.append([2, 42, u, 0, ai, start, start + len(arms[nm])])
    return covs, label, np.concatenate(rows), np.array(table), subject, exposed


class C7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.covs, cls.label, rows, table, cls.subject, cls.exposed = synth()
        c7.G.update({"covs": cls.covs, "label": cls.label, "rows": rows, "table": table})
        cls.res = [c7.run_task((2, 42, u)) for u in range(2)]

    def test_r4_path_equals_direct_fit(self):
        f = self.res[0]["folds"][0]
        T = f["T"]
        sel = c7.G["table"][(c7.G["table"][:, 2] == 0)]
        u, cnt = np.unique(c7._get(sel, "L"), return_counts=True)
        m = rm.TSLRw().fit(self.covs[u], self.label[u], cnt)
        np.testing.assert_array_equal(f["r4_pred_L"], m.predict(self.covs[T]).astype(np.int8))

    def test_law_tracks_exact_leaked(self):
        f = self.res[0]["folds"][0]
        dE = f["opt_p_L"] - f["opt_p_P"]
        dL = f["law_p_L"] - f["opt_p_P"]
        self.assertGreater(np.corrcoef(dE, dL)[0, 1], 0.9)
        self.assertLess(abs(dL.mean() / dE.mean() - 1), 0.3)
        self.assertGreater(dE[:20].mean(), dE[20:].mean())  # exposed trials gain more

    def test_multiplicity(self):
        f = self.res[0]["folds"][0]
        np.testing.assert_array_equal(f["mult_in_L"], np.r_[np.full(20, 3), np.zeros(20)])

    def test_assemble_and_analysis(self):
        A = c7_analysis.assemble(self.res, self.label.size)
        r4_pred = {"S2_42_L": A[(2, 42)]["r4_pred_L"], "S2_42_P": A[(2, 42)]["r4_pred_P"], "S2_42_C": A[(2, 42)]["r4_pred_C"]}
        saved = (c7_analysis.SPLITS, c7_analysis.SEEDS)
        c7_analysis.SPLITS, c7_analysis.SEEDS = {2: "S2"}, (42,)
        try:
            dist = {"subject": self.subject, "label": self.label}
            man = {"exposed_S2_42": self.exposed}
            pub = {"splits": {"S2": {k: {"mean_pp": 0.0} for k in ("primary_LEAKED_minus_PLACEBO", "PLACEBO_minus_CLEAN",
                                                                   "LEAKED_minus_CLEAN", "exposed_only_LEAKED_minus_PLACEBO",
                                                                   "unexposed_only_LEAKED_minus_PLACEBO")}}}
            out = c7_analysis.analyse(A, dist, man, r4_pred, pub)
        finally:
            c7_analysis.SPLITS, c7_analysis.SEEDS = saved
        o = out["S2"]
        self.assertTrue(o["R1_retrospective_reproduction_of_R4"]["all_hard_labels_equal"])
        self.assertIn("met", o["L1_law_benchmark_level"])
        self.assertNotIn("sorted_values_pp_unlabelled", o["probability_inflation_LEAKED_minus_PLACEBO_points"]["all_scored"]["exact_optimum"])
        self.assertAlmostEqual(o["accounting"]["fraction_of_scored_trials_exposed"], 0.5)


if __name__ == "__main__":
    unittest.main()
